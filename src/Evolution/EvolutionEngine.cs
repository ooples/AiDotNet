using System.Diagnostics;
using System.Globalization;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>
/// Orchestrates generic quality-diversity evolution with deterministic identity, bounded workers, islands,
/// evaluation caching, failure isolation, and optional checkpoint/resume.
/// </summary>
/// <typeparam name="TGenome">The task-specific immutable genome type.</typeparam>
public sealed partial class EvolutionEngine<TGenome>
{
    private readonly IEvolutionTask<TGenome> _task;
    private readonly IVariationOperator<TGenome> _variation;
    private readonly ICandidateRefiner<TGenome>? _refiner;
    private readonly ISelectionPolicy<TGenome> _selection;
    private readonly IMigrationPolicy<TGenome> _migration;
    private readonly IEvolutionObserver<TGenome>? _observer;
    private readonly IEvolutionCheckpointStore? _checkpointStore;
    private readonly IEvolutionGenomeCodec<TGenome>? _codec;
    private readonly EvolutionEngineOptions _options;
    private readonly IEvolutionArchive<TGenome>[] _islands;
    private readonly HashSet<string> _seen = new(StringComparer.Ordinal);
    private readonly Dictionary<string, EvolutionTaskResult> _cache = new(StringComparer.Ordinal);
    private readonly Dictionary<EvolutionEvaluationStatus, long> _statusCounts = new();
    private readonly Queue<EvolutionDiagnostic> _failures = new();
    private readonly string _configurationHash;
    private readonly string _compatibilityHash;
    private long _nextEvaluationId;
    private long _proposals;
    private long _evaluationAttempts;
    private long _completedEvaluations;
    private long _generation;
    private long _eventSequence;
    private int _batchesSinceMigration;
    private long _commitsSinceCheckpoint;
    private int _runStarted;

    /// <summary>Initializes an evolution engine and its independent island archives.</summary>
    /// <param name="task">Task-specific canonicalization and evaluation.</param>
    /// <param name="variation">Task-specific variation.</param>
    /// <param name="archiveFactory">Creates a distinct empty archive for each island index.</param>
    /// <param name="options">Run options, defensively copied during construction.</param>
    /// <param name="selection">Optional parent/inspiration policy.</param>
    /// <param name="refiner">Optional immutable inner optimizer.</param>
    /// <param name="migration">Optional island migration policy.</param>
    /// <param name="observer">Optional structured observer.</param>
    /// <param name="checkpointStore">Optional checkpoint store.</param>
    /// <param name="genomeCodec">Required when checkpointing or resume is enabled.</param>
    public EvolutionEngine(
        IEvolutionTask<TGenome> task,
        IVariationOperator<TGenome> variation,
        Func<int, IEvolutionArchive<TGenome>> archiveFactory,
        EvolutionEngineOptions options,
        ISelectionPolicy<TGenome>? selection = null,
        ICandidateRefiner<TGenome>? refiner = null,
        IMigrationPolicy<TGenome>? migration = null,
        IEvolutionObserver<TGenome>? observer = null,
        IEvolutionCheckpointStore? checkpointStore = null,
        IEvolutionGenomeCodec<TGenome>? genomeCodec = null)
    {
        Guard.NotNull(task);
        Guard.NotNull(variation);
        Guard.NotNull(archiveFactory);
        Guard.NotNull(options);
        ValidateComponent(task.Id, task.VersionHash, nameof(task));
        ValidateComponent(task.Id, task.EvaluatorVersionHash, nameof(task));
        ValidateComponent(variation.Id, variation.VersionHash, nameof(variation));

        _options = options.SnapshotAndValidate();
        _task = task;
        _variation = variation;
        _selection = selection ?? new UniformEvolutionSelectionPolicy<TGenome>();
        _refiner = refiner;
        _migration = migration ?? new RingMigrationPolicy<TGenome>();
        _observer = observer;
        _checkpointStore = checkpointStore;
        _codec = genomeCodec;
        ValidateComponent(_selection.Id, _selection.VersionHash, nameof(selection));
        ValidateComponent(_migration.Id, _migration.VersionHash, nameof(migration));
        if (_refiner is not null) ValidateComponent(_refiner.Id, _refiner.VersionHash, nameof(refiner));
        if (_codec is not null) ValidateComponent(_codec.Id, _codec.VersionHash, nameof(genomeCodec));
        if ((_checkpointStore is not null || _options.Resume) && _codec is null)
            throw new ArgumentException("A genome codec is required for checkpointing and resume.", nameof(genomeCodec));
        if (_options.Resume && _checkpointStore is null)
            throw new ArgumentException("Resume requires a checkpoint store.", nameof(checkpointStore));

        _islands = new IEvolutionArchive<TGenome>[_options.IslandCount];
        for (int i = 0; i < _islands.Length; i++)
        {
            _islands[i] = archiveFactory(i) ?? throw new ArgumentException("The archive factory returned null.", nameof(archiveFactory));
            if (_islands[i].Count != 0) throw new ArgumentException("New engine archives must be empty; use checkpoint import for prior state.", nameof(archiveFactory));
            for (int prior = 0; prior < i; prior++)
                if (ReferenceEquals(_islands[prior], _islands[i]))
                    throw new ArgumentException("The archive factory must return independent instances.", nameof(archiveFactory));
        }
        ValidateCompatibleArchives(_islands);

        string archiveDefinition = CanonicalArchiveDefinition(_islands[0]);
        _configurationHash = EvolutionHash.Compute(_options.ToCanonicalString());
        _compatibilityHash = EvolutionHash.Combine(new[]
        {
            EvolutionCheckpoint.CurrentSchemaVersion.ToString(CultureInfo.InvariantCulture),
            StableRandom.AlgorithmId,
            _task.Id, _task.VersionHash, _task.EvaluatorVersionHash,
            _variation.Id, _variation.VersionHash,
            _selection.Id, _selection.VersionHash,
            _refiner?.Id ?? "none", _refiner?.VersionHash ?? "none",
            _migration.Id, _migration.VersionHash,
            _codec?.Id ?? "none", _codec?.VersionHash ?? "none",
            archiveDefinition,
            _configurationHash
        });
    }

    /// <summary>Gets the checkpoint compatibility hash for this exact engine configuration.</summary>
    public string CompatibilityHash => _compatibilityHash;

    /// <summary>Runs evolution once using a finite initial seed set.</summary>
    /// <param name="initialGenomes">Finite task-specific seed genomes.</param>
    /// <param name="cancellationToken">Cancellation propagated through proposals, refinement, and evaluation.</param>
    public async Task<EvolutionRunResult<TGenome>> RunAsync(
        IEnumerable<TGenome> initialGenomes,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(initialGenomes);
        if (Interlocked.Exchange(ref _runStarted, 1) != 0)
            throw new InvalidOperationException("An EvolutionEngine instance can be run only once.");

        TGenome[] seeds = MaterializeSeeds(initialGenomes);
        int seedIndex = 0;
        if (_options.Resume)
        {
            RestoredSeeds restored = await RestoreCheckpointAsync(seeds, cancellationToken).ConfigureAwait(false);
            seeds = restored.Seeds;
            seedIndex = restored.SeedIndex;
        }
        CaptureSafeState(seeds, seedIndex);

        var runTimer = Stopwatch.StartNew();
        using var runCancellation = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        if (_options.TimeLimit.HasValue) runCancellation.CancelAfter(_options.TimeLimit.Value);
        CancellationToken runToken = runCancellation.Token;
        EvolutionStopReason stopReason;
        try
        {
            try
            {
                stopReason = await RunLoopAsync(seeds, seedIndex, runTimer, runToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested && runCancellation.IsCancellationRequested)
            {
                stopReason = EvolutionStopReason.TimeLimitReached;
            }
            await SaveCheckpointAsync(force: true,
                runCancellation.IsCancellationRequested ? CancellationToken.None : cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            await SaveCheckpointAsync(force: true, CancellationToken.None).ConfigureAwait(false);
            await NotifyAsync(new EvolutionEvent<TGenome>(EvolutionEventKind.Stopped, NextEventSequence(),
                message: EvolutionStopReason.Canceled.ToString()), CancellationToken.None).ConfigureAwait(false);
            throw;
        }

        string stateHash = ComputeStateHash();
        await NotifyAsync(new EvolutionEvent<TGenome>(EvolutionEventKind.Stopped, NextEventSequence(),
            message: stopReason.ToString()),
            runCancellation.IsCancellationRequested ? CancellationToken.None : cancellationToken).ConfigureAwait(false);
        return new EvolutionRunResult<TGenome>(stopReason, Array.AsReadOnly(_islands), CreateCounters(), stateHash);
    }

    private async Task<EvolutionStopReason> RunLoopAsync(TGenome[] seeds, int seedIndex, Stopwatch runTimer,
        CancellationToken cancellationToken)
    {
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            EvolutionStopReason? limit = GetLimitStopReason(runTimer);
            if (limit.HasValue) return limit.Value;

            BatchTransaction transaction = CaptureBatchTransaction();
            var batch = new List<WorkItem>(Math.Min(_options.ProposalBatchSize, 1024));
            try
            {
                while (batch.Count < _options.ProposalBatchSize)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    limit = GetLimitStopReason(runTimer);
                    if (limit.HasValue) break;

                    PreparedProposal? prepared;
                    if (seedIndex < seeds.Length)
                    {
                        prepared = await PrepareSeedAsync(seeds[seedIndex++], cancellationToken).ConfigureAwait(false);
                    }
                    else
                    {
                        if (_generation >= _options.MaxGenerations) break;
                        prepared = await PrepareVariationAsync(cancellationToken).ConfigureAwait(false);
                        if (prepared is null) break;
                    }

                    if (prepared is not null) batch.Add(prepared.Item);
                    if (_evaluationAttempts + batch.Count(item => item.RequiresEvaluation) >= _options.MaxEvaluationAttempts)
                        break;
                }

                if (batch.Count > 0)
                {
                    await EvaluateBatchAsync(batch, cancellationToken).ConfigureAwait(false);
                    cancellationToken.ThrowIfCancellationRequested();
                }
            }
            catch (OperationCanceledException)
            {
                RestoreBatchTransaction(transaction, batch);
                throw;
            }

            if (batch.Count == 0)
            {
                if (_evaluationAttempts >= _options.MaxEvaluationAttempts) return EvolutionStopReason.EvaluationBudgetReached;
                if (_proposals >= _options.MaxProposals) return EvolutionStopReason.ProposalBudgetReached;
                if (_generation >= _options.MaxGenerations) return EvolutionStopReason.GenerationLimitReached;
                return EvolutionStopReason.NoCandidates;
            }

            // A completed logical batch is one transaction: cancellation is observed before or after, never mid-commit.
            bool failedFast = await CommitBatchAsync(batch, CancellationToken.None).ConfigureAwait(false);
            if (_options.MigrationInterval > 0 && _islands.Length > 1)
            {
                _batchesSinceMigration++;
                await MigrateIfDueAsync(CancellationToken.None).ConfigureAwait(false);
            }
            CaptureSafeState(seeds, seedIndex);
            await SaveCheckpointAsync(force: false,
                cancellationToken.IsCancellationRequested ? CancellationToken.None : cancellationToken).ConfigureAwait(false);
            cancellationToken.ThrowIfCancellationRequested();
            if (failedFast) return EvolutionStopReason.CandidateFailure;
        }
    }

    private EvolutionStopReason? GetLimitStopReason(Stopwatch timer)
    {
        if (_evaluationAttempts >= _options.MaxEvaluationAttempts) return EvolutionStopReason.EvaluationBudgetReached;
        if (_proposals >= _options.MaxProposals) return EvolutionStopReason.ProposalBudgetReached;
        if (_options.TimeLimit.HasValue && timer.Elapsed >= _options.TimeLimit.Value) return EvolutionStopReason.TimeLimitReached;
        return null;
    }

    private TGenome[] MaterializeSeeds(IEnumerable<TGenome> initialGenomes)
    {
        var result = new List<TGenome>();
        using (IEnumerator<TGenome> enumerator = initialGenomes.GetEnumerator())
        {
            while (enumerator.MoveNext())
            {
                if (result.Count >= _options.MaxProposals)
                    throw new ArgumentException("Initial seed count exceeds MaxProposals.", nameof(initialGenomes));
                if (enumerator.Current is null) throw new ArgumentException("Initial genomes cannot contain null values.", nameof(initialGenomes));
                result.Add(enumerator.Current);
            }
        }
        return result.ToArray();
    }

    private EvolutionRunCounters CreateCounters() => new(_proposals, _evaluationAttempts, _completedEvaluations, _statusCounts);

    private long NextEventSequence() => _eventSequence++;

    private async ValueTask NotifyAsync(EvolutionEvent<TGenome> evolutionEvent, CancellationToken cancellationToken)
    {
        if (_observer is null) return;
        try
        {
            await _observer.OnEventAsync(evolutionEvent, cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception)
        {
            // Observer failures are deliberately isolated from deterministic engine state.
        }
    }

    private void RetainFailure(EvolutionDiagnostic diagnostic)
    {
        while (_failures.Count >= _options.MaxRetainedFailures) _failures.Dequeue();
        _failures.Enqueue(diagnostic);
    }

    private static void ValidateComponent(string id, string versionHash, string parameterName)
    {
        if (string.IsNullOrWhiteSpace(id)) throw new ArgumentException("Component IDs cannot be empty.", parameterName);
        if (string.IsNullOrWhiteSpace(versionHash)) throw new ArgumentException("Component version hashes cannot be empty.", parameterName);
    }

    private static void ValidateCompatibleArchives(IReadOnlyList<IEvolutionArchive<TGenome>> archives)
    {
        string expected = CanonicalArchiveDefinition(archives[0]);
        for (int i = 1; i < archives.Count; i++)
            if (!string.Equals(expected, CanonicalArchiveDefinition(archives[i]), StringComparison.Ordinal))
                throw new ArgumentException("Every island archive must have an identical definition.", nameof(archives));
    }

    private static string CanonicalArchiveDefinition(IEvolutionArchive<TGenome> archive)
    {
        if (string.IsNullOrWhiteSpace(archive.DefinitionHash))
            throw new ArgumentException("Archive definition hashes cannot be empty.", nameof(archive));
        return archive.DefinitionHash.Trim();
    }

    private sealed class PreparedProposal
    {
        public PreparedProposal(WorkItem item) => Item = item;
        public WorkItem Item { get; }
    }

    private sealed class RestoredSeeds
    {
        public RestoredSeeds(TGenome[] seeds, int seedIndex)
        {
            Seeds = seeds;
            SeedIndex = seedIndex;
        }

        public TGenome[] Seeds { get; }
        public int SeedIndex { get; }
    }

    private sealed class WorkItem
    {
        public long EvaluationId { get; set; }
        public int Island { get; set; }
        public EvolutionLineage Lineage { get; set; } = null!;
        public EvolutionCandidate<TGenome>? Candidate { get; set; }
        public EvolutionTaskResult? Result { get; set; }
        public EvolutionCacheStatus CacheStatus { get; set; }
        public bool RequiresEvaluation { get; set; }
        public int AttemptCount { get; set; }
        public TimeSpan Elapsed { get; set; }
        public double AccumulatedCostUnits { get; set; }
        public List<EvolutionDiagnostic> AttemptDiagnostics { get; } = new();
        public long CompletionOrder { get; set; }
        public bool AddedToSeen { get; set; }
    }

    private sealed class BatchTransaction
    {
        public long NextEvaluationId { get; set; }
        public long Proposals { get; set; }
        public long EvaluationAttempts { get; set; }
        public long Generation { get; set; }
        public long EventSequence { get; set; }
        public long CompletionSequence { get; set; }
        public EvolutionDiagnostic[] Failures { get; set; } = Array.Empty<EvolutionDiagnostic>();
    }

    private BatchTransaction CaptureBatchTransaction() => new()
    {
        NextEvaluationId = _nextEvaluationId,
        Proposals = _proposals,
        EvaluationAttempts = _evaluationAttempts,
        Generation = _generation,
        EventSequence = _eventSequence,
        CompletionSequence = _completionSequence,
        Failures = _failures.ToArray()
    };

    private void RestoreBatchTransaction(BatchTransaction transaction, IEnumerable<WorkItem> batch)
    {
        foreach (WorkItem item in batch)
            if (item.AddedToSeen && item.Candidate is not null) _seen.Remove(item.Candidate.CanonicalGenome.Id);
        _nextEvaluationId = transaction.NextEvaluationId;
        _proposals = transaction.Proposals;
        _evaluationAttempts = transaction.EvaluationAttempts;
        _generation = transaction.Generation;
        _eventSequence = transaction.EventSequence;
        _completionSequence = transaction.CompletionSequence;
        _failures.Clear();
        foreach (EvolutionDiagnostic failure in transaction.Failures) _failures.Enqueue(failure);
    }
}
