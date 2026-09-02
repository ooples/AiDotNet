using System.Diagnostics;
using System.Globalization;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Evolution;

public sealed partial class EvolutionEngine<TGenome>
{
    /// <summary>Monotonic completion counter used to order committed work items in non-deterministic mode.</summary>
    private long _completionSequence;

    /// <summary>Allocates an identity for a caller-supplied seed genome and prepares it for evaluation.</summary>
    private async Task<PreparedProposal> PrepareSeedAsync(TGenome genome, CancellationToken cancellationToken)
    {
        long evaluationId = AllocateProposalId();
        int island = (int)(evaluationId % _islands.Length);
        var lineage = new EvolutionLineage(null, null, "seed", _refiner?.Id, 0, island, (ulong)evaluationId);
        return new PreparedProposal(await PrepareGenomeAsync(genome, evaluationId, island, lineage, cancellationToken).ConfigureAwait(false));
    }

    /// <summary>
    /// Selects a parent and inspirations, runs the variation operator, and prepares the proposal;
    /// returns <c>null</c> when no island holds an elite to select from.
    /// </summary>
    private async Task<PreparedProposal?> PrepareVariationAsync(CancellationToken cancellationToken)
    {
        long evaluationId = _nextEvaluationId;
        int island = (int)(evaluationId % _islands.Length);
        int sourceIsland = FindSelectionIsland(island);
        if (sourceIsland < 0) return null;
        IEvolutionArchive<TGenome> sourceArchive = _islands[sourceIsland];
        if (_options.IslandAssignment == EvolutionIslandAssignmentStrategy.InheritParent) island = sourceIsland;

        StableRandom proposalRandom = StableRandom.CreateStream(_options.Seed, unchecked((ulong)evaluationId * 8UL));
        if (_selection is IEliteIndexAwareEvolutionSelectionPolicy<TGenome> eliteAwareSelection)
            eliteAwareSelection.UseEliteIndex(_globalElites.Entries, island);
        EvolutionSelection<TGenome>? selection = _selection.Select(sourceArchive, proposalRandom, _options.InspirationCount);
        if (selection is null) return null;
        long allocatedId = AllocateProposalId();
        if (allocatedId != evaluationId) throw new InvalidOperationException("Proposal identity allocation was not sequential.");
        long generation = ++_generation;
        _islandGenerations[island]++;
        string[] inspirationIds = selection.Inspirations.Select(entry => entry.Evaluation.GenomeId).ToArray();
        var lineage = new EvolutionLineage(
            new[] { selection.Parent.Evaluation.GenomeId },
            inspirationIds,
            _variation.Id,
            _refiner?.Id,
            generation,
            island,
            (ulong)evaluationId);

        IReadOnlyList<EvolutionArtifact> parentArtifacts = ConsumeArtifacts(selection.Parent.Evaluation.GenomeId);
        TGenome proposed;
        try
        {
            var context = new EvolutionVariationContext<TGenome>(selection.Parent, selection.Inspirations,
                proposalRandom, generation, island, parentArtifacts);
            proposed = await _variation.ProposeAsync(context, cancellationToken).ConfigureAwait(false);
            if (proposed is null) throw new InvalidOperationException("The variation operator returned null.");
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception)
        {
            return new PreparedProposal(CreatePreEvaluationFailure(evaluationId, island, lineage, "variation_failure"));
        }

        return new PreparedProposal(await PrepareGenomeAsync(proposed, evaluationId, island, lineage, cancellationToken).ConfigureAwait(false));
    }

    /// <summary>
    /// Refines and canonicalizes a genome, raises the proposed event, and resolves cache hits and duplicates
    /// before deciding whether the work item requires an evaluator call.
    /// </summary>
    private async Task<WorkItem> PrepareGenomeAsync(TGenome genome, long evaluationId, int island,
        EvolutionLineage lineage, CancellationToken cancellationToken)
    {
        try
        {
            if (_refiner is not null)
            {
                StableRandom refinerRandom = StableRandom.CreateStream(_options.Seed, unchecked((ulong)evaluationId * 8UL + 1UL));
                genome = await _refiner.RefineAsync(genome,
                    new EvolutionRefinementContext(evaluationId, refinerRandom), cancellationToken).ConfigureAwait(false);
                if (genome is null) throw new InvalidOperationException("The candidate refiner returned null.");
            }

            EvolutionCanonicalGenome<TGenome> canonical = await _task.CanonicalizeAsync(genome, cancellationToken).ConfigureAwait(false);
            if (canonical is null) throw new InvalidOperationException("The evolution task returned a null canonical genome.");
            var candidate = new EvolutionCandidate<TGenome>(evaluationId, canonical, lineage);
            await NotifyAsync(new EvolutionEvent<TGenome>(EvolutionEventKind.Proposed, NextEventSequence(), candidate),
                cancellationToken).ConfigureAwait(false);

            if (_options.EnableEvaluationCache && _cache.TryGetValue(canonical.Id, out EvolutionTaskResult? cached))
            {
                return new WorkItem(lineage)
                {
                    EvaluationId = evaluationId,
                    Island = island,
                    Candidate = candidate,
                    Result = CopyWithZeroCost(cached),
                    CacheStatus = EvolutionCacheStatus.Hit,
                    CompletionOrder = Interlocked.Increment(ref _completionSequence)
                };
            }

            if (!_seen.Add(canonical.Id))
            {
                return new WorkItem(lineage)
                {
                    EvaluationId = evaluationId,
                    Island = island,
                    Candidate = candidate,
                    Result = new EvolutionTaskResult(EvolutionEvaluationStatus.Duplicate),
                    CacheStatus = EvolutionCacheStatus.NotChecked,
                    CompletionOrder = Interlocked.Increment(ref _completionSequence)
                };
            }

            if (!IsStructurallyNovel(canonical, island))
            {
                _seen.Remove(canonical.Id);
                return new WorkItem(lineage)
                {
                    EvaluationId = evaluationId,
                    Island = island,
                    Candidate = candidate,
                    Result = new EvolutionTaskResult(EvolutionEvaluationStatus.Rejected,
                        diagnostics: new[] { new EvolutionDiagnostic("not_novel",
                            "The candidate was within the structural novelty threshold of an existing elite.") }),
                    CacheStatus = EvolutionCacheStatus.NotChecked,
                    CompletionOrder = Interlocked.Increment(ref _completionSequence)
                };
            }

            return new WorkItem(lineage)
            {
                EvaluationId = evaluationId,
                Island = island,
                Candidate = candidate,
                CacheStatus = _options.EnableEvaluationCache ? EvolutionCacheStatus.Miss : EvolutionCacheStatus.NotChecked,
                RequiresEvaluation = true,
                AddedToSeen = true
            };
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception)
        {
            return CreatePreEvaluationFailure(evaluationId, island, lineage, "canonicalization_failure");
        }
    }

    /// <summary>Creates a terminal failed work item for a candidate that could not be prepared.</summary>
    private WorkItem CreatePreEvaluationFailure(long evaluationId, int island, EvolutionLineage lineage, string code)
    {
        return new WorkItem(lineage)
        {
            EvaluationId = evaluationId,
            Island = island,
            Result = EvolutionTaskResult.Failed(code, "Candidate preparation failed."),
            CacheStatus = EvolutionCacheStatus.NotChecked,
            CompletionOrder = Interlocked.Increment(ref _completionSequence)
        };
    }

    /// <summary>
    /// Returns whether a canonical genome is far enough from every elite of its target island, using at most one
    /// distance call per occupied cell and never an evaluator, embedding, or network call.
    /// </summary>
    private bool IsStructurallyNovel(EvolutionCanonicalGenome<TGenome> canonical, int island)
    {
        if (_distance is null || _options.NoveltyDistanceThreshold <= 0) return true;
        foreach (EvolutionArchiveEntry<TGenome> entry in _islands[island].Entries)
        {
            double distance = _distance.Distance(canonical.Genome, entry.Candidate.CanonicalGenome.Genome);
            if (!EvolutionDescriptorDefinition.IsFinite(distance) || distance < 0)
                throw new InvalidOperationException("The genome distance metric returned a value that is not finite and non-negative.");
            if (distance < _options.NoveltyDistanceThreshold) return false;
        }
        return true;
    }

    /// <summary>Allocates the next sequential evaluation identifier and counts the proposal.</summary>
    private long AllocateProposalId()
    {
        long id = _nextEvaluationId++;
        _proposals++;
        return id;
    }

    /// <summary>
    /// Returns the preferred island when occupied, otherwise the next occupied island in ring order, or a negative
    /// value when every island is empty.
    /// </summary>
    private int FindSelectionIsland(int preferredIsland)
    {
        if (_islands[preferredIsland].Count > 0) return preferredIsland;
        for (int offset = 1; offset < _islands.Length; offset++)
        {
            int island = (preferredIsland + offset) % _islands.Length;
            if (_islands[island].Count > 0) return island;
        }
        return -1;
    }

    /// <summary>
    /// Evaluates the pending work items in bounded-parallel rounds, retrying failure-like results within the
    /// retry and attempt budgets, and marks anything left undispatched as skipped.
    /// </summary>
    /// <remarks>
    /// An attempt is charged to <c>MaxEvaluationAttempts</c> before dispatch so the round can never exceed the budget,
    /// and is refunded afterwards when a cascade stage rejected the candidate before the final stage and
    /// <c>Cascade.ChargeRejectedStagesToBudget</c> is clear. The refund keeps the budget a measure of full evaluations
    /// rather than of cheap screening calls, and cannot loop: a rejected candidate has a terminal
    /// <see cref="EvolutionEvaluationStatus.Skipped"/> status and is never re-queued.
    /// </remarks>
    private async Task EvaluateBatchAsync(List<WorkItem> batch, CancellationToken cancellationToken)
    {
        List<WorkItem> pending = batch.Where(item => item.RequiresEvaluation).OrderBy(item => item.EvaluationId).ToList();
        while (pending.Count > 0 && _evaluationAttempts < _options.MaxEvaluationAttempts)
        {
            int available = (int)Math.Min(pending.Count, _options.MaxEvaluationAttempts - _evaluationAttempts);
            WorkItem[] round = pending.Take(available).ToArray();
            pending.RemoveRange(0, available);
            int highestAttempt = 0;
            foreach (WorkItem item in round)
            {
                item.AttemptCount++;
                _evaluationAttempts++;
                highestAttempt = Math.Max(highestAttempt, item.AttemptCount);
            }

            TimeSpan retryDelay = RetryDelayForAttempt(highestAttempt);
            if (retryDelay > TimeSpan.Zero) await Task.Delay(retryDelay, cancellationToken).ConfigureAwait(false);

            using (var semaphore = new SemaphoreSlim(_options.MaxDegreeOfParallelism, _options.MaxDegreeOfParallelism))
            {
                Task[] tasks = round.Select(item => EvaluateWithSlotAsync(item, semaphore, cancellationToken)).ToArray();
                await Task.WhenAll(tasks).ConfigureAwait(false);
            }

            foreach (WorkItem item in round.OrderBy(item => item.EvaluationId))
            {
                if (item.CascadeRejectedStage.HasValue && !_options.Cascade.ChargeRejectedStagesToBudget)
                    _evaluationAttempts--;

                if (IsRetryable(item.Result) && item.AttemptCount <= _options.MaxRetries &&
                    _evaluationAttempts + pending.Count < _options.MaxEvaluationAttempts)
                {
                    pending.Add(item);
                }
                else
                {
                    item.RequiresEvaluation = false;
                }
            }
            pending = pending.OrderBy(item => item.EvaluationId).ToList();
        }

        foreach (WorkItem unevaluated in pending)
        {
            unevaluated.RequiresEvaluation = false;
            unevaluated.Result = new EvolutionTaskResult(EvolutionEvaluationStatus.Skipped,
                diagnostics: new[] { new EvolutionDiagnostic("budget_exhausted", "Evaluator budget was exhausted before dispatch.") });
            unevaluated.CompletionOrder = Interlocked.Increment(ref _completionSequence);
        }
    }

    /// <summary>Runs one evaluation attempt inside a parallelism slot and records its timing and result.</summary>
    private async Task EvaluateWithSlotAsync(WorkItem item, SemaphoreSlim semaphore, CancellationToken cancellationToken)
    {
        await semaphore.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            Stopwatch timer = Stopwatch.StartNew();
            EvolutionTaskResult result = await EvaluateAttemptAsync(item, cancellationToken).ConfigureAwait(false);
            timer.Stop();
            item.Elapsed += timer.Elapsed;
            AccumulateAttemptMetadata(item, result);
            item.Result = result;
            item.CompletionOrder = Interlocked.Increment(ref _completionSequence);
        }
        finally
        {
            semaphore.Release();
        }
    }

    /// <summary>
    /// Runs one evaluation attempt, using the staged cascade path when it is configured and the single-call evaluator
    /// otherwise.
    /// </summary>
    private async Task<EvolutionTaskResult> EvaluateAttemptAsync(WorkItem item, CancellationToken cancellationToken)
    {
        if (item.Candidate is null) return EvolutionTaskResult.Failed("missing_candidate", "The evaluator candidate was unavailable.");
        item.StageCostUnits = Array.Empty<double>();
        item.CascadeRejectedStage = null;
        if (_options.Cascade.Enabled && _cascadeTask is not null)
            return await EvaluateCascadeAsync(item, _cascadeTask, cancellationToken).ConfigureAwait(false);

        EvolutionCandidate<TGenome> candidate = item.Candidate;
        return await InvokeEvaluatorAsync(item, stage: 0, _options.EvaluationTimeout,
            (context, token) => _task.EvaluateAsync(candidate, context, token), cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Invokes one evaluator call under its cooperative timeout, converting cancellation, timeout, and evaluator
    /// exceptions into terminal results, and abandoning a call that ignores its token once the grace period elapses.
    /// </summary>
    private async Task<EvolutionTaskResult> InvokeEvaluatorAsync(WorkItem item, int stage, TimeSpan? timeout,
        Func<EvolutionEvaluationContext, CancellationToken, ValueTask<EvolutionTaskResult>> invoke,
        CancellationToken cancellationToken)
    {
        using (var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
        {
            if (timeout.HasValue) linked.CancelAfter(timeout.Value);
            Task<EvolutionTaskResult> work;
            try
            {
                var context = new EvolutionEvaluationContext(item.EvaluationId, _options.Seed,
                    unchecked((ulong)item.EvaluationId * 8UL + 2UL + (ulong)stage * 0x9E3779B97F4A7C15UL),
                    item.AttemptCount);
                work = invoke(context, linked.Token).AsTask();
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                return new EvolutionTaskResult(EvolutionEvaluationStatus.Canceled);
            }
            catch (Exception exception)
            {
                return EvaluatorException(exception, stage, item.AttemptCount);
            }

            if (timeout.HasValue && _options.EvaluationGracePeriod.HasValue)
            {
                using (var abandonment = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
                {
                    Task limit = Task.Delay(timeout.Value + _options.EvaluationGracePeriod.Value, abandonment.Token);
                    Task winner = await Task.WhenAny(work, limit).ConfigureAwait(false);
                    if (!ReferenceEquals(winner, work))
                    {
                        Interlocked.Increment(ref _abandonedEvaluations);
                        ObserveAbandoned(work);
                        return TimedOut(stage, item.AttemptCount, timeout.Value, abandoned: true);
                    }
                    abandonment.Cancel();
                }
            }

            try
            {
                EvolutionTaskResult result = await work.ConfigureAwait(false);
                return result ?? EvolutionTaskResult.Failed("null_result", "The task returned a null evaluation result.");
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                return new EvolutionTaskResult(EvolutionEvaluationStatus.Canceled);
            }
            catch (OperationCanceledException) when (linked.IsCancellationRequested)
            {
                return TimedOut(stage, item.AttemptCount, timeout, abandoned: false);
            }
            catch (Exception exception)
            {
                return EvaluatorException(exception, stage, item.AttemptCount);
            }
        }
    }

    /// <summary>Keeps an abandoned evaluator call from raising an unobserved task exception at finalization.</summary>
    private static void ObserveAbandoned(Task<EvolutionTaskResult> work) =>
        _ = work.ContinueWith(static completed => { _ = completed.Exception; }, CancellationToken.None,
            TaskContinuationOptions.ExecuteSynchronously, TaskScheduler.Default);

    /// <summary>Builds the structured timed-out result for one evaluator call.</summary>
    private static EvolutionTaskResult TimedOut(int stage, int attempt, TimeSpan? timeout, bool abandoned)
    {
        var data = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["stage"] = stage.ToString(CultureInfo.InvariantCulture),
            ["attempt"] = attempt.ToString(CultureInfo.InvariantCulture),
            ["timeout_ticks"] = (timeout?.Ticks ?? 0).ToString(CultureInfo.InvariantCulture),
            ["abandoned"] = abandoned ? "true" : "false"
        };
        return new EvolutionTaskResult(EvolutionEvaluationStatus.TimedOut,
            diagnostics: new[]
            {
                new EvolutionDiagnostic("evaluation_timeout",
                    abandoned
                        ? "The evaluator ignored its cancellation token and was abandoned after the grace period."
                        : "The evaluator exceeded its cooperative timeout.",
                    isRedacted: false, data: data)
            });
    }

    /// <summary>Builds the redacted failed result for an evaluator exception.</summary>
    private static EvolutionTaskResult EvaluatorException(Exception exception, int stage, int attempt) =>
        new(EvolutionEvaluationStatus.Failed,
            diagnostics: new[]
            {
                new EvolutionDiagnostic("evaluator_exception", $"Evaluator threw {exception.GetType().Name}.",
                    isRedacted: true, data: new Dictionary<string, string>(StringComparer.Ordinal)
                    {
                        ["stage"] = stage.ToString(CultureInfo.InvariantCulture),
                        ["attempt"] = attempt.ToString(CultureInfo.InvariantCulture),
                        ["exception"] = exception.GetType().Name
                    })
            });

    /// <summary>
    /// Commits a batch in deterministic or completion order: updates counters, archives, cache, and seen set,
    /// notifies observers, and returns whether a fail-fast failure was encountered.
    /// </summary>
    private async Task<bool> CommitBatchAsync(List<WorkItem> batch, CancellationToken cancellationToken)
    {
        IEnumerable<WorkItem> ordered = _options.ExecutionMode == EvolutionExecutionMode.Deterministic
            ? batch.OrderBy(item => item.EvaluationId)
            : batch.OrderBy(item => item.CompletionOrder).ThenBy(item => item.EvaluationId);
        bool failedFast = false;
        foreach (WorkItem item in ordered)
        {
            EvolutionTaskResult result = item.Result ?? EvolutionTaskResult.Failed("missing_result", "No terminal evaluator result was produced.");
            EvolutionEvaluation evaluation = BuildEvaluation(item, result);
            IncrementStatus(evaluation.Status);
            EvolutionArchiveInsertionResult? insertion = null;
            if (evaluation.Status == EvolutionEvaluationStatus.Completed)
            {
                _completedEvaluations++;
                if (item.Candidate is not null)
                {
                    insertion = _islands[item.Island].TryAdd(item.Candidate, evaluation);
                    if (_options.EnableEvaluationCache && item.CacheStatus != EvolutionCacheStatus.Hit)
                        _cache[item.Candidate.CanonicalGenome.Id] = WithoutArtifacts(result);
                    RecordCompletedEvaluation(item.Island, item.Candidate, evaluation);
                }
            }
            else if (item.Candidate is not null && !_options.DeduplicateFailedCandidates &&
                     IsFailureLike(evaluation.Status))
            {
                _seen.Remove(item.Candidate.CanonicalGenome.Id);
            }

            if (item.CacheStatus != EvolutionCacheStatus.Hit) QueueLineageArtifacts(item, evaluation);

            if (_selection is IOutcomeAwareEvolutionSelectionPolicy<TGenome> adaptiveSelection)
                adaptiveSelection.Observe(evaluation, insertion);

            if (IsFailureLike(evaluation.Status))
            {
                foreach (EvolutionDiagnostic diagnostic in evaluation.Diagnostics) RetainFailure(diagnostic);
                failedFast |= evaluation.Status != EvolutionEvaluationStatus.Canceled &&
                              _options.FailurePolicy == EvolutionFailurePolicy.FailFast;
            }

            await NotifyAsync(new EvolutionEvent<TGenome>(EvolutionEventKind.Evaluated, NextEventSequence(),
                item.Candidate, evaluation, insertion), cancellationToken).ConfigureAwait(false);
            if (insertion == EvolutionArchiveInsertionResult.Inserted ||
                insertion == EvolutionArchiveInsertionResult.Replaced ||
                insertion == EvolutionArchiveInsertionResult.InsertedWithEviction)
            {
                await NotifyAsync(new EvolutionEvent<TGenome>(EvolutionEventKind.ArchiveChanged, NextEventSequence(),
                    item.Candidate, evaluation, insertion), cancellationToken).ConfigureAwait(false);
            }

            if (_checkpointStore is not null && _options.CheckpointInterval > 0) _commitsSinceCheckpoint++;
        }
        return failedFast;
    }

    /// <summary>
    /// Adds the completed evaluation to the island's global elite index and bounded history, and reports any
    /// configured descriptor the task omitted so a silent archive rejection cannot go unnoticed.
    /// </summary>
    private void RecordCompletedEvaluation(int island, EvolutionCandidate<TGenome> candidate, EvolutionEvaluation evaluation)
    {
        IEvolutionArchive<TGenome> archive = _islands[island];
        EvolutionCellKey? cell = TryCreateCellKey(archive, evaluation.Descriptors);
        if (cell is null)
        {
            foreach (EvolutionDescriptorDefinition descriptor in archive.Descriptors)
                if (!evaluation.Descriptors.ContainsKey(descriptor.Name))
                    RetainFailure(new EvolutionDiagnostic("descriptor_missing:" + descriptor.Name,
                        "A completed evaluation omitted a configured archive descriptor and could not be placed in a cell."));
            return;
        }

        var entry = new EvolutionArchiveEntry<TGenome>(cell, candidate, evaluation);
        if (_globalElites.Capacity > 0) _globalElites.Consider(new EvolutionEliteRecord<TGenome>(island, entry));
        if (_histories[island].Capacity > 0)
        {
            string[] cellOwners = archive.Entries.Select(item => item.Evaluation.GenomeId).ToArray();
            _histories[island].Add(entry, cellOwners, archive.Best?.Evaluation.GenomeId);
        }
    }

    /// <summary>
    /// Computes an archive cell from descriptor values using the archive's own descriptor definitions, returning
    /// <c>null</c> when a value is missing or rejected by its out-of-range policy.
    /// </summary>
    private static EvolutionCellKey? TryCreateCellKey(IEvolutionArchiveView<TGenome> archive,
        IReadOnlyDictionary<string, double> descriptors)
    {
        IReadOnlyList<EvolutionDescriptorDefinition> definitions = archive.Descriptors;
        if (definitions.Count == 0) return null;
        var bins = new int[definitions.Count];
        for (int i = 0; i < definitions.Count; i++)
        {
            if (!descriptors.TryGetValue(definitions[i].Name, out double value) ||
                !definitions[i].TryGetBin(value, out bins[i]))
            {
                return null;
            }
        }
        return new EvolutionCellKey(bins);
    }

    /// <summary>Builds the immutable evaluation record for a work item from its terminal result and attempt metadata.</summary>
    private EvolutionEvaluation BuildEvaluation(WorkItem item, EvolutionTaskResult result)
    {
        string genomeId = item.Candidate?.CanonicalGenome.Id ?? $"unavailable:{item.EvaluationId}";
        bool cacheHit = item.CacheStatus == EvolutionCacheStatus.Hit;
        double costUnits = cacheHit ? 0 : item.AccumulatedCostUnits;
        IReadOnlyList<EvolutionDiagnostic> diagnostics = item.AttemptCount == 0
            ? result.Diagnostics
            : item.AttemptDiagnostics;
        return new EvolutionEvaluation(
            item.EvaluationId,
            genomeId,
            result.Status,
            result.Quality,
            result.Direction,
            WithQualityDescriptor(result),
            result.Objectives,
            result.ConstraintViolations,
            new EvolutionEvaluationCost(item.Elapsed, item.AttemptCount, costUnits,
                cacheHit ? Array.Empty<double>() : item.StageCostUnits, item.CascadeRejectedStage),
            item.Lineage,
            item.CacheStatus,
            diagnostics,
            _task.VersionHash,
            _task.EvaluatorVersionHash,
            _configurationHash,
            result.Metrics,
            BoundArtifacts(result.Artifacts));
    }

    /// <summary>
    /// The absolute number of transfers one migration round may ever carry, whatever the topology or island count.
    /// </summary>
    /// <remarks>
    /// The engine's real bound is the ordered island pairs times <c>MigrantsPerIsland</c>, which scales with the
    /// configured topology; this constant is the backstop that keeps a buggy policy from flooding the archives on a run
    /// configured with a very large island count.
    /// </remarks>
    private const int MaximumMigrationTransfers = 65_536;

    /// <summary>
    /// Runs the migration policy when the interval is due, validates every transfer against its source island and the
    /// topology-scaled transfer bounds, and applies the marked copies in a stable order before notifying observers.
    /// </summary>
    /// <remarks>
    /// A transfer is applied as a copy whose lineage records the source island, so a migrated elite stays
    /// distinguishable from one discovered locally and the marker survives the checkpoint. Each copy is offered to the
    /// destination archive under that archive's normal insertion rules and raises its own
    /// <see cref="EvolutionEventKind.ArchiveChanged"/> event when it is accepted, so an observer sees exactly which
    /// migrants changed which island rather than only a round summary.
    /// </remarks>
    private async Task MigrateIfDueAsync(CancellationToken cancellationToken)
    {
        if (_islands.Length < 2 || _options.MigrationInterval == 0 || !IsMigrationDue()) return;
        _batchesSinceMigration = 0;
        _lastMigrationGeneration = MaximumIslandGeneration();
        StableRandom random = StableRandom.CreateStream(_options.Seed, unchecked(0x8000000000000000UL + (ulong)_generation));
        IReadOnlyList<EvolutionMigration<TGenome>> migrations = _migration.CreateMigrations(
            Array.AsReadOnly(_islands), _options.MigrantsPerIsland, random)
            ?? throw new InvalidOperationException("The migration policy returned null.");
        if (migrations.Any(item => item is null))
            throw new InvalidOperationException("The migration policy returned a null transfer.");
        ValidateMigrationBounds(migrations);
        EvolutionMigration<TGenome>[] orderedMigrations = migrations
            .OrderBy(item => item.SourceIsland)
            .ThenBy(item => item.DestinationIsland)
            .ThenBy(item => item.Entry.Evaluation.GenomeId, StringComparer.Ordinal)
            .ToArray();
        foreach (EvolutionMigration<TGenome> migration in orderedMigrations)
        {
            if (migration.SourceIsland < 0 || migration.DestinationIsland < 0 ||
                migration.SourceIsland >= _islands.Length || migration.DestinationIsland >= _islands.Length ||
                migration.SourceIsland == migration.DestinationIsland)
                throw new InvalidOperationException("The migration policy produced an invalid island index.");
            bool belongsToSource = _islands[migration.SourceIsland].Entries.Any(entry =>
                entry.Cell.Equals(migration.Entry.Cell) &&
                entry.Evaluation.EvaluationId == migration.Entry.Evaluation.EvaluationId &&
                string.Equals(entry.Evaluation.GenomeId, migration.Entry.Evaluation.GenomeId, StringComparison.Ordinal));
            if (!belongsToSource)
                throw new InvalidOperationException("The migration policy produced an entry that is not in its source island.");
        }
        foreach (EvolutionMigration<TGenome> migration in orderedMigrations)
        {
            EvolutionArchiveEntry<TGenome> migrant = CreateMigrantEntry(migration);
            EvolutionArchiveInsertionResult insertion =
                _islands[migration.DestinationIsland].TryAdd(migrant.Candidate, migrant.Evaluation);
            if (insertion == EvolutionArchiveInsertionResult.Inserted ||
                insertion == EvolutionArchiveInsertionResult.Replaced ||
                insertion == EvolutionArchiveInsertionResult.InsertedWithEviction)
            {
                await NotifyAsync(new EvolutionEvent<TGenome>(EvolutionEventKind.ArchiveChanged, NextEventSequence(),
                    migrant.Candidate, migrant.Evaluation, insertion), cancellationToken).ConfigureAwait(false);
            }
        }
        await NotifyAsync(new EvolutionEvent<TGenome>(EvolutionEventKind.Migrated, NextEventSequence(),
            message: $"{migrations.Count} elite transfers"), cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Rejects a migration round that carries more transfers than the configured topology could ever justify.
    /// </summary>
    /// <remarks>
    /// The bound is two-sided. No ordered island pair may carry more than <c>MigrantsPerIsland</c> transfers, so no
    /// single destination can be swamped whatever the topology, and the round as a whole may not exceed the ordered
    /// island pairs times that per-pair bound, capped by <see cref="MaximumMigrationTransfers"/>. This replaces the
    /// earlier per-source-island bound, which a star or fully connected topology necessarily exceeds because one source
    /// legitimately feeds every other island.
    /// </remarks>
    private void ValidateMigrationBounds(IReadOnlyList<EvolutionMigration<TGenome>> migrations)
    {
        long orderedPairs = (long)_islands.Length * (_islands.Length - 1);
        long topologyBound = orderedPairs * _options.MigrantsPerIsland;
        long allowed = Math.Min(topologyBound, MaximumMigrationTransfers);
        if (migrations.Count > allowed)
            throw new InvalidOperationException("The migration policy exceeded the total transfer bound for its topology.");
        foreach (IGrouping<long, EvolutionMigration<TGenome>> pair in migrations.GroupBy(
            item => (long)item.SourceIsland * _islands.Length + item.DestinationIsland))
        {
            if (pair.Count() > _options.MigrantsPerIsland)
                throw new InvalidOperationException("The migration policy exceeded the per-destination transfer bound.");
        }
    }

    /// <summary>Copies one elite for its destination island, marking the copy with the island it came from.</summary>
    /// <remarks>
    /// The genome, cell, evaluation identifier, and every recorded metric are preserved exactly so the destination
    /// archive applies its normal insertion rules to the same evidence the source island holds. Only the lineage
    /// changes: its island becomes the destination and <see cref="EvolutionLineage.MigrationSourceIsland"/> records the
    /// origin. The source entry is untouched.
    /// </remarks>
    private static EvolutionArchiveEntry<TGenome> CreateMigrantEntry(EvolutionMigration<TGenome> migration)
    {
        EvolutionArchiveEntry<TGenome> source = migration.Entry;
        EvolutionEvaluation evaluation = source.Evaluation;
        EvolutionLineage lineage = evaluation.Lineage;
        var migrantLineage = new EvolutionLineage(
            lineage.ParentIds,
            lineage.InspirationIds,
            lineage.VariationOperatorId,
            lineage.RefinerId,
            lineage.Generation,
            migration.DestinationIsland,
            lineage.SeedStream,
            migration.SourceIsland);
        var candidate = new EvolutionCandidate<TGenome>(source.Candidate.EvaluationId,
            source.Candidate.CanonicalGenome, migrantLineage);
        var migrantEvaluation = new EvolutionEvaluation(
            evaluation.EvaluationId,
            evaluation.GenomeId,
            evaluation.Status,
            evaluation.Quality,
            evaluation.Direction,
            evaluation.Descriptors,
            evaluation.Objectives,
            evaluation.ConstraintViolations,
            evaluation.Cost,
            migrantLineage,
            evaluation.CacheStatus,
            evaluation.Diagnostics,
            evaluation.TaskVersionHash,
            evaluation.EvaluatorVersionHash,
            evaluation.ConfigurationHash,
            evaluation.Metrics,
            evaluation.Artifacts);
        return new EvolutionArchiveEntry<TGenome>(source.Cell, candidate, migrantEvaluation);
    }

    /// <summary>Returns whether the configured migration trigger has reached its interval.</summary>
    private bool IsMigrationDue() => _options.MigrationTrigger == EvolutionMigrationTrigger.IslandGenerations
        ? MaximumIslandGeneration() - _lastMigrationGeneration >= _options.MigrationInterval
        : _batchesSinceMigration >= _options.MigrationInterval;

    /// <summary>Returns the highest per-island generation counter.</summary>
    private long MaximumIslandGeneration()
    {
        long maximum = 0;
        foreach (long generation in _islandGenerations) maximum = Math.Max(maximum, generation);
        return maximum;
    }

    /// <summary>Increments the terminal counter for a status.</summary>
    private void IncrementStatus(EvolutionEvaluationStatus status)
    {
        _statusCounts.TryGetValue(status, out long current);
        _statusCounts[status] = current + 1;
    }

    /// <summary>Returns whether a result exists and has a failure-like status the configured retry set allows.</summary>
    private bool IsRetryable(EvolutionTaskResult? result) =>
        result is not null && IsFailureLike(result.Status) && (RetryFlag(result.Status) & _options.RetryOn) != 0;

    /// <summary>Maps a failure-like status onto its retry flag.</summary>
    private static EvolutionRetryStatuses RetryFlag(EvolutionEvaluationStatus status) => status switch
    {
        EvolutionEvaluationStatus.Failed => EvolutionRetryStatuses.Failed,
        EvolutionEvaluationStatus.TimedOut => EvolutionRetryStatuses.TimedOut,
        EvolutionEvaluationStatus.Canceled => EvolutionRetryStatuses.Canceled,
        _ => EvolutionRetryStatuses.None
    };

    /// <summary>Returns whether a status is failed, timed out, or canceled.</summary>
    private static bool IsFailureLike(EvolutionEvaluationStatus status) =>
        status == EvolutionEvaluationStatus.Failed ||
        status == EvolutionEvaluationStatus.TimedOut ||
        status == EvolutionEvaluationStatus.Canceled;

    /// <summary>Adds an attempt's cost units (saturating at the finite maximum) and diagnostics to the work item.</summary>
    private static void AccumulateAttemptMetadata(WorkItem item, EvolutionTaskResult result)
    {
        if (result.CostUnits > double.MaxValue - item.AccumulatedCostUnits)
        {
            item.AccumulatedCostUnits = double.MaxValue;
            AddAttemptDiagnostic(item, new EvolutionDiagnostic("cost_units_saturated",
                "Accumulated evaluator cost exceeded the finite numeric range."));
        }
        else
        {
            item.AccumulatedCostUnits += result.CostUnits;
        }

        foreach (EvolutionDiagnostic diagnostic in result.Diagnostics) AddAttemptDiagnostic(item, diagnostic);
    }

    /// <summary>Appends a diagnostic, replacing the last slot with a truncation marker once the public bound is reached.</summary>
    private static void AddAttemptDiagnostic(WorkItem item, EvolutionDiagnostic diagnostic)
    {
        const int maximumDiagnostics = 64;
        if (item.AttemptDiagnostics.Count < maximumDiagnostics)
        {
            item.AttemptDiagnostics.Add(diagnostic);
            return;
        }

        if (item.AttemptDiagnostics[maximumDiagnostics - 1].Code != "diagnostics_truncated")
        {
            item.AttemptDiagnostics[maximumDiagnostics - 1] = new EvolutionDiagnostic(
                "diagnostics_truncated", "Additional retry diagnostics were omitted to preserve the public bound.");
        }
    }

    /// <summary>
    /// Returns the result's descriptors, adding the configured quality descriptor when the option is enabled, the
    /// evaluation completed with a quality, and the task did not already supply that name itself.
    /// </summary>
    private IReadOnlyDictionary<string, double> WithQualityDescriptor(EvolutionTaskResult result)
    {
        string? name = _options.QualityDescriptorName;
        if (name is null || result.Status != EvolutionEvaluationStatus.Completed || !result.Quality.HasValue ||
            result.Descriptors.ContainsKey(name))
        {
            return result.Descriptors;
        }

        var merged = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, double> descriptor in result.Descriptors) merged[descriptor.Key] = descriptor.Value;
        merged[name] = result.Quality.Value;
        return merged;
    }

    /// <summary>Copies a cached result with zero cost units so cache hits do not re-bill the original evaluation.</summary>
    private static EvolutionTaskResult CopyWithZeroCost(EvolutionTaskResult result) => new(
        result.Status, result.Quality, result.Direction, result.Descriptors, result.Objectives,
        result.ConstraintViolations, 0, result.Diagnostics, result.Metrics);

    /// <summary>Strips artifacts from a result before it enters the evaluation cache.</summary>
    /// <remarks>
    /// The cache exists to avoid recomputation, and a cache hit never replays artifacts: they describe one specific
    /// evaluation run, and handing them to an unrelated later proposal would deliver a stale failure note. Dropping
    /// them at the point of caching also keeps the checkpoint small and keeps the cached result byte-identical across a
    /// checkpoint round trip, which the run state hash depends on.
    /// </remarks>
    private static EvolutionTaskResult WithoutArtifacts(EvolutionTaskResult result) => result.Artifacts.Count == 0
        ? result
        : new EvolutionTaskResult(result.Status, result.Quality, result.Direction, result.Descriptors,
            result.Objectives, result.ConstraintViolations, result.CostUnits, result.Diagnostics, result.Metrics);
}
