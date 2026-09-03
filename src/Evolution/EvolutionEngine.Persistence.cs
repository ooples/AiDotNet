using System.Globalization;
using System.Text;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using Newtonsoft.Json;

namespace AiDotNet.Evolution;

public sealed partial class EvolutionEngine<TGenome>
{
    // 5 added per-island descriptor ranges so a Grow axis's widened grid survives a resume.
    private const int EngineStateSchemaVersion = 5;
    private string? _safePayload;
    private long _safeSequence;

    private void CaptureSafeState(TGenome[] seeds, int seedIndex)
    {
        if (_checkpointStore is null) return;
        if (_codec is null) throw new InvalidOperationException("Checkpoint capture requires a genome codec.");
        var document = new EngineStateDocument
        {
            SchemaVersion = EngineStateSchemaVersion,
            SemanticOptions = _options.SemanticFields().Select(OptionFieldDocument.From).ToList(),
            BudgetOptions = _options.BudgetFields().Select(OptionFieldDocument.From).ToList(),
            SeedPayloads = seeds.Select(SerializeGenome).ToList(),
            SeedIndex = seedIndex,
            NextEvaluationId = _nextEvaluationId,
            Proposals = _proposals,
            EvaluationAttempts = _evaluationAttempts,
            CompletedEvaluations = _completedEvaluations,
            Generation = _generation,
            EventSequence = _eventSequence,
            CompletionSequence = _completionSequence,
            BatchesSinceMigration = _batchesSinceMigration,
            LastMigrationGeneration = _lastMigrationGeneration,
            IslandGenerations = _islandGenerations.ToList(),
            GlobalElites = _globalElites.Entries.Select(record => new EliteRecordDocument
            {
                Island = record.Island,
                Entry = ArchiveEntryDocument.From(record.Entry, SerializeGenome)
            }).ToList(),
            IslandHistories = _histories.Select(history => history.Entries
                .Select(entry => ArchiveEntryDocument.From(entry, SerializeGenome)).ToList()).ToList(),
            SelectionState = (_selection as IOutcomeAwareEvolutionSelectionPolicy<TGenome>)?.CaptureState(),
            StatusCounts = _statusCounts.OrderBy(pair => pair.Key).Select(pair => new StatusCountDocument
            {
                Status = pair.Key,
                Count = pair.Value
            }).ToList(),
            SeenGenomeIds = _seen.OrderBy(value => value, StringComparer.Ordinal).ToList(),
            Cache = _cache.OrderBy(pair => pair.Key, StringComparer.Ordinal).Select(pair => new CacheDocument
            {
                GenomeId = pair.Key,
                Result = TaskResultDocument.From(pair.Value)
            }).ToList(),
            Failures = _failures.Select(DiagnosticDocument.From).ToList(),
            EarlyStoppingBest = _earlyStoppingBest,
            EvaluationsSinceImprovement = _evaluationsSinceImprovement,
            AbandonedEvaluations = Interlocked.Read(ref _abandonedEvaluations),
            PendingArtifacts = _pendingArtifactOrder
                .Where(genomeId => _pendingArtifacts.ContainsKey(genomeId))
                .Select(genomeId => new PendingArtifactDocument
                {
                    GenomeId = genomeId,
                    Artifacts = _pendingArtifacts[genomeId].Select(ArtifactDocument.From).ToList()
                }).ToList(),
            Islands = _islands.Select(archive => ArchiveDocument.From(archive, SerializeGenome)).ToList()
        };
        _safePayload = JsonConvert.SerializeObject(document, Formatting.None);
        _safeSequence++;
    }

    private async Task SaveCheckpointAsync(bool force, CancellationToken cancellationToken)
    {
        if (_checkpointStore is null || _safePayload is null) return;
        if (!force && (_options.CheckpointInterval == 0 || _commitsSinceCheckpoint < _options.CheckpointInterval)) return;
        var checkpoint = new EvolutionCheckpoint(_options.RunId, _safeSequence, _compatibilityHash, _safePayload,
            EvolutionCheckpoint.CurrentSchemaVersion, BestQualityAcrossIslands(), _islands[0].Direction);
        await _checkpointStore.SaveAsync(checkpoint, cancellationToken).ConfigureAwait(false);
        _commitsSinceCheckpoint = 0;
        await NotifyAsync(new EvolutionEvent<TGenome>(EvolutionEventKind.Checkpointed, NextEventSequence(),
            message: $"checkpoint {_safeSequence}"), cancellationToken).ConfigureAwait(false);
    }

    private async Task<RestoredSeeds> RestoreCheckpointAsync(TGenome[] suppliedSeeds, CancellationToken cancellationToken)
    {
        if (_checkpointStore is null || _codec is null) throw new InvalidOperationException("Resume is not configured.");
        EvolutionCheckpoint? checkpoint = await _checkpointStore.LoadLatestAsync(_options.RunId, cancellationToken).ConfigureAwait(false);
        if (checkpoint is null) return new RestoredSeeds(suppliedSeeds, 0);
        checkpoint.Validate();
        if (!string.Equals(checkpoint.CompatibilityHash, _compatibilityHash, StringComparison.Ordinal))
            throw new InvalidDataException(DescribeIncompatibility(checkpoint));

        EngineStateDocument? state;
        try
        {
            state = JsonConvert.DeserializeObject<EngineStateDocument>(checkpoint.Payload);
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException("The evolution engine state payload is invalid.", exception);
        }
        if (state is null || state.SchemaVersion != EngineStateSchemaVersion)
            throw new InvalidDataException("The evolution engine state schema is invalid.");

        string[] checkpointSeedPayloads = state.SeedPayloads?.ToArray()
            ?? throw new InvalidDataException("The checkpoint seed list is missing.");
        TGenome[] seeds;
        if (suppliedSeeds.Length == 0)
        {
            seeds = checkpointSeedPayloads.Select(DeserializeGenome).ToArray();
        }
        else
        {
            string[] suppliedPayloads = suppliedSeeds.Select(SerializeGenome).ToArray();
            if (!suppliedPayloads.SequenceEqual(checkpointSeedPayloads, StringComparer.Ordinal))
                throw new InvalidDataException("The supplied initial genomes do not match the checkpointed seed sequence.");
            seeds = suppliedSeeds;
        }
        if (state.SeedIndex < 0 || state.SeedIndex > seeds.Length)
            throw new InvalidDataException("The checkpoint seed cursor is invalid.");

        ValidateNonnegative(state.NextEvaluationId, nameof(state.NextEvaluationId));
        ValidateNonnegative(state.Proposals, nameof(state.Proposals));
        ValidateNonnegative(state.EvaluationAttempts, nameof(state.EvaluationAttempts));
        ValidateNonnegative(state.CompletedEvaluations, nameof(state.CompletedEvaluations));
        ValidateNonnegative(state.Generation, nameof(state.Generation));
        ValidateNonnegative(state.EventSequence, nameof(state.EventSequence));
        ValidateNonnegative(state.CompletionSequence, nameof(state.CompletionSequence));
        if (state.BatchesSinceMigration < 0) throw new InvalidDataException("The checkpoint migration counter is invalid.");
        // Budgets are deliberately not compared: a raised limit continues the run, and a limit lowered below what the
        // run already spent restores the counters and stops immediately with the matching budget stop reason.
        if (state.NextEvaluationId != state.Proposals || state.CompletedEvaluations > state.Proposals ||
            state.SeedIndex > state.Proposals)
            throw new InvalidDataException("Checkpoint counters violate engine identity invariants.");
        if ((state.SeenGenomeIds?.Count ?? 0) > state.Proposals || (state.Cache?.Count ?? 0) > state.Proposals)
            throw new InvalidDataException("Checkpoint identity collections exceed the proposal count.");
        if ((state.Failures?.Count ?? 0) > _options.MaxRetainedFailures)
            throw new InvalidDataException("Checkpoint failure diagnostics exceed the configured bound.");

        _nextEvaluationId = state.NextEvaluationId;
        _proposals = state.Proposals;
        _evaluationAttempts = state.EvaluationAttempts;
        _completedEvaluations = state.CompletedEvaluations;
        _generation = state.Generation;
        _eventSequence = state.EventSequence;
        _completionSequence = state.CompletionSequence;
        _batchesSinceMigration = state.BatchesSinceMigration;
        RestoreIslandGenerations(state);
        if (_selection is IOutcomeAwareEvolutionSelectionPolicy<TGenome> adaptiveSelection)
        {
            if (state.SelectionState is null) throw new InvalidDataException("The checkpoint adaptive-selection state is missing.");
            adaptiveSelection.RestoreState(state.SelectionState);
        }
        else if (state.SelectionState is not null)
        {
            throw new InvalidDataException("The checkpoint contains state for a non-adaptive selection policy.");
        }
        _statusCounts.Clear();
        foreach (StatusCountDocument count in state.StatusCounts ?? new List<StatusCountDocument>())
        {
            if (!Enum.IsDefined(typeof(EvolutionEvaluationStatus), count.Status) || count.Count < 0 || _statusCounts.ContainsKey(count.Status))
                throw new InvalidDataException("The checkpoint status counters are invalid.");
            _statusCounts[count.Status] = count.Count;
        }
        long terminalCount;
        try { terminalCount = checked(_statusCounts.Values.Sum()); }
        catch (OverflowException exception) { throw new InvalidDataException("Checkpoint status counters overflowed.", exception); }
        if (terminalCount != state.Proposals ||
            (_statusCounts.TryGetValue(EvolutionEvaluationStatus.Completed, out long completedCount) ? completedCount : 0) !=
            state.CompletedEvaluations)
            throw new InvalidDataException("Checkpoint status counters do not match the run counters.");

        _seen.Clear();
        foreach (string id in state.SeenGenomeIds ?? new List<string>())
            if (string.IsNullOrWhiteSpace(id) || !_seen.Add(id)) throw new InvalidDataException("The checkpoint deduplication set is invalid.");

        _cache.Clear();
        foreach (CacheDocument cached in state.Cache ?? new List<CacheDocument>())
        {
            if (string.IsNullOrWhiteSpace(cached.GenomeId) || cached.Result is null || _cache.ContainsKey(cached.GenomeId))
                throw new InvalidDataException("The checkpoint evaluation cache is invalid.");
            if (!_seen.Contains(cached.GenomeId))
                throw new InvalidDataException("A checkpoint cache key is missing from the deduplication set.");
            EvolutionTaskResult result = cached.Result.ToTaskResult();
            if (result.Status != EvolutionEvaluationStatus.Completed)
                throw new InvalidDataException("Only completed evaluations may be cached.");
            _cache[cached.GenomeId] = result;
        }

        _failures.Clear();
        foreach (DiagnosticDocument diagnostic in state.Failures ?? new List<DiagnosticDocument>())
            RetainFailure(diagnostic.ToDiagnostic());
        RestoreTerminationState(state);
        RestorePendingArtifacts(state);

        List<ArchiveDocument> archiveDocuments = state.Islands ?? throw new InvalidDataException("Checkpoint islands are missing.");
        if (archiveDocuments.Count != _islands.Length) throw new InvalidDataException("Checkpoint island count is incompatible.");
        for (int island = 0; island < _islands.Length; island++)
        {
            if (!(_islands[island] is ICheckpointableEvolutionArchive<TGenome> restorable))
                throw new InvalidOperationException("Resume requires checkpointable archive implementations.");
            ArchiveDocument archiveDocument = archiveDocuments[island];
            if (archiveDocument.Version < 0) throw new InvalidDataException("Checkpoint archive version is invalid.");
            RestoreArchiveDescriptors(restorable, archiveDocument);
            var entries = new List<EvolutionArchiveEntry<TGenome>>();
            var islandGenomeIds = new HashSet<string>(StringComparer.Ordinal);
            foreach (ArchiveEntryDocument entryDocument in archiveDocument.Entries ?? new List<ArchiveEntryDocument>())
            {
                EvolutionArchiveEntry<TGenome> entry = RestoreArchiveEntry(entryDocument);
                // Naming which invariant broke, and on which entry, is what turns a failed resume from a dead end into
                // something diagnosable: the three causes need completely different fixes.
                if (entry.Evaluation.EvaluationId >= state.NextEvaluationId)
                    throw new InvalidDataException(
                        $"Checkpoint archive entry '{entry.Evaluation.GenomeId}' on island {island} has evaluation id " +
                        $"{entry.Evaluation.EvaluationId}, which the run had not yet issued " +
                        $"({state.NextEvaluationId} were issued in total).");
                if (!_seen.Contains(entry.Evaluation.GenomeId))
                    throw new InvalidDataException(
                        $"Checkpoint archive entry '{entry.Evaluation.GenomeId}' on island {island} is missing from the " +
                        "run's set of proposed genomes.");
                if (!islandGenomeIds.Add(entry.Evaluation.GenomeId))
                    throw new InvalidDataException(
                        $"Checkpoint archive entry '{entry.Evaluation.GenomeId}' occupies more than one cell on island " +
                        $"{island}; each genome may hold at most one cell.");
                entries.Add(entry);
            }
            restorable.Restore(entries, archiveDocument.Version);
        }

        RestoreGlobalElites(state);
        RestoreIslandHistories(state);
        _safeSequence = checkpoint.Sequence;
        _safePayload = checkpoint.Payload;
        return new RestoredSeeds(seeds, state.SeedIndex);
    }

    /// <summary>Adopts the descriptor ranges a checkpoint recorded before any elite is replayed into the archive.</summary>
    /// <param name="archive">The freshly built, still-empty island archive.</param>
    /// <param name="document">The checkpointed archive state for that island.</param>
    /// <remarks>
    /// A descriptor with <see cref="EvolutionOutOfRangePolicy.Grow"/> widens as a run meets values outside its
    /// configured range, so the grid the elites were binned on is part of the saved state. Adopting it first means the
    /// replay rebins every elite onto the grid it came from, and means the replay itself never triggers growth, which
    /// is what keeps a resumed run's archive identical to the uninterrupted one even though the replay walks cells in
    /// key order rather than commit order. Ranges identical to the configured ones need no growable archive, so runs
    /// with fixed descriptors are unaffected.
    /// </remarks>
    private void RestoreArchiveDescriptors(ICheckpointableEvolutionArchive<TGenome> archive, ArchiveDocument document)
    {
        if (document.Descriptors is null) return;

        EvolutionDescriptorDefinition[] restored = document.Descriptors
            .Select(descriptor => descriptor is null
                ? throw new InvalidDataException("A checkpoint descriptor definition is missing.")
                : descriptor.ToDefinition())
            .ToArray();

        IReadOnlyList<EvolutionDescriptorDefinition> live = archive.Descriptors;
        bool unchanged = restored.Length == live.Count;
        for (int i = 0; unchanged && i < restored.Length; i++)
        {
            unchanged = string.Equals(restored[i].ToCanonicalString(), live[i].ToCanonicalString(), StringComparison.Ordinal);
        }
        if (unchanged) return;

        if (archive is not IGrowableEvolutionArchive<TGenome> growable)
            throw new InvalidDataException(
                "The checkpoint recorded widened descriptor ranges, but the archive does not support restoring them. " +
                "Resume requires an archive implementing IGrowableEvolutionArchive when any descriptor can grow.");

        growable.RestoreDescriptorBounds(restored);
    }

    /// <summary>
    /// Builds the message for a refused resume, naming the semantic option that differs whenever the checkpoint
    /// recorded one.
    /// </summary>
    /// <remarks>
    /// The compatibility hash also covers the task, variation operator, selection, refiner, codec, distance metric, and
    /// archive definition, so a mismatch that no recorded option explains falls back to the general message. The option
    /// list is read best-effort: a payload that cannot be parsed never turns a refusal into a different failure.
    /// </remarks>
    private string DescribeIncompatibility(EvolutionCheckpoint checkpoint)
    {
        const string general =
            "The evolution checkpoint is incompatible with the current task or engine configuration.";
        EngineStateDocument? state;
        try
        {
            state = JsonConvert.DeserializeObject<EngineStateDocument>(checkpoint.Payload);
        }
        catch (JsonException)
        {
            return general;
        }
        List<OptionFieldDocument>? recorded = state?.SemanticOptions;
        if (recorded is null || recorded.Count == 0) return general;
        var fields = new List<KeyValuePair<string, string>>(recorded.Count);
        foreach (OptionFieldDocument field in recorded)
        {
            if (field is null || string.IsNullOrEmpty(field.Name)) return general;
            fields.Add(new KeyValuePair<string, string>(field.Name, field.Value ?? string.Empty));
        }
        string? difference = _options.DescribeSemanticDifference(fields);
        return difference is null ? general : "The evolution checkpoint cannot be resumed because " + difference + ".";
    }

    private void RestoreIslandGenerations(EngineStateDocument state)
    {
        List<long> generations = state.IslandGenerations ?? throw new InvalidDataException("Checkpoint island generations are missing.");
        if (generations.Count != _islandGenerations.Length)
            throw new InvalidDataException("Checkpoint island generation count is incompatible.");
        long total = 0;
        for (int island = 0; island < generations.Count; island++)
        {
            if (generations[island] < 0) throw new InvalidDataException("A checkpoint island generation counter is invalid.");
            total = checked(total + generations[island]);
            _islandGenerations[island] = generations[island];
        }
        if (total != state.Generation)
            throw new InvalidDataException("Checkpoint island generation counters do not sum to the run generation.");
        if (state.LastMigrationGeneration < 0 || state.LastMigrationGeneration > state.Generation)
            throw new InvalidDataException("The checkpoint migration generation marker is invalid.");
        _lastMigrationGeneration = state.LastMigrationGeneration;
    }

    private void RestoreTerminationState(EngineStateDocument state)
    {
        if (state.EvaluationsSinceImprovement < 0 || state.AbandonedEvaluations < 0)
            throw new InvalidDataException("The checkpoint termination counters are invalid.");
        if (state.EarlyStoppingBest.HasValue && !EvolutionDescriptorDefinition.IsFinite(state.EarlyStoppingBest.Value))
            throw new InvalidDataException("The checkpoint early-stopping metric is not finite.");
        _earlyStoppingBest = state.EarlyStoppingBest;
        _evaluationsSinceImprovement = state.EvaluationsSinceImprovement;
        Interlocked.Exchange(ref _abandonedEvaluations, state.AbandonedEvaluations);
    }

    private void RestorePendingArtifacts(EngineStateDocument state)
    {
        _pendingArtifacts.Clear();
        _pendingArtifactOrder.Clear();
        List<PendingArtifactDocument> pending = state.PendingArtifacts ?? new List<PendingArtifactDocument>();
        if (pending.Count > _options.Artifacts.MaxPendingCandidates)
            throw new InvalidDataException("The checkpoint pending-artifact queue exceeds its configured capacity.");
        foreach (PendingArtifactDocument document in pending)
        {
            if (string.IsNullOrWhiteSpace(document.GenomeId) || _pendingArtifacts.ContainsKey(document.GenomeId))
                throw new InvalidDataException("A checkpoint pending-artifact entry is invalid.");
            if (!_seen.Contains(document.GenomeId))
                throw new InvalidDataException("A checkpoint pending-artifact key is missing from the deduplication set.");
            EvolutionArtifact[] artifacts = (document.Artifacts ?? new List<ArtifactDocument>())
                .Select(item => item.ToArtifact()).ToArray();
            if (artifacts.Length > _options.Artifacts.MaxArtifactsPerEvaluation)
                throw new InvalidDataException("A checkpoint pending-artifact entry exceeds its configured artifact bound.");
            _pendingArtifacts[document.GenomeId] = artifacts;
            _pendingArtifactOrder.Add(document.GenomeId);
        }
    }

    private void RestoreGlobalElites(EngineStateDocument state)
    {
        List<EliteRecordDocument> records = state.GlobalElites ?? new List<EliteRecordDocument>();
        if (records.Count > _options.GlobalEliteCount)
            throw new InvalidDataException("The checkpoint global elite index exceeds the configured capacity.");
        var restored = new List<EvolutionEliteRecord<TGenome>>(records.Count);
        foreach (EliteRecordDocument record in records)
        {
            if (record.Entry is null || record.Island < 0 || record.Island >= _islands.Length)
                throw new InvalidDataException("A checkpoint global elite record is invalid.");
            EvolutionArchiveEntry<TGenome> entry = RestoreArchiveEntry(record.Entry);
            if (entry.Evaluation.EvaluationId >= state.NextEvaluationId || !_seen.Contains(entry.Evaluation.GenomeId))
                throw new InvalidDataException("A checkpoint global elite record violates engine identity invariants.");
            restored.Add(new EvolutionEliteRecord<TGenome>(record.Island, entry));
        }
        _globalElites.Restore(restored);
    }

    private void RestoreIslandHistories(EngineStateDocument state)
    {
        List<List<ArchiveEntryDocument>> histories = state.IslandHistories ?? new List<List<ArchiveEntryDocument>>();
        if (histories.Count == 0 && _options.HistorySize == 0) return;
        if (histories.Count != _histories.Length)
            throw new InvalidDataException("Checkpoint island history count is incompatible.");
        for (int island = 0; island < histories.Count; island++)
        {
            var entries = new List<EvolutionArchiveEntry<TGenome>>();
            foreach (ArchiveEntryDocument document in histories[island] ?? new List<ArchiveEntryDocument>())
            {
                EvolutionArchiveEntry<TGenome> entry = RestoreArchiveEntry(document);
                if (entry.Evaluation.EvaluationId >= state.NextEvaluationId || !_seen.Contains(entry.Evaluation.GenomeId))
                    throw new InvalidDataException("A checkpoint island history entry violates engine identity invariants.");
                entries.Add(entry);
            }
            _histories[island].Restore(entries);
        }
    }

    /// <summary>Reads the candidates a checkpoint holds, without starting or resuming a run.</summary>
    /// <param name="checkpoint">The checkpoint to read.</param>
    /// <param name="genomeCodec">The codec that wrote the checkpoint's genomes.</param>
    /// <returns>Every recovered candidate, with the island and the part of the checkpoint it came from.</returns>
    /// <remarks>
    /// <para>
    /// A checkpoint is written so a run can continue, but it is also the complete record of what that run held:
    /// every elite, every retained runner-up, each genome in full, and the lineage that produced it. Reading it back
    /// is how a finished run is audited, how a lineage is reconstructed after the fact, and how successful
    /// trajectories are harvested as training data - the post-hoc story the reference implementation gets from its
    /// per-program files and its checkpoint trace extractors.
    /// </para>
    /// <para>
    /// This is deliberately a reader on the same type that writes the checkpoint. The document schema is private,
    /// and a parallel reader that duplicated it would keep compiling while silently drifting the first time a field
    /// was renamed.
    /// </para>
    /// <para><b>For Beginners:</b> Point this at a saved run to see what it found, without running anything:
    /// <code>
    /// var checkpoint = await store.LoadLatestAsync("my-run");
    /// var contents = EvolutionEngine&lt;MyGenome&gt;.ReadCheckpoint(checkpoint, new MyGenomeCodec());
    /// foreach (var entry in contents.DistinctCandidates) Console.WriteLine(entry.Entry.Evaluation.Quality);
    /// </code>
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="checkpoint"/> or <paramref name="genomeCodec"/> is <c>null</c>.</exception>
    /// <exception cref="InvalidDataException">The checkpoint payload is not a readable engine state.</exception>
    public static EvolutionCheckpointContents<TGenome> ReadCheckpoint(
        EvolutionCheckpoint checkpoint, IEvolutionGenomeCodec<TGenome> genomeCodec)
    {
        Guard.NotNull(checkpoint);
        Guard.NotNull(genomeCodec);
        checkpoint.Validate();

        EngineStateDocument? state;
        try
        {
            state = JsonConvert.DeserializeObject<EngineStateDocument>(checkpoint.Payload);
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException("The evolution engine state payload is invalid.", exception);
        }

        if (state is null) throw new InvalidDataException("The evolution engine state payload is empty.");

        var entries = new List<EvolutionCheckpointEntry<TGenome>>();
        List<ArchiveDocument> islands = state.Islands ?? new List<ArchiveDocument>();
        for (int island = 0; island < islands.Count; island++)
        {
            foreach (ArchiveEntryDocument document in islands[island].Entries ?? new List<ArchiveEntryDocument>())
            {
                entries.Add(new EvolutionCheckpointEntry<TGenome>(island,
                    EvolutionCheckpointEntrySource.IslandArchive, ReadArchiveEntry(document, genomeCodec)));
            }
        }

        foreach (EliteRecordDocument record in state.GlobalElites ?? new List<EliteRecordDocument>())
        {
            if (record.Entry is null) throw new InvalidDataException("A checkpoint global elite record is invalid.");
            entries.Add(new EvolutionCheckpointEntry<TGenome>(Math.Max(0, record.Island),
                EvolutionCheckpointEntrySource.GlobalElite, ReadArchiveEntry(record.Entry, genomeCodec)));
        }

        List<List<ArchiveEntryDocument>> histories = state.IslandHistories ?? new List<List<ArchiveEntryDocument>>();
        for (int island = 0; island < histories.Count; island++)
        {
            foreach (ArchiveEntryDocument document in histories[island] ?? new List<ArchiveEntryDocument>())
            {
                entries.Add(new EvolutionCheckpointEntry<TGenome>(island,
                    EvolutionCheckpointEntrySource.IslandHistory, ReadArchiveEntry(document, genomeCodec)));
            }
        }

        return new EvolutionCheckpointContents<TGenome>(
            checkpoint.RunId, checkpoint.Sequence, checkpoint.CompatibilityHash, entries);
    }

    /// <summary>Rebuilds one archive entry from its document using a caller-supplied codec.</summary>
    private static EvolutionArchiveEntry<TGenome> ReadArchiveEntry(
        ArchiveEntryDocument document, IEvolutionGenomeCodec<TGenome> genomeCodec)
    {
        if (document.GenomePayload is null || string.IsNullOrWhiteSpace(document.GenomeId) || document.Lineage is null ||
            document.Evaluation is null || document.CellBins is null)
            throw new InvalidDataException("An archive entry is incomplete.");

        TGenome genome;
        try
        {
            genome = genomeCodec.Deserialize(document.GenomePayload);
        }
        catch (Exception exception) when (exception is not OutOfMemoryException and not StackOverflowException)
        {
            throw new InvalidDataException(
                $"The genome codec could not read the payload of '{document.GenomeId}'.", exception);
        }

        if (genome is null) throw new InvalidDataException("The genome codec returned null.");
        var canonical = new EvolutionCanonicalGenome<TGenome>(genome, document.GenomeId);
        EvolutionLineage lineage = document.Lineage.ToLineage();
        var candidate = new EvolutionCandidate<TGenome>(document.EvaluationId, canonical, lineage);
        EvolutionEvaluation evaluation = document.Evaluation.ToEvaluation(document.EvaluationId, document.GenomeId, lineage);
        return new EvolutionArchiveEntry<TGenome>(new EvolutionCellKey(document.CellBins), candidate, evaluation);
    }

    private EvolutionArchiveEntry<TGenome> RestoreArchiveEntry(ArchiveEntryDocument document)
    {
        if (document.GenomePayload is null || string.IsNullOrWhiteSpace(document.GenomeId) || document.Lineage is null ||
            document.Evaluation is null || document.CellBins is null)
            throw new InvalidDataException("An archive entry is incomplete.");
        TGenome genome = DeserializeGenome(document.GenomePayload);
        var canonical = new EvolutionCanonicalGenome<TGenome>(genome, document.GenomeId);
        EvolutionLineage lineage = document.Lineage.ToLineage();
        var candidate = new EvolutionCandidate<TGenome>(document.EvaluationId, canonical, lineage);
        EvolutionEvaluation evaluation = document.Evaluation.ToEvaluation(document.EvaluationId, document.GenomeId, lineage);
        return new EvolutionArchiveEntry<TGenome>(new EvolutionCellKey(document.CellBins), candidate, evaluation);
    }

    private string SerializeGenome(TGenome genome)
    {
        if (_codec is null) throw new InvalidOperationException("Checkpoint serialization requires a genome codec.");
        string payload = _codec.Serialize(genome);
        if (payload is null) throw new InvalidOperationException("The genome codec returned null.");
        return payload;
    }

    private TGenome DeserializeGenome(string payload)
    {
        if (_codec is null) throw new InvalidOperationException("Checkpoint deserialization requires a genome codec.");
        TGenome genome = _codec.Deserialize(payload);
        if (genome is null) throw new InvalidDataException("The genome codec deserialized a null genome.");
        return genome;
    }

    private string ComputeStateHash()
    {
        var builder = new StringBuilder();
        Append(builder, _compatibilityHash);
        Append(builder, _nextEvaluationId);
        Append(builder, _proposals);
        Append(builder, _evaluationAttempts);
        Append(builder, _completedEvaluations);
        Append(builder, _generation);
        Append(builder, _batchesSinceMigration);
        foreach (EvolutionEvaluationStatus status in Enum.GetValues(typeof(EvolutionEvaluationStatus)))
        {
            Append(builder, (int)status);
            Append(builder, _statusCounts.TryGetValue(status, out long count) ? count : 0);
        }
        Append(builder, "seen");
        Append(builder, _seen.Count);
        foreach (string id in _seen.OrderBy(value => value, StringComparer.Ordinal)) Append(builder, id);
        Append(builder, "cache");
        Append(builder, _cache.Count);
        foreach (KeyValuePair<string, EvolutionTaskResult> cached in _cache.OrderBy(pair => pair.Key, StringComparer.Ordinal))
        {
            Append(builder, "cache:" + cached.Key);
            AppendTaskResult(builder, cached.Value);
        }
        Append(builder, "selection");
        Append(builder, _selection is IOutcomeAwareEvolutionSelectionPolicy<TGenome> adaptiveSelection
            ? adaptiveSelection.CaptureState()
            : "stateless");
        Append(builder, "failures");
        Append(builder, _failures.Count);
        foreach (EvolutionDiagnostic failure in _failures) AppendDiagnostic(builder, failure);
        Append(builder, "islands");
        Append(builder, _islands.Length);
        for (int island = 0; island < _islands.Length; island++)
        {
            Append(builder, island);
            Append(builder, _islands[island].Version);
            Append(builder, _islandGenerations[island]);
            Append(builder, _islands[island].Entries.Count);
            foreach (EvolutionArchiveEntry<TGenome> entry in _islands[island].Entries.OrderBy(item => item.Cell.StableKey, StringComparer.Ordinal))
            {
                Append(builder, entry.Cell.StableKey);
                AppendEvaluation(builder, entry.Evaluation);
            }
        }
        Append(builder, "last-migration-generation");
        Append(builder, _lastMigrationGeneration);
        Append(builder, "global-elites");
        Append(builder, _globalElites.Capacity);
        IReadOnlyList<EvolutionEliteRecord<TGenome>> elites = _globalElites.Entries;
        Append(builder, elites.Count);
        foreach (EvolutionEliteRecord<TGenome> record in elites)
        {
            Append(builder, record.Island);
            Append(builder, record.Entry.Cell.StableKey);
            AppendEvaluation(builder, record.Entry.Evaluation);
        }
        Append(builder, "island-histories");
        for (int island = 0; island < _histories.Length; island++)
        {
            Append(builder, island);
            Append(builder, _histories[island].Capacity);
            IReadOnlyList<EvolutionArchiveEntry<TGenome>> history = _histories[island].Entries;
            Append(builder, history.Count);
            foreach (EvolutionArchiveEntry<TGenome> entry in history)
            {
                Append(builder, entry.Cell.StableKey);
                AppendEvaluation(builder, entry.Evaluation);
            }
        }
        // Abandonment depends on wall-clock timing, so its counter is reported but deliberately never hashed.
        Append(builder, "early-stopping");
        Append(builder, _earlyStoppingBest?.ToString("R", CultureInfo.InvariantCulture) ?? "none");
        Append(builder, _evaluationsSinceImprovement);
        Append(builder, "pending-artifacts");
        Append(builder, _pendingArtifactOrder.Count);
        foreach (string genomeId in _pendingArtifactOrder)
        {
            Append(builder, genomeId);
            if (!_pendingArtifacts.TryGetValue(genomeId, out EvolutionArtifact[]? artifacts)) continue;
            AppendArtifacts(builder, artifacts);
        }
        return EvolutionHash.Compute(builder.ToString());
    }

    private static void AppendTaskResult(StringBuilder builder, EvolutionTaskResult result)
    {
        Append(builder, (int)result.Status);
        Append(builder, result.Quality?.ToString("R", CultureInfo.InvariantCulture) ?? "none");
        Append(builder, (int)result.Direction);
        AppendNamedValues(builder, result.Descriptors);
        Append(builder, "objectives");
        Append(builder, result.Objectives.Count);
        foreach (double objective in result.Objectives) Append(builder, objective.ToString("R", CultureInfo.InvariantCulture));
        Append(builder, "violations");
        Append(builder, result.ConstraintViolations.Count);
        foreach (double violation in result.ConstraintViolations) Append(builder, violation.ToString("R", CultureInfo.InvariantCulture));
        Append(builder, result.CostUnits.ToString("R", CultureInfo.InvariantCulture));
        AppendDiagnostics(builder, result.Diagnostics);
        Append(builder, "metrics");
        AppendNamedValues(builder, result.Metrics);
        AppendArtifacts(builder, result.Artifacts);
    }

    private static void AppendEvaluation(StringBuilder builder, EvolutionEvaluation evaluation)
    {
        Append(builder, evaluation.EvaluationId);
        Append(builder, evaluation.GenomeId);
        Append(builder, (int)evaluation.Status);
        Append(builder, evaluation.Quality?.ToString("R", CultureInfo.InvariantCulture) ?? "none");
        Append(builder, (int)evaluation.Direction);
        AppendNamedValues(builder, evaluation.Descriptors);
        Append(builder, "objectives");
        Append(builder, evaluation.Objectives.Count);
        foreach (double objective in evaluation.Objectives) Append(builder, objective.ToString("R", CultureInfo.InvariantCulture));
        Append(builder, "violations");
        Append(builder, evaluation.ConstraintViolations.Count);
        foreach (double violation in evaluation.ConstraintViolations) Append(builder, violation.ToString("R", CultureInfo.InvariantCulture));
        Append(builder, evaluation.Cost.AttemptCount);
        Append(builder, evaluation.Cost.CostUnits.ToString("R", CultureInfo.InvariantCulture));
        Append(builder, "stage-costs");
        Append(builder, evaluation.Cost.StageCostUnits.Count);
        foreach (double stageCost in evaluation.Cost.StageCostUnits)
            Append(builder, stageCost.ToString("R", CultureInfo.InvariantCulture));
        Append(builder, evaluation.Cost.RejectedStage?.ToString(CultureInfo.InvariantCulture) ?? "none");
        Append(builder, (int)evaluation.CacheStatus);
        AppendDiagnostics(builder, evaluation.Diagnostics);
        Append(builder, "metrics");
        AppendNamedValues(builder, evaluation.Metrics);
        AppendArtifacts(builder, evaluation.Artifacts);
        Append(builder, evaluation.Lineage.VariationOperatorId);
        Append(builder, evaluation.Lineage.RefinerId ?? "none");
        Append(builder, evaluation.Lineage.Generation);
        Append(builder, evaluation.Lineage.Island);
        Append(builder, evaluation.Lineage.SeedStream);
        Append(builder, evaluation.Lineage.MigrationSourceIsland?.ToString(CultureInfo.InvariantCulture) ?? "local");
        Append(builder, "parents");
        Append(builder, evaluation.Lineage.ParentIds.Count);
        foreach (string parent in evaluation.Lineage.ParentIds) Append(builder, "p:" + parent);
        Append(builder, "inspirations");
        Append(builder, evaluation.Lineage.InspirationIds.Count);
        foreach (string inspiration in evaluation.Lineage.InspirationIds) Append(builder, "i:" + inspiration);
        Append(builder, evaluation.TaskVersionHash);
        Append(builder, evaluation.EvaluatorVersionHash);
        Append(builder, evaluation.ConfigurationHash);
    }

    private static void AppendNamedValues(StringBuilder builder, IReadOnlyDictionary<string, double> values)
    {
        Append(builder, "named-values");
        Append(builder, values.Count);
        foreach (KeyValuePair<string, double> value in values.OrderBy(pair => pair.Key, StringComparer.Ordinal))
        {
            Append(builder, value.Key);
            Append(builder, value.Value.ToString("R", CultureInfo.InvariantCulture));
        }
    }

    private static void AppendDiagnostics(StringBuilder builder, IReadOnlyList<EvolutionDiagnostic> diagnostics)
    {
        Append(builder, "diagnostics");
        Append(builder, diagnostics.Count);
        foreach (EvolutionDiagnostic diagnostic in diagnostics) AppendDiagnostic(builder, diagnostic);
    }

    private static void AppendDiagnostic(StringBuilder builder, EvolutionDiagnostic diagnostic)
    {
        Append(builder, diagnostic.Code);
        Append(builder, diagnostic.Message);
        Append(builder, diagnostic.IsRedacted ? 1 : 0);
        Append(builder, "data");
        Append(builder, diagnostic.Data.Count);
        foreach (KeyValuePair<string, string> entry in diagnostic.Data.OrderBy(pair => pair.Key, StringComparer.Ordinal))
        {
            Append(builder, entry.Key);
            Append(builder, entry.Value);
        }
    }

    private static void AppendArtifacts(StringBuilder builder, IReadOnlyList<EvolutionArtifact> artifacts)
    {
        Append(builder, "artifacts");
        Append(builder, artifacts.Count);
        foreach (EvolutionArtifact artifact in artifacts)
        {
            Append(builder, artifact.Key);
            Append(builder, artifact.Text);
            Append(builder, artifact.SizeBytes);
            Append(builder, artifact.IsTruncated ? 1 : 0);
            Append(builder, artifact.IsRedacted ? 1 : 0);
        }
    }

    private static void Append(StringBuilder builder, object value)
    {
        string text = Convert.ToString(value, CultureInfo.InvariantCulture) ?? string.Empty;
        builder.Append(text.Length.ToString(CultureInfo.InvariantCulture)).Append(':').Append(text).Append(';');
    }

    private static void ValidateNonnegative(long value, string name)
    {
        if (value < 0) throw new InvalidDataException($"Checkpoint counter '{name}' is invalid.");
    }

    private sealed class EngineStateDocument
    {
        public int SchemaVersion { get; set; }
        public List<OptionFieldDocument>? SemanticOptions { get; set; }
        public List<OptionFieldDocument>? BudgetOptions { get; set; }
        public List<string>? SeedPayloads { get; set; }
        public int SeedIndex { get; set; }
        public long NextEvaluationId { get; set; }
        public long Proposals { get; set; }
        public long EvaluationAttempts { get; set; }
        public long CompletedEvaluations { get; set; }
        public long Generation { get; set; }
        public long EventSequence { get; set; }
        public long CompletionSequence { get; set; }
        public int BatchesSinceMigration { get; set; }
        public long LastMigrationGeneration { get; set; }
        public List<long>? IslandGenerations { get; set; }
        public List<EliteRecordDocument>? GlobalElites { get; set; }
        public List<List<ArchiveEntryDocument>>? IslandHistories { get; set; }
        public string? SelectionState { get; set; }
        public List<StatusCountDocument>? StatusCounts { get; set; }
        public List<string>? SeenGenomeIds { get; set; }
        public List<CacheDocument>? Cache { get; set; }
        public List<DiagnosticDocument>? Failures { get; set; }
        public double? EarlyStoppingBest { get; set; }
        public long EvaluationsSinceImprovement { get; set; }
        public long AbandonedEvaluations { get; set; }
        public List<PendingArtifactDocument>? PendingArtifacts { get; set; }
        public List<ArchiveDocument>? Islands { get; set; }
    }

    private sealed class OptionFieldDocument
    {
        public string Name { get; set; } = string.Empty;
        public string Value { get; set; } = string.Empty;

        public static OptionFieldDocument From(KeyValuePair<string, string> field) =>
            new() { Name = field.Key, Value = field.Value };
    }

    private sealed class PendingArtifactDocument
    {
        public string GenomeId { get; set; } = string.Empty;
        public List<ArtifactDocument>? Artifacts { get; set; }
    }

    private sealed class ArtifactDocument
    {
        public string Key { get; set; } = string.Empty;
        public string Text { get; set; } = string.Empty;
        public bool IsTruncated { get; set; }
        public bool IsRedacted { get; set; }

        public static ArtifactDocument From(EvolutionArtifact artifact) => new()
        {
            Key = artifact.Key, Text = artifact.Text,
            IsTruncated = artifact.IsTruncated, IsRedacted = artifact.IsRedacted
        };

        public EvolutionArtifact ToArtifact()
        {
            try
            {
                return new EvolutionArtifact(Key, Text, IsTruncated, IsRedacted);
            }
            catch (ArgumentException exception)
            {
                throw new InvalidDataException("A checkpoint artifact is invalid.", exception);
            }
        }
    }

    private sealed class StatusCountDocument
    {
        public EvolutionEvaluationStatus Status { get; set; }
        public long Count { get; set; }
    }

    private sealed class CacheDocument
    {
        public string GenomeId { get; set; } = string.Empty;
        public TaskResultDocument? Result { get; set; }
    }

    private sealed class ArchiveDocument
    {
        public long Version { get; set; }
        public List<ArchiveEntryDocument>? Entries { get; set; }

        /// <summary>The descriptor ranges in force at checkpoint time, which a Grow axis widens during a run.</summary>
        public List<DescriptorDocument>? Descriptors { get; set; }

        public static ArchiveDocument From(IEvolutionArchive<TGenome> archive, Func<TGenome, string> serializeGenome) => new()
        {
            Version = archive.Version,
            Descriptors = archive.Descriptors.Select(DescriptorDocument.From).ToList(),
            Entries = archive.Entries.OrderBy(item => item.Cell.StableKey, StringComparer.Ordinal)
                .Select(entry => ArchiveEntryDocument.From(entry, serializeGenome)).ToList()
        };
    }

    private sealed class DescriptorDocument
    {
        public string Name { get; set; } = string.Empty;
        public double Minimum { get; set; }
        public double Maximum { get; set; }
        public int BinCount { get; set; }
        public EvolutionOutOfRangePolicy OutOfRangePolicy { get; set; }

        public static DescriptorDocument From(EvolutionDescriptorDefinition descriptor) => new()
        {
            Name = descriptor.Name,
            Minimum = descriptor.Minimum,
            Maximum = descriptor.Maximum,
            BinCount = descriptor.BinCount,
            OutOfRangePolicy = descriptor.OutOfRangePolicy
        };

        public EvolutionDescriptorDefinition ToDefinition()
        {
            if (string.IsNullOrWhiteSpace(Name) || BinCount <= 0 ||
                !Enum.IsDefined(typeof(EvolutionOutOfRangePolicy), OutOfRangePolicy))
                throw new InvalidDataException("A checkpoint descriptor definition is invalid.");
            try
            {
                return new EvolutionDescriptorDefinition(Name, Minimum, Maximum, BinCount, OutOfRangePolicy);
            }
            catch (Exception exception) when (exception is ArgumentException or ArgumentOutOfRangeException)
            {
                throw new InvalidDataException("A checkpoint descriptor definition is invalid.", exception);
            }
        }
    }

    private sealed class EliteRecordDocument
    {
        public int Island { get; set; }
        public ArchiveEntryDocument? Entry { get; set; }
    }

    private sealed class ArchiveEntryDocument
    {
        public int[]? CellBins { get; set; }
        public long EvaluationId { get; set; }
        public string GenomeId { get; set; } = string.Empty;
        public string? GenomePayload { get; set; }
        public LineageDocument? Lineage { get; set; }
        public EvaluationDocument? Evaluation { get; set; }

        public static ArchiveEntryDocument From(EvolutionArchiveEntry<TGenome> entry, Func<TGenome, string> serializeGenome) => new()
        {
            CellBins = entry.Cell.Bins.ToArray(),
            EvaluationId = entry.Evaluation.EvaluationId,
            GenomeId = entry.Evaluation.GenomeId,
            GenomePayload = serializeGenome(entry.Candidate.CanonicalGenome.Genome),
            Lineage = LineageDocument.From(entry.Evaluation.Lineage),
            Evaluation = EvaluationDocument.From(entry.Evaluation)
        };
    }

    private sealed class LineageDocument
    {
        public List<string>? ParentIds { get; set; }
        public List<string>? InspirationIds { get; set; }
        public string VariationOperatorId { get; set; } = string.Empty;
        public string? RefinerId { get; set; }
        public long Generation { get; set; }
        public int Island { get; set; }
        public ulong SeedStream { get; set; }
        public int? MigrationSourceIsland { get; set; }

        public static LineageDocument From(EvolutionLineage lineage) => new()
        {
            ParentIds = lineage.ParentIds.ToList(), InspirationIds = lineage.InspirationIds.ToList(),
            VariationOperatorId = lineage.VariationOperatorId, RefinerId = lineage.RefinerId,
            Generation = lineage.Generation, Island = lineage.Island, SeedStream = lineage.SeedStream,
            MigrationSourceIsland = lineage.MigrationSourceIsland
        };

        public EvolutionLineage ToLineage()
        {
            try
            {
                return new EvolutionLineage(ParentIds, InspirationIds, VariationOperatorId, RefinerId,
                    Generation, Island, SeedStream, MigrationSourceIsland);
            }
            catch (ArgumentException exception)
            {
                throw new InvalidDataException("A checkpoint lineage record is invalid.", exception);
            }
        }
    }

    private sealed class EvaluationDocument
    {
        public EvolutionEvaluationStatus Status { get; set; }
        public double? Quality { get; set; }
        public EvolutionOptimizationDirection Direction { get; set; }
        public Dictionary<string, double>? Descriptors { get; set; }
        public List<double>? Objectives { get; set; }
        public List<double>? ConstraintViolations { get; set; }
        public long ElapsedTicks { get; set; }
        public int AttemptCount { get; set; }
        public double CostUnits { get; set; }
        public List<double>? StageCostUnits { get; set; }
        public int? RejectedStage { get; set; }
        public EvolutionCacheStatus CacheStatus { get; set; }
        public List<DiagnosticDocument>? Diagnostics { get; set; }
        public Dictionary<string, double>? Metrics { get; set; }
        public List<ArtifactDocument>? Artifacts { get; set; }
        public string TaskVersionHash { get; set; } = string.Empty;
        public string EvaluatorVersionHash { get; set; } = string.Empty;
        public string ConfigurationHash { get; set; } = string.Empty;

        public static EvaluationDocument From(EvolutionEvaluation evaluation) => new()
        {
            Status = evaluation.Status, Quality = evaluation.Quality, Direction = evaluation.Direction,
            Descriptors = evaluation.Descriptors.ToDictionary(item => item.Key, item => item.Value, StringComparer.Ordinal),
            Objectives = evaluation.Objectives.ToList(), ConstraintViolations = evaluation.ConstraintViolations.ToList(),
            ElapsedTicks = evaluation.Cost.Elapsed.Ticks, AttemptCount = evaluation.Cost.AttemptCount,
            CostUnits = evaluation.Cost.CostUnits, StageCostUnits = evaluation.Cost.StageCostUnits.ToList(),
            RejectedStage = evaluation.Cost.RejectedStage, CacheStatus = evaluation.CacheStatus,
            Diagnostics = evaluation.Diagnostics.Select(DiagnosticDocument.From).ToList(),
            Metrics = evaluation.Metrics.ToDictionary(item => item.Key, item => item.Value, StringComparer.Ordinal),
            Artifacts = evaluation.Artifacts.Select(ArtifactDocument.From).ToList(),
            TaskVersionHash = evaluation.TaskVersionHash, EvaluatorVersionHash = evaluation.EvaluatorVersionHash,
            ConfigurationHash = evaluation.ConfigurationHash
        };

        public EvolutionEvaluation ToEvaluation(long evaluationId, string genomeId, EvolutionLineage lineage) => new(
            evaluationId, genomeId, Status, Quality, Direction,
            Descriptors ?? new Dictionary<string, double>(), Objectives ?? new List<double>(),
            ConstraintViolations ?? new List<double>(),
            new EvolutionEvaluationCost(TimeSpan.FromTicks(ElapsedTicks), AttemptCount, CostUnits,
                StageCostUnits ?? new List<double>(), RejectedStage),
            lineage, CacheStatus, (Diagnostics ?? new List<DiagnosticDocument>()).Select(item => item.ToDiagnostic()),
            TaskVersionHash, EvaluatorVersionHash, ConfigurationHash,
            Metrics ?? new Dictionary<string, double>(),
            (Artifacts ?? new List<ArtifactDocument>()).Select(item => item.ToArtifact()));
    }

    private sealed class TaskResultDocument
    {
        public EvolutionEvaluationStatus Status { get; set; }
        public double? Quality { get; set; }
        public EvolutionOptimizationDirection Direction { get; set; }
        public Dictionary<string, double>? Descriptors { get; set; }
        public List<double>? Objectives { get; set; }
        public List<double>? ConstraintViolations { get; set; }
        public double CostUnits { get; set; }
        public List<DiagnosticDocument>? Diagnostics { get; set; }
        public Dictionary<string, double>? Metrics { get; set; }

        public static TaskResultDocument From(EvolutionTaskResult result) => new()
        {
            Status = result.Status, Quality = result.Quality, Direction = result.Direction,
            Descriptors = result.Descriptors.ToDictionary(item => item.Key, item => item.Value, StringComparer.Ordinal),
            Objectives = result.Objectives.ToList(), ConstraintViolations = result.ConstraintViolations.ToList(),
            CostUnits = result.CostUnits, Diagnostics = result.Diagnostics.Select(DiagnosticDocument.From).ToList(),
            Metrics = result.Metrics.ToDictionary(item => item.Key, item => item.Value, StringComparer.Ordinal)
        };

        public EvolutionTaskResult ToTaskResult() => new(Status, Quality, Direction,
            Descriptors ?? new Dictionary<string, double>(), Objectives ?? new List<double>(),
            ConstraintViolations ?? new List<double>(), CostUnits,
            (Diagnostics ?? new List<DiagnosticDocument>()).Select(item => item.ToDiagnostic()),
            Metrics ?? new Dictionary<string, double>());
    }

    private sealed class DiagnosticDocument
    {
        public string Code { get; set; } = string.Empty;
        public string Message { get; set; } = string.Empty;
        public bool IsRedacted { get; set; }
        public Dictionary<string, string>? Data { get; set; }

        public static DiagnosticDocument From(EvolutionDiagnostic diagnostic) => new()
        {
            Code = diagnostic.Code, Message = diagnostic.Message, IsRedacted = diagnostic.IsRedacted,
            Data = diagnostic.Data.Count == 0
                ? null
                : diagnostic.Data.ToDictionary(item => item.Key, item => item.Value, StringComparer.Ordinal)
        };

        public EvolutionDiagnostic ToDiagnostic()
        {
            try
            {
                return new EvolutionDiagnostic(Code, Message, IsRedacted, Data);
            }
            catch (ArgumentException exception)
            {
                throw new InvalidDataException("A checkpoint diagnostic is invalid.", exception);
            }
        }
    }
}
