using System.Globalization;
using System.Text;
using Newtonsoft.Json;

namespace AiDotNet.Evolution;

public sealed partial class EvolutionEngine<TGenome>
{
    private const int EngineStateSchemaVersion = 1;
    private string? _safePayload;
    private long _safeSequence;

    private void CaptureSafeState(TGenome[] seeds, int seedIndex)
    {
        if (_checkpointStore is null) return;
        if (_codec is null) throw new InvalidOperationException("Checkpoint capture requires a genome codec.");
        var document = new EngineStateDocument
        {
            SchemaVersion = EngineStateSchemaVersion,
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
            Islands = _islands.Select(archive => ArchiveDocument.From(archive, SerializeGenome)).ToList()
        };
        _safePayload = JsonConvert.SerializeObject(document, Formatting.None);
        _safeSequence++;
    }

    private async Task SaveCheckpointAsync(bool force, CancellationToken cancellationToken)
    {
        if (_checkpointStore is null || _safePayload is null) return;
        if (!force && (_options.CheckpointInterval == 0 || _commitsSinceCheckpoint < _options.CheckpointInterval)) return;
        var checkpoint = new EvolutionCheckpoint(_options.RunId, _safeSequence, _compatibilityHash, _safePayload);
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
            throw new InvalidDataException("The evolution checkpoint is incompatible with the current task or engine configuration.");

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
        if (state.Proposals > _options.MaxProposals || state.EvaluationAttempts > _options.MaxEvaluationAttempts ||
            state.Generation > _options.MaxGenerations)
            throw new InvalidDataException("Checkpoint counters exceed the configured run budget.");
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

        List<ArchiveDocument> archiveDocuments = state.Islands ?? throw new InvalidDataException("Checkpoint islands are missing.");
        if (archiveDocuments.Count != _islands.Length) throw new InvalidDataException("Checkpoint island count is incompatible.");
        for (int island = 0; island < _islands.Length; island++)
        {
            if (!(_islands[island] is ICheckpointableEvolutionArchive<TGenome> restorable))
                throw new InvalidOperationException("Resume requires checkpointable archive implementations.");
            ArchiveDocument archiveDocument = archiveDocuments[island];
            if (archiveDocument.Version < 0) throw new InvalidDataException("Checkpoint archive version is invalid.");
            var entries = new List<EvolutionArchiveEntry<TGenome>>();
            var islandGenomeIds = new HashSet<string>(StringComparer.Ordinal);
            foreach (ArchiveEntryDocument entryDocument in archiveDocument.Entries ?? new List<ArchiveEntryDocument>())
            {
                EvolutionArchiveEntry<TGenome> entry = RestoreArchiveEntry(entryDocument);
                if (entry.Evaluation.EvaluationId >= state.NextEvaluationId || !_seen.Contains(entry.Evaluation.GenomeId) ||
                    !islandGenomeIds.Add(entry.Evaluation.GenomeId))
                    throw new InvalidDataException("A checkpoint archive entry violates engine identity invariants.");
                entries.Add(entry);
            }
            restorable.Restore(entries, archiveDocument.Version);
        }

        _safeSequence = checkpoint.Sequence;
        _safePayload = checkpoint.Payload;
        return new RestoredSeeds(seeds, state.SeedIndex);
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
        string payload = _codec!.Serialize(genome);
        if (payload is null) throw new InvalidOperationException("The genome codec returned null.");
        return payload;
    }

    private TGenome DeserializeGenome(string payload)
    {
        TGenome genome = _codec!.Deserialize(payload);
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
        foreach (EvolutionDiagnostic failure in _failures)
        {
            Append(builder, failure.Code);
            Append(builder, failure.Message);
            Append(builder, failure.IsRedacted ? 1 : 0);
        }
        Append(builder, "islands");
        Append(builder, _islands.Length);
        for (int island = 0; island < _islands.Length; island++)
        {
            Append(builder, island);
            Append(builder, _islands[island].Version);
            Append(builder, _islands[island].Entries.Count);
            foreach (EvolutionArchiveEntry<TGenome> entry in _islands[island].Entries.OrderBy(item => item.Cell.StableKey, StringComparer.Ordinal))
            {
                Append(builder, entry.Cell.StableKey);
                AppendEvaluation(builder, entry.Evaluation);
            }
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
        Append(builder, (int)evaluation.CacheStatus);
        AppendDiagnostics(builder, evaluation.Diagnostics);
        Append(builder, evaluation.Lineage.VariationOperatorId);
        Append(builder, evaluation.Lineage.RefinerId ?? "none");
        Append(builder, evaluation.Lineage.Generation);
        Append(builder, evaluation.Lineage.Island);
        Append(builder, evaluation.Lineage.SeedStream);
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
        foreach (EvolutionDiagnostic diagnostic in diagnostics)
        {
            Append(builder, diagnostic.Code);
            Append(builder, diagnostic.Message);
            Append(builder, diagnostic.IsRedacted ? 1 : 0);
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
        public string? SelectionState { get; set; }
        public List<StatusCountDocument>? StatusCounts { get; set; }
        public List<string>? SeenGenomeIds { get; set; }
        public List<CacheDocument>? Cache { get; set; }
        public List<DiagnosticDocument>? Failures { get; set; }
        public List<ArchiveDocument>? Islands { get; set; }
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

        public static ArchiveDocument From(IEvolutionArchive<TGenome> archive, Func<TGenome, string> serializeGenome) => new()
        {
            Version = archive.Version,
            Entries = archive.Entries.OrderBy(item => item.Cell.StableKey, StringComparer.Ordinal)
                .Select(entry => ArchiveEntryDocument.From(entry, serializeGenome)).ToList()
        };
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

        public static LineageDocument From(EvolutionLineage lineage) => new()
        {
            ParentIds = lineage.ParentIds.ToList(), InspirationIds = lineage.InspirationIds.ToList(),
            VariationOperatorId = lineage.VariationOperatorId, RefinerId = lineage.RefinerId,
            Generation = lineage.Generation, Island = lineage.Island, SeedStream = lineage.SeedStream
        };

        public EvolutionLineage ToLineage() => new(ParentIds, InspirationIds, VariationOperatorId, RefinerId,
            Generation, Island, SeedStream);
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
        public EvolutionCacheStatus CacheStatus { get; set; }
        public List<DiagnosticDocument>? Diagnostics { get; set; }
        public string TaskVersionHash { get; set; } = string.Empty;
        public string EvaluatorVersionHash { get; set; } = string.Empty;
        public string ConfigurationHash { get; set; } = string.Empty;

        public static EvaluationDocument From(EvolutionEvaluation evaluation) => new()
        {
            Status = evaluation.Status, Quality = evaluation.Quality, Direction = evaluation.Direction,
            Descriptors = evaluation.Descriptors.ToDictionary(item => item.Key, item => item.Value, StringComparer.Ordinal),
            Objectives = evaluation.Objectives.ToList(), ConstraintViolations = evaluation.ConstraintViolations.ToList(),
            ElapsedTicks = evaluation.Cost.Elapsed.Ticks, AttemptCount = evaluation.Cost.AttemptCount,
            CostUnits = evaluation.Cost.CostUnits, CacheStatus = evaluation.CacheStatus,
            Diagnostics = evaluation.Diagnostics.Select(DiagnosticDocument.From).ToList(),
            TaskVersionHash = evaluation.TaskVersionHash, EvaluatorVersionHash = evaluation.EvaluatorVersionHash,
            ConfigurationHash = evaluation.ConfigurationHash
        };

        public EvolutionEvaluation ToEvaluation(long evaluationId, string genomeId, EvolutionLineage lineage) => new(
            evaluationId, genomeId, Status, Quality, Direction,
            Descriptors ?? new Dictionary<string, double>(), Objectives ?? new List<double>(),
            ConstraintViolations ?? new List<double>(), new EvolutionEvaluationCost(TimeSpan.FromTicks(ElapsedTicks), AttemptCount, CostUnits),
            lineage, CacheStatus, (Diagnostics ?? new List<DiagnosticDocument>()).Select(item => item.ToDiagnostic()),
            TaskVersionHash, EvaluatorVersionHash, ConfigurationHash);
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

        public static TaskResultDocument From(EvolutionTaskResult result) => new()
        {
            Status = result.Status, Quality = result.Quality, Direction = result.Direction,
            Descriptors = result.Descriptors.ToDictionary(item => item.Key, item => item.Value, StringComparer.Ordinal),
            Objectives = result.Objectives.ToList(), ConstraintViolations = result.ConstraintViolations.ToList(),
            CostUnits = result.CostUnits, Diagnostics = result.Diagnostics.Select(DiagnosticDocument.From).ToList()
        };

        public EvolutionTaskResult ToTaskResult() => new(Status, Quality, Direction,
            Descriptors ?? new Dictionary<string, double>(), Objectives ?? new List<double>(),
            ConstraintViolations ?? new List<double>(), CostUnits,
            (Diagnostics ?? new List<DiagnosticDocument>()).Select(item => item.ToDiagnostic()));
    }

    private sealed class DiagnosticDocument
    {
        public string Code { get; set; } = string.Empty;
        public string Message { get; set; } = string.Empty;
        public bool IsRedacted { get; set; }

        public static DiagnosticDocument From(EvolutionDiagnostic diagnostic) => new()
        {
            Code = diagnostic.Code, Message = diagnostic.Message, IsRedacted = diagnostic.IsRedacted
        };

        public EvolutionDiagnostic ToDiagnostic() => new(Code, Message, IsRedacted);
    }
}
