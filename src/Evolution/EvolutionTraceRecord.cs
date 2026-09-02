using System.Collections.ObjectModel;
using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>One immutable, serializable trace entry describing a single evaluation of an evolution run.</summary>
/// <remarks>
/// <para>
/// A record is produced by <see cref="EvolutionTraceObserver{TGenome}"/> from an
/// <see cref="EvolutionEventKind.Evaluated"/> event and is written to and read back from a trace file by
/// <see cref="EvolutionTraceFile"/>. It carries the candidate's identity and outcome, its placement in the archive,
/// its cost, its ancestry, and - the part that makes a trace useful for reinforcement learning and post-hoc analysis -
/// the change in quality relative to its parent.
/// </para>
/// <para>
/// It records strictly more than OpenEvolve's <c>EvolutionTrace</c> dataclass (evolution_trace.py:24-60), which has
/// no status, no cost, no cache information, no archive placement, no attempt count, no inspiration identifiers, and
/// no task, evaluator, or configuration version hashes, and whose <c>code_diff</c> field is never populated. Two
/// coverage differences matter more than the field list. OpenEvolve writes a trace only for a child that evaluated
/// successfully and only while its parent is still present in the database (process_parallel.py:613-640), so failures,
/// timeouts, duplicates, and every trace whose parent has since been replaced are silently absent; this record is
/// written for every terminal <see cref="EvolutionEvaluationStatus"/>, and <see cref="ParentQuality"/> comes from the
/// observer's own bounded cache so a replaced parent still yields a delta. And every record is stamped with
/// <see cref="SchemaVersion"/> plus the three version hashes, so a trace can be checked against the run that produced
/// it rather than merely assumed to match.
/// </para>
/// <para>
/// A record never contains genome content: it is a metadata trail, so a trace stays small and cannot leak model
/// weights, source code, or prompts. Correlate a record with the archives through <see cref="GenomeId"/> or
/// <see cref="EvaluationId"/> when you need the candidate itself.
/// </para>
/// <para><b>For Beginners:</b> Think of one record as a single line in the run's diary: "candidate 42 came from
/// candidate 17, scored 0.83 which is 0.05 better than its parent, landed in archive cell 3,7 on island 1, took two
/// attempts and 1.2 seconds". Collect all of them and you can plot progress over time, find which operators actually
/// helped, or replay the search. You normally do not build these by hand - the trace observer creates them for you and
/// <see cref="EvolutionTraceFile"/> reads them back - but every field is public so a test or an analysis script can
/// construct exactly the record it wants to assert on.</para>
/// </remarks>
public sealed class EvolutionTraceRecord
{
    /// <summary>The schema version stamped on records written by this build.</summary>
    public const int CurrentSchemaVersion = 1;

    private readonly ReadOnlyDictionary<string, double> _descriptors =
        new(new Dictionary<string, double>(StringComparer.Ordinal));
    private readonly ReadOnlyDictionary<string, double> _metrics =
        new(new Dictionary<string, double>(StringComparer.Ordinal));
    private readonly ReadOnlyDictionary<string, double> _metricDeltas =
        new(new Dictionary<string, double>(StringComparer.Ordinal));
    private readonly ReadOnlyCollection<string> _parentIds = Array.AsReadOnly(Array.Empty<string>());
    private readonly ReadOnlyCollection<string> _inspirationIds = Array.AsReadOnly(Array.Empty<string>());
    private readonly ReadOnlyCollection<EvolutionDiagnostic> _diagnostics =
        Array.AsReadOnly(Array.Empty<EvolutionDiagnostic>());
    private readonly double? _quality;
    private readonly double? _parentQuality;
    private readonly double? _qualityDelta;
    private readonly double _costUnits;
    private readonly int _attemptCount;
    private readonly TimeSpan _elapsed;
    private readonly int? _rejectedStage;
    private readonly string? _cell;
    private readonly string? _parentGenomeId;
    private readonly string? _variationOperatorId;
    private readonly string? _refinerId;

    /// <summary>Initializes the identity and outcome fields every trace record carries.</summary>
    /// <param name="sequence">The non-negative observer event sequence, which totally orders a run's records.</param>
    /// <param name="evaluationId">The non-negative engine evaluation identifier.</param>
    /// <param name="genomeId">The non-blank canonical genome identity.</param>
    /// <param name="status">How the evaluation ended.</param>
    /// <param name="direction">Whether larger or smaller quality is better.</param>
    /// <param name="island">The non-negative zero-based island index.</param>
    /// <param name="generation">The non-negative logical generation.</param>
    /// <param name="cacheStatus">Whether the result was computed or served from the evaluation cache.</param>
    /// <param name="recordedAtUtc">When the record was created, in UTC.</param>
    /// <param name="taskVersionHash">The non-blank task version hash in effect for the evaluation.</param>
    /// <param name="evaluatorVersionHash">The non-blank evaluator version hash in effect for the evaluation.</param>
    /// <param name="configurationHash">The non-blank engine configuration hash in effect for the evaluation.</param>
    /// <exception cref="ArgumentNullException">A string argument is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">A string argument is empty or white space.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// A numeric argument is negative, an enumeration argument is undefined, or <paramref name="recordedAtUtc"/> is
    /// not expressed in UTC.
    /// </exception>
    public EvolutionTraceRecord(
        long sequence,
        long evaluationId,
        string genomeId,
        EvolutionEvaluationStatus status,
        EvolutionOptimizationDirection direction,
        int island,
        long generation,
        EvolutionCacheStatus cacheStatus,
        DateTimeOffset recordedAtUtc,
        string taskVersionHash,
        string evaluatorVersionHash,
        string configurationHash)
    {
        if (sequence < 0) throw new ArgumentOutOfRangeException(nameof(sequence));
        if (evaluationId < 0) throw new ArgumentOutOfRangeException(nameof(evaluationId));
        Guard.NotNullOrWhiteSpace(genomeId);
        if (!Enum.IsDefined(typeof(EvolutionEvaluationStatus), status)) throw new ArgumentOutOfRangeException(nameof(status));
        if (!Enum.IsDefined(typeof(EvolutionOptimizationDirection), direction)) throw new ArgumentOutOfRangeException(nameof(direction));
        if (island < 0) throw new ArgumentOutOfRangeException(nameof(island));
        if (generation < 0) throw new ArgumentOutOfRangeException(nameof(generation));
        if (!Enum.IsDefined(typeof(EvolutionCacheStatus), cacheStatus)) throw new ArgumentOutOfRangeException(nameof(cacheStatus));
        if (recordedAtUtc.Offset != TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(recordedAtUtc), "Trace timestamps must be expressed in UTC.");
        Guard.NotNullOrWhiteSpace(taskVersionHash);
        Guard.NotNullOrWhiteSpace(evaluatorVersionHash);
        Guard.NotNullOrWhiteSpace(configurationHash);

        Sequence = sequence;
        EvaluationId = evaluationId;
        GenomeId = genomeId.Trim();
        Status = status;
        Direction = direction;
        Island = island;
        Generation = generation;
        CacheStatus = cacheStatus;
        RecordedAtUtc = recordedAtUtc;
        TaskVersionHash = taskVersionHash.Trim();
        EvaluatorVersionHash = evaluatorVersionHash.Trim();
        ConfigurationHash = configurationHash.Trim();
    }

    /// <summary>Gets the schema version this record conforms to.</summary>
    public int SchemaVersion => CurrentSchemaVersion;

    /// <summary>Gets the observer event sequence, which totally orders every record of one run.</summary>
    public long Sequence { get; }

    /// <summary>Gets the engine evaluation identifier.</summary>
    public long EvaluationId { get; }

    /// <summary>Gets the canonical genome identity.</summary>
    public string GenomeId { get; }

    /// <summary>Gets how the evaluation ended.</summary>
    public EvolutionEvaluationStatus Status { get; }

    /// <summary>Gets whether larger or smaller quality is better.</summary>
    public EvolutionOptimizationDirection Direction { get; }

    /// <summary>Gets the zero-based island index the candidate belonged to.</summary>
    public int Island { get; }

    /// <summary>Gets the logical generation the candidate was produced in.</summary>
    public long Generation { get; }

    /// <summary>Gets whether the result was computed or served from the evaluation cache.</summary>
    public EvolutionCacheStatus CacheStatus { get; }

    /// <summary>Gets when the record was created, in UTC.</summary>
    public DateTimeOffset RecordedAtUtc { get; }

    /// <summary>Gets the task version hash in effect for the evaluation.</summary>
    public string TaskVersionHash { get; }

    /// <summary>Gets the evaluator version hash in effect for the evaluation.</summary>
    public string EvaluatorVersionHash { get; }

    /// <summary>Gets the engine configuration hash in effect for the evaluation.</summary>
    public string ConfigurationHash { get; }

    /// <summary>Gets the scalar quality, or <c>null</c> for a status that produced none.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The assigned value is not finite.</exception>
    public double? Quality
    {
        get => _quality;
        init
        {
            if (value.HasValue && !EvolutionDescriptorDefinition.IsFinite(value.Value))
                throw new ArgumentOutOfRangeException(nameof(value), "A trace quality must be finite.");
            _quality = value;
        }
    }

    /// <summary>Gets the archive insertion outcome, or <c>null</c> when no insertion was attempted.</summary>
    public EvolutionArchiveInsertionResult? InsertionResult { get; init; }

    /// <summary>Gets the archive cell the candidate mapped to, or <c>null</c> when it mapped to none.</summary>
    /// <remarks>
    /// Populated only when the observer was given the archive's descriptor definitions and
    /// <c>EvolutionTraceOptions.IncludeDescriptors</c> is set. The value is
    /// <see cref="EvolutionCellKey.StableKey"/>, a comma-separated bin index per descriptor.
    /// </remarks>
    /// <exception cref="ArgumentException">The assigned value is empty or white space.</exception>
    public string? Cell
    {
        get => _cell;
        init => _cell = NormalizeOptional(value, nameof(value));
    }

    /// <summary>Gets the behaviour descriptor values, ordered by name.</summary>
    /// <exception cref="ArgumentException">A key is blank, duplicated, or a value is not finite.</exception>
    public IReadOnlyDictionary<string, double> Descriptors
    {
        get => _descriptors;
        init => _descriptors = CopyFiniteValues(value, nameof(value));
    }

    /// <summary>Gets the reporting metrics that took no part in archive placement, ordered by name.</summary>
    /// <exception cref="ArgumentException">A key is blank, duplicated, or a value is not finite.</exception>
    public IReadOnlyDictionary<string, double> Metrics
    {
        get => _metrics;
        init => _metrics = CopyFiniteValues(value, nameof(value));
    }

    /// <summary>Gets the canonical identity of the parent the deltas are measured against, when one was known.</summary>
    /// <exception cref="ArgumentException">The assigned value is empty or white space.</exception>
    public string? ParentGenomeId
    {
        get => _parentGenomeId;
        init => _parentGenomeId = NormalizeOptional(value, nameof(value));
    }

    /// <summary>Gets the parent's quality, when the observer's bounded cache still held it.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The assigned value is not finite.</exception>
    public double? ParentQuality
    {
        get => _parentQuality;
        init
        {
            if (value.HasValue && !EvolutionDescriptorDefinition.IsFinite(value.Value))
                throw new ArgumentOutOfRangeException(nameof(value), "A trace parent quality must be finite.");
            _parentQuality = value;
        }
    }

    /// <summary>Gets the raw quality change from parent to child, without applying the optimization direction.</summary>
    /// <remarks>
    /// The sign is always <c>child - parent</c>, matching OpenEvolve's <c>calculate_improvement</c>
    /// (evolution_trace.py:51-60). Use <see cref="IsImprovement"/> for the direction-aware verdict, which OpenEvolve
    /// does not compute at all: it treats any positive delta as an improvement, so every minimising objective is
    /// scored backwards in its statistics (evolution_trace.py:214-219).
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">The assigned value is not finite.</exception>
    public double? QualityDelta
    {
        get => _qualityDelta;
        init
        {
            if (value.HasValue && !EvolutionDescriptorDefinition.IsFinite(value.Value))
                throw new ArgumentOutOfRangeException(nameof(value), "A trace quality delta must be finite.");
            _qualityDelta = value;
        }
    }

    /// <summary>Gets whether the candidate improved on its parent in the run's optimization direction.</summary>
    public bool IsImprovement { get; init; }

    /// <summary>Gets the per-metric change from parent to child, ordered by name.</summary>
    /// <exception cref="ArgumentException">A key is blank, duplicated, or a value is not finite.</exception>
    public IReadOnlyDictionary<string, double> MetricDeltas
    {
        get => _metricDeltas;
        init => _metricDeltas = CopyFiniteValues(value, nameof(value));
    }

    /// <summary>Gets the canonical identities of the candidate's direct parents.</summary>
    /// <exception cref="ArgumentException">An identity is empty or white space.</exception>
    public IReadOnlyList<string> ParentIds
    {
        get => _parentIds;
        init => _parentIds = CopyIdentities(value, nameof(value));
    }

    /// <summary>Gets the canonical identities of the elites supplied to variation as inspirations.</summary>
    /// <exception cref="ArgumentException">An identity is empty or white space.</exception>
    public IReadOnlyList<string> InspirationIds
    {
        get => _inspirationIds;
        init => _inspirationIds = CopyIdentities(value, nameof(value));
    }

    /// <summary>Gets the identifier of the variation operator that produced the candidate.</summary>
    /// <exception cref="ArgumentException">The assigned value is empty or white space.</exception>
    public string? VariationOperatorId
    {
        get => _variationOperatorId;
        init => _variationOperatorId = NormalizeOptional(value, nameof(value));
    }

    /// <summary>Gets the identifier of the refiner that adjusted the candidate, when one did.</summary>
    /// <exception cref="ArgumentException">The assigned value is empty or white space.</exception>
    public string? RefinerId
    {
        get => _refinerId;
        init => _refinerId = NormalizeOptional(value, nameof(value));
    }

    /// <summary>Gets the deterministic random stream the candidate was produced from.</summary>
    public ulong SeedStream { get; init; }

    /// <summary>Gets how many evaluator attempts the canonical candidate consumed.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The assigned value is negative.</exception>
    public int AttemptCount
    {
        get => _attemptCount;
        init
        {
            if (value < 0) throw new ArgumentOutOfRangeException(nameof(value));
            _attemptCount = value;
        }
    }

    /// <summary>Gets the task-defined resource units the evaluation consumed.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The assigned value is negative or not finite.</exception>
    public double CostUnits
    {
        get => _costUnits;
        init
        {
            if (!EvolutionDescriptorDefinition.IsFinite(value) || value < 0)
                throw new ArgumentOutOfRangeException(nameof(value), "Trace cost units must be finite and non-negative.");
            _costUnits = value;
        }
    }

    /// <summary>Gets the wall-clock evaluator time the evaluation consumed.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The assigned value is negative.</exception>
    public TimeSpan Elapsed
    {
        get => _elapsed;
        init
        {
            if (value < TimeSpan.Zero) throw new ArgumentOutOfRangeException(nameof(value));
            _elapsed = value;
        }
    }

    /// <summary>Gets the cascade stage that rejected the candidate, or <c>null</c> when none did.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The assigned value is negative.</exception>
    public int? RejectedStage
    {
        get => _rejectedStage;
        init
        {
            if (value.HasValue && value.Value < 0) throw new ArgumentOutOfRangeException(nameof(value));
            _rejectedStage = value;
        }
    }

    /// <summary>Gets the diagnostics attached to the evaluation, in the order the engine reported them.</summary>
    /// <exception cref="ArgumentException">An element is <c>null</c>.</exception>
    public IReadOnlyList<EvolutionDiagnostic> Diagnostics
    {
        get => _diagnostics;
        init => _diagnostics = CopyDiagnostics(value, nameof(value));
    }

    private static ReadOnlyCollection<EvolutionDiagnostic> CopyDiagnostics(
        IReadOnlyList<EvolutionDiagnostic>? values, string parameterName)
    {
        if (values is null) return Array.AsReadOnly(Array.Empty<EvolutionDiagnostic>());
        EvolutionDiagnostic[] copy = values.ToArray();
        foreach (EvolutionDiagnostic diagnostic in copy)
            if (diagnostic is null)
                throw new ArgumentException("Trace diagnostics cannot contain null values.", parameterName);
        return Array.AsReadOnly(copy);
    }

    private static string? NormalizeOptional(string? value, string parameterName)
    {
        if (value is null) return null;
        if (string.IsNullOrWhiteSpace(value))
            throw new ArgumentException("A trace identifier cannot be empty or white space.", parameterName);
        return value.Trim();
    }

    private static ReadOnlyDictionary<string, double> CopyFiniteValues(
        IReadOnlyDictionary<string, double>? values, string parameterName)
    {
        var copy = new Dictionary<string, double>(StringComparer.Ordinal);
        if (values is null) return new ReadOnlyDictionary<string, double>(copy);
        foreach (KeyValuePair<string, double> pair in values.OrderBy(item => item.Key, StringComparer.Ordinal))
        {
            if (string.IsNullOrWhiteSpace(pair.Key))
                throw new ArgumentException("Trace value names cannot be empty or white space.", parameterName);
            if (!EvolutionDescriptorDefinition.IsFinite(pair.Value))
                throw new ArgumentException("Trace values must be finite.", parameterName);
            string key = pair.Key.Trim();
            if (copy.ContainsKey(key)) throw new ArgumentException("Trace value names must be unique.", parameterName);
            copy.Add(key, pair.Value);
        }
        return new ReadOnlyDictionary<string, double>(copy);
    }

    private static ReadOnlyCollection<string> CopyIdentities(IReadOnlyList<string>? values, string parameterName)
    {
        if (values is null) return Array.AsReadOnly(Array.Empty<string>());
        string[] copy = values.ToArray();
        foreach (string value in copy)
            if (string.IsNullOrWhiteSpace(value))
                throw new ArgumentException("Trace identities cannot be empty or white space.", parameterName);
        return Array.AsReadOnly(copy.Select(value => value.Trim()).ToArray());
    }
}
