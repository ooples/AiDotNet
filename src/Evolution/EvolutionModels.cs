using System.Collections.ObjectModel;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>Associates an immutable task genome with its stable canonical identity.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public sealed class EvolutionCanonicalGenome<TGenome>
{
    /// <summary>Initializes a canonical genome.</summary>
    /// <param name="genome">An immutable genome snapshot.</param>
    /// <param name="id">A stable identity for its semantics rather than its incidental representation.</param>
    public EvolutionCanonicalGenome(TGenome genome, string id)
    {
        if (genome is null) throw new ArgumentNullException(nameof(genome));
        Guard.NotNullOrWhiteSpace(id);
        Genome = genome;
        Id = id.Trim();
    }

    /// <summary>Gets the immutable task-specific genome.</summary>
    public TGenome Genome { get; }

    /// <summary>Gets the stable canonical genome identity.</summary>
    public string Id { get; }
}

/// <summary>Immutable ancestry and deterministic-stream metadata for a candidate.</summary>
public sealed class EvolutionLineage
{
    private readonly ReadOnlyCollection<string> _parentIds;
    private readonly ReadOnlyCollection<string> _inspirationIds;

    /// <summary>Initializes lineage metadata.</summary>
    public EvolutionLineage(
        IEnumerable<string>? parentIds,
        IEnumerable<string>? inspirationIds,
        string variationOperatorId,
        string? refinerId,
        long generation,
        int island,
        ulong seedStream)
    {
        Guard.NotNullOrWhiteSpace(variationOperatorId);
        if (generation < 0) throw new ArgumentOutOfRangeException(nameof(generation));
        if (island < 0) throw new ArgumentOutOfRangeException(nameof(island));
        _parentIds = Array.AsReadOnly(CopyIds(parentIds, nameof(parentIds)));
        _inspirationIds = Array.AsReadOnly(CopyIds(inspirationIds, nameof(inspirationIds)));
        VariationOperatorId = variationOperatorId.Trim();
        RefinerId = string.IsNullOrWhiteSpace(refinerId) ? null : refinerId!.Trim();
        Generation = generation;
        Island = island;
        SeedStream = seedStream;
    }

    /// <summary>Gets canonical IDs of direct parents.</summary>
    public IReadOnlyList<string> ParentIds => _parentIds;

    /// <summary>Gets canonical IDs supplied as variation inspirations.</summary>
    public IReadOnlyList<string> InspirationIds => _inspirationIds;

    /// <summary>Gets the variation operator identifier.</summary>
    public string VariationOperatorId { get; }

    /// <summary>Gets the optional refiner identifier.</summary>
    public string? RefinerId { get; }

    /// <summary>Gets the logical generation number.</summary>
    public long Generation { get; }

    /// <summary>Gets the zero-based island index.</summary>
    public int Island { get; }

    /// <summary>Gets the deterministic random stream identifier.</summary>
    public ulong SeedStream { get; }

    private static string[] CopyIds(IEnumerable<string>? values, string parameterName)
    {
        if (values is null) return Array.Empty<string>();
        string[] result = values.ToArray();
        if (result.Any(string.IsNullOrWhiteSpace))
            throw new ArgumentException("Identity collections cannot contain empty values.", parameterName);
        return result.Select(value => value.Trim()).ToArray();
    }
}

/// <summary>An immutable candidate assigned by the evolution engine.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public sealed class EvolutionCandidate<TGenome>
{
    /// <summary>Initializes a candidate.</summary>
    public EvolutionCandidate(long evaluationId, EvolutionCanonicalGenome<TGenome> canonicalGenome, EvolutionLineage lineage)
    {
        if (evaluationId < 0) throw new ArgumentOutOfRangeException(nameof(evaluationId));
        Guard.NotNull(canonicalGenome);
        Guard.NotNull(lineage);
        EvaluationId = evaluationId;
        CanonicalGenome = canonicalGenome;
        Lineage = lineage;
    }

    /// <summary>Gets the monotonically increasing evaluation identifier.</summary>
    public long EvaluationId { get; }

    /// <summary>Gets the canonical genome snapshot.</summary>
    public EvolutionCanonicalGenome<TGenome> CanonicalGenome { get; }

    /// <summary>Gets the immutable lineage.</summary>
    public EvolutionLineage Lineage { get; }
}

/// <summary>One bounded diagnostic attached to an evaluation.</summary>
public sealed class EvolutionDiagnostic
{
    /// <summary>Initializes a diagnostic.</summary>
    /// <param name="code">A stable machine-readable code.</param>
    /// <param name="message">A bounded human-readable message.</param>
    /// <param name="isRedacted">Whether sensitive detail was removed.</param>
    public EvolutionDiagnostic(string code, string message, bool isRedacted = false)
    {
        Guard.NotNullOrWhiteSpace(code);
        Guard.NotNull(message);
        if (code.Length > 128) throw new ArgumentException("Diagnostic codes cannot exceed 128 characters.", nameof(code));
        if (message.Length > 4096) throw new ArgumentException("Diagnostic messages cannot exceed 4096 characters.", nameof(message));
        Code = code.Trim();
        Message = message;
        IsRedacted = isRedacted;
    }

    /// <summary>Gets the stable diagnostic code.</summary>
    public string Code { get; }

    /// <summary>Gets the human-readable diagnostic message.</summary>
    public string Message { get; }

    /// <summary>Gets whether sensitive content was redacted.</summary>
    public bool IsRedacted { get; }
}

/// <summary>Task-produced terminal metrics before engine identity, timing, and cache metadata are attached.</summary>
public sealed class EvolutionTaskResult
{
    private readonly ReadOnlyDictionary<string, double> _descriptors;
    private readonly ReadOnlyCollection<double> _objectives;
    private readonly ReadOnlyCollection<double> _constraintViolations;
    private readonly ReadOnlyCollection<EvolutionDiagnostic> _diagnostics;

    /// <summary>Initializes a task result.</summary>
    public EvolutionTaskResult(
        EvolutionEvaluationStatus status,
        double? quality = null,
        EvolutionOptimizationDirection direction = EvolutionOptimizationDirection.Maximize,
        IReadOnlyDictionary<string, double>? descriptors = null,
        IEnumerable<double>? objectives = null,
        IEnumerable<double>? constraintViolations = null,
        double costUnits = 0,
        IEnumerable<EvolutionDiagnostic>? diagnostics = null)
    {
        if (!Enum.IsDefined(typeof(EvolutionEvaluationStatus), status)) throw new ArgumentOutOfRangeException(nameof(status));
        if (!Enum.IsDefined(typeof(EvolutionOptimizationDirection), direction)) throw new ArgumentOutOfRangeException(nameof(direction));
        if (!EvolutionDescriptorDefinition.IsFinite(costUnits) || costUnits < 0) throw new ArgumentOutOfRangeException(nameof(costUnits));
        if (status == EvolutionEvaluationStatus.Completed && (!quality.HasValue || !EvolutionDescriptorDefinition.IsFinite(quality.Value)))
            throw new ArgumentException("Completed evaluations require a finite quality value.", nameof(quality));
        if (quality.HasValue && !EvolutionDescriptorDefinition.IsFinite(quality.Value))
            throw new ArgumentOutOfRangeException(nameof(quality));

        var descriptorCopy = new Dictionary<string, double>(StringComparer.Ordinal);
        if (descriptors is not null)
        {
            foreach (KeyValuePair<string, double> descriptor in descriptors)
            {
                Guard.NotNullOrWhiteSpace(descriptor.Key);
                if (!EvolutionDescriptorDefinition.IsFinite(descriptor.Value))
                    throw new ArgumentOutOfRangeException(nameof(descriptors), $"Descriptor '{descriptor.Key}' must be finite.");
                string descriptorName = descriptor.Key.Trim();
                if (descriptorCopy.ContainsKey(descriptorName))
                    throw new ArgumentException("Descriptor names must be unique.", nameof(descriptors));
                descriptorCopy.Add(descriptorName, descriptor.Value);
            }
        }

        double[] objectiveCopy = CopyFinite(objectives, nameof(objectives), nonnegative: false);
        double[] violationCopy = CopyFinite(constraintViolations, nameof(constraintViolations), nonnegative: true);
        EvolutionDiagnostic[] diagnosticCopy = diagnostics?.ToArray() ?? Array.Empty<EvolutionDiagnostic>();
        if (diagnosticCopy.Length > 64) throw new ArgumentException("At most 64 diagnostics may be attached to one result.", nameof(diagnostics));
        if (diagnosticCopy.Any(item => item is null)) throw new ArgumentException("Diagnostics cannot contain null entries.", nameof(diagnostics));

        Status = status;
        Quality = quality;
        Direction = direction;
        _descriptors = new ReadOnlyDictionary<string, double>(descriptorCopy);
        _objectives = Array.AsReadOnly(objectiveCopy);
        _constraintViolations = Array.AsReadOnly(violationCopy);
        CostUnits = costUnits;
        _diagnostics = Array.AsReadOnly(diagnosticCopy);
    }

    /// <summary>Gets the terminal status.</summary>
    public EvolutionEvaluationStatus Status { get; }

    /// <summary>Gets the scalar quality, when present.</summary>
    public double? Quality { get; }

    /// <summary>Gets the scalar optimization direction.</summary>
    public EvolutionOptimizationDirection Direction { get; }

    /// <summary>Gets named quality-diversity descriptor values.</summary>
    public IReadOnlyDictionary<string, double> Descriptors => _descriptors;

    /// <summary>Gets optional multi-objective values retained independently of scalar archive quality.</summary>
    public IReadOnlyList<double> Objectives => _objectives;

    /// <summary>Gets nonnegative constraint violations.</summary>
    public IReadOnlyList<double> ConstraintViolations => _constraintViolations;

    /// <summary>Gets task-defined resource units charged by this evaluation.</summary>
    public double CostUnits { get; }

    /// <summary>Gets bounded diagnostics.</summary>
    public IReadOnlyList<EvolutionDiagnostic> Diagnostics => _diagnostics;

    /// <summary>Creates a successful scalar-quality result.</summary>
    public static EvolutionTaskResult Completed(
        double quality,
        IReadOnlyDictionary<string, double> descriptors,
        EvolutionOptimizationDirection direction = EvolutionOptimizationDirection.Maximize,
        double costUnits = 0) => new(EvolutionEvaluationStatus.Completed, quality, direction, descriptors, costUnits: costUnits);

    /// <summary>Creates a recoverable failed result.</summary>
    public static EvolutionTaskResult Failed(string code, string message) => new(
        EvolutionEvaluationStatus.Failed,
        diagnostics: new[] { new EvolutionDiagnostic(code, message) });

    private static double[] CopyFinite(IEnumerable<double>? values, string parameterName, bool nonnegative)
    {
        double[] result = values?.ToArray() ?? Array.Empty<double>();
        if (result.Any(value => !EvolutionDescriptorDefinition.IsFinite(value) || (nonnegative && value < 0)))
            throw new ArgumentOutOfRangeException(parameterName);
        return result;
    }
}

/// <summary>Immutable evaluation cost and elapsed-time metadata.</summary>
public sealed class EvolutionEvaluationCost
{
    /// <summary>Initializes cost metadata.</summary>
    public EvolutionEvaluationCost(TimeSpan elapsed, int attemptCount, double costUnits)
    {
        if (elapsed < TimeSpan.Zero) throw new ArgumentOutOfRangeException(nameof(elapsed));
        if (attemptCount < 0) throw new ArgumentOutOfRangeException(nameof(attemptCount));
        if (!EvolutionDescriptorDefinition.IsFinite(costUnits) || costUnits < 0) throw new ArgumentOutOfRangeException(nameof(costUnits));
        Elapsed = elapsed;
        AttemptCount = attemptCount;
        CostUnits = costUnits;
    }

    /// <summary>Gets wall-clock evaluator time.</summary>
    public TimeSpan Elapsed { get; }

    /// <summary>Gets the one-based attempt count for this canonical candidate.</summary>
    public int AttemptCount { get; }

    /// <summary>Gets task-defined resource units.</summary>
    public double CostUnits { get; }
}

/// <summary>A complete immutable engine evaluation suitable for archives, observers, and checkpoints.</summary>
public sealed class EvolutionEvaluation
{
    private readonly ReadOnlyDictionary<string, double> _descriptors;
    private readonly ReadOnlyCollection<double> _objectives;
    private readonly ReadOnlyCollection<double> _constraintViolations;
    private readonly ReadOnlyCollection<EvolutionDiagnostic> _diagnostics;

    /// <summary>Initializes a complete evaluation.</summary>
    public EvolutionEvaluation(
        long evaluationId,
        string genomeId,
        EvolutionEvaluationStatus status,
        double? quality,
        EvolutionOptimizationDirection direction,
        IReadOnlyDictionary<string, double> descriptors,
        IEnumerable<double> objectives,
        IEnumerable<double> constraintViolations,
        EvolutionEvaluationCost cost,
        EvolutionLineage lineage,
        EvolutionCacheStatus cacheStatus,
        IEnumerable<EvolutionDiagnostic> diagnostics,
        string taskVersionHash,
        string evaluatorVersionHash,
        string configurationHash)
    {
        if (evaluationId < 0) throw new ArgumentOutOfRangeException(nameof(evaluationId));
        Guard.NotNullOrWhiteSpace(genomeId);
        Guard.NotNull(descriptors);
        Guard.NotNull(objectives);
        Guard.NotNull(constraintViolations);
        Guard.NotNull(cost);
        Guard.NotNull(lineage);
        Guard.NotNull(diagnostics);
        Guard.NotNullOrWhiteSpace(taskVersionHash);
        Guard.NotNullOrWhiteSpace(evaluatorVersionHash);
        Guard.NotNullOrWhiteSpace(configurationHash);
        if (!Enum.IsDefined(typeof(EvolutionCacheStatus), cacheStatus)) throw new ArgumentOutOfRangeException(nameof(cacheStatus));

        EvolutionTaskResult validated = new(status, quality, direction, descriptors, objectives,
            constraintViolations, cost.CostUnits, diagnostics);
        EvaluationId = evaluationId;
        GenomeId = genomeId.Trim();
        Status = validated.Status;
        Quality = validated.Quality;
        Direction = validated.Direction;
        _descriptors = new ReadOnlyDictionary<string, double>(validated.Descriptors.ToDictionary(
            item => item.Key, item => item.Value, StringComparer.Ordinal));
        _objectives = Array.AsReadOnly(validated.Objectives.ToArray());
        _constraintViolations = Array.AsReadOnly(validated.ConstraintViolations.ToArray());
        Cost = cost;
        Lineage = lineage;
        CacheStatus = cacheStatus;
        _diagnostics = Array.AsReadOnly(validated.Diagnostics.ToArray());
        TaskVersionHash = taskVersionHash.Trim();
        EvaluatorVersionHash = evaluatorVersionHash.Trim();
        ConfigurationHash = configurationHash.Trim();
    }

    /// <summary>Gets the stable evaluation ID.</summary>
    public long EvaluationId { get; }
    /// <summary>Gets the canonical genome ID.</summary>
    public string GenomeId { get; }
    /// <summary>Gets the terminal status.</summary>
    public EvolutionEvaluationStatus Status { get; }
    /// <summary>Gets the scalar quality, when present.</summary>
    public double? Quality { get; }
    /// <summary>Gets the scalar optimization direction.</summary>
    public EvolutionOptimizationDirection Direction { get; }
    /// <summary>Gets named descriptor values.</summary>
    public IReadOnlyDictionary<string, double> Descriptors => _descriptors;
    /// <summary>Gets optional objective values.</summary>
    public IReadOnlyList<double> Objectives => _objectives;
    /// <summary>Gets constraint violations.</summary>
    public IReadOnlyList<double> ConstraintViolations => _constraintViolations;
    /// <summary>Gets cost and timing metadata.</summary>
    public EvolutionEvaluationCost Cost { get; }
    /// <summary>Gets lineage metadata.</summary>
    public EvolutionLineage Lineage { get; }
    /// <summary>Gets cache metadata.</summary>
    public EvolutionCacheStatus CacheStatus { get; }
    /// <summary>Gets bounded diagnostics.</summary>
    public IReadOnlyList<EvolutionDiagnostic> Diagnostics => _diagnostics;
    /// <summary>Gets the task version hash.</summary>
    public string TaskVersionHash { get; }
    /// <summary>Gets the evaluator version hash.</summary>
    public string EvaluatorVersionHash { get; }
    /// <summary>Gets the engine configuration hash.</summary>
    public string ConfigurationHash { get; }
}

/// <summary>One immutable elite stored in a quality-diversity archive.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public sealed class EvolutionArchiveEntry<TGenome>
{
    /// <summary>Initializes an archive entry.</summary>
    public EvolutionArchiveEntry(EvolutionCellKey cell, EvolutionCandidate<TGenome> candidate, EvolutionEvaluation evaluation)
    {
        Guard.NotNull(cell);
        Guard.NotNull(candidate);
        Guard.NotNull(evaluation);
        if (candidate.EvaluationId != evaluation.EvaluationId || candidate.CanonicalGenome.Id != evaluation.GenomeId)
            throw new ArgumentException("Candidate and evaluation identities must match.", nameof(evaluation));
        Cell = cell;
        Candidate = candidate;
        Evaluation = evaluation;
    }

    /// <summary>Gets the occupied cell.</summary>
    public EvolutionCellKey Cell { get; }
    /// <summary>Gets the canonical candidate.</summary>
    public EvolutionCandidate<TGenome> Candidate { get; }
    /// <summary>Gets the completed evaluation.</summary>
    public EvolutionEvaluation Evaluation { get; }
}
