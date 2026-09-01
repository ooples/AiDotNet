using System.Collections.ObjectModel;
using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

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
