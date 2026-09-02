using System.Collections.ObjectModel;
using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>A complete immutable engine evaluation suitable for archives, observers, and checkpoints.</summary>
/// <remarks>
/// <para>
/// An <see cref="EvolutionTaskResult"/> holds only what the task reported. The engine wraps it into this type,
/// adding the stable evaluation and canonical genome identities, wall-clock and attempt cost, lineage, cache
/// status, and the task, evaluator, and configuration version hashes that make the record verifiable when a
/// checkpoint is resumed. Construction re-validates the metrics through <see cref="EvolutionTaskResult"/>, so a
/// <see cref="EvolutionEvaluationStatus.Completed"/> status always carries a finite quality, and every collection
/// is stored as a defensive read-only copy with ordinal descriptor keys.
/// </para>
/// <para><b>For Beginners:</b> This is the full report card for one candidate. It records the score
/// (<see cref="Quality"/>) and which way is better (<see cref="Direction"/>), the behavior descriptors that decide
/// which archive cell it competes for, how the evaluation ended (<see cref="Status"/>), how long and how many
/// attempts it took (<see cref="Cost"/>), which earlier candidates it descended from (<see cref="Lineage"/>), and
/// whether the score was recomputed or reused from the cache (<see cref="CacheStatus"/>). For example, after a run
/// you can read <c>archive.Best.Evaluation.Quality</c> for the winning score and
/// <c>archive.Best.Evaluation.Lineage.ParentIds</c> to trace its ancestry. Because instances never change after
/// construction, they are safe to share between threads, observers, and persisted checkpoints.</para>
/// </remarks>
public sealed class EvolutionEvaluation
{
    private readonly ReadOnlyDictionary<string, double> _descriptors;
    private readonly ReadOnlyDictionary<string, double> _metrics;
    private readonly ReadOnlyCollection<double> _objectives;
    private readonly ReadOnlyCollection<double> _constraintViolations;
    private readonly ReadOnlyCollection<EvolutionDiagnostic> _diagnostics;
    private readonly ReadOnlyCollection<EvolutionArtifact> _artifacts;

    /// <summary>Initializes a complete evaluation.</summary>
    /// <param name="evaluationId">The nonnegative stable evaluation identifier.</param>
    /// <param name="genomeId">The canonical genome identity; trimmed on construction.</param>
    /// <param name="status">The terminal status of the evaluation.</param>
    /// <param name="quality">The scalar quality, required and finite when <paramref name="status"/> is completed.</param>
    /// <param name="direction">Whether larger or smaller quality is better.</param>
    /// <param name="descriptors">Named finite behavior descriptor values.</param>
    /// <param name="objectives">Optional finite multi-objective values.</param>
    /// <param name="constraintViolations">Nonnegative finite constraint violation magnitudes.</param>
    /// <param name="cost">Elapsed time, attempt count, and task cost units.</param>
    /// <param name="lineage">Ancestry and deterministic-stream metadata for the candidate.</param>
    /// <param name="cacheStatus">Whether the result was computed or served from the evaluation cache.</param>
    /// <param name="diagnostics">Up to 64 diagnostics attached to the result.</param>
    /// <param name="taskVersionHash">The task version hash in effect for this evaluation.</param>
    /// <param name="evaluatorVersionHash">The evaluator version hash in effect for this evaluation.</param>
    /// <param name="configurationHash">The engine configuration hash in effect for this evaluation.</param>
    /// <param name="metrics">Optional named, finite reporting metrics that do not take part in archive placement.</param>
    /// <param name="artifacts">Optional bounded, untrusted text artifacts retained by the engine.</param>
    /// <exception cref="ArgumentNullException">A reference argument is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">
    /// A string argument is empty or white space, a completed status lacks a finite quality, descriptor names
    /// collide, or more than 64 diagnostics are supplied.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="evaluationId"/> is negative, <paramref name="cacheStatus"/> is undefined, or a metric value
    /// is not finite.
    /// </exception>
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
        string configurationHash,
        IReadOnlyDictionary<string, double>? metrics = null,
        IEnumerable<EvolutionArtifact>? artifacts = null)
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
            constraintViolations, cost.CostUnits, diagnostics, metrics, artifacts);
        EvaluationId = evaluationId;
        GenomeId = genomeId.Trim();
        Status = validated.Status;
        Quality = validated.Quality;
        Direction = validated.Direction;
        _descriptors = new ReadOnlyDictionary<string, double>(validated.Descriptors.ToDictionary(
            item => item.Key, item => item.Value, StringComparer.Ordinal));
        _metrics = new ReadOnlyDictionary<string, double>(validated.Metrics.ToDictionary(
            item => item.Key, item => item.Value, StringComparer.Ordinal));
        _objectives = Array.AsReadOnly(validated.Objectives.ToArray());
        _constraintViolations = Array.AsReadOnly(validated.ConstraintViolations.ToArray());
        Cost = cost;
        Lineage = lineage;
        CacheStatus = cacheStatus;
        _diagnostics = Array.AsReadOnly(validated.Diagnostics.ToArray());
        _artifacts = Array.AsReadOnly(validated.Artifacts.ToArray());
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
    /// <summary>Gets named reporting metrics that never take part in archive placement.</summary>
    public IReadOnlyDictionary<string, double> Metrics => _metrics;
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
    /// <summary>Gets the bounded, untrusted text artifacts the engine retained for this evaluation.</summary>
    /// <remarks>
    /// Empty unless <c>EvolutionEngineOptions.Artifacts.Enabled</c> is set. The content comes from the evaluated
    /// candidate, so treat it as data rather than as instructions or executable text.
    /// </remarks>
    public IReadOnlyList<EvolutionArtifact> Artifacts => _artifacts;
    /// <summary>Gets the task version hash.</summary>
    public string TaskVersionHash { get; }
    /// <summary>Gets the evaluator version hash.</summary>
    public string EvaluatorVersionHash { get; }
    /// <summary>Gets the engine configuration hash.</summary>
    public string ConfigurationHash { get; }
}
