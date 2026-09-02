using System.Collections.ObjectModel;
using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>Task-produced terminal metrics before engine identity, timing, and cache metadata are attached.</summary>
/// <remarks>
/// <para>
/// An <see cref="AiDotNet.Interfaces.IEvolutionTask{TGenome}"/> returns one instance from its evaluate method
/// for each candidate, and the engine wraps it into an <see cref="EvolutionEvaluation"/> by attaching identity,
/// timing, attempt, and cache metadata. The constructor validates everything up front: the status and
/// direction must be defined enum values, <see cref="CostUnits"/> must be finite and non-negative, a
/// <see cref="EvolutionEvaluationStatus.Completed"/> result must carry a finite <see cref="Quality"/>,
/// descriptor names are trimmed and must be unique with finite values, objectives must be finite, constraint
/// violations must be finite and non-negative, and at most 64 non-null diagnostics may be attached. Every
/// collection is defensively copied and exposed read-only, so a result is immutable once constructed and can
/// safely be cached by the engine and reused across islands.
/// </para>
/// <para><b>For Beginners:</b> When you plug your own problem into the evolution engine, the engine hands you
/// a candidate and asks how good it is; this class is how you answer. For a candidate that was evaluated
/// successfully, call <see cref="Completed"/> with its score, the descriptor values that place it in the
/// MAP-Elites grid (for example model size and inference latency), and whether a higher or lower score is
/// better. For a candidate that crashed or broke a rule, call <see cref="Failed"/> with a short machine-readable
/// code and a message so the failure is recorded without stopping the run. Use the full constructor when you
/// also want to report several objectives, constraint violations, or how many resource units the evaluation
/// consumed. It is like a referee's scorecard: it reports the outcome of one match, and the engine attaches
/// who played and when.</para>
/// </remarks>
public sealed class EvolutionTaskResult
{
    private readonly ReadOnlyDictionary<string, double> _descriptors;
    private readonly ReadOnlyCollection<double> _objectives;
    private readonly ReadOnlyCollection<double> _constraintViolations;
    private readonly ReadOnlyCollection<EvolutionDiagnostic> _diagnostics;

    /// <summary>Initializes a task result.</summary>
    /// <param name="status">The terminal evaluation status.</param>
    /// <param name="quality">The finite scalar quality; required when <paramref name="status"/> is <see cref="EvolutionEvaluationStatus.Completed"/>.</param>
    /// <param name="direction">Whether larger or smaller <paramref name="quality"/> values are better.</param>
    /// <param name="descriptors">Optional named, finite descriptor values used to place the candidate in an archive.</param>
    /// <param name="objectives">Optional finite multi-objective values retained alongside the scalar quality.</param>
    /// <param name="constraintViolations">Optional finite, non-negative constraint violation magnitudes.</param>
    /// <param name="costUnits">Finite, non-negative task-defined resource units charged by this evaluation.</param>
    /// <param name="diagnostics">Optional diagnostics, at most 64 and none <c>null</c>.</param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// An enum argument is undefined, or a numeric argument is not finite or violates its sign constraint.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// A completed result lacks a finite quality, descriptor names collide, or the diagnostics are too many or contain <c>null</c>.
    /// </exception>
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
    /// <param name="quality">The finite scalar quality of the candidate.</param>
    /// <param name="descriptors">Named, finite descriptor values used to place the candidate in an archive.</param>
    /// <param name="direction">Whether larger or smaller <paramref name="quality"/> values are better.</param>
    /// <param name="costUnits">Finite, non-negative task-defined resource units charged by this evaluation.</param>
    /// <returns>A result with <see cref="EvolutionEvaluationStatus.Completed"/> status.</returns>
    public static EvolutionTaskResult Completed(
        double quality,
        IReadOnlyDictionary<string, double> descriptors,
        EvolutionOptimizationDirection direction = EvolutionOptimizationDirection.Maximize,
        double costUnits = 0) => new(EvolutionEvaluationStatus.Completed, quality, direction, descriptors, costUnits: costUnits);

    /// <summary>Creates a recoverable failed result.</summary>
    /// <param name="code">A stable machine-readable failure code of at most 128 characters.</param>
    /// <param name="message">A human-readable message of at most 4096 characters.</param>
    /// <returns>A result with <see cref="EvolutionEvaluationStatus.Failed"/> status and one diagnostic.</returns>
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
