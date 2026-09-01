using System.Collections.ObjectModel;
using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

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
