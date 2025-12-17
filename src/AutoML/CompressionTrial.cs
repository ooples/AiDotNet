using AiDotNet.Enums;
using AiDotNet.ModelCompression;

namespace AiDotNet.AutoML;

/// <summary>
/// Represents a compression configuration to be evaluated.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class CompressionTrial<T>
{
    /// <summary>
    /// Gets or sets the compression technique used.
    /// </summary>
    public CompressionType Technique { get; set; }

    /// <summary>
    /// Gets or sets the hyperparameters for the compression technique.
    /// </summary>
    public Dictionary<string, object> Hyperparameters { get; set; } = new();

    /// <summary>
    /// Gets or sets the resulting compression metrics.
    /// </summary>
    public CompressionMetrics<T>? Metrics { get; set; }

    /// <summary>
    /// Gets or sets the fitness score for this trial.
    /// </summary>
    public T FitnessScore { get; set; } = default!;

    /// <summary>
    /// Gets or sets whether this trial completed successfully.
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Gets or sets any error message from the trial.
    /// </summary>
    public string? ErrorMessage { get; set; }
}

