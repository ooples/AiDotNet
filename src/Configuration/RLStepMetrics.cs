using AiDotNet.Helpers;

namespace AiDotNet.Configuration;

/// <summary>
/// Metrics for a single RL training step.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class RLStepMetrics<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Creates a new instance with default values.
    /// </summary>
    public RLStepMetrics()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        Reward = _numOps.Zero;
    }

    /// <summary>
    /// Current episode number.
    /// </summary>
    public int Episode { get; init; }

    /// <summary>
    /// Step number within the current episode.
    /// </summary>
    public int Step { get; init; }

    /// <summary>
    /// Total steps across all episodes.
    /// </summary>
    public int TotalSteps { get; init; }

    /// <summary>
    /// Reward received for this step.
    /// </summary>
    public T Reward { get; init; }

    /// <summary>
    /// Training loss (if training occurred this step).
    /// </summary>
    public T? Loss { get; init; }

    /// <summary>
    /// Whether training occurred this step.
    /// </summary>
    public bool DidTrain { get; init; }
}
