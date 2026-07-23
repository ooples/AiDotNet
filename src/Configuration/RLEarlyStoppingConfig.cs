using AiDotNet.Helpers;

namespace AiDotNet.Configuration;

/// <summary>
/// Configuration for early stopping during RL training.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class RLEarlyStoppingConfig<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Creates a new instance with default values.
    /// </summary>
    public RLEarlyStoppingConfig()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        MinImprovement = _numOps.FromDouble(0.01);
    }

    /// <summary>
    /// Stop training if average reward exceeds this threshold.
    /// </summary>
    public T? RewardThreshold { get; set; }

    /// <summary>
    /// Stop if no improvement for this many episodes.
    /// </summary>
    public int PatienceEpisodes { get; set; } = 100;

    /// <summary>
    /// Minimum improvement to reset patience counter.
    /// </summary>
    public T MinImprovement { get; set; }
}
