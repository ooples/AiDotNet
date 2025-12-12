using AiDotNet.Helpers;

namespace AiDotNet.Configuration;

/// <summary>
/// Metrics for a completed RL episode.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class RLEpisodeMetrics<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Creates a new instance with default values.
    /// </summary>
    public RLEpisodeMetrics()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        TotalReward = _numOps.Zero;
        AverageLoss = _numOps.Zero;
        AverageRewardRecent = _numOps.Zero;
    }

    /// <summary>
    /// The episode number (1-indexed).
    /// </summary>
    public int Episode { get; init; }

    /// <summary>
    /// Total reward accumulated in this episode.
    /// </summary>
    public T TotalReward { get; init; }

    /// <summary>
    /// Number of steps taken in this episode.
    /// </summary>
    public int Steps { get; init; }

    /// <summary>
    /// Average loss during training in this episode.
    /// </summary>
    public T AverageLoss { get; init; }

    /// <summary>
    /// Whether the episode ended naturally (vs hitting max steps).
    /// </summary>
    public bool TerminatedNaturally { get; init; }

    /// <summary>
    /// Running average reward over recent episodes (smoothed metric).
    /// </summary>
    public T AverageRewardRecent { get; init; }

    /// <summary>
    /// Total time elapsed since training started.
    /// </summary>
    public TimeSpan ElapsedTime { get; init; }
}
