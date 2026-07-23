using AiDotNet.Helpers;

namespace AiDotNet.Configuration;

/// <summary>
/// Summary of completed RL training.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class RLTrainingSummary<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Creates a new instance with default values.
    /// </summary>
    public RLTrainingSummary()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        AverageReward = _numOps.Zero;
        BestReward = _numOps.Zero;
        FinalAverageReward = _numOps.Zero;
        AverageLoss = _numOps.Zero;
    }

    /// <summary>
    /// Total episodes completed.
    /// </summary>
    public int TotalEpisodes { get; init; }

    /// <summary>
    /// Total steps across all episodes.
    /// </summary>
    public int TotalSteps { get; init; }

    /// <summary>
    /// Average reward across all episodes.
    /// </summary>
    public T AverageReward { get; init; }

    /// <summary>
    /// Best reward achieved in any episode.
    /// </summary>
    public T BestReward { get; init; }

    /// <summary>
    /// Average reward over the last 100 episodes.
    /// </summary>
    public T FinalAverageReward { get; init; }

    /// <summary>
    /// Average loss across training.
    /// </summary>
    public T AverageLoss { get; init; }

    /// <summary>
    /// Total training time.
    /// </summary>
    public TimeSpan TotalTime { get; init; }

    /// <summary>
    /// Whether early stopping was triggered.
    /// </summary>
    public bool EarlyStopTriggered { get; init; }
}
