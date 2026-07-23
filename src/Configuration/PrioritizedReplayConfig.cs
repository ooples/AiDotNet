using AiDotNet.Helpers;

namespace AiDotNet.Configuration;

/// <summary>
/// Configuration for prioritized experience replay (PER).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> PER samples experiences based on their TD-error (surprise).
/// High-error experiences are sampled more often because they have more to teach.
/// </remarks>
public class PrioritizedReplayConfig<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Creates a new instance with default values.
    /// </summary>
    public PrioritizedReplayConfig()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        Alpha = _numOps.FromDouble(0.6);
        InitialBeta = _numOps.FromDouble(0.4);
        FinalBeta = _numOps.FromDouble(1.0);
        PriorityEpsilon = _numOps.FromDouble(1e-6);
    }

    /// <summary>
    /// Alpha parameter controlling prioritization strength (0 = uniform, 1 = full prioritization).
    /// </summary>
    /// <remarks>
    /// Common value: 0.6. Higher = stronger prioritization.
    /// </remarks>
    public T Alpha { get; set; }

    /// <summary>
    /// Initial beta for importance sampling correction.
    /// </summary>
    /// <remarks>
    /// Starts low and anneals to 1.0. Common initial: 0.4.
    /// </remarks>
    public T InitialBeta { get; set; }

    /// <summary>
    /// Final beta value (should reach 1.0 by end of training).
    /// </summary>
    public T FinalBeta { get; set; }

    /// <summary>
    /// Steps over which to anneal beta from initial to final.
    /// </summary>
    public int BetaAnnealingSteps { get; set; } = 100000;

    /// <summary>
    /// Small constant added to priorities to prevent zero probabilities.
    /// </summary>
    public T PriorityEpsilon { get; set; }
}
