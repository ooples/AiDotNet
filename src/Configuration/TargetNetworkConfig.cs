using AiDotNet.Helpers;

namespace AiDotNet.Configuration;

/// <summary>
/// Configuration for target network updates in DQN-family algorithms.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> DQN uses two networks - a main network for selecting actions
/// and a target network for computing stable Q-value targets. This configuration
/// controls how often and how the target network gets updated.
/// </remarks>
public class TargetNetworkConfig<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Creates a new instance with default values.
    /// </summary>
    public TargetNetworkConfig()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        Tau = _numOps.FromDouble(0.005);
    }

    /// <summary>
    /// Update target network every N steps.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Higher values mean more stable targets but slower learning.
    /// Common values: 1000-10000 for hard updates, 1 for soft updates.
    /// Default: 1000 steps.
    /// </remarks>
    public int UpdateFrequency { get; set; } = 1000;

    /// <summary>
    /// Whether to use soft updates (Polyak averaging) instead of hard updates.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b>
    /// - Hard update: Copy entire network weights at once
    /// - Soft update: Gradually blend weights each step (more stable)
    /// Default: false (hard updates).
    /// </remarks>
    public bool UseSoftUpdate { get; set; }

    /// <summary>
    /// Tau parameter for soft updates (0 to 1).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> For soft updates: target = tau * main + (1 - tau) * target.
    /// Lower tau = slower, more stable updates. Common values: 0.001 to 0.01.
    /// Default: 0.005.
    /// </remarks>
    public T Tau { get; set; }
}
