using AiDotNet.Helpers;

namespace AiDotNet.Configuration;

/// <summary>
/// Configuration for reward clipping.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Clipping rewards to a fixed range can stabilize training
/// when reward magnitudes vary widely. The famous Atari DQN paper clipped rewards to [-1, 1].
/// </remarks>
public class RewardClippingConfig<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Creates a new instance with default values.
    /// </summary>
    public RewardClippingConfig()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        MinReward = _numOps.FromDouble(-1.0);
        MaxReward = _numOps.FromDouble(1.0);
    }

    /// <summary>
    /// Minimum reward value after clipping.
    /// </summary>
    public T MinReward { get; set; }

    /// <summary>
    /// Maximum reward value after clipping.
    /// </summary>
    public T MaxReward { get; set; }

    /// <summary>
    /// Whether to clip rewards (vs just scaling them).
    /// </summary>
    public bool UseClipping { get; set; } = true;
}
