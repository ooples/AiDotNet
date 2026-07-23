namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for REINFORCE agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// REINFORCE is the simplest policy gradient algorithm. It directly optimizes
/// the policy by following the gradient of expected returns.
/// </para>
/// <para><b>For Beginners:</b>
/// REINFORCE is the "hello world" of policy gradient methods. It's simple but powerful:
/// - Play an entire episode
/// - See which actions led to good rewards
/// - Make those actions more likely in the future
///
/// Think of it like learning to play a game: you play a round, see your score,
/// then adjust your strategy to do better next time.
///
/// Simple, but can be slow to learn and high variance.
/// Modern algorithms like PPO improve on REINFORCE's ideas.
/// </para>
/// </remarks>
public class REINFORCEOptions<T> : ModelOptions
{
    /// <summary>Dimension of the environment state vector.</summary>
    public int StateSize { get; set; } = 4;

    /// <summary>Number of discrete actions available to the agent.</summary>
    public int ActionSize { get; set; } = 2;

    /// <summary>Whether the action space is continuous (true) or discrete (false).</summary>
    public bool IsContinuous { get; set; } = false;

    /// <summary>Learning rate for policy gradient updates.</summary>
    /// <value>Default: 0.001.</value>
    public T LearningRate { get; set; }

    /// <summary>Discount factor (gamma) for future rewards.</summary>
    /// <value>Default: 0.99.</value>
    public T DiscountFactor { get; set; }

    /// <summary>Hidden layer sizes for the policy network.</summary>
    public List<int> HiddenLayers { get; set; } = new List<int> { 32, 32 };

    public REINFORCEOptions()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        LearningRate = numOps.FromDouble(0.001);
        DiscountFactor = numOps.FromDouble(0.99);
    }
}
