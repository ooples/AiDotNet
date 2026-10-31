using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Advantage Actor-Critic (A2C) agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// A2C is a synchronous version of A3C that is simpler and often more sample-efficient.
/// It combines policy gradients with value function learning for stable, efficient training.
/// </para>
/// <para><b>For Beginners:</b>
/// A2C learns two things simultaneously:
/// - **Actor (Policy)**: What action to take in each state
/// - **Critic (Value Function)**: How good each state is
///
/// The critic helps the actor learn faster by providing better feedback than just rewards alone.
/// Think of the critic as a coach giving targeted advice rather than just "good" or "bad".
///
/// A2C is the foundation for many modern RL algorithms including PPO.
/// </para>
/// </remarks>
public class A2COptions<T> : ModelOptions
{
    /// <summary>Number of features in one observation of the environment state.</summary>
    public int StateSize { get; set; }
    /// <summary>Number of discrete actions, or the dimension of the action vector for a continuous action space.</summary>
    public int ActionSize { get; set; }
    /// <summary>Whether the action space is continuous rather than discrete.</summary>
    public bool IsContinuous { get; set; } = false;
    /// <summary>Learning rate of the policy network.</summary>
    public T PolicyLearningRate { get; set; }
    /// <summary>Learning rate of the value network.</summary>
    public T ValueLearningRate { get; set; }
    /// <summary>Discount factor (gamma) applied to future rewards.</summary>
    public T DiscountFactor { get; set; }
    /// <summary>Weight of the entropy bonus in the objective, which encourages exploration.</summary>
    public T EntropyCoefficient { get; set; }
    /// <summary>Weight of the value loss in the combined objective.</summary>
    public T ValueLossCoefficient { get; set; }
    /// <summary>Number of environment steps collected before each update (the rollout length n of n-step A2C).</summary>
    public int StepsPerUpdate { get; set; } = 5;
    /// <summary>Loss function used to train the value network.</summary>
    public ILossFunction<T> ValueLossFunction { get; set; } = new MeanSquaredErrorLoss<T>();
    /// <summary>Hidden layer sizes of the policy network.</summary>
    public List<int> PolicyHiddenLayers { get; set; } = new List<int> { 64, 64 };
    /// <summary>Hidden layer sizes of the value network.</summary>
    public List<int> ValueHiddenLayers { get; set; } = new List<int> { 64, 64 };

    /// <summary>Creates options with the default values.</summary>
    public A2COptions()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        PolicyLearningRate = numOps.FromDouble(0.0007);
        ValueLearningRate = numOps.FromDouble(0.001);
        DiscountFactor = numOps.FromDouble(0.99);
        EntropyCoefficient = numOps.FromDouble(0.01);
        ValueLossCoefficient = numOps.FromDouble(0.5);
    }
}
