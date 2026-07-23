using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Proximal Policy Optimization (PPO) agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PPO is a state-of-the-art policy gradient algorithm that achieves a balance between
/// sample efficiency, simplicity, and reliability. It uses a clipped surrogate objective
/// to prevent destructively large policy updates.
/// </para>
/// <para><b>For Beginners:</b>
/// PPO learns a policy (strategy for choosing actions) by making careful, controlled updates.
/// It's like learning to drive - you make small adjustments to your steering rather than
/// jerking the wheel wildly. This makes learning stable and efficient.
///
/// Key features:
/// - **Actor-Critic**: Learns both a policy (actor) and value function (critic)
/// - **Clipped Updates**: Prevents too-large changes that could break learning
/// - **GAE**: Generalized Advantage Estimation for better gradient estimates
/// - **Multi-Epoch**: Reuses collected experience multiple times
///
/// Famous for: OpenAI's ChatGPT uses PPO for RLHF (Reinforcement Learning from Human Feedback)
/// </para>
/// </remarks>
public class PPOOptions<T> : ModelOptions
{
    /// <summary>
    /// Size of the state observation space.
    /// </summary>
    public int StateSize { get; set; }

    /// <summary>
    /// Number of possible actions (discrete) or action dimensions (continuous).
    /// </summary>
    public int ActionSize { get; set; }

    /// <summary>
    /// Whether the action space is continuous (true) or discrete (false).
    /// </summary>
    public bool IsContinuous { get; set; } = false;

    /// <summary>
    /// Learning rate for the policy network.
    /// </summary>
    public T PolicyLearningRate { get; set; }

    /// <summary>
    /// Learning rate for the value network.
    /// </summary>
    public T ValueLearningRate { get; set; }

    /// <summary>
    /// Discount factor (gamma) for future rewards.
    /// </summary>
    /// <remarks>
    /// Typical values: 0.95-0.99.
    /// </remarks>
    public T DiscountFactor { get; set; }

    /// <summary>
    /// GAE (Generalized Advantage Estimation) lambda parameter.
    /// </summary>
    /// <remarks>
    /// Typical values: 0.95-0.99.
    /// Controls bias-variance tradeoff in advantage estimation.
    /// Higher values = lower bias, higher variance.
    /// </remarks>
    public T GaeLambda { get; set; }

    /// <summary>
    /// PPO clipping parameter (epsilon).
    /// </summary>
    /// <remarks>
    /// Typical values: 0.1-0.3.
    /// Limits how much the policy can change in one update.
    /// Smaller = more conservative updates, more stable.
    /// </remarks>
    public T ClipEpsilon { get; set; }

    /// <summary>
    /// Entropy coefficient for exploration.
    /// </summary>
    /// <remarks>
    /// Typical values: 0.01-0.1.
    /// Encourages exploration by penalizing deterministic policies.
    /// Higher = more exploration.
    /// </remarks>
    public T EntropyCoefficient { get; set; }

    /// <summary>
    /// Value function loss coefficient.
    /// </summary>
    /// <remarks>
    /// Typical values: 0.5-1.0.
    /// Weight of value loss relative to policy loss.
    /// </remarks>
    public T ValueLossCoefficient { get; set; }

    /// <summary>
    /// Maximum gradient norm for gradient clipping.
    /// </summary>
    /// <remarks>
    /// Typical values: 0.5-5.0.
    /// Prevents exploding gradients.
    /// </remarks>
    public double MaxGradNorm { get; set; } = 0.5;

    /// <summary>
    /// Number of steps to collect before each training update.
    /// </summary>
    /// <remarks>
    /// Typical values: 128-2048.
    /// PPO collects trajectories, then trains on them.
    /// </remarks>
    public int StepsPerUpdate { get; set; } = 2048;

    /// <summary>
    /// Mini-batch size for training.
    /// </summary>
    /// <remarks>
    /// Typical values: 32-256.
    /// Should divide StepsPerUpdate evenly.
    /// </remarks>
    public int MiniBatchSize { get; set; } = 64;

    /// <summary>
    /// Number of epochs to train on collected data.
    /// </summary>
    /// <remarks>
    /// Typical values: 3-10.
    /// PPO reuses collected experiences multiple times.
    /// </remarks>
    public int TrainingEpochs { get; set; } = 10;

    /// <summary>
    /// Loss function for value network (typically MSE).
    /// </summary>
    public ILossFunction<T> ValueLossFunction { get; set; } = new MeanSquaredErrorLoss<T>();

    /// <summary>
    /// Hidden layer sizes for policy network.
    /// </summary>
    public List<int> PolicyHiddenLayers { get; set; } = new List<int> { 64, 64 };

    /// <summary>
    /// Hidden layer sizes for value network.
    /// </summary>
    public List<int> ValueHiddenLayers { get; set; } = new List<int> { 64, 64 };

    public PPOOptions()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        PolicyLearningRate = numOps.FromDouble(0.0003);
        ValueLearningRate = numOps.FromDouble(0.001);
        DiscountFactor = numOps.FromDouble(0.99);
        GaeLambda = numOps.FromDouble(0.95);
        ClipEpsilon = numOps.FromDouble(0.2);
        EntropyCoefficient = numOps.FromDouble(0.01);
        ValueLossCoefficient = numOps.FromDouble(0.5);
    }
}
