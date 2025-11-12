using AiDotNet.LossFunctions;

namespace AiDotNet.ReinforcementLearning.Agents.PPO;

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
public class PPOOptions<T>
{
    /// <summary>
    /// Size of the state observation space.
    /// </summary>
    public int StateSize { get; init; }

    /// <summary>
    /// Number of possible actions (discrete) or action dimensions (continuous).
    /// </summary>
    public int ActionSize { get; init; }

    /// <summary>
    /// Whether the action space is continuous (true) or discrete (false).
    /// </summary>
    public bool IsContinuous { get; init; } = false;

    /// <summary>
    /// Learning rate for the policy network.
    /// </summary>
    public T PolicyLearningRate { get; init; }

    /// <summary>
    /// Learning rate for the value network.
    /// </summary>
    public T ValueLearningRate { get; init; }

    /// <summary>
    /// Discount factor (gamma) for future rewards.
    /// </summary>
    /// <remarks>
    /// Typical values: 0.95-0.99.
    /// </remarks>
    public T DiscountFactor { get; init; }

    /// <summary>
    /// GAE (Generalized Advantage Estimation) lambda parameter.
    /// </summary>
    /// <remarks>
    /// Typical values: 0.95-0.99.
    /// Controls bias-variance tradeoff in advantage estimation.
    /// Higher values = lower bias, higher variance.
    /// </remarks>
    public T GaeLambda { get; init; }

    /// <summary>
    /// PPO clipping parameter (epsilon).
    /// </summary>
    /// <remarks>
    /// Typical values: 0.1-0.3.
    /// Limits how much the policy can change in one update.
    /// Smaller = more conservative updates, more stable.
    /// </remarks>
    public T ClipEpsilon { get; init; }

    /// <summary>
    /// Entropy coefficient for exploration.
    /// </summary>
    /// <remarks>
    /// Typical values: 0.01-0.1.
    /// Encourages exploration by penalizing deterministic policies.
    /// Higher = more exploration.
    /// </remarks>
    public T EntropyCoefficient { get; init; }

    /// <summary>
    /// Value function loss coefficient.
    /// </summary>
    /// <remarks>
    /// Typical values: 0.5-1.0.
    /// Weight of value loss relative to policy loss.
    /// </remarks>
    public T ValueLossCoefficient { get; init; }

    /// <summary>
    /// Maximum gradient norm for gradient clipping.
    /// </summary>
    /// <remarks>
    /// Typical values: 0.5-5.0.
    /// Prevents exploding gradients.
    /// </remarks>
    public double MaxGradNorm { get; init; } = 0.5;

    /// <summary>
    /// Number of steps to collect before each training update.
    /// </summary>
    /// <remarks>
    /// Typical values: 128-2048.
    /// PPO collects trajectories, then trains on them.
    /// </remarks>
    public int StepsPerUpdate { get; init; } = 2048;

    /// <summary>
    /// Mini-batch size for training.
    /// </summary>
    /// <remarks>
    /// Typical values: 32-256.
    /// Should divide StepsPerUpdate evenly.
    /// </remarks>
    public int MiniBatchSize { get; init; } = 64;

    /// <summary>
    /// Number of epochs to train on collected data.
    /// </summary>
    /// <remarks>
    /// Typical values: 3-10.
    /// PPO reuses collected experiences multiple times.
    /// </remarks>
    public int TrainingEpochs { get; init; } = 10;

    /// <summary>
    /// Loss function for value network (typically MSE).
    /// </summary>
    public ILossFunction<T> ValueLossFunction { get; init; }

    /// <summary>
    /// Hidden layer sizes for policy network.
    /// </summary>
    public int[] PolicyHiddenLayers { get; init; } = new[] { 64, 64 };

    /// <summary>
    /// Hidden layer sizes for value network.
    /// </summary>
    public int[] ValueHiddenLayers { get; init; } = new[] { 64, 64 };

    /// <summary>
    /// Random seed for reproducibility (optional).
    /// </summary>
    public int? Seed { get; init; }

    /// <summary>
    /// Creates default options for PPO with common hyperparameters.
    /// </summary>
    public static PPOOptions<T> Default(
        int stateSize,
        int actionSize,
        T policyLr,
        T valueLr,
        T gamma,
        bool isContinuous = false)
    {
        var numOps = NumericOperations<T>.Instance;
        return new PPOOptions<T>
        {
            StateSize = stateSize,
            ActionSize = actionSize,
            IsContinuous = isContinuous,
            PolicyLearningRate = policyLr,
            ValueLearningRate = valueLr,
            DiscountFactor = gamma,
            GaeLambda = numOps.FromDouble(0.95),
            ClipEpsilon = numOps.FromDouble(0.2),
            EntropyCoefficient = numOps.FromDouble(0.01),
            ValueLossCoefficient = numOps.FromDouble(0.5),
            MaxGradNorm = 0.5,
            StepsPerUpdate = 2048,
            MiniBatchSize = 64,
            TrainingEpochs = 10,
            ValueLossFunction = new MeanSquaredError<T>(),
            PolicyHiddenLayers = new[] { 64, 64 },
            ValueHiddenLayers = new[] { 64, 64 }
        };
    }
}
