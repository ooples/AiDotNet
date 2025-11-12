using AiDotNet.LossFunctions;

namespace AiDotNet.ReinforcementLearning.Agents.SAC;

/// <summary>
/// Configuration options for Soft Actor-Critic (SAC) agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SAC is a state-of-the-art off-policy actor-critic algorithm that combines maximum
/// entropy RL with stable off-policy learning. It's particularly effective for
/// continuous control tasks and is known for excellent sample efficiency and robustness.
/// </para>
/// <para><b>For Beginners:</b>
/// SAC is one of the best algorithms for continuous control (like robot movement).
///
/// Key innovations:
/// - **Maximum Entropy**: Encourages exploration by being "random on purpose"
/// - **Off-Policy**: Learns from old experiences (sample efficient)
/// - **Twin Q-Networks**: Uses two Q-functions to prevent overestimation
/// - **Automatic Tuning**: Adjusts exploration automatically
///
/// Think of it like learning to drive while staying diverse in your driving style -
/// you don't just learn one way to drive, you stay flexible and adaptable.
///
/// Used by: Robotic manipulation, dexterous control, autonomous systems
/// </para>
/// </remarks>
public class SACOptions<T>
{
    /// <summary>
    /// Size of the state observation space.
    /// </summary>
    public int StateSize { get; init; }

    /// <summary>
    /// Size of the continuous action space.
    /// </summary>
    public int ActionSize { get; init; }

    /// <summary>
    /// Learning rate for policy network.
    /// </summary>
    public T PolicyLearningRate { get; init; }

    /// <summary>
    /// Learning rate for Q-networks.
    /// </summary>
    public T QLearningRate { get; init; }

    /// <summary>
    /// Learning rate for temperature parameter (alpha).
    /// </summary>
    public T AlphaLearningRate { get; init; }

    /// <summary>
    /// Discount factor (gamma) for future rewards.
    /// </summary>
    /// <remarks>
    /// Typical values: 0.95-0.99.
    /// </remarks>
    public T DiscountFactor { get; init; }

    /// <summary>
    /// Soft target update coefficient (tau).
    /// </summary>
    /// <remarks>
    /// Typical values: 0.005-0.01.
    /// Controls how quickly target networks track main networks.
    /// </remarks>
    public T TargetUpdateTau { get; init; }

    /// <summary>
    /// Initial temperature (alpha) for entropy regularization.
    /// </summary>
    /// <remarks>
    /// Typical values: 0.2-1.0.
    /// Higher = more exploration.
    /// Can be automatically tuned if AutoTuneTemperature is true.
    /// </remarks>
    public T InitialTemperature { get; init; }

    /// <summary>
    /// Whether to automatically tune the temperature parameter.
    /// </summary>
    /// <remarks>
    /// Recommended: true.
    /// Automatically adjusts exploration based on entropy target.
    /// </remarks>
    public bool AutoTuneTemperature { get; init; } = true;

    /// <summary>
    /// Target entropy for automatic temperature tuning.
    /// </summary>
    /// <remarks>
    /// Typical: -ActionSize (for continuous actions).
    /// If null, uses -ActionSize as default.
    /// </remarks>
    public T? TargetEntropy { get; init; }

    /// <summary>
    /// Mini-batch size for training.
    /// </summary>
    /// <remarks>
    /// Typical values: 256-512.
    /// </remarks>
    public int BatchSize { get; init; } = 256;

    /// <summary>
    /// Capacity of the experience replay buffer.
    /// </summary>
    /// <remarks>
    /// Typical values: 100,000-1,000,000.
    /// </remarks>
    public int ReplayBufferSize { get; init; } = 1000000;

    /// <summary>
    /// Number of warmup steps before starting training.
    /// </summary>
    /// <remarks>
    /// Typical values: 1,000-10,000.
    /// Collects random experiences before training begins.
    /// </remarks>
    public int WarmupSteps { get; init; } = 10000;

    /// <summary>
    /// Number of gradient steps per environment step.
    /// </summary>
    /// <remarks>
    /// Typical value: 1.
    /// Can be > 1 for faster learning from collected experiences.
    /// </remarks>
    public int GradientSteps { get; init; } = 1;

    /// <summary>
    /// Loss function for Q-networks (typically MSE).
    /// </summary>
    public ILossFunction<T> QLossFunction { get; init; }

    /// <summary>
    /// Hidden layer sizes for policy network.
    /// </summary>
    public int[] PolicyHiddenLayers { get; init; } = new[] { 256, 256 };

    /// <summary>
    /// Hidden layer sizes for Q-networks.
    /// </summary>
    public int[] QHiddenLayers { get; init; } = new[] { 256, 256 };

    /// <summary>
    /// Random seed for reproducibility (optional).
    /// </summary>
    public int? Seed { get; init; }

    /// <summary>
    /// Creates default options for SAC with common hyperparameters.
    /// </summary>
    public static SACOptions<T> Default(
        int stateSize,
        int actionSize,
        T policyLr,
        T qLr,
        T gamma)
    {
        var numOps = NumericOperations<T>.Instance;
        return new SACOptions<T>
        {
            StateSize = stateSize,
            ActionSize = actionSize,
            PolicyLearningRate = policyLr,
            QLearningRate = qLr,
            AlphaLearningRate = numOps.FromDouble(0.0003),
            DiscountFactor = gamma,
            TargetUpdateTau = numOps.FromDouble(0.005),
            InitialTemperature = numOps.FromDouble(0.2),
            AutoTuneTemperature = true,
            TargetEntropy = numOps.FromDouble(-actionSize),
            BatchSize = 256,
            ReplayBufferSize = 1000000,
            WarmupSteps = 10000,
            GradientSteps = 1,
            QLossFunction = new MeanSquaredError<T>(),
            PolicyHiddenLayers = new[] { 256, 256 },
            QHiddenLayers = new[] { 256, 256 }
        };
    }
}
