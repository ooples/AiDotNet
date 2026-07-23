using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

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
public class SACOptions<T> : ModelOptions
{
    /// <summary>
    /// Size of the state observation space.
    /// </summary>
    public int StateSize { get; set; }

    /// <summary>
    /// Size of the continuous action space.
    /// </summary>
    public int ActionSize { get; set; }

    /// <summary>
    /// Learning rate for policy network.
    /// </summary>
    public T PolicyLearningRate { get; set; }

    /// <summary>
    /// Learning rate for Q-networks.
    /// </summary>
    public T QLearningRate { get; set; }

    /// <summary>
    /// Learning rate for temperature parameter (alpha).
    /// </summary>
    public T AlphaLearningRate { get; set; }

    /// <summary>
    /// Discount factor (gamma) for future rewards.
    /// </summary>
    /// <remarks>
    /// Typical values: 0.95-0.99.
    /// </remarks>
    public T DiscountFactor { get; set; }

    /// <summary>
    /// Soft target update coefficient (tau).
    /// </summary>
    /// <remarks>
    /// Typical values: 0.005-0.01.
    /// Controls how quickly target networks track main networks.
    /// </remarks>
    public T TargetUpdateTau { get; set; }

    /// <summary>
    /// Initial temperature (alpha) for entropy regularization.
    /// </summary>
    /// <remarks>
    /// Typical values: 0.2-1.0.
    /// Higher = more exploration.
    /// Can be automatically tuned if AutoTuneTemperature is true.
    /// </remarks>
    public T InitialTemperature { get; set; }

    /// <summary>
    /// Whether to automatically tune the temperature parameter.
    /// </summary>
    /// <remarks>
    /// Recommended: true.
    /// Automatically adjusts exploration based on entropy target.
    /// </remarks>
    public bool AutoTuneTemperature { get; set; } = true;

    /// <summary>
    /// Target entropy for automatic temperature tuning.
    /// </summary>
    /// <remarks>
    /// Typical: -ActionSize (for continuous actions).
    /// If null, uses -ActionSize as default.
    /// </remarks>
    public T? TargetEntropy { get; set; }

    /// <summary>
    /// Mini-batch size for training.
    /// </summary>
    /// <remarks>
    /// Typical values: 256-512.
    /// </remarks>
    public int BatchSize { get; set; } = 256;

    /// <summary>
    /// Capacity of the experience replay buffer.
    /// </summary>
    /// <remarks>
    /// Typical values: 100,000-1,000,000.
    /// </remarks>
    public int ReplayBufferSize { get; set; } = 1000000;

    /// <summary>
    /// Number of warmup steps before starting training.
    /// </summary>
    /// <remarks>
    /// Typical values: 1,000-10,000.
    /// Collects random experiences before training begins.
    /// </remarks>
    public int WarmupSteps { get; set; } = 10000;

    /// <summary>
    /// Number of gradient steps per environment step.
    /// </summary>
    /// <remarks>
    /// Typical value: 1.
    /// Can be > 1 for faster learning from collected experiences.
    /// </remarks>
    public int GradientSteps { get; set; } = 1;

    /// <summary>
    /// Loss function for Q-networks (typically MSE).
    /// </summary>
    /// <remarks>
    /// MSE (Mean Squared Error) is the standard loss for SAC Q-networks as it minimizes
    /// the Bellman error: L = E[(Q(s,a) - (r + γ * Q_target(s',a')))^2].
    /// This is the correct loss function for value-based RL algorithms.
    /// </remarks>
    public ILossFunction<T> QLossFunction { get; set; } = new MeanSquaredErrorLoss<T>();

    /// <summary>
    /// Hidden layer sizes for policy network.
    /// </summary>
    public List<int> PolicyHiddenLayers { get; set; } = new List<int> { 256, 256 };

    /// <summary>
    /// Hidden layer sizes for Q-networks.
    /// </summary>
    public List<int> QHiddenLayers { get; set; } = new List<int> { 256, 256 };

    public SACOptions()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        PolicyLearningRate = numOps.FromDouble(0.0003);
        QLearningRate = numOps.FromDouble(0.0003);
        AlphaLearningRate = numOps.FromDouble(0.0003);
        DiscountFactor = numOps.FromDouble(0.99);
        TargetUpdateTau = numOps.FromDouble(0.005);
        InitialTemperature = numOps.FromDouble(0.2);
    }
}
