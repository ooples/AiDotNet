using AiDotNet.LossFunctions;

namespace AiDotNet.ReinforcementLearning.Agents.DDPG;

/// <summary>
/// Configuration options for DDPG agent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DDPGOptions<T>
{
    /// <summary>
    /// Dimension of the state space.
    /// </summary>
    public int StateSize { get; set; }

    /// <summary>
    /// Dimension of the continuous action space.
    /// </summary>
    public int ActionSize { get; set; }

    /// <summary>
    /// Learning rate for actor network.
    /// </summary>
    public T ActorLearningRate { get; set; } = default!;

    /// <summary>
    /// Learning rate for critic network.
    /// </summary>
    public T CriticLearningRate { get; set; } = default!;

    /// <summary>
    /// Discount factor (gamma) for future rewards.
    /// </summary>
    public T DiscountFactor { get; set; } = default!;

    /// <summary>
    /// Soft update parameter (tau) for target networks.
    /// </summary>
    public T TargetUpdateTau { get; set; } = default!;

    /// <summary>
    /// Loss function for critic training.
    /// </summary>
    public ILossFunction<T> CriticLossFunction { get; set; } = new MeanSquaredError<T>();

    /// <summary>
    /// Batch size for training.
    /// </summary>
    public int BatchSize { get; set; } = 64;

    /// <summary>
    /// Maximum size of replay buffer.
    /// </summary>
    public int ReplayBufferSize { get; set; } = 1000000;

    /// <summary>
    /// Number of steps before training begins.
    /// </summary>
    public int WarmupSteps { get; set; } = 1000;

    /// <summary>
    /// Standard deviation for exploration noise (Ornstein-Uhlenbeck process).
    /// </summary>
    public double ExplorationNoise { get; set; } = 0.1;

    /// <summary>
    /// Hidden layer sizes for actor network.
    /// </summary>
    public int[] ActorHiddenLayers { get; set; } = new[] { 400, 300 };

    /// <summary>
    /// Hidden layer sizes for critic network.
    /// </summary>
    public int[] CriticHiddenLayers { get; set; } = new[] { 400, 300 };

    /// <summary>
    /// Random seed for reproducibility.
    /// </summary>
    public int? Seed { get; set; }
}
