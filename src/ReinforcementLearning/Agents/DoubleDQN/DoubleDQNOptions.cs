using AiDotNet.LossFunctions;

namespace AiDotNet.ReinforcementLearning.Agents.DoubleDQN;

/// <summary>
/// Configuration options for Double DQN agent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DoubleDQNOptions<T>
{
    /// <summary>
    /// Dimension of the state space.
    /// </summary>
    public int StateSize { get; set; }

    /// <summary>
    /// Number of discrete actions.
    /// </summary>
    public int ActionSize { get; set; }

    /// <summary>
    /// Learning rate for Q-network updates.
    /// </summary>
    public T LearningRate { get; set; } = default!;

    /// <summary>
    /// Discount factor (gamma) for future rewards.
    /// </summary>
    public T DiscountFactor { get; set; } = default!;

    /// <summary>
    /// Loss function for Q-network training.
    /// </summary>
    public ILossFunction<T> LossFunction { get; set; } = new MeanSquaredError<T>();

    /// <summary>
    /// Starting value of exploration rate.
    /// </summary>
    public double EpsilonStart { get; set; } = 1.0;

    /// <summary>
    /// Minimum value of exploration rate.
    /// </summary>
    public double EpsilonEnd { get; set; } = 0.01;

    /// <summary>
    /// Decay rate for epsilon.
    /// </summary>
    public double EpsilonDecay { get; set; } = 0.995;

    /// <summary>
    /// Batch size for training.
    /// </summary>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Maximum size of replay buffer.
    /// </summary>
    public int ReplayBufferSize { get; set; } = 10000;

    /// <summary>
    /// Frequency (in steps) to update target network.
    /// </summary>
    public int TargetUpdateFrequency { get; set; } = 1000;

    /// <summary>
    /// Number of steps before training begins.
    /// </summary>
    public int WarmupSteps { get; set; } = 1000;

    /// <summary>
    /// Hidden layer sizes for Q-network.
    /// </summary>
    public int[] HiddenLayers { get; set; } = new[] { 64, 64 };

    /// <summary>
    /// Random seed for reproducibility.
    /// </summary>
    public int? Seed { get; set; }
}
