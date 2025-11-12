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
    public int StateSize { get; init; }

    /// <summary>
    /// Number of discrete actions.
    /// </summary>
    public int ActionSize { get; init; }

    /// <summary>
    /// Learning rate for Q-network updates.
    /// </summary>
    public T LearningRate { get; init; }

    /// <summary>
    /// Discount factor (gamma) for future rewards.
    /// </summary>
    public T DiscountFactor { get; init; }

    /// <summary>
    /// Loss function for Q-network training.
    /// </summary>
    public ILossFunction<T> LossFunction { get; init; }

    /// <summary>
    /// Starting value of exploration rate.
    /// </summary>
    public double EpsilonStart { get; init; } = 1.0;

    /// <summary>
    /// Minimum value of exploration rate.
    /// </summary>
    public double EpsilonEnd { get; init; } = 0.01;

    /// <summary>
    /// Decay rate for epsilon.
    /// </summary>
    public double EpsilonDecay { get; init; } = 0.995;

    /// <summary>
    /// Batch size for training.
    /// </summary>
    public int BatchSize { get; init; } = 32;

    /// <summary>
    /// Maximum size of replay buffer.
    /// </summary>
    public int ReplayBufferSize { get; init; } = 10000;

    /// <summary>
    /// Frequency (in steps) to update target network.
    /// </summary>
    public int TargetUpdateFrequency { get; init; } = 1000;

    /// <summary>
    /// Number of steps before training begins.
    /// </summary>
    public int WarmupSteps { get; init; } = 1000;

    /// <summary>
    /// Hidden layer sizes for Q-network.
    /// </summary>
    public int[] HiddenLayers { get; init; } = new[] { 64, 64 };

    /// <summary>
    /// Random seed for reproducibility.
    /// </summary>
    public int? Seed { get; init; }

    /// <summary>
    /// Creates default options for Double DQN with common hyperparameters.
    /// </summary>
    public static DoubleDQNOptions<T> Default(int stateSize, int actionSize, T learningRate, T gamma)
    {
        return new DoubleDQNOptions<T>
        {
            StateSize = stateSize,
            ActionSize = actionSize,
            LearningRate = learningRate,
            DiscountFactor = gamma,
            LossFunction = new MeanSquaredError<T>(),
            EpsilonStart = 1.0,
            EpsilonEnd = 0.01,
            EpsilonDecay = 0.995,
            BatchSize = 32,
            ReplayBufferSize = 10000,
            TargetUpdateFrequency = 1000,
            WarmupSteps = 1000,
            HiddenLayers = new[] { 64, 64 }
        };
    }
}
