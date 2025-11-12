using AiDotNet.LossFunctions;

namespace AiDotNet.ReinforcementLearning.Agents.DQN;

/// <summary>
/// Configuration options for Deep Q-Network (DQN) agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DQN is a foundational value-based RL algorithm that combines Q-learning with deep neural networks.
/// It learns to estimate the value (expected future reward) of taking each action in each state.
/// </para>
/// <para><b>For Beginners:</b>
/// DQN learns a "quality" function (Q-function) that tells the agent how good each action is.
/// Think of it like a scorecard - for every situation, it scores each possible action,
/// and the agent picks the action with the highest score.
///
/// Key innovations in DQN:
/// - **Experience Replay**: Stores past experiences and learns from random samples (breaks correlations)
/// - **Target Network**: Uses a separate, slowly-updating network for stable learning
/// - **Epsilon-Greedy**: Balances exploring new actions vs. exploiting known good ones
/// </para>
/// </remarks>
public class DQNOptions<T>
{
    /// <summary>
    /// Size of the state observation space.
    /// </summary>
    public int StateSize { get; init; }

    /// <summary>
    /// Number of possible actions.
    /// </summary>
    public int ActionSize { get; init; }

    /// <summary>
    /// Learning rate for the Q-network optimizer.
    /// </summary>
    public T LearningRate { get; init; }

    /// <summary>
    /// Discount factor (gamma) for future rewards.
    /// </summary>
    /// <remarks>
    /// Typical values: 0.95-0.99.
    /// Higher values make the agent more "far-sighted" (values long-term rewards more).
    /// </remarks>
    public T DiscountFactor { get; init; }

    /// <summary>
    /// Initial exploration rate (epsilon) for epsilon-greedy action selection.
    /// </summary>
    /// <remarks>
    /// Typical value: 1.0 (100% random exploration at start).
    /// </remarks>
    public double EpsilonStart { get; init; } = 1.0;

    /// <summary>
    /// Final exploration rate (epsilon) after decay.
    /// </summary>
    /// <remarks>
    /// Typical value: 0.01 (1% random exploration after training).
    /// </remarks>
    public double EpsilonEnd { get; init; } = 0.01;

    /// <summary>
    /// Exploration decay rate per episode.
    /// </summary>
    /// <remarks>
    /// Typical value: 0.995 (epsilon decays slowly over episodes).
    /// </remarks>
    public double EpsilonDecay { get; init; } = 0.995;

    /// <summary>
    /// Size of mini-batches for training.
    /// </summary>
    /// <remarks>
    /// Typical values: 32-256.
    /// </remarks>
    public int BatchSize { get; init; } = 64;

    /// <summary>
    /// Capacity of the experience replay buffer.
    /// </summary>
    /// <remarks>
    /// Typical values: 10,000-1,000,000.
    /// Larger buffers provide more diverse experiences but use more memory.
    /// </remarks>
    public int ReplayBufferSize { get; init; } = 100000;

    /// <summary>
    /// Frequency (in steps) of updating the target network.
    /// </summary>
    /// <remarks>
    /// Typical values: 100-10,000 steps.
    /// More frequent updates may cause instability; less frequent updates slow learning.
    /// </remarks>
    public int TargetUpdateFrequency { get; init; } = 1000;

    /// <summary>
    /// Number of warmup steps before starting training.
    /// </summary>
    /// <remarks>
    /// Typical values: 1,000-10,000.
    /// Allows the buffer to fill with diverse experiences before training begins.
    /// </remarks>
    public int WarmupSteps { get; init; } = 1000;

    /// <summary>
    /// Loss function to use for training the Q-network.
    /// </summary>
    /// <remarks>
    /// Typically Mean Squared Error (MSE) for regression of Q-values.
    /// </remarks>
    public ILossFunction<T> LossFunction { get; init; }

    /// <summary>
    /// Hidden layer sizes for the Q-network.
    /// </summary>
    /// <remarks>
    /// Example: [128, 128] creates a network with two hidden layers of 128 units each.
    /// </remarks>
    public int[] HiddenLayers { get; init; } = new[] { 128, 128 };

    /// <summary>
    /// Random seed for reproducibility (optional).
    /// </summary>
    public int? Seed { get; init; }

    /// <summary>
    /// Creates default options for DQN with common hyperparameters.
    /// </summary>
    public static DQNOptions<T> Default(int stateSize, int actionSize, T learningRate, T discountFactor)
    {
        var numOps = NumericOperations<T>.Instance;
        return new DQNOptions<T>
        {
            StateSize = stateSize,
            ActionSize = actionSize,
            LearningRate = learningRate,
            DiscountFactor = discountFactor,
            LossFunction = new MeanSquaredError<T>(),
            EpsilonStart = 1.0,
            EpsilonEnd = 0.01,
            EpsilonDecay = 0.995,
            BatchSize = 64,
            ReplayBufferSize = 100000,
            TargetUpdateFrequency = 1000,
            WarmupSteps = 1000,
            HiddenLayers = new[] { 128, 128 }
        };
    }
}
