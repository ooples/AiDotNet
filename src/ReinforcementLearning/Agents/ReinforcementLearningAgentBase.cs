using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.ReinforcementLearning.Agents;

/// <summary>
/// Base class for all reinforcement learning agents, providing common functionality and structure.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// This abstract base class defines the core structure that all RL agents must follow, ensuring
/// consistency across different RL algorithms while allowing for specialized implementations.
/// It integrates deeply with AiDotNet's existing architecture, using Vector, Matrix, and Tensor types,
/// and following established patterns like OptimizerBase and NeuralNetworkBase.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation for all RL agents in AiDotNet.
///
/// Think of this base class as the blueprint that defines what every RL agent must be able to do:
/// - Select actions based on observations
/// - Store experiences for learning
/// - Train/update from experiences
/// - Save and load trained models
/// - Integrate with AiDotNet's neural networks and optimizers
///
/// All specific RL algorithms (DQN, PPO, SAC, etc.) inherit from this base and implement
/// their own unique learning logic while sharing common functionality.
/// </para>
/// </remarks>
public abstract class ReinforcementLearningAgentBase<T> : IDisposable
{
    /// <summary>
    /// Numeric operations provider for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Random number generator for stochastic operations.
    /// </summary>
    protected readonly Random Random;

    /// <summary>
    /// The primary neural network(s) used by this agent.
    /// </summary>
    protected readonly List<INeuralNetwork<T>> Networks;

    /// <summary>
    /// Loss function used for training.
    /// </summary>
    protected readonly ILossFunction<T> LossFunction;

    /// <summary>
    /// Learning rate for gradient updates.
    /// </summary>
    protected T LearningRate;

    /// <summary>
    /// Discount factor (gamma) for future rewards.
    /// </summary>
    protected T DiscountFactor;

    /// <summary>
    /// Number of training steps completed.
    /// </summary>
    protected int TrainingSteps;

    /// <summary>
    /// Number of episodes completed.
    /// </summary>
    protected int Episodes;

    /// <summary>
    /// History of losses during training.
    /// </summary>
    protected readonly List<T> LossHistory;

    /// <summary>
    /// History of episode rewards.
    /// </summary>
    protected readonly List<T> RewardHistory;

    /// <summary>
    /// Configuration options for this agent.
    /// </summary>
    protected readonly ReinforcementLearningOptions<T> Options;

    /// <summary>
    /// Initializes a new instance of the ReinforcementLearningAgentBase class.
    /// </summary>
    /// <param name="options">Configuration options for the agent.</param>
    protected ReinforcementLearningAgentBase(ReinforcementLearningOptions<T> options)
    {
        Options = options ?? throw new ArgumentNullException(nameof(options));
        NumOps = NumericOperations<T>.Instance;
        Random = options.Seed.HasValue ? new Random(options.Seed.Value) : new Random();
        Networks = new List<INeuralNetwork<T>>();
        LossFunction = options.LossFunction;
        LearningRate = options.LearningRate;
        DiscountFactor = options.DiscountFactor;
        TrainingSteps = 0;
        Episodes = 0;
        LossHistory = new List<T>();
        RewardHistory = new List<T>();
    }

    /// <summary>
    /// Selects an action given the current state observation.
    /// </summary>
    /// <param name="state">The current state observation as a Vector.</param>
    /// <param name="training">Whether the agent is in training mode (affects exploration).</param>
    /// <returns>Action as a Vector (can be discrete or continuous).</returns>
    public abstract Vector<T> SelectAction(Vector<T> state, bool training = true);

    /// <summary>
    /// Stores an experience tuple for later learning.
    /// </summary>
    /// <param name="state">The state before action.</param>
    /// <param name="action">The action taken.</param>
    /// <param name="reward">The reward received.</param>
    /// <param name="nextState">The state after action.</param>
    /// <param name="done">Whether the episode terminated.</param>
    public abstract void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done);

    /// <summary>
    /// Performs one training step, updating the agent's policy/value function.
    /// </summary>
    /// <returns>The training loss for monitoring.</returns>
    public abstract T Train();

    /// <summary>
    /// Resets episode-specific state (if any).
    /// </summary>
    public virtual void Reset()
    {
        // Base implementation - can be overridden by derived classes
    }

    /// <summary>
    /// Saves the agent's state to a file.
    /// </summary>
    /// <param name="filepath">Path to save the agent.</param>
    public abstract void Save(string filepath);

    /// <summary>
    /// Loads the agent's state from a file.
    /// </summary>
    /// <param name="filepath">Path to load the agent from.</param>
    public abstract void Load(string filepath);

    /// <summary>
    /// Gets the current training metrics.
    /// </summary>
    /// <returns>Dictionary of metric names to values.</returns>
    public virtual Dictionary<string, T> GetMetrics()
    {
        return new Dictionary<string, T>
        {
            { "TrainingSteps", NumOps.FromDouble(TrainingSteps) },
            { "Episodes", NumOps.FromDouble(Episodes) },
            { "AverageLoss", LossHistory.Count > 0 ? ComputeAverage(LossHistory.TakeLast(100)) : NumOps.Zero },
            { "AverageReward", RewardHistory.Count > 0 ? ComputeAverage(RewardHistory.TakeLast(100)) : NumOps.Zero }
        };
    }

    /// <summary>
    /// Computes the average of a collection of values.
    /// </summary>
    protected T ComputeAverage(IEnumerable<T> values)
    {
        var list = values.ToList();
        if (list.Count == 0) return NumOps.Zero;

        T sum = NumOps.Zero;
        foreach (var value in list)
        {
            sum = NumOps.Add(sum, value);
        }
        return NumOps.Divide(sum, NumOps.FromDouble(list.Count));
    }

    /// <summary>
    /// Disposes of resources used by the agent.
    /// </summary>
    public virtual void Dispose()
    {
        foreach (var network in Networks)
        {
            if (network is IDisposable disposable)
            {
                disposable.Dispose();
            }
        }
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// Configuration options for reinforcement learning agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ReinforcementLearningOptions<T>
{
    /// <summary>
    /// Learning rate for gradient updates.
    /// </summary>
    public T LearningRate { get; init; }

    /// <summary>
    /// Discount factor (gamma) for future rewards.
    /// </summary>
    public T DiscountFactor { get; init; }

    /// <summary>
    /// Loss function to use for training.
    /// </summary>
    public ILossFunction<T> LossFunction { get; init; }

    /// <summary>
    /// Random seed for reproducibility (optional).
    /// </summary>
    public int? Seed { get; init; }

    /// <summary>
    /// Batch size for training updates.
    /// </summary>
    public int BatchSize { get; init; } = 32;

    /// <summary>
    /// Size of the replay buffer (if applicable).
    /// </summary>
    public int ReplayBufferSize { get; init; } = 100000;

    /// <summary>
    /// Frequency of target network updates (if applicable).
    /// </summary>
    public int TargetUpdateFrequency { get; init; } = 100;

    /// <summary>
    /// Whether to use prioritized experience replay.
    /// </summary>
    public bool UsePrioritizedReplay { get; init; } = false;

    /// <summary>
    /// Initial exploration rate (for epsilon-greedy policies).
    /// </summary>
    public double EpsilonStart { get; init; } = 1.0;

    /// <summary>
    /// Final exploration rate.
    /// </summary>
    public double EpsilonEnd { get; init} = 0.01;

    /// <summary>
    /// Exploration decay rate.
    /// </summary>
    public double EpsilonDecay { get; init; } = 0.995;

    /// <summary>
    /// Number of warmup steps before training.
    /// </summary>
    public int WarmupSteps { get; init; } = 1000;

    /// <summary>
    /// Maximum gradient norm for clipping (0 = no clipping).
    /// </summary>
    public double MaxGradientNorm { get; init; } = 0.5;
}
