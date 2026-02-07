using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
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
public abstract class ReinforcementLearningAgentBase<T> : IRLAgent<T>, IConfigurableModel<T>, IDisposable
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
    protected ReinforcementLearningOptions<T> Options;

    /// <inheritdoc/>
    public virtual ModelOptions GetOptions() => Options;

    /// <summary>
    /// Initializes a new instance of the ReinforcementLearningAgentBase class.
    /// </summary>
    /// <param name="options">Configuration options for the agent.</param>
    protected ReinforcementLearningAgentBase(ReinforcementLearningOptions<T> options)
    {
        Options = options ?? throw new ArgumentNullException(nameof(options));
        NumOps = MathHelper.GetNumericOperations<T>();
        Random = options.Seed.HasValue ? RandomHelper.CreateSeededRandom(options.Seed.Value) : RandomHelper.CreateSecureRandom();

        // Ensure required properties are provided
        if (options.LossFunction is null)
            throw new ArgumentNullException(nameof(options), "LossFunction must be provided in options.");
        if (options.LearningRate is null)
            throw new ArgumentNullException(nameof(options), "LearningRate must be provided in options.");
        if (options.DiscountFactor is null)
            throw new ArgumentNullException(nameof(options), "DiscountFactor must be provided in options.");

        LossFunction = options.LossFunction;
        LearningRate = options.LearningRate;
        DiscountFactor = options.DiscountFactor;
        TrainingSteps = 0;
        Episodes = 0;
        LossHistory = new List<T>();
        RewardHistory = new List<T>();
    }

    // ===== IRLAgent<T> Implementation =====

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
    public virtual void ResetEpisode()
    {
        // Base implementation - can be overridden by derived classes
    }

    // ===== IFullModel<T, Vector<T>, Vector<T>> Implementation =====

    /// <summary>
    /// Makes a prediction using the trained agent.
    /// </summary>
    public virtual Vector<T> Predict(Vector<T> input)
    {
        return SelectAction(input, training: false);
    }

    /// <summary>
    /// Gets the default loss function for this agent.
    /// </summary>
    public virtual ILossFunction<T> DefaultLossFunction => LossFunction;

    /// <summary>
    /// Gets model metadata.
    /// </summary>
    public abstract ModelMetadata<T> GetModelMetadata();

    /// <summary>
    /// Trains the agent with supervised learning (not supported for RL agents).
    /// </summary>
    public virtual void Train(Vector<T> input, Vector<T> output)
    {
        throw new NotSupportedException(
            "RL agents are trained via reinforcement learning using Train() method (no parameters), " +
            "not supervised learning. Use BuildAsync(episodes) with an environment instead.");
    }

    /// <summary>
    /// Serializes the agent to bytes.
    /// </summary>
    public abstract byte[] Serialize();

    /// <summary>
    /// Deserializes the agent from bytes.
    /// </summary>
    public abstract void Deserialize(byte[] data);

    /// <summary>
    /// Gets the agent's parameters.
    /// </summary>
    public abstract Vector<T> GetParameters();

    /// <summary>
    /// Sets the agent's parameters.
    /// </summary>
    public abstract void SetParameters(Vector<T> parameters);

    /// <summary>
    /// Gets the number of parameters in the agent.
    /// </summary>
    /// <remarks>
    /// Deep RL agents return parameter counts from neural networks.
    /// Classical RL agents (tabular, linear) may have different implementations.
    /// </remarks>
    public abstract int ParameterCount { get; }

    /// <summary>
    /// Gets the number of input features (state dimensions).
    /// </summary>
    public abstract int FeatureCount { get; }

    /// <summary>
    /// Gets the names of input features.
    /// </summary>
    public virtual string[] FeatureNames => Enumerable.Range(0, FeatureCount)
        .Select(i => $"State_{i}")
        .ToArray();

    /// <summary>
    /// Gets feature importance scores.
    /// </summary>
    public virtual Dictionary<string, T> GetFeatureImportance()
    {
        var importance = new Dictionary<string, T>();
        for (int i = 0; i < FeatureCount; i++)
        {
            importance[$"State_{i}"] = NumOps.One;  // Placeholder
        }
        return importance;
    }

    /// <summary>
    /// Gets the indices of active features.
    /// </summary>
    public virtual IEnumerable<int> GetActiveFeatureIndices()
    {
        return Enumerable.Range(0, FeatureCount);
    }

    /// <summary>
    /// Checks if a feature is used by the agent.
    /// </summary>
    public virtual bool IsFeatureUsed(int featureIndex)
    {
        return featureIndex >= 0 && featureIndex < FeatureCount;
    }

    /// <summary>
    /// Sets the active feature indices.
    /// </summary>
    public virtual void SetActiveFeatureIndices(IEnumerable<int> indices)
    {
        // Default implementation - can be overridden by derived classes
    }

    /// <summary>
    /// Clones the agent.
    /// </summary>
    public abstract IFullModel<T, Vector<T>, Vector<T>> Clone();

    /// <summary>
    /// Creates a deep copy of the agent.
    /// </summary>
    public virtual IFullModel<T, Vector<T>, Vector<T>> DeepCopy()
    {
        return Clone();
    }

    /// <summary>
    /// Creates a new instance with the specified parameters.
    /// </summary>
    public virtual IFullModel<T, Vector<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var clone = Clone();
        clone.SetParameters(parameters);
        return clone;
    }

    /// <summary>
    /// Computes gradients for the agent.
    /// </summary>
    public abstract Vector<T> ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null);

    /// <summary>
    /// Applies gradients to update the agent.
    /// </summary>
    public abstract void ApplyGradients(Vector<T> gradients, T learningRate);

    /// <summary>
    /// Saves the agent's state to a file.
    /// </summary>
    /// <param name="filepath">Path to save the agent.</param>
    public abstract void SaveModel(string filepath);

    /// <summary>
    /// Loads the agent's state from a file.
    /// </summary>
    /// <param name="filepath">Path to load the agent from.</param>
    public abstract void LoadModel(string filepath);

    /// <summary>
    /// Gets the current training metrics.
    /// </summary>
    /// <returns>Dictionary of metric names to values.</returns>
    public virtual Dictionary<string, T> GetMetrics()
    {
        // Use Skip/Take instead of TakeLast for net462 compatibility
        var recentLosses = LossHistory.Count > 0
            ? LossHistory.Skip(Math.Max(0, LossHistory.Count - 100)).Take(100)
            : Enumerable.Empty<T>();
        var recentRewards = RewardHistory.Count > 0
            ? RewardHistory.Skip(Math.Max(0, RewardHistory.Count - 100)).Take(100)
            : Enumerable.Empty<T>();

        return new Dictionary<string, T>
        {
            { "TrainingSteps", NumOps.FromDouble(TrainingSteps) },
            { "Episodes", NumOps.FromDouble(Episodes) },
            { "AverageLoss", LossHistory.Count > 0 ? ComputeAverage(recentLosses) : NumOps.Zero },
            { "AverageReward", RewardHistory.Count > 0 ? ComputeAverage(recentRewards) : NumOps.Zero }
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
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Saves the agent's current state (parameters and configuration) to a stream.
    /// </summary>
    /// <param name="stream">The stream to write the agent state to.</param>
    public virtual void SaveState(Stream stream)
    {
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));

        if (!stream.CanWrite)
            throw new ArgumentException("Stream must be writable.", nameof(stream));

        try
        {
            var data = this.Serialize();
            stream.Write(data, 0, data.Length);
            stream.Flush();
        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to save agent state to stream: {ex.Message}", ex);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Unexpected error while saving agent state: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Loads the agent's state (parameters and configuration) from a stream.
    /// </summary>
    /// <param name="stream">The stream to read the agent state from.</param>
    public virtual void LoadState(Stream stream)
    {
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));

        if (!stream.CanRead)
            throw new ArgumentException("Stream must be readable.", nameof(stream));

        try
        {
            using var ms = new MemoryStream();
            stream.CopyTo(ms);
            var data = ms.ToArray();

            if (data.Length == 0)
                throw new InvalidOperationException("Stream contains no data.");

            this.Deserialize(data);
        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to read agent state from stream: {ex.Message}", ex);
        }
        catch (InvalidOperationException)
        {
            throw;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Failed to deserialize agent state. The stream may contain corrupted or incompatible data: {ex.Message}", ex);
        }
    }

    // ===== IJitCompilable<T, Vector<T>, Vector<T>> Implementation =====

    /// <summary>
    /// Gets whether this RL agent supports JIT compilation.
    /// </summary>
    /// <value>
    /// False for the base class. Derived classes may override to return true if they support JIT compilation.
    /// </value>
    /// <remarks>
    /// <para>
    /// Most RL agents do not directly support JIT compilation because:
    /// - They use layer-based neural networks without direct computation graph export
    /// - Tabular methods use lookup tables rather than mathematical operations
    /// - Policy selection often involves dynamic branching based on exploration strategies
    /// </para>
    /// <para>
    /// Deep RL agents that use neural networks (DQN, PPO, SAC, etc.) may override this
    /// to delegate JIT compilation to their underlying policy or value networks if those
    /// networks support computation graph export.
    /// </para>
    /// <para><b>For Beginners:</b> JIT compilation speeds up models by converting them to optimized code.
    ///
    /// RL agents typically don't support JIT compilation directly because:
    /// - They combine multiple networks (policy, value, target networks)
    /// - They use exploration strategies with random decisions
    /// - The action selection process is complex and dynamic
    ///
    /// However, the underlying neural networks used by deep RL agents (like the Q-network in DQN)
    /// can potentially be JIT compiled separately for faster inference.
    /// </para>
    /// </remarks>
    public virtual bool SupportsJitCompilation => false;

    /// <summary>
    /// Exports the agent's computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the agent's prediction.</returns>
    /// <exception cref="NotSupportedException">
    /// RL agents do not support direct JIT compilation. Use the underlying neural network for JIT compilation if needed.
    /// </exception>
    /// <remarks>
    /// <para>
    /// The base RL agent class does not support JIT compilation because RL agents are complex
    /// systems that combine multiple components:
    /// - Policy networks (select actions)
    /// - Value networks (estimate state/action values)
    /// - Target networks (provide stable training targets)
    /// - Exploration strategies (epsilon-greedy, noise injection, etc.)
    /// - Experience replay buffers
    /// </para>
    /// <para>
    /// The action selection process in RL involves:
    /// 1. Forward pass through policy/value network
    /// 2. Exploration decision (random vs greedy)
    /// 3. Action sampling or selection
    /// 4. Potential action noise injection
    ///
    /// This complex pipeline with dynamic branching is not suitable for JIT compilation.
    /// </para>
    /// <para><b>Workaround for Deep RL Agents:</b>
    /// If you need to accelerate inference for deep RL agents (DQN, PPO, SAC, etc.),
    /// consider JIT compiling the underlying neural networks separately:
    ///
    /// <code>
    /// // For DQN agent with Q-network
    /// var dqnAgent = new DQNAgent&lt;double&gt;(options);
    ///
    /// // Access the Q-network directly if exposed
    /// // (This requires the agent to expose its networks publicly or via a property)
    /// var qNetwork = dqnAgent.QNetwork; // hypothetical property
    ///
    /// // JIT compile the Q-network for faster inference
    /// if (qNetwork.SupportsJitCompilation)
    /// {
    ///     var inputNodes = new List&lt;ComputationNode&lt;double&gt;&gt;();
    ///     var graphOutput = qNetwork.ExportComputationGraph(inputNodes);
    ///     var jitCompiler = new JitCompiler&lt;double&gt;(graphOutput, inputNodes);
    ///     // Use jitCompiler.Evaluate() for fast Q-value computation
    /// }
    /// </code>
    /// </para>
    /// <para><b>For Tabular RL Agents:</b>
    /// Tabular methods (Q-Learning, SARSA, etc.) use lookup tables rather than neural networks.
    /// They perform dictionary lookups which cannot be JIT compiled. These agents are already
    /// very fast for small state spaces and do not benefit from JIT compilation.
    /// </para>
    /// </remarks>
    public virtual Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException(
            "RL agents do not support direct JIT compilation. " +
            "The agent's action selection involves complex processes including exploration strategies, " +
            "multiple neural networks (policy, value, target), and dynamic branching that cannot be " +
            "represented as a static computation graph. " +
            "\n\n" +
            "For deep RL agents (DQN, PPO, SAC, etc.), if you need faster inference, consider: " +
            "\n1. Disabling exploration during inference (set training=false in SelectAction) " +
            "\n2. Using the agent's Predict() method which uses the greedy policy " +
            "\n3. JIT compiling the underlying neural networks separately if they are exposed " +
            "\n\n" +
            "For tabular RL agents (Q-Learning, SARSA, etc.), JIT compilation is not applicable " +
            "as they use lookup tables which are already very fast for small state spaces.");
    }
}


/// <summary>
/// Configuration options for reinforcement learning agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ReinforcementLearningOptions<T> : ModelOptions
{
    /// <summary>
    /// Learning rate for gradient updates.
    /// </summary>
    public T? LearningRate { get; init; }

    /// <summary>
    /// Discount factor (gamma) for future rewards.
    /// </summary>
    public T? DiscountFactor { get; init; }

    /// <summary>
    /// Loss function to use for training.
    /// </summary>
    public ILossFunction<T>? LossFunction { get; init; }

    // Note: Seed property is inherited from ModelOptions.

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
    public double EpsilonEnd { get; init; } = 0.01;

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
