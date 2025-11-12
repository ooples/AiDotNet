using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.ReinforcementLearning.Interfaces;
using System.IO;

namespace AiDotNet.ReinforcementLearning.Agents;

/// <summary>
/// Implements a Deep Q-Network (DQN) agent that learns to make decisions through interaction with an environment.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// The DQN agent is a complete reinforcement learning agent that combines a neural network for Q-value estimation,
/// experience replay for stable learning, a target network for reducing overestimation, and configurable exploration
/// strategies. It implements the Deep Q-Network algorithm, which was the first deep RL algorithm to successfully
/// learn control policies directly from high-dimensional sensory input.
/// </para>
/// <para><b>For Beginners:</b> This is a complete AI agent that learns to make decisions in an environment.
///
/// Key components:
/// - Q-Network: A neural network that estimates how good each action is
/// - Target Network: A stable copy used to generate learning targets
/// - Replay Buffer: Memory that stores past experiences
/// - Policy: Strategy for balancing exploration and exploitation
///
/// How it learns:
/// 1. Observe the environment and choose actions using the policy
/// 2. Store experiences (state, action, reward, next state) in the replay buffer
/// 3. Periodically sample random batches of experiences and learn from them
/// 4. Update the Q-network to better predict rewards
/// 5. Occasionally sync the target network with the Q-network
///
/// Think of it like learning to play a video game:
/// - The Q-network is your brain estimating "this button press will give me X points"
/// - The replay buffer is your memory of what happened when you tried different things
/// - The policy decides when to try new strategies vs. use what works
/// - Learning happens by reviewing your memories and adjusting your predictions
///
/// This agent can learn to play games, control robots, optimize processes, and more!
/// </para>
/// </remarks>
public class DQNAgent<T> : IRLAgent<T>
{
    private readonly DeepQNetwork<T> _qNetwork;
    private readonly DeepQNetwork<T> _targetNetwork;
    private readonly IReplayBuffer<T> _replayBuffer;
    private readonly IPolicy<T> _policy;
    private readonly INumericOperations<T> _numOps;
    private readonly int _batchSize;
    private readonly T _gamma; // Discount factor
    private readonly T _learningRate;
    private readonly int _targetUpdateFrequency;
    private int _trainingSteps;

    /// <summary>
    /// Initializes a new instance of the <see cref="DQNAgent{T}"/> class.
    /// </summary>
    /// <param name="qNetwork">The Q-network used for action selection and learning.</param>
    /// <param name="targetNetwork">The target network used for generating target Q-values.</param>
    /// <param name="replayBuffer">The replay buffer for storing and sampling experiences.</param>
    /// <param name="policy">The policy used for action selection.</param>
    /// <param name="batchSize">The number of experiences to sample for each training step. Default is 32.</param>
    /// <param name="gamma">The discount factor for future rewards (0-1). Default is 0.99.</param>
    /// <param name="learningRate">The learning rate for network updates. Default is 0.001.</param>
    /// <param name="targetUpdateFrequency">How often (in training steps) to update the target network. Default is 100.</param>
    /// <remarks>
    /// <para>
    /// Creates a new DQN agent with the specified components. The Q-network and target network should have
    /// identical architectures. The target network parameters will be periodically synchronized with the
    /// Q-network during training.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up a complete DQN learning agent.
    ///
    /// Parameters explained:
    /// - qNetwork: The "brain" that makes decisions (neural network)
    /// - targetNetwork: A stable copy of the brain for better learning
    /// - replayBuffer: The agent's memory (stores experiences)
    /// - policy: The exploration strategy (e.g., epsilon-greedy)
    /// - batchSize: How many memories to review at once (32 is typical)
    /// - gamma: How much to value future rewards vs immediate ones
    ///   * 0.0 = only care about immediate rewards
    ///   * 0.99 = value future rewards almost as much as immediate ones
    ///   * 1.0 = value all future rewards equally (rarely used)
    /// - learningRate: How big the learning steps are (0.001 is typical)
    ///   * Too high: Unstable learning, might not converge
    ///   * Too low: Very slow learning
    /// - targetUpdateFrequency: How often to sync the target network (100-1000 is typical)
    ///
    /// Example setup:
    /// ```
    /// var qNetwork = new DeepQNetwork<double>(architecture);
    /// var targetNetwork = new DeepQNetwork<double>(architecture);
    /// var buffer = new UniformReplayBuffer<double>(100000);
    /// var policy = new EpsilonGreedyPolicy<double>(actionSpaceSize);
    /// var agent = new DQNAgent<double>(qNetwork, targetNetwork, buffer, policy);
    /// ```
    /// </para>
    /// </remarks>
    public DQNAgent(
        DeepQNetwork<T> qNetwork,
        DeepQNetwork<T> targetNetwork,
        IReplayBuffer<T> replayBuffer,
        IPolicy<T> policy,
        int batchSize = 32,
        double gamma = 0.99,
        double learningRate = 0.001,
        int targetUpdateFrequency = 100)
    {
        _qNetwork = qNetwork ?? throw new ArgumentNullException(nameof(qNetwork));
        _targetNetwork = targetNetwork ?? throw new ArgumentNullException(nameof(targetNetwork));
        _replayBuffer = replayBuffer ?? throw new ArgumentNullException(nameof(replayBuffer));
        _policy = policy ?? throw new ArgumentNullException(nameof(policy));
        _numOps = NumericOperations<T>.Instance;
        _batchSize = batchSize;
        _gamma = _numOps.FromDouble(gamma);
        _learningRate = _numOps.FromDouble(learningRate);
        _targetUpdateFrequency = targetUpdateFrequency;
        _trainingSteps = 0;

        // Initialize target network with same weights as Q-network
        SyncTargetNetwork();
    }

    /// <inheritdoc/>
    public int SelectAction(Tensor<T> state, bool training = true)
    {
        if (state == null)
        {
            throw new ArgumentNullException(nameof(state));
        }

        // Get Q-values from the Q-network
        var qValues = _qNetwork.GetQValues(state);

        // Use policy to select action
        if (training)
        {
            return _policy.SelectAction(state, qValues);
        }
        else
        {
            // During evaluation, always select best action (greedy)
            int bestAction = 0;
            T bestValue = qValues[0];
            for (int i = 1; i < qValues.Length; i++)
            {
                if (_numOps.GreaterThan(qValues[i], bestValue))
                {
                    bestValue = qValues[i];
                    bestAction = i;
                }
            }
            return bestAction;
        }
    }

    /// <inheritdoc/>
    public T Train()
    {
        // Check if we have enough experiences to sample
        if (!_replayBuffer.CanSample(_batchSize))
        {
            return _numOps.Zero;
        }

        // Sample a batch of experiences
        var batch = _replayBuffer.Sample(_batchSize);

        // Prepare arrays for states and target Q-values
        var states = new Tensor<T>[_batchSize];
        var targetQValues = new Tensor<T>[_batchSize];

        // Compute target Q-values for each experience
        for (int i = 0; i < _batchSize; i++)
        {
            var experience = batch[i];
            states[i] = experience.State;

            // Get current Q-values
            var currentQValues = _qNetwork.GetQValues(experience.State);
            var updatedQValues = currentQValues.Clone();

            // Compute target Q-value for the action taken
            T targetQ;
            if (experience.Done)
            {
                // Terminal state: target is just the reward
                targetQ = experience.Reward;
            }
            else
            {
                // Non-terminal: target is reward + gamma * max Q(s', a')
                var nextQValues = _targetNetwork.GetQValues(experience.NextState);

                // Find max Q-value for next state
                T maxNextQ = nextQValues[0];
                for (int j = 1; j < nextQValues.Length; j++)
                {
                    if (_numOps.GreaterThan(nextQValues[j], maxNextQ))
                    {
                        maxNextQ = nextQValues[j];
                    }
                }

                targetQ = _numOps.Add(experience.Reward, _numOps.Multiply(_gamma, maxNextQ));
            }

            // Update only the Q-value for the action that was taken
            updatedQValues[experience.Action] = targetQ;
            targetQValues[i] = updatedQValues;
        }

        // Stack into batches
        var statesBatch = Tensor<T>.Stack(states);
        var targetsBatch = Tensor<T>.Stack(targetQValues);

        // Train the Q-network
        _qNetwork.Train(statesBatch, targetsBatch);

        // Get loss for monitoring
        T loss = _qNetwork.LastLoss;

        // Increment training steps
        _trainingSteps++;

        // Periodically update target network
        if (_trainingSteps % _targetUpdateFrequency == 0)
        {
            SyncTargetNetwork();
        }

        // Update policy (e.g., decay epsilon)
        _policy.Update();

        return loss;
    }

    /// <inheritdoc/>
    public void StoreExperience(Tensor<T> state, int action, T reward, Tensor<T> nextState, bool done)
    {
        var experience = new Experience<T>(state, action, reward, nextState, done);
        _replayBuffer.Add(experience);
    }

    /// <inheritdoc/>
    public void Reset()
    {
        // DQN doesn't have episode-specific state to reset
        // This method can be left empty or used for subclasses that need it
    }

    /// <inheritdoc/>
    public void Save(string filepath)
    {
        if (string.IsNullOrWhiteSpace(filepath))
        {
            throw new ArgumentException("Filepath cannot be null or empty", nameof(filepath));
        }

        // Use the Q-network's serialization
        using var stream = File.Create(filepath);
        using var writer = new BinaryWriter(stream);

        // Serialize Q-network
        _qNetwork.Serialize(writer);

        // Also save training metadata
        writer.Write(_trainingSteps);
    }

    /// <inheritdoc/>
    public void Load(string filepath)
    {
        if (string.IsNullOrWhiteSpace(filepath))
        {
            throw new ArgumentException("Filepath cannot be null or empty", nameof(filepath));
        }

        if (!File.Exists(filepath))
        {
            throw new FileNotFoundException($"File not found: {filepath}");
        }

        using var stream = File.OpenRead(filepath);
        using var reader = new BinaryReader(stream);

        // Deserialize Q-network
        _qNetwork.Deserialize(reader);

        // Load training metadata
        _trainingSteps = reader.ReadInt32();

        // Sync target network with loaded Q-network
        SyncTargetNetwork();
    }

    /// <summary>
    /// Synchronizes the target network parameters with the Q-network parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method copies all parameters (weights and biases) from the Q-network to the target network.
    /// This provides stable learning targets and is performed periodically during training.
    /// </para>
    /// <para><b>For Beginners:</b> This updates the stable copy of the brain with the latest version.
    ///
    /// Why we do this:
    /// - The Q-network is constantly learning and changing
    /// - If we use it to generate our own learning targets, we're chasing a moving target
    /// - The target network stays stable for many steps, providing consistent targets
    /// - Every so often (e.g., every 100 steps), we update it with the latest Q-network
    ///
    /// Think of it like:
    /// - Q-network: Your current best understanding (constantly updating)
    /// - Target network: The "textbook" version you study from (updates less frequently)
    /// - Periodically, you update the textbook with your latest understanding
    ///
    /// This technique greatly stabilizes learning and is a key innovation in DQN.
    /// </para>
    /// </remarks>
    private void SyncTargetNetwork()
    {
        for (int i = 0; i < _qNetwork.Layers.Count; i++)
        {
            var parameters = _qNetwork.Layers[i].GetParameters();
            _targetNetwork.Layers[i].SetParameters(parameters);
        }
    }

    /// <summary>
    /// Gets the current training step count.
    /// </summary>
    /// <value>The number of training steps completed.</value>
    /// <remarks>
    /// <para>
    /// This property tracks how many times the Train() method has been called. It's useful for
    /// monitoring progress, implementing custom update schedules, and debugging.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many times the agent has learned from experiences.
    ///
    /// Use this to:
    /// - Monitor training progress
    /// - Know when to evaluate performance
    /// - Implement custom learning schedules
    /// - Debug learning issues
    ///
    /// The training step count increases by 1 each time Train() is called.
    /// </para>
    /// </remarks>
    public int TrainingSteps => _trainingSteps;
}
