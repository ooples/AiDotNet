using AiDotNet.Factories;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ReinforcementLearning.Interfaces;
using AiDotNet.ReinforcementLearning.Memory;
using AiDotNet.ReinforcementLearning.Models.Options;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Models;
using AiDotNet.Enums;
using AiDotNet.Interpretability;
using System.Threading.Tasks;

namespace AiDotNet.ReinforcementLearning.Agents;

/// <summary>
/// Deep Q-Network (DQN) agent for reinforcement learning.
/// </summary>
/// <typeparam name="TState">The type of the state representation.</typeparam>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DQNAgent<TState, T> : AgentBase<TState, int, T>
    where TState : Tensor<T>
{
    protected readonly DQNOptions _options;
    protected readonly IReplayBuffer<TState, int, T> _replayBuffer;
    protected IQNetwork<T, Tensor<T>> _qNetwork = null!;
    protected IQNetwork<T, Tensor<T>> _targetQNetwork = null!;
    private readonly OptimizerType _optimizerType = default!;
    protected readonly int _updateFrequency;
    protected readonly bool _useSoftUpdate;
    protected readonly T _tau;
    protected readonly T _gamma;
    private readonly bool _useDoubleDQN;
    private readonly bool _useDuelingDQN;
    private readonly bool _useNStepReturns;
    private readonly int _nSteps;
    private readonly bool _clipRewards;
    protected readonly bool _usePrioritizedReplay;
    private readonly double _prioritizedReplayAlpha;
    private readonly double _prioritizedReplayBetaInitial;
    private readonly double _prioritizedReplayBetaSteps;
    // _batchSize is now managed by the base class
    protected readonly int StateSize;
    protected readonly int ActionSize;
    protected readonly double _explorationFraction;
    
    protected IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer = null!;
    protected int _steps;
    protected int _updateCounter;
    protected bool _isTraining;
    protected double _currentEpsilon;
    protected double _initialEpsilon;
    protected double _finalEpsilon;
    protected T _prioritizedReplayBeta;
    protected Random _random;
    protected T _lastLoss = default!;
    
    // LastLoss is now managed by the base class

    /// <summary>
    /// Initializes a new instance of the <see cref="DQNAgent{TState, T}"/> class.
    /// </summary>
    /// <param name="options">The options for the DQN algorithm.</param>
    public DQNAgent(DQNOptions options)
        : base(options.Gamma, options.UseSoftTargetUpdate ? 0.001 : 1.0, options.BatchSize, options.Seed)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        StateSize = options.StateSize;
        ActionSize = options.ActionSize;
        
        _tau = NumOps.FromDouble(options.UseSoftTargetUpdate ? 0.001 : 1.0);
        _gamma = NumOps.FromDouble(options.Gamma);
        _random = new Random(options.Seed ?? DateTime.Now.Millisecond);
        
        _optimizerType = options.OptimizerType;
        _updateFrequency = options.TargetNetworkUpdateFrequency;
        _useSoftUpdate = options.UseSoftTargetUpdate;
        _useDoubleDQN = options.UseDoubleDQN;
        _useDuelingDQN = options.UseDuelingDQN;
        _useNStepReturns = options.UseNStepReturns;
        _nSteps = options.NSteps;
        _clipRewards = options.ClipRewards;
        _explorationFraction = options.ExplorationFraction;
        
        _usePrioritizedReplay = options.UsePrioritizedReplay;
        _prioritizedReplayAlpha = options.PrioritizedReplayAlpha;
        _prioritizedReplayBetaInitial = options.PrioritizedReplayBetaInitial;
        _prioritizedReplayBetaSteps = options.PrioritizedReplayBetaSteps;
        _prioritizedReplayBeta = NumOps.FromDouble(_prioritizedReplayBetaInitial);
        
        _updateCounter = 0;
        _initialEpsilon = options.InitialExplorationRate;
        _finalEpsilon = options.FinalExplorationRate;
        _currentEpsilon = _initialEpsilon;

        // Create replay buffer
        if (_usePrioritizedReplay)
        {
            _replayBuffer = new PrioritizedReplayBuffer<TState, int, T>(
                options.ReplayBufferCapacity,
                _prioritizedReplayAlpha);
        }
        else
        {
            _replayBuffer = new StandardReplayBuffer<TState, int, T>(options.ReplayBufferCapacity);
        }

        // Initialize networks
        InitializeNetworks();
        
        // Copy parameters from online to target network
        UpdateTargetNetwork(NumOps.One);
        
        // Create optimizer
        var optimizerOptions = new GradientDescentOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = options.LearningRate
        };
        _optimizer = OptimizerFactory<T, Tensor<T>, Tensor<T>>.CreateOptimizer(_optimizerType, optimizerOptions);
    }
    
    /// <summary>
    /// Initializes the Q-network and target network.
    /// </summary>
    protected virtual void InitializeNetworks()
    {
        _qNetwork = new QNetwork(StateSize, ActionSize, _options.NetworkArchitecture, _options.ActivationFunction, _options.UseDuelingDQN);
        _targetQNetwork = new QNetwork(StateSize, ActionSize, _options.NetworkArchitecture, _options.ActivationFunction, _options.UseDuelingDQN);
    }

    /// <summary>
    /// Selects an action based on the current state.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <param name="isTraining">Whether the agent should explore (true) or exploit (false).</param>
    /// <returns>The selected action.</returns>
    public override int SelectAction(TState state, bool isTraining = true)
    {
        // Epsilon-greedy exploration during training
        if (isTraining && Random.NextDouble() < _currentEpsilon)
        {
            // Random action
            return Random.Next(ActionSize);
        }
        
        // Forward pass through Q-network
        var output = _qNetwork.Predict(state);
        Vector<T> qValues;
        if (output.Shape.Length == 2 && output.Shape[0] == 1)
        {
            // Single batch, extract the Q-values for all actions
            qValues = new Vector<T>(ActionSize);
            for (int i = 0; i < ActionSize; i++)
            {
                qValues[i] = output[0, i];
            }
        }
        else if (output.Shape.Length == 1)
        {
            // Already a 1D tensor
            qValues = new Vector<T>(output.Shape[0]);
            for (int i = 0; i < output.Shape[0]; i++)
            {
                qValues[i] = output[i];
            }
        }
        else
        {
            throw new InvalidOperationException($"Unexpected output shape: [{string.Join(", ", output.Shape)}]");
        }
        
        // Select action with highest Q-value
        var actionIndex = 0;
        var maxQ = qValues[0];
        
        for (int i = 1; i < ActionSize; i++)
        {
            if (NumOps.GreaterThan(qValues[i], maxQ))
            {
                maxQ = qValues[i];
                actionIndex = i;
            }
        }
        
        return actionIndex;
    }

    /// <summary>
    /// Learns from an experience tuple.
    /// </summary>
    /// <param name="state">The state before the action was taken.</param>
    /// <param name="action">The action that was taken.</param>
    /// <param name="reward">The reward received after taking the action.</param>
    /// <param name="nextState">The state after the action was taken.</param>
    /// <param name="done">A flag indicating whether the episode ended after this action.</param>
    public override void Learn(TState state, int action, T reward, TState nextState, bool done)
    {
        // Add experience to replay buffer
        _replayBuffer.Add(state, action, reward, nextState, done);
        
        // Check if we have enough samples to start learning
        if (_replayBuffer.Size < BatchSize)
        {
            return;
        }
        
        // Update exploration rate
        UpdateExplorationRate();
        
        // Update step counter
        IncrementStepCounter();
        _updateCounter++;
        
        // Update prioritized replay beta parameter
        if (_usePrioritizedReplay)
        {
            UpdatePrioritizedReplayBeta();
        }
        
        // Sample batch from replay buffer
        ReplayBatch<TState, int, T> batch;
        T[]? importanceWeights = null;
        int[]? indices = null;
        
        if (_usePrioritizedReplay && _replayBuffer is PrioritizedReplayBuffer<TState, int, T> prioritizedBuffer)
        {
            var prioritizedBatch = prioritizedBuffer.SamplePrioritized(BatchSize, _prioritizedReplayBeta);
            batch = prioritizedBatch;
            importanceWeights = prioritizedBatch.Weights;
            indices = prioritizedBatch.Indices;
        }
        else
        {
            batch = _replayBuffer.SampleBatch(BatchSize);
        }
        
        // Convert batch to tensors
        var statesArray = new Tensor<T>[BatchSize];
        var nextStatesArray = new Tensor<T>[BatchSize];
        
        for (int i = 0; i < BatchSize; i++)
        {
            statesArray[i] = batch.States[i];
            nextStatesArray[i] = batch.NextStates[i];
        }
        
        var statesBatch = Tensor<T>.Stack(statesArray);
        var nextStatesBatch = Tensor<T>.Stack(nextStatesArray);
        var actionsBatch = new Vector<T>(BatchSize);
        var rewardsBatch = new Vector<T>(BatchSize);
        var donesBatch = new Vector<T>(BatchSize);
        
        for (int i = 0; i < BatchSize; i++)
        {
            actionsBatch[i] = NumOps.FromDouble(batch.Actions[i]);
            rewardsBatch[i] = batch.Rewards[i];
            donesBatch[i] = batch.Dones[i] ? NumOps.One : NumOps.Zero;
        }
        
        // Compute Q-values for current states
        // Predict Q-values for the batch
        var qValuesTensor = _qNetwork.Predict(statesBatch);
        var currentQValues = new Vector<T>[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            currentQValues[i] = new Vector<T>(ActionSize);
            for (int j = 0; j < ActionSize; j++)
            {
                currentQValues[i][j] = qValuesTensor[i, j];
            }
        }
        
        // Compute target Q-values
        var targetQValues = ComputeTargets(nextStatesBatch, rewardsBatch, donesBatch);
        
        // Compute loss and update priorities if using prioritized replay
        var losses = new Vector<T>(BatchSize);
        for (int i = 0; i < BatchSize; i++)
        {
            int actionIdx = NumOps.ToInt32(actionsBatch[i]);
            var currentQ = currentQValues[i][actionIdx];
            var targetQ = targetQValues[i];
            
            // Compute squared error
            var diff = NumOps.Subtract(currentQ, targetQ);
            var squaredError = NumOps.Multiply(diff, diff);
            // Apply importance sampling weights if using prioritized replay
            if (_usePrioritizedReplay && importanceWeights != null)
            {
                losses[i] = NumOps.Multiply(squaredError, importanceWeights[i]);
            }
            else
            {
                losses[i] = squaredError;
            }
            
            // Update priorities in replay buffer if using prioritized replay
            if (_usePrioritizedReplay && _replayBuffer is PrioritizedReplayBuffer<TState, int, T> prBuffer && indices != null)
            {
                var priority = NumOps.Add(squaredError, NumOps.FromDouble(1e-6)); // Add small constant to avoid zero priority
                prBuffer.UpdatePriority(indices[i], priority);
            }
        }
        
        // Train the Q-network with the computed targets
        // We need to create a batch of expected outputs based on the targets
        var expectedOutputs = new Tensor<T>(new[] { BatchSize, ActionSize });
        
        // Copy current Q-values and update only the taken actions
        for (int i = 0; i < BatchSize; i++)
        {
            for (int j = 0; j < ActionSize; j++)
            {
                expectedOutputs[i, j] = currentQValues[i][j];
            }
            int actionIdx = NumOps.ToInt32(actionsBatch[i]);
            expectedOutputs[i, actionIdx] = targetQValues[i];
        }
        
        // Train the network
        _qNetwork.Train(statesBatch, expectedOutputs);
        
        // Update target network if needed
        if (_useSoftUpdate)
        {
            // Soft update
            UpdateTargetNetwork(_tau);
        }
        else if (_updateCounter >= _updateFrequency)
        {
            // Hard update
            UpdateTargetNetwork(NumOps.One);
            _updateCounter = 0;
        }
        
        // Calculate and store average loss
        T totalLoss = NumOps.Zero;
        for (int i = 0; i < losses.Length; i++)
        {
            totalLoss = NumOps.Add(totalLoss, losses[i]);
        }
        
        LastLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(losses.Length));
    }

    /// <summary>
    /// Computes target Q-values for a batch of experiences.
    /// </summary>
    /// <param name="nextStates">Batch of next states.</param>
    /// <param name="rewards">Batch of rewards.</param>
    /// <param name="dones">Batch of done flags.</param>
    /// <returns>Target Q-values.</returns>
    protected virtual Vector<T> ComputeTargets(Tensor<T> nextStates, Vector<T> rewards, Vector<T> dones)
    {
        var targets = new Vector<T>(BatchSize);
        
        if (_useDoubleDQN)
        {
            // Double DQN: Use online network to select actions and target network to evaluate them
            // Predict Q-values using both networks
            var nextQOnlineTensor = _qNetwork.Predict(nextStates);
            var nextQTargetTensor = _targetQNetwork.Predict(nextStates);
            
            var nextQValuesOnline = new Vector<T>[BatchSize];
            var nextQValuesTarget = new Vector<T>[BatchSize];
            for (int i = 0; i < BatchSize; i++)
            {
                nextQValuesOnline[i] = new Vector<T>(ActionSize);
                nextQValuesTarget[i] = new Vector<T>(ActionSize);
                for (int j = 0; j < ActionSize; j++)
                {
                    nextQValuesOnline[i][j] = nextQOnlineTensor[i, j];
                    nextQValuesTarget[i][j] = nextQTargetTensor[i, j];
                }
            }
            
            for (int i = 0; i < BatchSize; i++)
            {
                // Find action with maximum Q-value according to online network
                var bestAction = 0;
                var maxQ = nextQValuesOnline[i][0];
                
                for (int a = 1; a < ActionSize; a++)
                {
                    if (NumOps.GreaterThan(nextQValuesOnline[i][a], maxQ))
                    {
                        maxQ = nextQValuesOnline[i][a];
                        bestAction = a;
                    }
                }
                
                // Evaluate that action using the target network
                var targetQ = nextQValuesTarget[i][bestAction];
                
                // Compute target: reward + gamma * Q(s', a') if not done, else reward
                targets[i] = NumOps.Add(rewards[i], NumOps.Multiply(NumOps.Subtract(NumOps.One, dones[i]), NumOps.Multiply(Gamma, targetQ)));
            }
        }
        else
        {
            // Standard DQN: Use target network for both selection and evaluation
            // Predict Q-values using target network
            var nextQTensor = _targetQNetwork.Predict(nextStates);
            var nextQValues = new Vector<T>[BatchSize];
            for (int i = 0; i < BatchSize; i++)
            {
                nextQValues[i] = new Vector<T>(ActionSize);
                for (int j = 0; j < ActionSize; j++)
                {
                    nextQValues[i][j] = nextQTensor[i, j];
                }
            }
            
            for (int i = 0; i < BatchSize; i++)
            {
                // Find maximum Q-value for next state
                var maxQ = nextQValues[i][0];
                
                for (int a = 1; a < ActionSize; a++)
                {
                    if (NumOps.GreaterThan(nextQValues[i][a], maxQ))
                    {
                        maxQ = nextQValues[i][a];
                    }
                }
                
                // Compute target: reward + gamma * max(Q(s', a')) if not done, else reward
                targets[i] = NumOps.Add(rewards[i], NumOps.Multiply(NumOps.Subtract(NumOps.One, dones[i]), NumOps.Multiply(Gamma, maxQ)));
            }
        }
        
        return targets;
    }

    /// <summary>
    /// Updates the target network parameters from the online network.
    /// </summary>
    /// <param name="tau">The update factor (1.0 for hard update, smaller for soft update).</param>
    protected virtual void UpdateTargetNetwork(T tau)
    {
        _targetQNetwork.CopyFrom(_qNetwork, tau);
    }

    /// <summary>
    /// Updates the exploration rate (epsilon) based on training progress.
    /// </summary>
    protected virtual void UpdateExplorationRate()
    {
        if (_explorationFraction <= 0)
        {
            _currentEpsilon = _finalEpsilon;
            return;
        }
        
        var totalSteps = _options.MaxSteps;
        var annealingSteps = (int)(totalSteps * _explorationFraction);
        
        if (_steps >= annealingSteps)
        {
            _currentEpsilon = _finalEpsilon;
        }
        else
        {
            var fraction = (double)_steps / annealingSteps;
            _currentEpsilon = _initialEpsilon + fraction * (_finalEpsilon - _initialEpsilon);
        }
    }

    /// <summary>
    /// Updates the prioritized replay beta parameter.
    /// </summary>
    protected virtual void UpdatePrioritizedReplayBeta()
    {
        if (_prioritizedReplayBetaSteps <= 0)
        {
            _prioritizedReplayBeta = NumOps.One;
            return;
        }
        
        var fraction = Math.Min(1.0, (double)_steps / _prioritizedReplayBetaSteps);
        var beta = _prioritizedReplayBetaInitial + fraction * (1.0 - _prioritizedReplayBetaInitial);
        _prioritizedReplayBeta = NumOps.FromDouble(beta);
    }

    /// <summary>
    /// Sets the training mode of the agent.
    /// </summary>
    /// <param name="isTraining">Whether the agent should be in training mode.</param>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        _qNetwork.SetTrainingMode(isTraining);
        _targetQNetwork.SetTrainingMode(false); // Target network is always in evaluation mode
    }

    /// <summary>
    /// Saves the model to the specified path.
    /// </summary>
    /// <param name="filePath">The path to save the model to.</param>
    public override void Save(string filePath)
    {
        // Save the networks using their serialization
        var qNetworkData = _qNetwork.Serialize();
        var targetNetworkData = _targetQNetwork.Serialize();
        
        File.WriteAllBytes($"{filePath}_online.dat", qNetworkData);
        File.WriteAllBytes($"{filePath}_target.dat", targetNetworkData);
    }

    /// <summary>
    /// Loads the model from the specified path.
    /// </summary>
    /// <param name="filePath">The path to load the model from.</param>
    public override void Load(string filePath)
    {
        // Load the networks using their deserialization
        if (File.Exists($"{filePath}_online.dat") && File.Exists($"{filePath}_target.dat"))
        {
            var qNetworkData = File.ReadAllBytes($"{filePath}_online.dat");
            var targetNetworkData = File.ReadAllBytes($"{filePath}_target.dat");
            
            _qNetwork.Deserialize(qNetworkData);
            _targetQNetwork.Deserialize(targetNetworkData);
        }
    }

    /// <summary>
    /// Gets the Q-network.
    /// </summary>
    public IQNetwork<T, Tensor<T>> QNetworkModel => _qNetwork;

    /// <summary>
    /// Gets the target Q-network.
    /// </summary>
    public IQNetwork<T, Tensor<T>> TargetQNetworkModel => _targetQNetwork;
    
    /// <summary>
    /// Gets the last computed loss value.
    /// </summary>
    /// <returns>The last computed loss value.</returns>
    public override T GetLatestLoss()
    {
        return base.GetLatestLoss();
    }
    
    /// <summary>
    /// Trains the agent on a batch of experiences.
    /// </summary>
    /// <param name="states">The batch of states.</param>
    /// <param name="actions">The batch of actions.</param>
    /// <param name="rewards">The batch of rewards.</param>
    /// <param name="nextStates">The batch of next states.</param>
    /// <param name="dones">The batch of done flags.</param>
    /// <returns>The average loss value from the training.</returns>
    public T Train(Tensor<T> states, int[] actions, Vector<T> rewards, Tensor<T> nextStates, bool[] dones)
    {
        if (!_isTraining)
        {
            return NumOps.Zero;
        }
        
        // Convert done booleans to tensor
        var donesTensor = new Vector<T>(dones.Length);
        for (int i = 0; i < dones.Length; i++)
        {
            donesTensor[i] = dones[i] ? NumOps.One : NumOps.Zero;
        }
        
        // Compute Q-values for current states
        var currentQValues = _qNetwork.Predict(states);
        
        // Compute target Q-values
        var targetQValues = ComputeTargets(nextStates, rewards, donesTensor);
        
        // Compute losses
        var losses = new Vector<T>(actions.Length);
        T totalLoss = NumOps.Zero;
        
        for (int i = 0; i < actions.Length; i++)
        {
            int actionIdx = actions[i];
            var currentQ = currentQValues[i, actionIdx];
            var targetQ = targetQValues[i];
            
            // Compute squared error
            var diff = NumOps.Subtract(currentQ, targetQ);
            var squaredError = NumOps.Multiply(diff, diff);
            losses[i] = squaredError;
            
            totalLoss = NumOps.Add(totalLoss, squaredError);
        }
        
        // Train the Q-network with the computed targets  
        // We need to create expected outputs based on the targets
        var expectedOutputs = new Tensor<T>(new[] { actions.Length, ActionSize });
        
        // Get current Q-values and update only the taken actions
        for (int i = 0; i < actions.Length; i++)
        {
            for (int j = 0; j < ActionSize; j++)
            {
                expectedOutputs[i, j] = currentQValues[i, j];
            }
            expectedOutputs[i, actions[i]] = targetQValues[i];
        }
        
        // Train the network
        _qNetwork.Train(states, expectedOutputs);
        
        // Update target network if needed
        if (_useSoftUpdate)
        {
            // Soft update
            UpdateTargetNetwork(_tau);
        }
        else if (_updateCounter >= _updateFrequency)
        {
            // Hard update
            UpdateTargetNetwork(NumOps.One);
            _updateCounter = 0;
        }
        
        // Calculate average loss
        LastLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(actions.Length));
        return LastLoss;
    }
    
    /// <summary>
    /// Gets the parameters of the agent as a single vector.
    /// </summary>
    /// <returns>A vector containing all parameters of the agent.</returns>
    public Vector<T> GetParameters()
    {
        // TODO: Implement parameter extraction from the networks
        // This is a placeholder implementation
        return new Vector<T>(0);
    }
    
    /// <summary>
    /// Sets the parameters of the agent from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    public void SetParameters(Vector<T> parameters)
    {
        // TODO: Implement parameter setting for the networks
        // This is a placeholder implementation
    }

    /// <summary>
    /// Neural network for approximating Q-values.
    /// </summary>
    public class QNetwork : IQNetwork<T, Tensor<T>>
    {
        private NeuralNetwork<T> _network = default!;
        
        /// <summary>
        /// Gets the numeric operations for type T.
        /// </summary>
        protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
        
        /// <summary>
        /// Gets the size of the state space.
        /// </summary>
        public int StateSize { get; }

        /// <summary>
        /// Gets the size of the action space.
        /// </summary>
        public int ActionSize { get; }

        /// <summary>
        /// Gets a value indicating whether this network uses a dueling architecture.
        /// </summary>
        public bool IsDueling { get; }
        
        private Vector<T> _lastActionValues = default!;

        /// <summary>
        /// Initializes a new instance of the <see cref="QNetwork"/> class.
        /// </summary>
        /// <param name="stateSize">The size of the state space.</param>
        /// <param name="actionSize">The size of the action space.</param>
        /// <param name="hiddenLayerSizes">The sizes of the hidden layers.</param>
        /// <param name="activationFunction">The activation function to use.</param>
        /// <param name="isDueling">Whether to use a dueling architecture.</param>
        public QNetwork(int stateSize, int actionSize, int[] hiddenLayerSizes, ActivationFunction activationFunction, bool isDueling)
        {
            StateSize = stateSize;
            ActionSize = actionSize;
            IsDueling = isDueling;
            
            // Create layers list
            var layers = new List<ILayer<T>>();
            
            // Input layer
            layers.Add(new InputLayer<T>(stateSize));
            
            // Hidden layers
            int previousSize = stateSize;
            for (int i = 0; i < hiddenLayerSizes.Length; i++)
            {
                layers.Add(new DenseLayer<T>(previousSize, hiddenLayerSizes[i], ActivationFunctionFactory<T>.CreateActivationFunction(activationFunction)));
                previousSize = hiddenLayerSizes[i];
            }
            
            if (isDueling)
            {
                // Dueling network architecture
                // Value stream
                layers.Add(new DenseLayer<T>(previousSize, hiddenLayerSizes[hiddenLayerSizes.Length - 1], ActivationFunctionFactory<T>.CreateActivationFunction(activationFunction)));
                layers.Add(new DenseLayer<T>(hiddenLayerSizes[hiddenLayerSizes.Length - 1], 1, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.Linear)));
                
                // Advantage stream
                layers.Add(new DenseLayer<T>(previousSize, hiddenLayerSizes[hiddenLayerSizes.Length - 1], ActivationFunctionFactory<T>.CreateActivationFunction(activationFunction)));
                layers.Add(new DenseLayer<T>(hiddenLayerSizes[hiddenLayerSizes.Length - 1], actionSize, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.Linear)));
                
                // Combine value and advantage
                // This is typically done in the forward pass
            }
            else
            {
                // Standard architecture
                layers.Add(new DenseLayer<T>(previousSize, actionSize, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.Linear)));
            }
            
            // Create neural network architecture
            var architecture = new NeuralNetworkArchitecture<T>(
                complexity: NetworkComplexity.Medium,
                taskType: NeuralNetworkTaskType.Regression,
                shouldReturnFullSequence: false,
                layers: layers,
                isDynamicSampleCount: true,
                isPlaceholder: false);
            
            // Create neural network
            _network = new NeuralNetwork<T>(architecture);
            _lastActionValues = new Vector<T>(ActionSize);
        }

        /// <summary>
        /// Predicts Q-values for the given input state(s).
        /// </summary>
        /// <param name="input">The input tensor representing one or more states.</param>
        /// <returns>Q-values tensor with shape [batch_size, action_size] or [action_size] for single input.</returns>
        public Tensor<T> Predict(Tensor<T> input)
        {
            // Ensure input is properly shaped
            if (input.Rank == 1)
            {
                // Single state: reshape to [1, state_size]
                input = input.Reshape(new[] { 1, input.Length });
            }
            else if (input.Rank != 2)
            {
                throw new ArgumentException("Input must be either 1D (single state) or 2D (batch of states)");
            }
            
            // Use the underlying neural network to predict
            var output = _network.Predict(input);
            
            // For dueling architecture, we need to combine value and advantage streams
            if (IsDueling)
            {
                // TODO: Implement dueling architecture combination
                // For now, just return the output
                return output;
            }
            
            return output;
        }
        
        /// <summary>
        /// Performs a forward pass through the network for a single input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>Q-values for each action.</returns>
        public Vector<T> ForwardSingle(Tensor<T> input)
        {
            var result = Predict(input);
            // If single input, extract the vector
            var qValues = new Vector<T>(ActionSize);
            if (result.Shape.Length == 2 && result.Shape[0] == 1)
            {
                // Extract Q-values from batch dimension
                for (int i = 0; i < ActionSize; i++)
                {
                    qValues[i] = result[0, i];
                }
            }
            else if (result.Shape.Length == 1)
            {
                // Already a 1D tensor
                for (int i = 0; i < ActionSize; i++)
                {
                    qValues[i] = result[i];
                }
            }
            else
            {
                throw new InvalidOperationException($"Unexpected result shape: [{string.Join(", ", result.Shape)}]");
            }
            return qValues;
        }
        
        /// <summary>
        /// Performs a forward pass for a batch of inputs.
        /// </summary>
        /// <param name="batchInput">Batch of input tensors.</param>
        /// <returns>Q-values for each action for each input in the batch.</returns>
        public Vector<T>[] ForwardBatch(Tensor<T> batchInput)
        {
            var output = Predict(batchInput);
            var batchSize = output.Shape[0];
            var result = new Vector<T>[batchSize];
            
            // Extract vectors for each sample in the batch
            for (int i = 0; i < batchSize; i++)
            {
                result[i] = new Vector<T>(ActionSize);
                for (int j = 0; j < ActionSize; j++)
                {
                    result[i][j] = output[i, j];
                }
            }
            
            return result;
        }

        /// <summary>
        /// Performs backward propagation through the network.
        /// </summary>
        /// <param name="losses">The losses to backpropagate.</param>
        /// <param name="optimizer">The optimizer to use for updating parameters.</param>
        public void Backward(Vector<T> losses, IOptimizer<T, Tensor<T>, Tensor<T>> optimizer)
        {
            // Since the neural network doesn't directly support this method,
            // we would need to implement custom backpropagation logic here.
            // For now, this is a placeholder that converts the vector losses to a tensor
            // and uses the network's built-in training mechanism.
            throw new NotImplementedException("Direct backward propagation is not implemented. Use Train method instead.");
        }

        /// <summary>
        /// Copies parameters from another network.
        /// </summary>
        /// <param name="source">The source network.</param>
        /// <param name="tau">The update factor (1.0 for hard update, smaller for soft update).</param>
        public void CopyFrom(IQNetwork<T, Tensor<T>> other, T tau)
        {
            var sourceParams = other.GetUnderlyingNetwork().GetParameters();
            var currentParams = _network.GetParameters();
            
            // Soft update: θ' = τ*θ + (1-τ)*θ'
            for (int i = 0; i < sourceParams.Length; i++)
            {
                currentParams[i] = NumOps.Add(
                    NumOps.Multiply(tau, sourceParams[i]),
                    NumOps.Multiply(NumOps.Subtract(NumOps.One, tau), currentParams[i])
                );
            }
            
            _network.UpdateParameters(currentParams);
        }

        /// <summary>
        /// Sets the training mode of the network.
        /// </summary>
        /// <param name="isTraining">Whether the network should be in training mode.</param>
        public void SetTrainingMode(bool isTraining)
        {
            _network.SetTrainingMode(isTraining);
        }

        /// <summary>
        /// Saves the network to the specified path.
        /// </summary>
        /// <param name="path">The path to save the network to.</param>
        public void Save(string path)
        {
            var data = _network.Serialize();
            File.WriteAllBytes(path, data);
        }

        /// <summary>
        /// Loads the network from the specified path.
        /// </summary>
        /// <param name="path">The path to load the network from.</param>
        public void Load(string path)
        {
            var data = File.ReadAllBytes(path);
            _network.Deserialize(data);
        }

        /// <summary>
        /// Gets the underlying neural network.
        /// </summary>
        /// <returns>The underlying neural network model.</returns>
        public INeuralNetworkModel<T> GetUnderlyingNetwork()
        {
            return _network;
        }

        // IFullModel implementation
        public void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            _network.Train(input, expectedOutput);
        }

        public byte[] Serialize()
        {
            return _network.Serialize();
        }

        public void Deserialize(byte[] data)
        {
            _network.Deserialize(data);
        }

        public ModelMetadata<T> ComputeMetaData()
        {
            return new ModelMetadata<T>
            {
                ModelType = ModelType.NeuralNetwork,
                Description = $"Q-Network with {StateSize} states and {ActionSize} actions",
                FeatureCount = StateSize,
                Complexity = _network.GetParameterCount(),
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "StateSize", StateSize },
                    { "ActionSize", ActionSize },
                    { "IsDueling", IsDueling },
                    { "ParameterCount", _network.GetParameterCount() },
                    { "CreatedDate", DateTime.Now },
                    { "Version", "1.0" }
                }
            };
        }

        public void UpdateParameters(Tensor<T> parameters)
        {
            // Convert tensor to vector for the neural network
            var vectorParams = new Vector<T>(parameters.Length);
            for (int i = 0; i < parameters.Length; i++)
            {
                vectorParams[i] = parameters[i];
            }
            _network.UpdateParameters(vectorParams);
        }

        public int ParameterCount => _network.GetParameterCount();

        public string[] FeatureNames { get; set; } = Array.Empty<string>();

        public IFullModel<T, Tensor<T>, Tensor<T>> Clone()
        {
            var clone = new QNetwork(StateSize, ActionSize, new int[0], ActivationFunction.ReLU, IsDueling);
            clone._network = (NeuralNetwork<T>)_network.Clone();
            return clone;
        }

        public IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
        {
            return Clone();
        }

        public ModelMetadata<T> GetModelMetadata()
        {
            return ComputeMetaData();
        }

        public Vector<T> GetParameters()
        {
            return _network.GetParameters();
        }

        public void SetParameters(Vector<T> parameters)
        {
            if (parameters == null)
            {
                throw new ArgumentNullException(nameof(parameters));
            }
            
            _network.UpdateParameters(parameters);
        }

        public IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
        {
            var clone = (QNetwork)Clone();
            var tensorParams = new Tensor<T>(new[] { parameters.Length });
            for (int i = 0; i < parameters.Length; i++)
            {
                tensorParams[i] = parameters[i];
            }
            clone.UpdateParameters(tensorParams);
            return clone;
        }

        public IEnumerable<int> GetActiveFeatureIndices()
        {
            return _network.GetActiveFeatureIndices();
        }

        public bool IsFeatureUsed(int featureIndex)
        {
            return _network.IsFeatureUsed(featureIndex);
        }

        public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
        {
            _network.SetActiveFeatureIndices(featureIndices);
        }

        #region IInterpretableModel Implementation

        protected readonly HashSet<InterpretationMethod> _enabledMethods = new();
        protected Vector<int> _sensitiveFeatures;
        protected readonly List<FairnessMetric> _fairnessMetrics = new();
        protected IModel<Tensor<T>, Tensor<T>, ModelMetadata<T>> _baseModel;

        /// <summary>
        /// Gets the global feature importance across all predictions.
        /// </summary>
        public virtual async Task<Dictionary<int, T>> GetGlobalFeatureImportanceAsync()
        {
            return await InterpretableModelHelper.GetGlobalFeatureImportanceAsync(this, _enabledMethods);
        }

        /// <summary>
        /// Gets the local feature importance for a specific input.
        /// </summary>
        public virtual async Task<Dictionary<int, T>> GetLocalFeatureImportanceAsync(Tensor<T> input)
        {
            return await InterpretableModelHelper.GetLocalFeatureImportanceAsync(this, _enabledMethods, input);
        }

        /// <summary>
        /// Gets SHAP values for the given inputs.
        /// </summary>
        public virtual async Task<Matrix<T>> GetShapValuesAsync(Tensor<T> inputs)
        {
            return await InterpretableModelHelper.GetShapValuesAsync(this, _enabledMethods);
        }

        /// <summary>
        /// Gets LIME explanation for a specific input.
        /// </summary>
        public virtual async Task<LimeExplanation<T>> GetLimeExplanationAsync(Tensor<T> input, int numFeatures = 10)
        {
            return await InterpretableModelHelper.GetLimeExplanationAsync<T>(_enabledMethods, numFeatures);
        }

        /// <summary>
        /// Gets partial dependence data for specified features.
        /// </summary>
        public virtual async Task<PartialDependenceData<T>> GetPartialDependenceAsync(Vector<int> featureIndices, int gridResolution = 20)
        {
            return await InterpretableModelHelper.GetPartialDependenceAsync<T>(_enabledMethods, featureIndices, gridResolution);
        }

        /// <summary>
        /// Gets counterfactual explanation for a given input and desired output.
        /// </summary>
        public virtual async Task<CounterfactualExplanation<T>> GetCounterfactualAsync(Tensor<T> input, Tensor<T> desiredOutput, int maxChanges = 5)
        {
            return await InterpretableModelHelper.GetCounterfactualAsync<T>(_enabledMethods, maxChanges);
        }

        /// <summary>
        /// Gets model-specific interpretability information.
        /// </summary>
        public virtual async Task<Dictionary<string, object>> GetModelSpecificInterpretabilityAsync()
        {
            return await InterpretableModelHelper.GetModelSpecificInterpretabilityAsync(this);
        }

        /// <summary>
        /// Generates a text explanation for a prediction.
        /// </summary>
        public virtual async Task<string> GenerateTextExplanationAsync(Tensor<T> input, Tensor<T> prediction)
        {
            return await InterpretableModelHelper.GenerateTextExplanationAsync(this, input, prediction);
        }

        /// <summary>
        /// Gets feature interaction effects between two features.
        /// </summary>
        public virtual async Task<T> GetFeatureInteractionAsync(int feature1Index, int feature2Index)
        {
            return await InterpretableModelHelper.GetFeatureInteractionAsync<T>(_enabledMethods, feature1Index, feature2Index);
        }

        /// <summary>
        /// Validates fairness metrics for the given inputs.
        /// </summary>
        public virtual async Task<FairnessMetrics<T>> ValidateFairnessAsync(Tensor<T> inputs, int sensitiveFeatureIndex)
        {
            return await InterpretableModelHelper.ValidateFairnessAsync<T>(_fairnessMetrics);
        }

        /// <summary>
        /// Gets anchor explanation for a given input.
        /// </summary>
        public virtual async Task<AnchorExplanation<T>> GetAnchorExplanationAsync(Tensor<T> input, T threshold)
        {
            return await InterpretableModelHelper.GetAnchorExplanationAsync(_enabledMethods, threshold);
        }

        /// <summary>
        /// Sets the base model for interpretability analysis.
        /// </summary>
        public virtual void SetBaseModel(IModel<Tensor<T>, Tensor<T>, ModelMetadata<T>> model)
        {
            _baseModel = model ?? throw new ArgumentNullException(nameof(model));
        }

        /// <summary>
        /// Enables specific interpretation methods.
        /// </summary>
        public virtual void EnableMethod(params InterpretationMethod[] methods)
        {
            foreach (var method in methods)
            {
                _enabledMethods.Add(method);
            }
        }

        /// <summary>
        /// Configures fairness evaluation settings.
        /// </summary>
        public virtual void ConfigureFairness(Vector<int> sensitiveFeatures, params FairnessMetric[] fairnessMetrics)
        {
            _sensitiveFeatures = sensitiveFeatures ?? throw new ArgumentNullException(nameof(sensitiveFeatures));
            _fairnessMetrics.Clear();
            _fairnessMetrics.AddRange(fairnessMetrics);
        }

        #endregion
    }
}