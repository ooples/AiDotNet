global using AiDotNet.ReinforcementLearning.ReplayBuffers;
namespace AiDotNet.ReinforcementLearning.Agents
{
    /// <summary>
    /// Distributional DQN agent implementing the C51 algorithm for value distribution learning.
    /// Instead of learning expected values, it learns a full probability distribution over returns.
    /// </summary>
    /// <typeparam name="TState">Type of state representation.</typeparam>
    /// <typeparam name="T">Numeric type for rewards and values.</typeparam>
    public class DistributionalDQNAgent<TState, T> : DQNAgent<TState, T> 
        where TState : Tensor<T>
    {
        private readonly DistributionalDQNOptions<T> _distOptions = default!;
        private readonly int _atomCount;
        private readonly T[] _supportValues;
        private readonly T _vMin = default!;
        private readonly T _vMax = default!;
        private readonly T _supportDelta = default!;

        /// <summary>
        /// Initializes a new instance of the <see cref="DistributionalDQNAgent{TState, T}"/> class.
        /// </summary>
        /// <param name="options">Distributional DQN algorithm options.</param>
        public DistributionalDQNAgent(DistributionalDQNOptions<T> options)
            : base(options)
        {
            _distOptions = options;
            _atomCount = options.AtomCount;
            _vMin = options.ValueRangeMin;
            _vMax = options.ValueRangeMax;
            
            // Create support values (atom positions)
            _supportDelta = NumOps.Divide(NumOps.Subtract(_vMax, _vMin), NumOps.FromDouble(_atomCount - 1));
            _supportValues = new T[_atomCount];
            
            for (int i = 0; i < _atomCount; i++)
            {
                _supportValues[i] = NumOps.Add(_vMin, NumOps.Multiply(_supportDelta, NumOps.FromDouble(i)));
            }
            
            // Override network initialization to use distributional architecture
            InitializeNetworks();
        }

        /// <summary>
        /// Initializes the Q-network and target network with distributional architecture.
        /// </summary>
        protected override void InitializeNetworks()
        {
            // Create Q-Network with distributional output
            _qNetwork = CreateDistributionalNetwork(false);
            
            // Create Target Network
            _targetQNetwork = CreateDistributionalNetwork(true);
            
            // Sync target network with policy network
            UpdateTargetNetwork(NumOps.One);
        }

        /// <summary>
        /// Creates a neural network with distributional output layer.
        /// </summary>
        /// <param name="isTarget">Whether this is the target network.</param>
        /// <returns>Neural network with distributional architecture.</returns>
        private QNetwork CreateDistributionalNetwork(bool isTarget)
        {
            // Build layers list
            var layers = new List<ILayer<T>>
            {
                // Input layer
                new InputLayer<T>(StateSize)
            };
            
            // Hidden layers
            for (int i = 0; i < _options.NetworkArchitecture.Length; i++)
            {
                int inputSize = (i == 0) ? StateSize : _options.NetworkArchitecture[i - 1];
                layers.Add(new DenseLayer<T>(inputSize, _options.NetworkArchitecture[i], ActivationFunctionFactory<T>.CreateActivationFunction(_options.ActivationFunction)));
            }
            
            // Add distributional output layer
            layers.Add(new DistributionalLayer<T>(new[] { _options.NetworkArchitecture[_options.NetworkArchitecture.Length - 1] }, ActionSize, _atomCount));
            
            // Create architecture with layers
            var architecture = new NeuralNetworkArchitecture<T>(
                complexity: NetworkComplexity.Medium,
                taskType: NeuralNetworkTaskType.Regression,
                shouldReturnFullSequence: false,
                layers: layers);
            
            // Create name for the network
            string networkName = isTarget ? "DistributionalDQN_Target" : "DistributionalDQN_Policy";
            
            return new QNetwork(StateSize, ActionSize, _options.NetworkArchitecture, _options.ActivationFunction, _options.UseDuelingDQN);
        }

        /// <summary>
        /// Selects an action for the given state based on the current policy.
        /// </summary>
        /// <param name="state">The current state.</param>
        /// <param name="isTraining">Whether the agent is in training mode.</param>
        /// <returns>Selected action.</returns>
        public override int SelectAction(TState state, bool isTraining = true)
        {
            // Epsilon-greedy exploration
            if (isTraining && _random.NextDouble() < _currentEpsilon)
            {
                return _random.Next(ActionSize);
            }
            
            // Convert state to tensor
            var stateTensor = state;
            
            // Forward pass to get action distributions
            var distributions = _qNetwork.Predict(stateTensor);
            
            // Calculate expected values for each action
            var expectedValues = new T[ActionSize];
            
            for (int a = 0; a < ActionSize; a++)
            {
                expectedValues[a] = NumOps.Zero;
                
                for (int z = 0; z < _atomCount; z++)
                {
                    expectedValues[a] = NumOps.Add(expectedValues[a], 
                        NumOps.Multiply(distributions[0, a, z], _supportValues[z]));
                }
            }
            
            // Return the action with the highest expected value
            return ArgMax(expectedValues);
        }

        /// <summary>
        /// Learns from an experience tuple and updates the agent's policy.
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
            
            // Update counter
            _steps++;
            _updateCounter++;
            
            // Update prioritized replay beta parameter
            if (_usePrioritizedReplay)
            {
                UpdatePrioritizedReplayBeta();
            }
            
            LearnFromBuffer();
        }
        
        /// <summary>
        /// Updates the agent's policy based on experiences from the replay buffer.
        /// </summary>
        private void LearnFromBuffer()
        {
            // If we don't have enough experiences yet, skip learning
            if (_replayBuffer.Size < BatchSize)
                return;
            
            // Sample batch of experiences from replay buffer
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
            
            // Compute distributional targets using the Bellman update
            var targets = ComputeDistributionalTargets(batch, nextStatesBatch);
            
            // Compute current distributions
            var currentDistributions = _qNetwork.Predict(statesBatch);
            
            // Compute cross-entropy loss between current and target distributions
            var losses = new Vector<T>(BatchSize);
            
            for (int i = 0; i < BatchSize; i++)
            {
                int action = batch.Actions[i];
                T batchLoss = NumOps.Zero;
                
                for (int j = 0; j < _atomCount; j++)
                {
                    // Cross-entropy loss: -target * log(current)
                    var current = currentDistributions[i, action, j];
                    var target = targets[i, action, j];
                    
                    // Avoid log(0) by adding a small epsilon
                    var epsilon = NumOps.FromDouble(1e-6);
                    var logProb = NumOps.Log(NumOps.Add(current, epsilon));
                    batchLoss = NumOps.Add(batchLoss, NumOps.Multiply(NumOps.Negate(target), logProb));
                }
                
                // Apply importance sampling weights if using prioritized replay
                if (_usePrioritizedReplay && importanceWeights != null)
                {
                    losses[i] = NumOps.Multiply(batchLoss, importanceWeights[i]);
                }
                else
                {
                    losses[i] = batchLoss;
                }
                
                // Update priorities in replay buffer if using prioritized replay
                if (_usePrioritizedReplay && _replayBuffer is PrioritizedReplayBuffer<TState, int, T> prBuffer && indices != null)
                {
                    var priority = NumOps.Add(batchLoss, NumOps.FromDouble(1e-6)); // Add small constant
                    prBuffer.UpdatePriority(indices[i], priority);
                }
            }
            
            // Backward pass and optimization
            _qNetwork.Backward(losses, _optimizer);
            
            // Update the target network periodically
            _steps++;
            _updateCounter++;
            
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
            
            // Update exploration rate
            UpdateExplorationRate();
            
            // Update prioritized replay beta parameter
            if (_usePrioritizedReplay)
            {
                UpdatePrioritizedReplayBeta();
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
        /// Computes distributional targets using the Categorical Algorithm from the C51 paper.
        /// </summary>
        /// <param name="batch">Batch of experiences.</param>
        /// <param name="nextStatesBatch">Batch of next states as tensor.</param>
        /// <returns>Target distributions for each action in the batch.</returns>
        private Tensor<T> ComputeDistributionalTargets(ReplayBatch<TState, int, T> batch, Tensor<T> nextStatesBatch)
        {
            var batchSize = batch.States.Length;
            
            // Get next state action distributions from target network
            var nextDistributions = _targetQNetwork.Predict(nextStatesBatch);
            
            // For double DQN, use online network to select actions
            var nextQValues = new T[batchSize, ActionSize];
            
            for (int i = 0; i < batchSize; i++)
            {
                for (int a = 0; a < ActionSize; a++)
                {
                    nextQValues[i, a] = NumOps.Zero;
                    for (int z = 0; z < _atomCount; z++)
                    {
                        nextQValues[i, a] = NumOps.Add(nextQValues[i, a], 
                            NumOps.Multiply(nextDistributions[i, a, z], _supportValues[z]));
                    }
                }
            }
            
            // Get optimal actions using online network's value estimates
            var optimalActions = new int[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                T maxValue = NumOps.FromDouble(double.MinValue);
                optimalActions[i] = 0;
                
                for (int a = 0; a < ActionSize; a++)
                {
                    if (NumOps.GreaterThan(nextQValues[i, a], maxValue))
                    {
                        maxValue = nextQValues[i, a];
                        optimalActions[i] = a;
                    }
                }
            }
            
            // Initialize target distributions
            var targetDistributions = new Tensor<T>(new[] { batchSize, ActionSize, _atomCount });
            
            // Compute target distributions for each sample in batch
            for (int i = 0; i < batchSize; i++)
            {
                var reward = batch.Rewards[i];
                var done = batch.Dones[i];
                int optimalAction = optimalActions[i];
                
                // Create target distribution projected onto support
                var projectedDistribution = new T[_atomCount];
                
                if (done)
                {
                    // If terminal state, target is just the reward
                    var targetValue = reward;
                    projectedDistribution = ProjectValue(targetValue);
                }
                else
                {
                    // For non-terminal states, apply Bellman update for each atom
                    for (int z = 0; z < _atomCount; z++)
                    {
                        // Get probability for this atom
                        var atomProb = nextDistributions[i, optimalAction, z];
                        
                        // Apply Bellman update: z' = r + γ*z
                        var targetValue = NumOps.Add(reward, 
                            NumOps.Multiply(_gamma, _supportValues[z]));
                        
                        // Project target value onto support and add weighted probabilities
                        var projectedAtom = ProjectValue(targetValue);
                        
                        for (int j = 0; j < _atomCount; j++)
                        {
                            projectedDistribution[j] = NumOps.Add(projectedDistribution[j], 
                                NumOps.Multiply(atomProb, projectedAtom[j]));
                        }
                    }
                }
                
                // Set target distribution for the action that was taken
                for (int j = 0; j < _atomCount; j++)
                {
                    targetDistributions[i, batch.Actions[i], j] = projectedDistribution[j];
                }
            }
            
            return targetDistributions;
        }

        /// <summary>
        /// Projects a scalar value onto the categorical support using the approach from the C51 paper.
        /// </summary>
        /// <param name="value">Value to project.</param>
        /// <returns>Probability mass distribution over the support.</returns>
        private T[] ProjectValue(T value)
        {
            var distribution = new T[_atomCount];
            
            // Clamp value to be within support range
            value = MathHelper.Clamp(value, _vMin, _vMax);
            
            // Find which bin the value falls into
            var bj = NumOps.Divide(NumOps.Subtract(value, _vMin), _supportDelta);
            var j = MathHelper.Floor(bj);
            int lowerIndex = Math.Min(Math.Max(0, (int)NumOps.ToInt32(j)), _atomCount - 2);
            int upperIndex = lowerIndex + 1;
            
            // Calculate probability mass for upper index (linear interpolation)
            var upperProb = NumOps.Subtract(bj, j);
            
            // Set probability mass
            distribution[lowerIndex] = NumOps.Subtract(NumOps.One, upperProb);
            distribution[upperIndex] = upperProb;
            
            return distribution;
        }

        /// <summary>
        /// Returns the index of the maximum value in an array.
        /// </summary>
        /// <param name="values">Array of values.</param>
        /// <returns>Index of the maximum value.</returns>
        private int ArgMax(T[] values)
        {
            int maxIndex = 0;
            T maxValue = values[0];
            
            for (int i = 1; i < values.Length; i++)
            {
                if (NumOps.GreaterThan(values[i], maxValue))
                {
                    maxValue = values[i];
                    maxIndex = i;
                }
            }
            
            return maxIndex;
        }
    }

    /// <summary>
    /// Options for the Distributional DQN (C51) algorithm.
    /// </summary>
    /// <typeparam name="T">Numeric type for rewards and values.</typeparam>
    public class DistributionalDQNOptions<T> : DQNOptions
    {
        private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
        
        /// <summary>
        /// Gets or sets the number of atoms in the categorical distribution.
        /// </summary>
        public int AtomCount { get; set; } = 51;
        
        /// <summary>
        /// Gets or sets the minimum value in the support range.
        /// </summary>
        public T ValueRangeMin { get; set; }
        
        /// <summary>
        /// Gets or sets the maximum value in the support range.
        /// </summary>
        public T ValueRangeMax { get; set; }
        
        /// <summary>
        /// Initializes a new instance of the <see cref="DistributionalDQNOptions{T}"/> class.
        /// </summary>
        public DistributionalDQNOptions()
        {
            ValueRangeMin = _numOps.FromDouble(-10);
            ValueRangeMax = _numOps.FromDouble(10);
        }
    }
}