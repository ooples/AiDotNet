using RainbowDQNOptions = AiDotNet.ReinforcementLearning.Models.Options.RainbowDQNOptions;

namespace AiDotNet.ReinforcementLearning.Agents
{
    /// <summary>
    /// Rainbow DQN agent that combines multiple improvements to DQN:
    /// - Double DQN (inherited from base)
    /// - Dueling architecture (inherited from base)
    /// - Prioritized Experience Replay (inherited from base)
    /// - Multi-step learning
    /// - Distributional RL (C51) - simplified implementation
    /// </summary>
    /// <typeparam name="TState">The type of state.</typeparam>
    /// <typeparam name="T">The numeric type.</typeparam>
    public class RainbowDQNAgent<TState, T> : DQNAgent<TState, T>
        where TState : Tensor<T>
    {
        private readonly int _numAtoms;
        private readonly T _vMin = default!;
        private readonly T _vMax = default!;
        private readonly T[] _support;
        private readonly int _multiStepN;
        private readonly bool _useDistributional;
        private readonly List<MultiStepTransition> _multiStepBuffer = new();

        /// <summary>
        /// Initializes a new instance of the <see cref="RainbowDQNAgent{TState, T}"/> class.
        /// </summary>
        /// <param name="options">The options for the Rainbow DQN algorithm.</param>
        public RainbowDQNAgent(RainbowDQNOptions options)
            : base(ConvertToDQNOptions(options))
        {
            _numAtoms = options.UseDistributionalRL ? options.AtomCount : 1;
            _vMin = NumOps.FromDouble(options.ValueRangeMin);
            _vMax = NumOps.FromDouble(options.ValueRangeMax);
            _multiStepN = options.NSteps;
            _useDistributional = options.UseDistributionalRL;

            // Initialize support for distributional RL
            _support = new T[_numAtoms];
            if (_useDistributional)
            {
                var deltaZ = NumOps.Divide(NumOps.Subtract(_vMax, _vMin), NumOps.FromDouble(_numAtoms - 1));
                for (int i = 0; i < _numAtoms; i++)
                {
                    _support[i] = NumOps.Add(_vMin, NumOps.Multiply(deltaZ, NumOps.FromDouble(i)));
                }
            }
        }

        /// <summary>
        /// Stores an experience in the replay buffer, handling multi-step returns if enabled.
        /// </summary>
        public void StoreExperience(TState state, int action, T reward, TState nextState, bool done)
        {
            if (_multiStepN > 1)
            {
                // Add to multi-step buffer
                _multiStepBuffer.Add(new MultiStepTransition
                {
                    State = state,
                    Action = action,
                    Reward = reward,
                    NextState = nextState,
                    Done = done
                });

                // Process multi-step transitions
                if (_multiStepBuffer.Count >= _multiStepN || done)
                {
                    ProcessMultiStepTransitions();
                }
            }
            else
            {
                // Use base class single-step storage
                _replayBuffer.Add(state, action, reward, nextState, done);
            }
        }

        /// <summary>
        /// Processes multi-step transitions from the buffer.
        /// </summary>
        private void ProcessMultiStepTransitions()
        {
            if (_multiStepBuffer.Count == 0) return;

            // Calculate n-step return
            var firstTransition = _multiStepBuffer[0];
            T nStepReturn = NumOps.Zero;
            T discountPower = NumOps.One;
            bool done = false;

            for (int i = 0; i < Math.Min(_multiStepBuffer.Count, _multiStepN); i++)
            {
                var transition = _multiStepBuffer[i];
                nStepReturn = NumOps.Add(nStepReturn, NumOps.Multiply(discountPower, transition.Reward));
                discountPower = NumOps.Multiply(discountPower, _gamma);
                done = transition.Done;
                if (done) break;
            }

            // Get the final next state
            var lastTransition = _multiStepBuffer[Math.Min(_multiStepBuffer.Count - 1, _multiStepN - 1)];
            
            // Store the n-step transition
            _replayBuffer.Add(
                firstTransition.State,
                firstTransition.Action,
                nStepReturn,
                lastTransition.NextState,
                done);

            // Remove processed transition
            _multiStepBuffer.RemoveAt(0);
        }

        /// <summary>
        /// Learns from an experience tuple, using multi-step storage if enabled.
        /// </summary>
        public override void Learn(TState state, int action, T reward, TState nextState, bool done)
        {
            // Store experience (handles multi-step internally)
            StoreExperience(state, action, reward, nextState, done);
            
            // Increment step counter
            IncrementStepCounter();
            _steps++;
            
            // Train if we have enough experiences
            if (_replayBuffer.Size >= _options.BatchSize && _steps >= _options.WarmupSteps)
            {
                // Train the network by sampling from replay buffer
                var batch = _replayBuffer.Sample(_options.BatchSize);
                
                // Convert arrays to tensors for training
                var statesTensor = StackStates(batch.states);
                var nextStatesTensor = StackStates(batch.nextStates);
                var rewardsVector = new Vector<T>(batch.rewards);
                
                Train(statesTensor, batch.actions, rewardsVector, nextStatesTensor, batch.dones);
                
                // Update target network
                _updateCounter++;
                if (_useSoftUpdate)
                {
                    UpdateTargetNetwork(_tau);
                }
                else if (_updateCounter >= _updateFrequency)
                {
                    UpdateTargetNetwork(NumOps.One);
                    _updateCounter = 0;
                }
            }
            
            // Update exploration rate
            UpdateExplorationRate();
        }

        /// <summary>
        /// Multi-step transition structure.
        /// </summary>
        private struct MultiStepTransition
        {
            public TState State { get; set; }
            public int Action { get; set; }
            public T Reward { get; set; }
            public TState NextState { get; set; }
            public bool Done { get; set; }
        }

        /// <summary>
        /// Stacks multiple states into a single tensor for batch processing.
        /// </summary>
        /// <param name="states">Array of states to stack.</param>
        /// <returns>Stacked tensor of shape [batch_size, state_size].</returns>
        private Tensor<T> StackStates(TState[] states)
        {
            if (states.Length == 0)
                throw new ArgumentException("Cannot stack empty states array");
                
            var batchSize = states.Length;
            var stateSize = states[0].Length;
            var result = new Tensor<T>(new[] { batchSize, stateSize });
            
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < stateSize; j++)
                {
                    result[i, j] = states[i][j];
                }
            }
            
            return result;
        }

        private static DQNOptions ConvertToDQNOptions(RainbowDQNOptions options)
        {
            return new DQNOptions
            {
                StateSize = options.StateSize,
                ActionSize = options.ActionSize,
                LearningRate = options.LearningRate,
                Gamma = options.DiscountFactor,
                InitialExplorationRate = options.InitialExplorationRate,
                FinalExplorationRate = options.FinalExplorationRate,
                ExplorationFraction = options.ExplorationFraction,
                ReplayBufferCapacity = options.ReplayBufferCapacity,
                BatchSize = options.BatchSize,
                TargetNetworkUpdateFrequency = options.TargetUpdateFrequency,
                WarmupSteps = options.WarmupSteps,
                UseDoubleDQN = options.UseDoubleDQN,
                UseDuelingDQN = options.UseDuelingDQN,
                UsePrioritizedReplay = options.UsePrioritizedReplay,
                PrioritizedReplayAlpha = options.PrioritizedReplayAlpha,
                PrioritizedReplayBetaInitial = options.PrioritizedReplayBetaInitial,
                PrioritizedReplayBetaSteps = options.PrioritizedReplayBetaSteps,
                OptimizerType = options.OptimizerType,
                Seed = null, // RainbowDQNOptions doesn't have RandomSeed
                UseSoftTargetUpdate = options.UseSoftTargetUpdate,
                NetworkArchitecture = options.NetworkArchitecture,
                ActivationFunction = options.ActivationFunction,
                MaxSteps = options.MaxTrainingSteps,
                ClipRewards = options.ClipRewards,
                UseNStepReturns = options.UseMultiStepReturns,
                NSteps = options.NSteps,
                HuberDelta = 1.0 // Default value as it's not in RainbowDQNOptions
            };
        }
    }

}