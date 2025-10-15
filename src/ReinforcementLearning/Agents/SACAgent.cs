using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Interfaces;
using AiDotNet.ReinforcementLearning.Models.Options;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.ReinforcementLearning.Policies;
using AiDotNet.Interfaces;
using AiDotNet.Helpers;
using System;
using System.Collections.Generic;

namespace AiDotNet.ReinforcementLearning.Agents
{
    /// <summary>
    /// Implementation of the Soft Actor-Critic (SAC) algorithm for continuous control.
    /// </summary>
    /// <typeparam name="TState">The type used to represent the environment state.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// <para>
    /// SAC is an off-policy actor-critic algorithm based on the maximum entropy reinforcement learning framework.
    /// It incorporates three key components:
    /// 1. An actor-critic architecture with a stochastic policy
    /// 2. A maximum entropy objective that encourages exploration
    /// 3. A policy that maximizes both expected return and entropy
    /// </para>
    /// <para>
    /// This implementation includes:
    /// - Automatic entropy coefficient tuning
    /// - Twin critics for regularization (similar to TD3)
    /// - Squashed Gaussian policy for bounded actions
    /// - Prioritized experience replay
    /// </para>
    /// </remarks>
    public class SACAgent<TState, T> : StochasticActorCriticAgentBase<TState, Vector<T>, T>
        where TState : Tensor<T>
    {
        private readonly SACOptions _options = default!;
        private readonly bool _updateAfterEachStep;
        private readonly int _trainingFrequency;
        private readonly int _gradientsStepsPerUpdate;
        private readonly bool _useFixedWarmUpExploration;
        private readonly T _warmUpExplorationStdDev = default!;
        private T _logEntropyCoefficient;  // log alpha, used for auto-tuning

        /// <summary>
        /// Initializes a new instance of the <see cref="SACAgent{TState, T}"/> class.
        /// </summary>
        /// <param name="options">Options for the SAC algorithm.</param>
        public SACAgent(SACOptions options)
            : base(
                  // Actor network (Continuous stochastic policy)
                  actor: new StochasticPolicy<TState, T>(
                      options.StateSize,
                      options.ActionSize,
                      options.ActorNetworkArchitecture,
                      options.ActorActivationFunction,
                      null, // Default action bounds (-1 to 1)
                      null, // Default action bounds (-1 to 1)
                      options.UseStateDependentExploration, // Learn std dev if state-dependent exploration
                      0.5, // Initial std dev
                      options.MinLogProb > 0 ? Math.Exp(options.MinLogProb) : 0.01, // Convert log bounds to std dev bounds
                      options.MaxLogProb > 0 ? Math.Exp(options.MaxLogProb) : 2.0, // Convert log bounds to std dev bounds
                      options.Seed),
                  // Critic networks
                  critic1: new QNetwork<TState, Vector<T>, T>(
                      options.StateSize,
                      options.ActionSize,
                      options.CriticNetworkArchitecture,
                      options.CriticActivationFunction,
                      options.Seed),
                  critic1Target: new QNetwork<TState, Vector<T>, T>(
                      options.StateSize,
                      options.ActionSize,
                      options.CriticNetworkArchitecture,
                      options.CriticActivationFunction,
                      options.Seed),
                  critic2: new QNetwork<TState, Vector<T>, T>(
                      options.StateSize,
                      options.ActionSize,
                      options.CriticNetworkArchitecture,
                      options.CriticActivationFunction,
                      options.UseSeparateQNetworks ? (options.Seed.HasValue ? options.Seed.Value + 1 : null) : options.Seed),
                  critic2Target: new QNetwork<TState, Vector<T>, T>(
                      options.StateSize,
                      options.ActionSize,
                      options.CriticNetworkArchitecture,
                      options.CriticActivationFunction,
                      options.UseSeparateQNetworks ? (options.Seed.HasValue ? options.Seed.Value + 1 : null) : options.Seed),
                  // Replay buffer
                  replayBuffer: options.UsePrioritizedReplay
                      ? new PrioritizedReplayBuffer<TState, Vector<T>, T>(
                          options.ReplayBufferCapacity,
                          options.PrioritizedReplayAlpha,
                          options.PrioritizedReplayBetaInitial)
                      : new ReplayBufferBase<TState, Vector<T>, T>(options.ReplayBufferCapacity),
                  // Learning parameters
                  gamma: options.Gamma,
                  tau: options.Tau,
                  batchSize: options.BatchSize,
                  warmUpSteps: options.WarmUpSteps,
                  useGradientClipping: options.UseGradientClipping,
                  maxGradientNorm: options.MaxGradientNorm,
                  policyUpdateFrequency: 1, // SAC typically updates policy every step
                  useMinimumQValue: true, // SAC always uses minimum Q-value
                  entropyCoefficient: options.InitialEntropyCoefficient,
                  autoTuneEntropyCoefficient: options.AutoTuneEntropyCoefficient,
                  targetEntropy: options.TargetEntropy ?? -options.ActionSize, // Default target entropy is -dim(A)
                  entropyLearningRate: options.EntropyLearningRate,
                  clippedLogProbs: options.ClipLogProbs,
                  minLogProb: options.MinLogProb,
                  maxLogProb: options.MaxLogProb,
                  seed: options.Seed)
        {
            _options = options;
            _updateAfterEachStep = options.UpdateAfterEachStep;
            _trainingFrequency = options.TrainingFrequency;
            _gradientsStepsPerUpdate = options.GradientsStepsPerUpdate;
            _useFixedWarmUpExploration = options.UseFixedWarmUpExploration;
            _warmUpExplorationStdDev = NumOps.FromDouble(options.WarmUpExplorationStdDev);
            _logEntropyCoefficient = NumOps.Log(EntropyCoefficient);
            
            // Initialize target networks
            InitializeTargetNetworks();
        }
        
        /// <summary>
        /// Initializes the target networks with the same weights as the online networks.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This is an implementation of the abstract method from DualCriticAgentBase.
        /// </para>
        /// </remarks>
        protected override void InitializeTargetNetworks()
        {
            // Copy parameters from critics to target critics
            Critic1Target.CopyParametersFrom(Critic1);
            Critic2Target.CopyParametersFrom(Critic2);
        }
        
        /// <summary>
        /// Updates the target networks using the soft update approach.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This is an implementation of the abstract method from DualCriticAgentBase.
        /// The soft update blends the parameters of the target networks with the online networks:
        /// target_params = (1 - tau) * target_params + tau * online_params
        /// </para>
        /// </remarks>
        protected override void UpdateTargetNetworks()
        {
            // Soft update target networks using the Tau parameter
            Critic1Target.SoftUpdate(Critic1, Tau);
            Critic2Target.SoftUpdate(Critic2, Tau);
        }

        /// <summary>
        /// Selects an action based on the current state.
        /// </summary>
        /// <param name="state">The current state observation.</param>
        /// <param name="isTraining">A flag indicating whether the agent is in training mode.</param>
        /// <returns>The selected action.</returns>
        public override Vector<T> SelectAction(TState state, bool isTraining = true)
        {
            // During warmup, either use random actions or fixed exploration
            if (isTraining && TotalSteps < WarmUpSteps)
            {
                if (_useFixedWarmUpExploration)
                {
                    // Use the policy with high exploration
                    var action = Actor.SelectAction(state);
                    
                    // Add extra noise
                    for (int i = 0; i < action.Length; i++)
                    {
                        T noise = NumOps.Multiply(GenerateGaussianNoise(), _warmUpExplorationStdDev);
                        action[i] = MathHelper.Clamp(NumOps.Add(action[i], noise), NumOps.Negate(NumOps.One), NumOps.One);
                    }
                    
                    return action;
                }
                else
                {
                    // Use purely random actions during warm-up
                    return GenerateRandomAction();
                }
            }
            
            // During training, use stochastic actions from the policy
            if (isTraining)
            {
                return Actor.SelectAction(state);
            }
            else
            {
                // During evaluation, use deterministic actions (mean of the policy)
                return Actor.SelectDeterministicAction(state);
            }
        }

        /// <summary>
        /// Generates a random action vector for exploration.
        /// </summary>
        /// <returns>A random action vector.</returns>
        protected override Vector<T> GenerateRandomAction()
        {
            int actionDimension = _options.ActionSize;
            var action = Vector<T>.CreateDefault(actionDimension, NumOps.Zero);
            for (int i = 0; i < actionDimension; i++)
            {
                // Random values in [-1, 1]
                action[i] = NumOps.FromDouble(Random.NextDouble() * 2.0 - 1.0);
            }
            return action;
        }

        /// <summary>
        /// Updates the agent's knowledge based on an experience tuple.
        /// </summary>
        /// <param name="state">The state before the action was taken.</param>
        /// <param name="action">The action that was taken.</param>
        /// <param name="reward">The reward received after taking the action.</param>
        /// <param name="nextState">The state after the action was taken.</param>
        /// <param name="done">A flag indicating whether the episode ended after this action.</param>
        public override void Learn(TState state, Vector<T> action, T reward, TState nextState, bool done)
        {
            if (!IsTraining)
                return;

            // Increment step counter and store in replay buffer (handled by base class)
            base.Learn(state, action, reward, nextState, done);

            // Check training frequency
            if (!_updateAfterEachStep && TotalSteps % _trainingFrequency != 0)
                return;

            // Perform multiple gradient updates per environment step if specified
            for (int i = 0; i < _gradientsStepsPerUpdate; i++)
            {
                // Only start training after collecting enough samples
                if (TotalSteps < WarmUpSteps || ReplayBuffer.Size < BatchSize)
                    return;

                // Sample a batch of experiences from the replay buffer
                var batch = ReplayBuffer.SampleBatch(BatchSize);
                var states = batch.States;
                var actions = batch.Actions;
                var rewards = batch.Rewards;
                var nextStates = batch.NextStates;
                var dones = batch.Dones;

                // Update critic and actor networks
                // For now, use uniform weights if not using prioritized replay
                T[]? weights = null;
                int[]? indices = null;
                
                if (ReplayBuffer is IPrioritizedReplayBuffer<TState, Vector<T>, T> prioritizedBuffer)
                {
                    var prioritizedBatch = prioritizedBuffer.SampleBatch(BatchSize) as PrioritizedReplayBatch<TState, Vector<T>, T>;
                    if (prioritizedBatch != null)
                    {
                        weights = prioritizedBatch.Weights;
                        indices = prioritizedBatch.Indices;
                    }
                }
                
                UpdateNetworks(states, actions, rewards, nextStates, dones, 
                              weights ?? Array.Empty<T>(), 
                              indices ?? Array.Empty<int>());

                // Update target networks
                UpdateTargetNetworks();
            }
        }

        /// <summary>
        /// Updates the critic and actor networks based on a batch of experiences.
        /// </summary>
        /// <param name="states">Batch of states.</param>
        /// <param name="actions">Batch of actions.</param>
        /// <param name="rewards">Batch of rewards.</param>
        /// <param name="nextStates">Batch of next states.</param>
        /// <param name="dones">Batch of episode termination flags.</param>
        /// <param name="weights">Importance sampling weights (for prioritized replay).</param>
        /// <param name="indices">Indices of the sampled experiences (for prioritized replay).</param>
        protected override void UpdateNetworks(
            TState[] states,
            Vector<T>[] actions,
            T[] rewards,
            TState[] nextStates,
            bool[] dones,
            T[] weights,
            int[] indices)
        {
            // Compute target Q-values using next state actions from the current policy
            var targetQValues = new Vector<T>(states.Length);
            
            // Get next actions and their log probabilities from the policy
            var nextActions = new Vector<T>[nextStates.Length];
            var nextLogProbs = new T[nextStates.Length];
            
            for (int i = 0; i < nextStates.Length; i++)
            {
                // Sample next action from the policy
                nextActions[i] = Actor.SelectAction(nextStates[i]);
                
                // Calculate log probability of the action
                nextLogProbs[i] = Actor.LogProbability(nextStates[i], nextActions[i]);
            }
            
            // Get Q-values from both target critics for the next actions
            Vector<T> targetQ1 = Critic1Target.PredictQValues(nextStates, nextActions);
            Vector<T> targetQ2 = Critic2Target.PredictQValues(nextStates, nextActions);
            
            // Use minimum of twin Q-values to reduce overestimation bias
            for (int i = 0; i < states.Length; i++)
            {
                // If done, only consider immediate reward
                if (dones[i])
                {
                    targetQValues[i] = rewards[i];
                }
                else
                {
                    // Calculate minimum Q-value from the two critics
                    T minQValue = MathHelper.Min(targetQ1[i], targetQ2[i]);
                    
                    // SAC target includes the entropy term: r + γ(min(Q1', Q2')(s', a') - α*logπ(a'|s'))
                    targetQValues[i] = NumOps.Add(rewards[i], 
                        NumOps.Multiply(Gamma, 
                            NumOps.Subtract(minQValue, 
                                NumOps.Multiply(EntropyCoefficient, nextLogProbs[i]))));
                }
            }

            // Update both critics
            T critic1Loss = Critic1.Update(states, actions, targetQValues, weights);
            T critic2Loss = Critic2.Update(states, actions, targetQValues, weights);
            
            // Update priorities if using prioritized replay
            if (ReplayBuffer is PrioritizedReplayBuffer<TState, Vector<T>, T> prioritizedBuffer)
            {
                // Compute TD errors for priority updates (using average of both critics)
                Vector<T> currentQ1 = Critic1.PredictQValues(states, actions);
                Vector<T> currentQ2 = Critic2.PredictQValues(states, actions);
                var tdErrors = new Vector<T>(states.Length);
                
                for (int i = 0; i < states.Length; i++)
                {
                    T currentAvgQ = NumOps.Multiply(NumOps.Add(currentQ1[i], currentQ2[i]), NumOps.FromDouble(0.5));
                    tdErrors[i] = NumOps.Abs(NumOps.Subtract(targetQValues[i], currentAvgQ));
                }
                
                // Update priorities
                prioritizedBuffer.UpdatePriorities(indices, tdErrors);
            }

            // Update actor (policy)
            UpdateActorAndEntropyCoefficient(states);
        }

        /// <summary>
        /// Updates the actor (policy) network and entropy coefficient.
        /// </summary>
        /// <param name="states">Batch of states.</param>
        private void UpdateActorAndEntropyCoefficient(TState[] states)
        {
            T totalLoss = NumOps.Zero;
            T totalEntropy = NumOps.Zero;
            var policyGradients = new List<(TState state, Vector<T> actionMeanGradient, Vector<T> actionLogStdGradient)>();
            
            for (int i = 0; i < states.Length; i++)
            {
                // Sample action from the policy (reparameterization trick is used inside)
                Vector<T> action = Actor.SelectAction(states[i]);
                
                // Calculate log probability of the action
                T logProb = Actor.LogProbability(states[i], action);
                
                // Get the entropy for this state
                T entropy = Actor.GetEntropy(states[i]);
                totalEntropy = NumOps.Add(totalEntropy, entropy);
                
                // Get Q-values from both critics
                T q1 = Critic1.PredictQValue(states[i], action);
                T q2 = Critic2.PredictQValue(states[i], action);
                
                // Use the minimum Q-value (conservative estimation)
                T minQ = MathHelper.Min(q1, q2);
                
                // Calculate loss: policy tries to maximize Q - α*logπ
                // For gradient descent, we negate this to get a minimization problem
                T sampleLoss = NumOps.Subtract(NumOps.Multiply(EntropyCoefficient, logProb), minQ);
                totalLoss = NumOps.Add(totalLoss, sampleLoss);
                
                // Get policy gradients for this sample
                var (actionMeanGrad, actionLogStdGrad) = Actor.CalculatePolicyGradients(
                    states[i], 
                    action, 
                    minQ, 
                    EntropyCoefficient);
                    
                policyGradients.Add((states[i], actionMeanGrad, actionLogStdGrad));
            }
            
            // Apply the gradients to update the policy
            Actor.UpdateParameters(policyGradients, UseGradientClipping, MaxGradientNorm);
            
            // Update entropy coefficient if auto-tuning is enabled
            if (AutoTuneEntropyCoefficient)
            {
                // Calculate average entropy
                T averageEntropy = NumOps.Divide(totalEntropy, NumOps.FromDouble(states.Length));
                
                // Update entropy coefficient using the base class method
                UpdateEntropyCoefficient(averageEntropy);
            }
        }

        /// <summary>
        /// Updates the entropy coefficient (alpha) based on the policy entropy.
        /// </summary>
        /// <param name="averageEntropy">The average entropy across the batch.</param>
        protected override void UpdateEntropyCoefficient(T policyEntropy)
        {
            base.UpdateEntropyCoefficient(policyEntropy); // Use the base implementation
            
            // Additionally track the log entropy coefficient
            _logEntropyCoefficient = NumOps.Log(EntropyCoefficient);
        }

        /// <summary>
        /// Generates a random sample from a standard Gaussian distribution.
        /// </summary>
        /// <returns>A random sample from N(0, 1).</returns>
        private T GenerateGaussianNoise()
        {
            double u1 = 1.0 - Random.NextDouble(); // (0, 1] -> (0, 1]
            double u2 = 1.0 - Random.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); // Box-Muller transform
            return NumOps.FromDouble(z);
        }

        /// <summary>
        /// Saves the agent's state to a file.
        /// </summary>
        /// <param name="filePath">The path where the agent's state should be saved.</param>
        public override void Save(string filePath)
        {
            // TODO: Implement serialization
            base.Save(filePath);
        }

        /// <summary>
        /// Loads the agent's state from a file.
        /// </summary>
        /// <param name="filePath">The path from which to load the agent's state.</param>
        public override void Load(string filePath)
        {
            // TODO: Implement deserialization
            base.Load(filePath);
        }

        /// <summary>
        /// Gets the actor network.
        /// </summary>
        /// <returns>The actor network.</returns>
        public IStochasticPolicy<TState, Vector<T>, T> GetActor()
        {
            return Actor;
        }

        /// <summary>
        /// Gets the first critic network.
        /// </summary>
        /// <returns>The first critic network.</returns>
        public IActionValueFunction<TState, Vector<T>, T> GetCritic1()
        {
            return Critic1;
        }
        
        /// <summary>
        /// Gets the second critic network.
        /// </summary>
        /// <returns>The second critic network.</returns>
        public IActionValueFunction<TState, Vector<T>, T> GetCritic2()
        {
            return Critic2;
        }
        
        /// <summary>
        /// Gets the current entropy coefficient (alpha).
        /// </summary>
        /// <returns>The entropy coefficient.</returns>
        public T GetEntropyCoefficient()
        {
            return EntropyCoefficient;
        }

        /// <summary>
        /// Trains the agent on a batch of experiences.
        /// </summary>
        /// <param name="states">The states experienced.</param>
        /// <param name="actions">The actions taken in those states.</param>
        /// <param name="rewards">The rewards received.</param>
        /// <param name="nextStates">The next states after taking the actions.</param>
        /// <param name="dones">Whether each transition was terminal.</param>
        /// <returns>The average loss across the batch.</returns>
        public T Train(TState[] states, Vector<T>[] actions, T[] rewards, TState[] nextStates, bool[] dones)
        {
            // Add experiences to replay buffer
            for (int i = 0; i < states.Length; i++)
            {
                ReplayBuffer.Add(states[i], actions[i], rewards[i], nextStates[i], dones[i]);
            }

            // Only train if we have enough samples
            if (ReplayBuffer.Size < BatchSize)
            {
                return NumOps.Zero;
            }

            T totalLoss = NumOps.Zero;
            int updateCount = 0;

            // Perform training updates based on configuration
            if (_updateAfterEachStep)
            {
                // Update after each experience
                for (int i = 0; i < states.Length && i < _gradientsStepsPerUpdate; i++)
                {
                    if (TotalSteps % _trainingFrequency == 0)
                    {
                        var loss = PerformTrainingStep();
                        totalLoss = NumOps.Add(totalLoss, loss);
                        updateCount++;
                    }
                }
            }
            else
            {
                // Update once per batch
                if (TotalSteps % _trainingFrequency == 0)
                {
                    for (int i = 0; i < _gradientsStepsPerUpdate; i++)
                    {
                        var loss = PerformTrainingStep();
                        totalLoss = NumOps.Add(totalLoss, loss);
                        updateCount++;
                    }
                }
            }

            // Return average loss
            return updateCount > 0 ? NumOps.Divide(totalLoss, NumOps.FromDouble(updateCount)) : NumOps.Zero;
        }

        /// <summary>
        /// Performs a single training step by sampling from the replay buffer and updating networks.
        /// </summary>
        /// <returns>The loss value for this training step.</returns>
        private T PerformTrainingStep()
        {
            // Sample a batch from replay buffer
            var batch = ReplayBuffer.Sample(BatchSize);
            
            // Update critics and policy using the base class method
            UpdateNetworks(batch.Item1, batch.Item2, batch.Item3, batch.Item4, batch.Item5, 
                          Array.Empty<T>(), Array.Empty<int>());
            
            // Update target networks
            UpdateTargetNetworks();
            
            return LastLoss;
        }

        
        /// <summary>
        /// Q-network implementation for critics.
        /// </summary>
        /// <typeparam name="TStateType">The type used to represent the environment state.</typeparam>
        /// <typeparam name="TActionType">The type used to represent actions.</typeparam>
        /// <typeparam name="TNumeric">The numeric type used for calculations.</typeparam>
        private class QNetwork<TStateType, TActionType, TNumeric> : IActionValueFunction<TStateType, TActionType, TNumeric>
            where TStateType : Tensor<TNumeric>
            where TActionType : Vector<TNumeric>
        {
            /// <summary>
            /// Gets the numeric operations for type TNumeric.
            /// </summary>
            protected INumericOperations<TNumeric> NumOps => MathHelper.GetNumericOperations<TNumeric>();
            private readonly List<LayerBase<TNumeric>> _layers = default!;
            private readonly TNumeric _learningRate = default!;
            private readonly Random _random = default!;
            
            /// <summary>
            /// Gets the number of actions in the action space.
            /// </summary>
            public int ActionSize { get; }
            
            /// <summary>
            /// Gets a value indicating whether the action space is continuous.
            /// </summary>
            public bool IsContinuous => true;

            /// <summary>
            /// Initializes a new instance of the <see cref="QNetwork{TStateType, TActionType, TNumeric}"/> class.
            /// </summary>
            /// <param name="stateSize">The size of the state space.</param>
            /// <param name="actionSize">The size of the action space.</param>
            /// <param name="hiddenSizes">The sizes of hidden layers.</param>
            /// <param name="activation">The activation function to use.</param>
            /// <param name="seed">Optional random seed for reproducibility.</param>
            public QNetwork(
                int stateSize,
                int actionSize,
                int[] hiddenSizes,
                ActivationFunction activation,
                int? seed = null)
            {
                _layers = new List<LayerBase<TNumeric>>();
                _learningRate = NumOps.FromDouble(0.0003);
                _random = seed.HasValue ? new Random(seed.Value) : new Random();
                ActionSize = actionSize;
                
                // First layer (state input)
                int inputSize = stateSize + actionSize; // Concatenate state and action
                
                for (int i = 0; i < hiddenSizes.Length; i++)
                {
                    _layers.Add(new DenseLayer<TNumeric>(inputSize, hiddenSizes[i], ActivationFunctionFactory<TNumeric>.CreateActivationFunction(activation)));
                    inputSize = hiddenSizes[i];
                }
                
                // Output layer (single Q-value)
                _layers.Add(new DenseLayer<TNumeric>(inputSize, 1, ActivationFunctionFactory<TNumeric>.CreateActivationFunction(ActivationFunction.Identity)));
            }

            /// <summary>
            /// Predicts the Q-value for a given state-action pair.
            /// </summary>
            /// <param name="state">The state.</param>
            /// <param name="action">The action.</param>
            /// <returns>The predicted Q-value.</returns>
            public TNumeric PredictQValue(TStateType state, TActionType action)
            {
                // Forward pass
                Tensor<TNumeric> output = Forward(state, action);
                return output.ToVector()[0];
            }

            /// <summary>
            /// Predicts Q-values for a batch of state-action pairs.
            /// </summary>
            /// <param name="states">The batch of states.</param>
            /// <param name="actions">The batch of actions.</param>
            /// <returns>The predicted Q-values for each state-action pair.</returns>
            public Vector<TNumeric> PredictQValues(TStateType[] states, TActionType[] actions)
            {
                if (states.Length != actions.Length)
                {
                    throw new ArgumentException("Number of states and actions must match");
                }
                
                var qValues = new Vector<TNumeric>(states.Length);
                for (int i = 0; i < states.Length; i++)
                {
                    qValues[i] = PredictQValue(states[i], actions[i]);
                }
                
                return qValues;
            }

            /// <summary>
            /// Updates the Q-function based on target Q-values.
            /// </summary>
            /// <param name="states">The states.</param>
            /// <param name="actions">The actions taken in each state.</param>
            /// <param name="targets">The target Q-values for each state-action pair.</param>
            /// <param name="weights">Optional importance sampling weights for prioritized replay.</param>
            /// <returns>The loss value after the update.</returns>
            public TNumeric Update(TStateType[] states, TActionType[] actions, Vector<TNumeric> targets, TNumeric[]? weights = null)
            {
                if (states.Length != actions.Length || states.Length != targets.Length)
                {
                    throw new ArgumentException("Batch sizes must match");
                }
                
                // Use uniform weights if none provided
                if (weights == null)
                {
                    weights = new TNumeric[states.Length];
                    for (int i = 0; i < states.Length; i++)
                    {
                        weights[i] = NumOps.One;
                    }
                }
                
                TNumeric totalLoss = NumOps.Zero;
                
                // Process each sample
                for (int i = 0; i < states.Length; i++)
                {
                    // Forward pass
                    Tensor<TNumeric> input = ConcatenateTensors(states[i], Tensor<TNumeric>.FromVector(actions[i]));
                    
                    var layerInputs = new List<Tensor<TNumeric>> { input };
                    var layerOutputs = new List<Tensor<TNumeric>>();
                    
                    Tensor<TNumeric> output = input;
                    foreach (var layer in _layers)
                    {
                        output = layer.Forward(output);
                        layerOutputs.Add(output);
                    }
                    
                    // Compute loss
                    TNumeric prediction = output.ToVector()[0];
                    TNumeric target = targets[i];
                    TNumeric error = NumOps.Subtract(prediction, target);
                    TNumeric loss = NumOps.Multiply(NumOps.Multiply(error, error), weights[i]);  // Weighted MSE
                    totalLoss = NumOps.Add(totalLoss, loss);
                    
                    // Backward pass
                    Tensor<TNumeric> gradient = new Tensor<TNumeric>(new[] { 1 });
                    gradient[0] = NumOps.Multiply(NumOps.FromDouble(2.0), NumOps.Multiply(error, weights[i]));
                    
                    for (int j = _layers.Count - 1; j >= 0; j--)
                    {
                        if (_layers[j] is DenseLayer<TNumeric> denseLayer)
                        {
                            gradient = denseLayer.Backward(gradient);
                            denseLayer.UpdateParameters(_learningRate);
                        }
                    }
                }
                
                // Return average loss
                return NumOps.Divide(totalLoss, NumOps.FromDouble(states.Length));
            }

            /// <summary>
            /// Updates the Q-function based on target Q-values.
            /// </summary>
            /// <param name="states">The states.</param>
            /// <param name="actions">The actions taken in each state.</param>
            /// <param name="targets">The target Q-values for each state-action pair.</param>
            /// <returns>The loss value after the update.</returns>
            public TNumeric UpdateQ(TStateType[] states, TActionType[] actions, Vector<TNumeric> targets)
            {
                // Redirects to the main Update method
                return Update(states, actions, targets);
            }

            /// <summary>
            /// Computes the gradients of the Q-value with respect to the action.
            /// </summary>
            /// <param name="state">The state to evaluate.</param>
            /// <param name="action">The action to evaluate.</param>
            /// <returns>The gradients of the Q-value with respect to each action dimension.</returns>
            public Vector<TNumeric> ActionGradients(TStateType state, TActionType action)
            {
                // First, perform a forward pass
                Tensor<TNumeric> input = ConcatenateTensors(state, Tensor<TNumeric>.FromVector(action));
                
                var layerInputs = new List<Tensor<TNumeric>> { input };
                var layerOutputs = new List<Tensor<TNumeric>>();
                
                Tensor<TNumeric> output = input;
                foreach (var layer in _layers)
                {
                    output = layer.Forward(output);
                    layerOutputs.Add(output);
                }
                
                // Initialize gradient for backward pass
                Tensor<TNumeric> gradient = new Tensor<TNumeric>(new[] { 1 });
                gradient[0] = NumOps.One;  // Gradient of output w.r.t. itself is 1
                
                // Backward pass to compute gradients
                for (int i = _layers.Count - 1; i >= 0; i--)
                {
                    if (_layers[i] is DenseLayer<TNumeric> denseLayer)
                    {
                        gradient = denseLayer.Backward(gradient);
                    }
                }
                
                // Extract the gradients corresponding to the action part of the input
                // Calculate state size from the input tensor dimensions
                int totalSize = input.Length;
                int actionSize = action.Length;
                int stateSize = totalSize - actionSize;
                var actionGradients = new Vector<TNumeric>(actionSize);
                
                for (int i = 0; i < actionSize; i++)
                {
                    actionGradients[i] = gradient[stateSize + i];
                }
                
                return actionGradients;
            }

            /// <summary>
            /// Gets the parameters of the Q-network.
            /// </summary>
            /// <returns>The parameters as a flat vector.</returns>
            public Vector<TNumeric> GetParameters()
            {
                List<TNumeric> parameters = new List<TNumeric>();
                
                foreach (var layer in _layers)
                {
                    if (layer is DenseLayer<TNumeric> denseLayer)
                    {
                        // Add weights
                        var weights = denseLayer.GetWeights();
                        for (int i = 0; i < weights.Shape[0]; i++)
                        {
                            for (int j = 0; j < weights.Shape[1]; j++)
                            {
                                parameters.Add(weights[i, j]);
                            }
                        }
                        
                        // Add biases
                        var biases = denseLayer.GetBiases();
                        for (int i = 0; i < biases.Length; i++)
                        {
                            parameters.Add(biases[i]);
                        }
                    }
                }
                
                return new Vector<TNumeric>(parameters.ToArray());
            }

            /// <summary>
            /// Sets the parameters of the Q-network.
            /// </summary>
            /// <param name="parameters">The new parameter values.</param>
            public void SetParameters(Vector<TNumeric> parameters)
            {
                int index = 0;
                
                foreach (var layer in _layers)
                {
                    if (layer is DenseLayer<TNumeric> denseLayer)
                    {
                        // Get current shapes
                        var weights = denseLayer.GetWeights();
                        var biases = denseLayer.GetBiases();
                        
                        // Set weights
                        var newWeights = new Tensor<TNumeric>(weights.Shape);
                        for (int i = 0; i < weights.Shape[0]; i++)
                        {
                            for (int j = 0; j < weights.Shape[1]; j++)
                            {
                                newWeights[i, j] = parameters[index++];
                            }
                        }
                        denseLayer.SetWeights(newWeights);
                        
                        // Set biases
                        var newBiases = new Tensor<TNumeric>(new[] { biases.Length });
                        for (int i = 0; i < biases.Length; i++)
                        {
                            newBiases[i] = parameters[index++];
                        }
                        denseLayer.SetBiases(newBiases);
                    }
                }
            }

            /// <summary>
            /// Copies the parameters from another Q-network.
            /// </summary>
            /// <param name="source">The source Q-network.</param>
            public void CopyParametersFrom(IActionValueFunction<TStateType, TActionType, TNumeric> source)
            {
                var parameters = source.GetParameters();
                SetParameters(parameters);
            }

            /// <summary>
            /// Performs a soft update of parameters from another Q-network.
            /// </summary>
            /// <param name="source">The source Q-network.</param>
            /// <param name="tau">The soft update factor (between 0 and 1).</param>
            public void SoftUpdate(IActionValueFunction<TStateType, TActionType, TNumeric> source, TNumeric tau)
            {
                // Get parameters from both networks
                var targetParams = GetParameters();
                var sourceParams = source.GetParameters();
                
                // Ensure the parameter vectors have the same length
                if (targetParams.Length != sourceParams.Length)
                {
                    throw new InvalidOperationException("Parameter vectors must have the same length for soft update");
                }
                
                // Apply soft update: target_params = (1 - tau) * target_params + tau * source_params
                for (int i = 0; i < targetParams.Length; i++)
                {
                    targetParams[i] = NumOps.Add(
                        NumOps.Multiply(NumOps.Subtract(NumOps.One, tau), targetParams[i]),
                        NumOps.Multiply(tau, sourceParams[i]));
                }
                
                // Set the updated parameters
                SetParameters(targetParams);
            }

            /// <summary>
            /// Copies the parameters from another value function.
            /// </summary>
            /// <param name="source">The source value function.</param>
            public void CopyParametersFrom(IValueFunction<TStateType, TNumeric> source)
            {
                if (source is IActionValueFunction<TStateType, TActionType, TNumeric> actionValueFunction)
                {
                    CopyParametersFrom(actionValueFunction);
                }
                else
                {
                    throw new ArgumentException("Source must be an IActionValueFunction");
                }
            }

            /// <summary>
            /// Performs a soft update of parameters from another value function.
            /// </summary>
            /// <param name="source">The source value function.</param>
            /// <param name="tau">The soft update factor (between 0 and 1).</param>
            public void SoftUpdate(IValueFunction<TStateType, TNumeric> source, TNumeric tau)
            {
                if (source is IActionValueFunction<TStateType, TActionType, TNumeric> actionValueFunction)
                {
                    SoftUpdate(actionValueFunction, tau);
                }
                else
                {
                    throw new ArgumentException("Source must be an IActionValueFunction");
                }
            }

            /// <summary>
            /// Predicts the value for a given state.
            /// </summary>
            /// <param name="state">The state for which to predict the value.</param>
            /// <returns>The predicted value.</returns>
            public TNumeric PredictValue(TStateType state)
            {
                // SAC uses Q-functions rather than value functions
                // Without access to the policy, we cannot compute a meaningful value
                // This method is required by the interface but not used in SAC
                throw new NotSupportedException(
                    "SAC uses Q-functions rather than value functions. " +
                    "Use PredictQValue with a specific action instead.");
            }

            /// <summary>
            /// Predicts values for a batch of states.
            /// </summary>
            /// <param name="states">The batch of states.</param>
            /// <returns>The predicted values for each state.</returns>
            public Vector<TNumeric> PredictValues(TStateType[] states)
            {
                // SAC uses Q-functions rather than value functions
                // Without access to the policy, we cannot compute meaningful values
                // This method is required by the interface but not used in SAC
                throw new NotSupportedException(
                    "SAC uses Q-functions rather than value functions. " +
                    "Use PredictQValue with specific actions instead.");
            }

            /// <summary>
            /// Predicts Q-values for all possible actions in a given state.
            /// </summary>
            /// <param name="state">The state.</param>
            /// <returns>A vector of Q-values, one for each possible action.</returns>
            public Vector<TNumeric> PredictQValues(TStateType state)
            {
                // This method is designed for discrete action spaces where we can enumerate all actions
                // SAC works with continuous action spaces where this is not possible
                throw new NotSupportedException(
                    "PredictQValues is not applicable for continuous action spaces. " +
                    "Use PredictQValue(state, action) to evaluate specific actions.");
            }

            /// <summary>
            /// Predicts Q-values for all possible actions for a batch of states.
            /// </summary>
            /// <param name="states">The batch of states.</param>
            /// <returns>A matrix of Q-values, where each row corresponds to a state and each column to an action.</returns>
            public Matrix<TNumeric> PredictQValuesBatch(TStateType[] states)
            {
                // This method is designed for discrete action spaces where we can enumerate all actions
                // SAC works with continuous action spaces where this is not possible
                throw new NotSupportedException(
                    "PredictQValuesBatch is not applicable for continuous action spaces. " +
                    "Use PredictQValue(state, action) to evaluate specific actions.");
            }

            /// <summary>
            /// Gets the best action for a given state (the action with the highest Q-value).
            /// </summary>
            /// <param name="state">The state.</param>
            /// <returns>The best action.</returns>
            public TActionType GetBestAction(TStateType state)
            {
                // SAC uses a separate policy network that is not part of the Q-network
                // The Q-network evaluates state-action pairs, it doesn't select actions
                // For continuous action spaces, we cannot enumerate all possible actions
                // This method is required by the interface but not meaningful for SAC's Q-network
                
                // Return a zero action as a placeholder
                return (TActionType)(object)new Vector<TNumeric>(ActionSize);
            }

            /// <summary>
            /// Updates the value function based on target values.
            /// </summary>
            /// <param name="states">The states for which to update values.</param>
            /// <param name="targets">The target values for each state.</param>
            /// <returns>The loss value after the update.</returns>
            public TNumeric Update(TStateType[] states, Vector<TNumeric> targets)
            {
                // This method is for value functions, but SAC uses Q-functions
                // For compatibility, we need actions to update Q-values
                throw new NotSupportedException(
                    "SAC uses Q-functions which require state-action pairs. " +
                    "Use Update(states, actions, targets) method instead.");
            }

            /// <summary>
            /// Processes a state-action pair through the network.
            /// </summary>
            /// <param name="state">The state.</param>
            /// <param name="action">The action.</param>
            /// <returns>The output tensor.</returns>
            private Tensor<TNumeric> Forward(TStateType state, TActionType action)
            {
                // Concatenate state and action
                Tensor<TNumeric> input = ConcatenateTensors(state, Tensor<TNumeric>.FromVector(action));
                
                // Forward pass through all layers
                Tensor<TNumeric> output = input;
                foreach (var layer in _layers)
                {
                    output = layer.Forward(output);
                }
                
                return output;
            }

            /// <summary>
            /// Concatenates two tensors.
            /// </summary>
            /// <param name="a">The first tensor.</param>
            /// <param name="b">The second tensor.</param>
            /// <returns>The concatenated tensor.</returns>
            private Tensor<TNumeric> ConcatenateTensors(Tensor<TNumeric> a, Tensor<TNumeric> b)
            {
                // Flatten both tensors
                Vector<TNumeric> aVector = a.ToVector();
                Vector<TNumeric> bVector = b.ToVector();
                
                // Create a new vector with the combined size
                Vector<TNumeric> combined = new Vector<TNumeric>(aVector.Length + bVector.Length);
                
                // Copy values from both vectors
                for (int i = 0; i < aVector.Length; i++)
                {
                    combined[i] = aVector[i];
                }
                
                for (int i = 0; i < bVector.Length; i++)
                {
                    combined[aVector.Length + i] = bVector[i];
                }
                
                // Return as tensor
                return Tensor<TNumeric>.FromVector(combined);
            }
        }

        /// <summary>
        /// Implements a stochastic policy for continuous action spaces using neural networks.
        /// </summary>
        /// <typeparam name="TStateType">The type used to represent the environment state.</typeparam>
        /// <typeparam name="TNumeric">The numeric type used for calculations.</typeparam>
        public class StochasticPolicy<TStateType, TNumeric> : IStochasticPolicy<TStateType, Vector<TNumeric>, TNumeric>
            where TStateType : Tensor<TNumeric>
        {
            /// <summary>
            /// Gets the numeric operations for type TNumeric.
            /// </summary>
            protected INumericOperations<TNumeric> NumOps => MathHelper.GetNumericOperations<TNumeric>();
            private readonly List<LayerBase<TNumeric>> _commonLayers; // Shared layers
            private readonly List<LayerBase<TNumeric>> _meanLayers; // Layers for action means
            private readonly List<LayerBase<TNumeric>> _logStdLayers; // Layers for log standard deviations
            private readonly TanhActivation<TNumeric> _finalMeanActivation; // Tanh for bounding action means
            private readonly Random _random = default!;
            private readonly Vector<TNumeric>? _actionLowerBound;
            private readonly Vector<TNumeric>? _actionUpperBound;
            private readonly bool _learnStdDev;
            private readonly TNumeric _fixedLogStd = default!;
            private readonly TNumeric _minLogStd = default!;
            private readonly TNumeric _maxLogStd = default!;
            private readonly int _stateSize;
            private readonly int _actionSize;

            public bool IsStochastic => true;
            public bool IsContinuous => true;

            public StochasticPolicy(
                int stateSize,
                int actionSize,
                int[] hiddenSizes,
                ActivationFunction activation,
                Vector<TNumeric>? actionLowerBound,
                Vector<TNumeric>? actionUpperBound,
                bool learnStdDev,
                double initialStdDev,
                double minStdDev,
                double maxStdDev,
                int? seed = null)
            {
                _stateSize = stateSize;
                _actionSize = actionSize;
                _commonLayers = new List<LayerBase<TNumeric>>();
                _meanLayers = new List<LayerBase<TNumeric>>();
                _logStdLayers = new List<LayerBase<TNumeric>>();
                _finalMeanActivation = new TanhActivation<TNumeric>();
                _random = seed.HasValue ? new Random(seed.Value) : new Random();
                _actionLowerBound = actionLowerBound;
                _actionUpperBound = actionUpperBound;
                _learnStdDev = learnStdDev;
                _fixedLogStd = NumOps.FromDouble(Math.Log(initialStdDev));
                _minLogStd = NumOps.FromDouble(Math.Log(minStdDev));
                _maxLogStd = NumOps.FromDouble(Math.Log(maxStdDev));

                var activationFunc = ActivationFunctionFactory<TNumeric>.CreateActivationFunction(activation);

                // Build common layers
                int inputSize = stateSize;
                for (int i = 0; i < hiddenSizes.Length; i++)
                {
                    _commonLayers.Add(new DenseLayer<TNumeric>(inputSize, hiddenSizes[i], activationFunc));
                    inputSize = hiddenSizes[i];
                }

                // Mean output head
                _meanLayers.Add(new DenseLayer<TNumeric>(inputSize, actionSize, _finalMeanActivation as IActivationFunction<TNumeric>));

                // Log std output head (if learning std)
                if (_learnStdDev)
                {
                    _logStdLayers.Add(new DenseLayer<TNumeric>(inputSize, actionSize, new IdentityActivation<TNumeric>() as IActivationFunction<TNumeric>));
                }
            }

            public Vector<TNumeric> SelectAction(TStateType state)
            {
                return SelectAction(state, false);
            }

            public Vector<TNumeric> SelectAction(TStateType state, bool deterministic)
            {
                var (mean, logStd) = Forward(state);
                
                if (deterministic)
                {
                    return mean;
                }

                // Sample from Gaussian distribution
                var action = new Vector<TNumeric>(_actionSize);
                for (int i = 0; i < _actionSize; i++)
                {
                    TNumeric std = NumOps.Exp(logStd[i]);
                    TNumeric noise = NumOps.FromDouble(NormalDistribution());
                    action[i] = NumOps.Add(mean[i], NumOps.Multiply(std, noise));
                }

                // Clip actions if bounds are specified
                if (_actionLowerBound != null && _actionUpperBound != null)
                {
                    for (int i = 0; i < _actionSize; i++)
                    {
                        if (NumOps.LessThan(action[i], _actionLowerBound[i]))
                        {
                            action[i] = _actionLowerBound[i];
                        }
                        if (NumOps.GreaterThan(action[i], _actionUpperBound[i]))
                        {
                            action[i] = _actionUpperBound[i];
                        }
                    }
                }

                return action;
            }

            public (Vector<TNumeric> action, TNumeric logProb) SelectActionWithLogProb(TStateType state)
            {
                var action = SelectAction(state, false);
                var logProb = GetLogProbability(state, action);
                return (action, logProb);
            }

            public TNumeric GetLogProbability(TStateType state, Vector<TNumeric> action)
            {
                var (mean, logStd) = Forward(state);
                
                TNumeric logProb = NumOps.Zero;
                for (int i = 0; i < _actionSize; i++)
                {
                    TNumeric std = NumOps.Exp(logStd[i]);
                    TNumeric diff = NumOps.Subtract(action[i], mean[i]);
                    TNumeric normalized = NumOps.Divide(diff, std);
                    
                    // Log probability of Gaussian: -0.5 * (normalized^2 + log(2*pi) + 2*log(std))
                    TNumeric logPi = NumOps.FromDouble(Math.Log(2 * Math.PI));
                    TNumeric term1 = NumOps.Multiply(NumOps.FromDouble(-0.5), NumOps.Multiply(normalized, normalized));
                    TNumeric term2 = NumOps.Multiply(NumOps.FromDouble(-0.5), logPi);
                    TNumeric term3 = NumOps.Negate(logStd[i]);
                    
                    logProb = NumOps.Add(logProb, NumOps.Add(term1, NumOps.Add(term2, term3)));
                }

                return logProb;
            }

            public TNumeric GetEntropy(TStateType state)
            {
                var (_, logStd) = Forward(state);
                
                // Entropy of Gaussian: 0.5 * log(2*pi*e) + log(std)
                TNumeric logTwoPiE = NumOps.FromDouble(Math.Log(2 * Math.PI * Math.E));
                TNumeric entropy = NumOps.Zero;
                
                for (int i = 0; i < _actionSize; i++)
                {
                    entropy = NumOps.Add(entropy, NumOps.Add(NumOps.Multiply(NumOps.FromDouble(0.5), logTwoPiE), logStd[i]));
                }

                return entropy;
            }

            public Vector<TNumeric> SelectDeterministicAction(TStateType state)
            {
                var (mean, _) = Forward(state);
                return mean;
            }

            public (Vector<TNumeric> meanGradient, Vector<TNumeric> logStdGradient) CalculatePolicyGradients(
                TStateType state, Vector<TNumeric> action, TNumeric qValue, TNumeric entropyCoefficient)
            {
                // This is a simplified version - in practice, you'd need proper backpropagation
                var meanGradient = new Vector<TNumeric>(_actionSize);
                var logStdGradient = new Vector<TNumeric>(_actionSize);
                
                // Placeholder implementation
                for (int i = 0; i < _actionSize; i++)
                {
                    meanGradient[i] = NumOps.Zero;
                    logStdGradient[i] = NumOps.Zero;
                }
                
                return (meanGradient, logStdGradient);
            }

            public void UpdateParameters(List<(TStateType state, Vector<TNumeric> meanGradient, Vector<TNumeric> logStdGradient)> gradients,
                                       bool clampGradients, TNumeric maxGradientNorm)
            {
                // Simplified parameter update - in practice, you'd update network weights
            }

            public void CopyParametersFrom(IStochasticPolicy<TStateType, Vector<TNumeric>, TNumeric> other)
            {
                if (other is StochasticPolicy<TStateType, TNumeric> otherPolicy)
                {
                    // Copy parameters from layers - simplified implementation
                    // In practice, you'd need to implement proper parameter copying
                }
            }

            public void SoftUpdate(IStochasticPolicy<TStateType, Vector<TNumeric>, TNumeric> other, TNumeric tau)
            {
                // Implement soft update tau * self + (1 - tau) * other
                // Simplified implementation
            }

            public object EvaluatePolicy(TStateType state)
            {
                var (mean, logStd) = Forward(state);
                return (mean, logStd);
            }

            public TNumeric LogProbability(TStateType state, Vector<TNumeric> action)
            {
                return GetLogProbability(state, action);
            }

            public void UpdateParameters(object gradients, TNumeric learningRate)
            {
                // Simplified update
            }

            public void Save(string path)
            {
                // Save network parameters
            }

            public void Load(string path)
            {
                // Load network parameters
            }

            public void UpdateParameters(Vector<TNumeric> parameters)
            {
                // Update from parameter vector
            }

            public Vector<TNumeric> GetParameters()
            {
                var allParameters = new List<TNumeric>();
                
                // Flatten common layer parameters
                foreach (var layer in _commonLayers)
                {
                    if (layer is DenseLayer<TNumeric> denseLayer)
                    {
                        var paramVector = denseLayer.GetParameters();
                        for (int i = 0; i < paramVector.Length; i++)
                        {
                            allParameters.Add(paramVector[i]);
                        }
                    }
                }
                
                // Flatten mean layer parameters
                foreach (var layer in _meanLayers)
                {
                    if (layer is DenseLayer<TNumeric> denseLayer)
                    {
                        var paramVector = denseLayer.GetParameters();
                        for (int i = 0; i < paramVector.Length; i++)
                        {
                            allParameters.Add(paramVector[i]);
                        }
                    }
                }
                
                // Flatten log std layer parameters if learned
                if (_learnStdDev)
                {
                    foreach (var layer in _logStdLayers)
                    {
                        if (layer is DenseLayer<TNumeric> denseLayer)
                        {
                            var paramVector = denseLayer.GetParameters();
                            for (int i = 0; i < paramVector.Length; i++)
                            {
                                allParameters.Add(paramVector[i]);
                            }
                        }
                    }
                }
                
                return new Vector<TNumeric>([.. allParameters]);
            }

            public void CopyParametersFrom(IParameterizable<TNumeric, TStateType, Vector<TNumeric>> other)
            {
                if (other is IStochasticPolicy<TStateType, Vector<TNumeric>, TNumeric> stochasticOther)
                {
                    CopyParametersFrom(stochasticOther);
                }
            }

            private (Vector<TNumeric> mean, Vector<TNumeric> logStd) Forward(TStateType state)
            {
                // Forward through common layers
                Tensor<TNumeric> output = state;
                foreach (var layer in _commonLayers)
                {
                    output = layer.Forward(output);
                }

                // Forward through mean head
                Tensor<TNumeric> meanOutput = output;
                foreach (var layer in _meanLayers)
                {
                    meanOutput = layer.Forward(meanOutput);
                }
                var mean = meanOutput.ToVector();

                // Get log std
                Vector<TNumeric> logStd;
                if (_learnStdDev)
                {
                    Tensor<TNumeric> logStdOutput = output;
                    foreach (var layer in _logStdLayers)
                    {
                        logStdOutput = layer.Forward(logStdOutput);
                    }
                    logStd = logStdOutput.ToVector();
                    
                    // Clamp log std
                    for (int i = 0; i < _actionSize; i++)
                    {
                        if (NumOps.LessThan(logStd[i], _minLogStd))
                        {
                            logStd[i] = _minLogStd;
                        }
                        if (NumOps.GreaterThan(logStd[i], _maxLogStd))
                        {
                            logStd[i] = _maxLogStd;
                        }
                    }
                }
                else
                {
                    logStd = new Vector<TNumeric>(_actionSize);
                    for (int i = 0; i < _actionSize; i++)
                    {
                        logStd[i] = _fixedLogStd;
                    }
                }

                return (mean, logStd);
            }

            private double NormalDistribution()
            {
                // Box-Muller transform for generating normal distribution
                double u1 = 1.0 - _random.NextDouble();
                double u2 = 1.0 - _random.NextDouble();
                return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            }
            
            /// <summary>
            /// Sets the parameters of the stochastic policy from a flattened vector.
            /// </summary>
            /// <param name="parameters">The parameters to set.</param>
            public void SetParameters(Vector<TNumeric> parameters)
            {
                int offset = 0;
                
                // Set common layer parameters
                foreach (var layer in _commonLayers)
                {
                    if (layer is DenseLayer<TNumeric> denseLayer)
                    {
                        var layerParams = denseLayer.GetParameters();
                        var newParams = new Vector<TNumeric>(layerParams.Length);
                        for (int i = 0; i < layerParams.Length; i++)
                        {
                            newParams[i] = parameters[offset++];
                        }
                        denseLayer.SetParameters(newParams);
                    }
                }
                
                // Set mean layer parameters
                foreach (var layer in _meanLayers)
                {
                    if (layer is DenseLayer<TNumeric> denseLayer)
                    {
                        var layerParams = denseLayer.GetParameters();
                        var newParams = new Vector<TNumeric>(layerParams.Length);
                        for (int i = 0; i < layerParams.Length; i++)
                        {
                            newParams[i] = parameters[offset++];
                        }
                        denseLayer.SetParameters(newParams);
                    }
                }
                
                // Set log std layer parameters if learned
                if (_learnStdDev)
                {
                    foreach (var layer in _logStdLayers)
                    {
                        if (layer is DenseLayer<TNumeric> denseLayer)
                        {
                            var layerParams = denseLayer.GetParameters();
                            var newParams = new Vector<TNumeric>(layerParams.Length);
                            for (int i = 0; i < layerParams.Length; i++)
                            {
                                newParams[i] = parameters[offset++];
                            }
                            denseLayer.SetParameters(newParams);
                        }
                    }
                }
                
                // Check if we used all parameters
                if (offset != parameters.Length)
                {
                    throw new ArgumentException($"Parameter count mismatch. Expected {offset} parameters, got {parameters.Length}");
                }
            }
        }
    }
}