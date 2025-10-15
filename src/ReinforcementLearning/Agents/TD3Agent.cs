using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.ReinforcementLearning.Models.Options;
using AiDotNet.ReinforcementLearning.Exploration;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Interfaces;
using AiDotNet.Helpers;
using System;
using System.Collections.Generic;

namespace AiDotNet.ReinforcementLearning.Agents
{
    /// <summary>
    /// Implementation of the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm for continuous control.
    /// </summary>
    /// <typeparam name="TState">The type used to represent the environment state.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// <para>
    /// TD3 is an algorithm that addresses the function approximation error in DDPG through three key improvements:
    /// 1. Clipped Double Q-Learning: Maintaining two separate critics and using the minimum of their estimates
    /// 2. Delayed Policy Updates: Updating the policy less frequently than the critics
    /// 3. Target Policy Smoothing: Adding noise to the target actions to prevent exploitation of Q-function errors
    /// </para>
    /// <para>
    /// These improvements help to reduce overestimation bias and variance, making TD3 more stable than DDPG.
    /// </para>
    /// </remarks>
    public class TD3Agent<TState, T> : DualCriticAgentBase<TState, Vector<T>, T, IDeterministicPolicy<TState, Vector<T>, T>>
        where TState : Tensor<T>
    {
        private readonly TD3Options _options = default!;
        private readonly T _targetPolicyNoiseScale = default!;
        private readonly T _targetPolicyNoiseClip = default!;
        private readonly bool _useAverageCriticForActorUpdate;
        private readonly bool _useDelayedTargetUpdate;
        private readonly int _targetUpdateFrequency;
        private readonly bool _clippedCriticValueDuringActorUpdate;
        private readonly T _maxCriticValueDuringActorUpdate = default!;
        private readonly bool _useExtraWarmupNoise;
        private readonly T _extraWarmupNoiseScale = default!;
        
        private T _noiseScale = default!;
        private readonly T _noiseDecayRate = default!;
        private readonly T _minNoiseScale = default!;

        /// <summary>
        /// Initializes a new instance of the <see cref="TD3Agent{TState, T}"/> class.
        /// </summary>
        /// <param name="options">Options for the TD3 algorithm.</param>
        public TD3Agent(TD3Options options)
            : base(
                  // Actor networks
                  actor: new DDPGAgent<TState, T>.DeterministicPolicy<TState, T>(
                      options.StateSize,
                      options.ActionSize,
                      options.ActorNetworkArchitecture,
                      options.ActorActivationFunction,
                      options.ActorFinalActivationFunction,
                      options.UseLayerNormalization,
                      options.Seed),
                  actorTarget: new DDPGAgent<TState, T>.DeterministicPolicy<TState, T>(
                      options.StateSize,
                      options.ActionSize,
                      options.ActorNetworkArchitecture,
                      options.ActorActivationFunction,
                      options.ActorFinalActivationFunction,
                      options.UseLayerNormalization,
                      options.Seed),
                  // Critic networks (two for TD3)
                  critic1: new DDPGAgent<TState, T>.QNetwork<TState, Vector<T>, T>(
                      options.StateSize,
                      options.ActionSize,
                      options.CriticNetworkArchitecture,
                      options.CriticActivationFunction,
                      options.UseLayerNormalization,
                      options.Seed),
                  critic1Target: new DDPGAgent<TState, T>.QNetwork<TState, Vector<T>, T>(
                      options.StateSize,
                      options.ActionSize,
                      options.CriticNetworkArchitecture,
                      options.CriticActivationFunction,
                      options.UseLayerNormalization,
                      options.Seed),
                  critic2: new DDPGAgent<TState, T>.QNetwork<TState, Vector<T>, T>(
                      options.StateSize,
                      options.ActionSize,
                      options.CriticNetworkArchitecture,
                      options.CriticActivationFunction,
                      options.UseLayerNormalization,
                      options.UseDifferentCriticInitializations ? 
                          (options.Seed.HasValue ? options.Seed.Value + 1 : null) : 
                          options.Seed),
                  critic2Target: new DDPGAgent<TState, T>.QNetwork<TState, Vector<T>, T>(
                      options.StateSize,
                      options.ActionSize,
                      options.CriticNetworkArchitecture,
                      options.CriticActivationFunction,
                      options.UseLayerNormalization,
                      options.UseDifferentCriticInitializations ? 
                          (options.Seed.HasValue ? options.Seed.Value + 1 : null) : 
                          options.Seed),
                  // Replay buffer
                  replayBuffer: options.UsePrioritizedReplay
                      ? new PrioritizedReplayBuffer<TState, Vector<T>, T>(
                          options.ReplayBufferCapacity,
                          options.PrioritizedReplayAlpha,
                          options.PrioritizedReplayBetaInitial)
                      : new ReplayBufferBase<TState, Vector<T>, T>(options.ReplayBufferCapacity),
                  // Exploration strategy
                  explorationStrategy: options.UseGaussianNoise
                      ? new DDPGAgent<TState, T>.GaussianNoiseStrategy<Vector<T>, T>(
                          MathHelper.GetNumericOperations<T>().FromDouble(options.GaussianNoiseStdDev),
                          options.Seed)
                      : new OrnsteinUhlenbeckNoiseStrategy<T>(
                          options.ActionSize,
                          options.OUNoiseTheta,
                          options.OUNoiseSigma,
                          seed: options.Seed),
                  // Learning parameters
                  gamma: options.Gamma,
                  tau: options.Tau,
                  batchSize: options.BatchSize,
                  warmUpSteps: options.WarmUpSteps,
                  useGradientClipping: options.UseGradientClipping,
                  maxGradientNorm: options.MaxGradientNorm,
                  policyUpdateFrequency: options.PolicyUpdateFrequency,
                  useMinimumQValue: options.UseMinimumQValue,
                  seed: options.Seed)
        {
            _options = options;
            _targetPolicyNoiseScale = NumOps.FromDouble(options.TargetPolicyNoiseScale);
            _targetPolicyNoiseClip = NumOps.FromDouble(options.TargetPolicyNoiseClip);
            _useAverageCriticForActorUpdate = options.UseAverageCriticForActorUpdate;
            _useDelayedTargetUpdate = options.UseDelayedTargetUpdate;
            _targetUpdateFrequency = options.TargetUpdateFrequency;
            _clippedCriticValueDuringActorUpdate = options.ClipCriticValueDuringActorUpdate;
            _maxCriticValueDuringActorUpdate = NumOps.FromDouble(options.MaxCriticValueDuringActorUpdate);
            _useExtraWarmupNoise = options.UseExtraWarmupNoise;
            _extraWarmupNoiseScale = NumOps.FromDouble(options.ExtraWarmupNoiseScale);
            _noiseScale = NumOps.One;
            _noiseDecayRate = NumOps.FromDouble(options.NoiseDecayRate);
            _minNoiseScale = NumOps.FromDouble(options.MinNoiseScale);
        }

        /// <summary>
        /// Selects an action based on the current state.
        /// </summary>
        /// <param name="state">The current state observation.</param>
        /// <param name="isTraining">A flag indicating whether the agent is in training mode.</param>
        /// <returns>The selected action.</returns>
        public override Vector<T> SelectAction(TState state, bool isTraining = true)
        {
            // Get action from the actor
            Vector<T> action = Actor.SelectAction(state);

            // Add exploration noise during training
            if (isTraining)
            {
                if (TotalSteps < WarmUpSteps)
                {
                    // During warm-up phase, use random actions or extra noise for better exploration
                    if (_useExtraWarmupNoise)
                    {
                        // Add extra noise during warm-up
                        T extraScale = NumOps.Multiply(_noiseScale, _extraWarmupNoiseScale);
                        action = ExplorationStrategy.ApplyExploration(action, TotalSteps);
                    }
                    else
                    {
                        // Use purely random actions during warm-up
                        action = GenerateRandomAction();
                    }
                }
                else
                {
                    // Apply standard exploration noise
                    action = ExplorationStrategy.ApplyExploration(action, TotalSteps);
                }
                
                // Ensure actions are clipped to valid range (assuming [-1, 1])
                for (int i = 0; i < action.Length; i++)
                {
                    action[i] = MathHelper.Clamp(action[i], NumOps.Negate(NumOps.One), NumOps.One);
                }
            }

            return action;
        }

        /// <summary>
        /// Generates a random action vector for exploration.
        /// </summary>
        /// <returns>A random action vector.</returns>
        protected override Vector<T> GenerateRandomAction()
        {
            int actionDimension = _options.ActionSize;
            var action = new Vector<T>(actionDimension);
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

            // Decay noise scale
            _noiseScale = MathHelper.Max(_minNoiseScale, NumOps.Multiply(_noiseScale, _noiseDecayRate));
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
            // Update both critics
            T critic1Loss = UpdateCritic(Critic1, Critic1Target, Critic2Target, states, actions, rewards, nextStates, dones, weights, indices);
            T critic2Loss = UpdateCritic(Critic2, Critic2Target, Critic1Target, states, actions, rewards, nextStates, dones, weights, indices);

            // Delayed policy updates (TD3's key feature)
            if (ShouldUpdatePolicy())
            {
                // Only update the actor every policy_update_frequency steps
                T actorLoss = UpdateActor(states);
            }
        }

        /// <summary>
        /// Updates a critic network.
        /// </summary>
        /// <param name="critic">The critic network to update.</param>
        /// <param name="criticTarget">The target network for the critic being updated.</param>
        /// <param name="otherCriticTarget">The target network for the other critic.</param>
        /// <param name="states">Batch of states.</param>
        /// <param name="actions">Batch of actions.</param>
        /// <param name="rewards">Batch of rewards.</param>
        /// <param name="nextStates">Batch of next states.</param>
        /// <param name="dones">Batch of episode termination flags.</param>
        /// <param name="weights">Importance sampling weights (for prioritized replay).</param>
        /// <param name="indices">Indices of the sampled experiences (for prioritized replay).</param>
        /// <returns>The critic loss.</returns>
        private T UpdateCritic(
            IActionValueFunction<TState, Vector<T>, T> critic,
            IActionValueFunction<TState, Vector<T>, T> criticTarget,
            IActionValueFunction<TState, Vector<T>, T> otherCriticTarget,
            TState[] states,
            Vector<T>[] actions,
            T[] rewards,
            TState[] nextStates,
            bool[] dones,
            T[] weights,
            int[] indices)
        {
            // Compute target Q-values
            var targetQValues = new Vector<T>(states.Length);
            
            // Generate smooth target actions with noise for TD3's target policy smoothing
            var nextActionsWithNoise = new Vector<T>[nextStates.Length];
            
            for (int i = 0; i < nextStates.Length; i++)
            {
                // Get next action from target actor
                Vector<T> nextAction = ActorTarget.SelectAction(nextStates[i]);
                
                // Add clipped noise to target actions (target policy smoothing)
                nextActionsWithNoise[i] = new Vector<T>(nextAction.Length);
                for (int j = 0; j < nextAction.Length; j++)
                {
                    // Generate Gaussian noise
                    T noise = NumOps.Multiply(GenerateGaussianNoise(), _targetPolicyNoiseScale);
                    
                    // Clip noise
                    noise = MathHelper.Clamp(noise, NumOps.Negate(_targetPolicyNoiseClip), _targetPolicyNoiseClip);
                    
                    // Add noise to action
                    nextActionsWithNoise[i][j] = MathHelper.Clamp(
                        NumOps.Add(nextAction[j], noise), 
                        NumOps.Negate(NumOps.One), 
                        NumOps.One);
                }
            }
            
            // Get Q-values from both target critics for the noisy actions
            Vector<T> targetQ1 = criticTarget.PredictQValues(nextStates, nextActionsWithNoise);
            Vector<T> targetQ2 = otherCriticTarget.PredictQValues(nextStates, nextActionsWithNoise);
            
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
                    // TD3's clipped double Q-learning
                    T nextQValue = UseMinimumQValue ?
                        MathHelper.Min(targetQ1[i], targetQ2[i]) :
                        targetQ1[i]; // Use just the current critic's target if not using min Q
                    
                    // Q-target = r + gamma * min(Q1', Q2')(s', a' + noise)
                    targetQValues[i] = NumOps.Add(rewards[i], NumOps.Multiply(Gamma, nextQValue));
                }
            }

            // Update critic
            T criticLoss = critic.Update(states, actions, targetQValues, weights);
            
            // Update priorities if using prioritized replay
            if (ReplayBuffer is PrioritizedReplayBuffer<TState, Vector<T>, T> prioritizedBuffer)
            {
                // Compute TD errors for priority updates
                Vector<T> currentQValues = critic.PredictQValues(states, actions);
                var tdErrors = new Vector<T>(states.Length);
                
                for (int i = 0; i < states.Length; i++)
                {
                    tdErrors[i] = NumOps.Abs(NumOps.Subtract(targetQValues[i], currentQValues[i]));
                }
                
                // Update priorities
                prioritizedBuffer.UpdatePriorities(indices, tdErrors);
            }
            
            return criticLoss;
        }

        /// <summary>
        /// Updates the actor network.
        /// </summary>
        /// <param name="states">Batch of states.</param>
        /// <returns>The actor loss.</returns>
        private T UpdateActor(TState[] states)
        {
            // Calculate policy gradient
            var policyGradients = new List<(TState state, Vector<T> actionGradient)>();
            T totalLoss = NumOps.Zero;
            
            for (int i = 0; i < states.Length; i++)
            {
                // Get action from current policy
                Vector<T> action = Actor.SelectAction(states[i]);
                
                // Use first critic by default
                Vector<T> actionGradient = Critic1.ActionGradients(states[i], action);
                T qValue = Critic1.PredictQValue(states[i], action);
                
                // If using average of critics for actor update
                if (_useAverageCriticForActorUpdate)
                {
                    Vector<T> actionGradient2 = Critic2.ActionGradients(states[i], action);
                    T qValue2 = Critic2.PredictQValue(states[i], action);
                    
                    // Average gradients and Q-values
                    for (int j = 0; j < actionGradient.Length; j++)
                    {
                        actionGradient[j] = NumOps.Multiply(
                            NumOps.Add(actionGradient[j], actionGradient2[j]), 
                            NumOps.FromDouble(0.5));
                    }
                    qValue = NumOps.Multiply(NumOps.Add(qValue, qValue2), NumOps.FromDouble(0.5));
                }
                
                // Clip critic value if enabled
                if (_clippedCriticValueDuringActorUpdate)
                {
                    qValue = MathHelper.Min(qValue, _maxCriticValueDuringActorUpdate);
                }
                
                // Negate gradient since we want to maximize Q (minimize -Q)
                for (int j = 0; j < actionGradient.Length; j++)
                {
                    actionGradient[j] = NumOps.Negate(actionGradient[j]);
                }
                
                // Store gradients for actor update
                policyGradients.Add((states[i], actionGradient));
                
                // Compute loss (for monitoring)
                totalLoss = NumOps.Add(totalLoss, NumOps.Negate(qValue));
            }
            
            // Update actor with the policy gradients
            Actor.UpdateFromPolicyGradients(policyGradients, UseGradientClipping, MaxGradientNorm);
            
            // Return average loss
            return NumOps.Divide(totalLoss, NumOps.FromDouble(states.Length));
        }

        /// <summary>
        /// Updates target networks using soft update.
        /// </summary>
        protected override void UpdateTargetNetworks()
        {
            // Soft update: θ' = (1 - τ) * θ' + τ * θ
            ActorTarget.SoftUpdate(Actor, Tau);
            Critic1Target.SoftUpdate(Critic1, Tau);
            Critic2Target.SoftUpdate(Critic2, Tau);
        }
        
        /// <summary>
        /// Initializes the target networks by copying parameters from the main networks.
        /// </summary>
        protected override void InitializeTargetNetworks()
        {
            ActorTarget.CopyParametersFrom(Actor);
            Critic1Target.CopyParametersFrom(Critic1);
            Critic2Target.CopyParametersFrom(Critic2);
        }
        
        /// <summary>
        /// Determines whether the policy should be updated in the current step.
        /// </summary>
        /// <returns>True if the policy should be updated, otherwise false.</returns>
        protected override bool ShouldUpdatePolicy()
        {
            return TotalSteps % _options.PolicyUpdateFrequency == 0;
        }

        /// <summary>
        /// Determines whether target networks should be updated in the current step.
        /// </summary>
        /// <returns>True if target networks should be updated, otherwise false.</returns>
        protected override bool ShouldUpdateTargets()
        {
            if (_useDelayedTargetUpdate)
            {
                return TotalSteps % _targetUpdateFrequency == 0 && ShouldUpdatePolicy();
            }
            
            return ShouldUpdatePolicy();
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

            // Sample a batch from replay buffer
            var batch = ReplayBuffer.Sample(BatchSize);
            
            // Update critics using the base class method
            // Note: TD3 uses delayed policy updates, so we pass empty arrays for importance weights and indices
            UpdateNetworks(batch.Item1, batch.Item2, batch.Item3, batch.Item4, batch.Item5, 
                          Array.Empty<T>(), Array.Empty<int>());
            var criticLoss = LastLoss;
            totalLoss = NumOps.Add(totalLoss, criticLoss);
            updateCount++;
            
            // Update policy and target networks with delayed frequency
            if (TotalSteps % _options.PolicyUpdateFrequency == 0)
            {
                // Policy update is handled in UpdateNetworks
                // Update target networks
                UpdateTargetNetworks();
            }
            
            IncrementStepCounter();

            // Decay noise scale
            _noiseScale = MathHelper.Max(_minNoiseScale, NumOps.Multiply(_noiseScale, _noiseDecayRate));

            // Return average loss
            return updateCount > 0 ? NumOps.Divide(totalLoss, NumOps.FromDouble(updateCount)) : NumOps.Zero;
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
        public IDeterministicPolicy<TState, Vector<T>, T> GetActor()
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
    }
}