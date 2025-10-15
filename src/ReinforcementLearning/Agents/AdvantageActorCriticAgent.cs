using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Interfaces;
using AiDotNet.ReinforcementLearning.Interfaces;
using AiDotNet.ReinforcementLearning.Models.Options;
using AiDotNet.ReinforcementLearning.Policies;
using AiDotNet.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;

namespace AiDotNet.ReinforcementLearning.Agents
{
    /// <summary>
    /// Implements the Advantage Actor-Critic (A2C) algorithm for reinforcement learning.
    /// </summary>
    /// <typeparam name="TState">The type used to represent the environment state.</typeparam>
    /// <typeparam name="TAction">The type used to represent actions.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// <para>
    /// The Advantage Actor-Critic algorithm combines policy gradient methods with value function approximation.
    /// It uses an advantage function (difference between the actual returns and the state value) to reduce
    /// the variance of policy gradient estimates while maintaining an acceptable level of bias.
    /// </para>
    /// <para>
    /// This implementation supports both discrete and continuous action spaces, n-step returns,
    /// and Generalized Advantage Estimation (GAE).
    /// </para>
    /// </remarks>
    public class AdvantageActorCriticAgent<TState, TAction, T> : AgentBase<TState, TAction, T>
        where TState : Tensor<T>
    {
        private readonly IPolicy<TState, TAction, T> _actor = default!;
        private readonly IValueFunction<TState, T> _critic = default!;
        private readonly IValueFunction<TState, T>? _criticTarget;
        private readonly T _actorLearningRate = default!;
        private readonly T _criticLearningRate = default!;
        private readonly T _entropyCoefficient = default!;
        private readonly T _valueLossCoefficient = default!;
        private readonly bool _useGAE;
        private readonly T _gaeParameter = default!;
        private readonly bool _normalizeAdvantages;
        private readonly bool _standardizeRewards;
        private readonly bool _useCriticTargetNetwork;
        private readonly int _stepsPerUpdate;
        private readonly bool _useNStepReturns;
        private readonly int _nSteps;
        private readonly T _maxGradientNorm = default!;

        // Buffer to store experiences until it's time to update
        private readonly List<(TState state, TAction action, T reward, TState nextState, bool done)> _experienceBuffer;
        private int _stepsCollected;

        /// <summary>
        /// Initializes a new instance of the <see cref="AdvantageActorCriticAgent{TState, TAction, T}"/> class.
        /// </summary>
        /// <param name="options">The options for the Actor-Critic algorithm.</param>
        public AdvantageActorCriticAgent(ActorCriticOptions<T> options)
            : base(options.Gamma, options.CriticTargetUpdateTau, options.BatchSize, options.Seed)
        {
            // Copy basic options
            _actorLearningRate = NumOps.FromDouble(options.ActorLearningRate);
            _criticLearningRate = NumOps.FromDouble(options.CriticLearningRate);
            _entropyCoefficient = NumOps.FromDouble(options.EntropyCoefficient);
            _valueLossCoefficient = NumOps.FromDouble(options.ValueLossCoefficient);
            _useGAE = options.UseGAE;
            _gaeParameter = NumOps.FromDouble(options.GAELambda);
            _normalizeAdvantages = options.NormalizeAdvantages;
            _standardizeRewards = options.StandardizeRewards;
            _useCriticTargetNetwork = options.UseCriticTargetNetwork;
            _stepsPerUpdate = options.StepsPerUpdate;
            _useNStepReturns = options.UseNStepReturns;
            _nSteps = options.NSteps;
            _maxGradientNorm = NumOps.FromDouble(options.MaxGradientNorm);
            _experienceBuffer = new List<(TState, TAction, T, TState, bool)>();
            _stepsCollected = 0;

            // Create actor (policy) based on action space type
            if (!options.IsContinuous)
            {
                // Discrete action space
                if (typeof(TAction) != typeof(int))
                {
                    throw new ArgumentException("For discrete action spaces, TAction must be int");
                }

                _actor = (IPolicy<TState, TAction, T>)new DiscreteStochasticPolicy<T>(
                    options.StateSize,
                    options.ActionSize,
                    options.ActorNetworkArchitecture,
                    options.ActorActivationFunction,
                    options.Seed);
            }
            else
            {
                // Continuous action space
                if (typeof(TAction) != typeof(Vector<T>))
                {
                    throw new ArgumentException("For continuous action spaces, TAction must be Vector<T>");
                }

                _actor = (IPolicy<TState, TAction, T>)new ContinuousStochasticPolicy<T>(
                    options.StateSize,
                    options.ActionSize,
                    options.ActorNetworkArchitecture,
                    options.ActorActivationFunction,
                    null, // Default action bounds [-1, 1]
                    null,
                    options.LearnPolicyStdDev,
                    options.InitialPolicyStdDev,
                    0.01, // Min std dev
                    2.0,  // Max std dev
                    options.Seed);
            }

            // Create critic (value function)
            _critic = new ValueNetwork(
                options.StateSize,
                options.CriticNetworkArchitecture,
                options.CriticActivationFunction,
                options.Seed);

            // Create target critic if needed
            if (_useCriticTargetNetwork)
            {
                _criticTarget = new ValueNetwork(
                    options.StateSize,
                    options.CriticNetworkArchitecture,
                    options.CriticActivationFunction,
                    options.Seed);
                _criticTarget.CopyParametersFrom(_critic);
            }
            else
            {
                _criticTarget = null;
            }
        }

        /// <summary>
        /// Selects an action based on the current state.
        /// </summary>
        /// <param name="state">The current state observation.</param>
        /// <param name="isTraining">A flag indicating whether the agent is in training mode.</param>
        /// <returns>The selected action.</returns>
        public override TAction SelectAction(TState state, bool isTraining = true)
        {
            // Always use the policy to select actions
            // During training, this will incorporate exploration
            return _actor.SelectAction(state);
        }

        /// <summary>
        /// Updates the agent's knowledge based on an experience tuple.
        /// </summary>
        /// <param name="state">The state before the action was taken.</param>
        /// <param name="action">The action that was taken.</param>
        /// <param name="reward">The reward received after taking the action.</param>
        /// <param name="nextState">The state after the action was taken.</param>
        /// <param name="done">A flag indicating whether the episode ended after this action.</param>
        public override void Learn(TState state, TAction action, T reward, TState nextState, bool done)
        {
            if (!IsTraining)
                return;

            // Store the experience
            _experienceBuffer.Add((state, action, reward, nextState, done));
            _stepsCollected++;
            IncrementStepCounter();

            // Check if it's time to update
            if (_stepsCollected >= _stepsPerUpdate || done)
            {
                UpdateNetworks();
                _stepsCollected = 0;
            }
        }

        /// <summary>
        /// Updates the actor and critic networks based on collected experiences.
        /// </summary>
        private void UpdateNetworks()
        {
            if (_experienceBuffer.Count == 0)
                return;

            // Calculate advantages and returns
            (Vector<T> advantages, Vector<T> returns) = CalculateAdvantagesAndReturns();

            // Update the critic network using returns as targets
            var states = new TState[_experienceBuffer.Count];
            for (int i = 0; i < _experienceBuffer.Count; i++)
            {
                states[i] = _experienceBuffer[i].state;
            }
            T criticLoss = _critic.Update(states, returns);

            // Update the actor network using advantages
            UpdateActor(advantages);

            // Update target network if used
            if (_useCriticTargetNetwork && _criticTarget != null)
            {
                _criticTarget.SoftUpdate(_critic, Tau);
            }

            // Store the loss 
            LastLoss = criticLoss;

            // Clear the experience buffer
            _experienceBuffer.Clear();
        }

        /// <summary>
        /// Calculates advantages and returns for the collected experiences.
        /// </summary>
        /// <returns>A tuple containing the advantages and returns vectors.</returns>
        private (Vector<T> advantages, Vector<T> returns) CalculateAdvantagesAndReturns()
        {
            var advantages = new Vector<T>(_experienceBuffer.Count);
            var returns = new Vector<T>(_experienceBuffer.Count);

            if (_useGAE)
            {
                // Generalized Advantage Estimation
                returns = CalculateReturns();
                advantages = CalculateGAEAdvantages();
            }
            else if (_useNStepReturns)
            {
                // N-step returns
                returns = CalculateNStepReturns();
                
                // Calculate advantages as returns - baseline
                for (int i = 0; i < _experienceBuffer.Count; i++)
                {
                    T baselineValue = _critic.PredictValue(_experienceBuffer[i].state);
                    advantages[i] = NumOps.Subtract(returns[i], baselineValue);
                }
            }
            else
            {
                // Simple one-step TD error as advantage
                for (int i = 0; i < _experienceBuffer.Count; i++)
                {
                    var (state, _, reward, nextState, done) = _experienceBuffer[i];
                    
                    // Calculate the current state value
                    T currentValue = _critic.PredictValue(state);
                    
                    // Calculate the next state value (or 0 if terminal)
                    T nextValue = done ? NumOps.Zero : (_useCriticTargetNetwork && _criticTarget != null) 
                        ? _criticTarget.PredictValue(nextState) 
                        : _critic.PredictValue(nextState);
                    
                    // Calculate the TD error (advantage)
                    advantages[i] = NumOps.Subtract(NumOps.Add(reward, NumOps.Multiply(Gamma, nextValue)), currentValue);
                    
                    // Return is the TD target
                    returns[i] = NumOps.Add(reward, NumOps.Multiply(Gamma, nextValue));
                }
            }

            // Normalize advantages if needed
            if (_normalizeAdvantages)
            {
                NormalizeVector(advantages);
            }

            // Standardize returns if needed
            if (_standardizeRewards)
            {
                NormalizeVector(returns);
            }

            return (advantages, returns);
        }

        /// <summary>
        /// Calculates n-step returns for the collected experiences.
        /// </summary>
        /// <returns>A vector of n-step returns.</returns>
        private Vector<T> CalculateNStepReturns()
        {
            var returns = new Vector<T>(_experienceBuffer.Count);

            for (int i = 0; i < _experienceBuffer.Count; i++)
            {
                // Start with the immediate reward
                T nStepReturn = _experienceBuffer[i].reward;
                
                // Add discounted future rewards up to n steps
                T discountFactor = Gamma;
                for (int j = 1; j < _nSteps && (i + j) < _experienceBuffer.Count; j++)
                {
                    nStepReturn = NumOps.Add(nStepReturn, NumOps.Multiply(discountFactor, _experienceBuffer[i + j].reward));
                    discountFactor = NumOps.Multiply(discountFactor, Gamma);

                    // If we reach a terminal state, stop adding future rewards
                    if (_experienceBuffer[i + j].done)
                        break;
                }

                // Add the bootstrapped value if we haven't reached a terminal state or the end of buffer
                if ((i + _nSteps) < _experienceBuffer.Count && !_experienceBuffer[i + _nSteps - 1].done)
                {
                    TState nStepState = _experienceBuffer[i + _nSteps].state;
                    T nStepValue = (_useCriticTargetNetwork && _criticTarget != null)
                        ? _criticTarget.PredictValue(nStepState)
                        : _critic.PredictValue(nStepState);
                    nStepReturn = NumOps.Add(nStepReturn, NumOps.Multiply(discountFactor, nStepValue));
                }

                returns[i] = nStepReturn;
            }

            return returns;
        }

        /// <summary>
        /// Calculates advantages using Generalized Advantage Estimation (GAE).
        /// </summary>
        /// <returns>A vector of GAE advantages.</returns>
        private Vector<T> CalculateGAEAdvantages()
        {
            var advantages = new Vector<T>(_experienceBuffer.Count);
            
            // Initialize the next state value and advantage for bootstrapping
            T nextValue = NumOps.Zero;
            T nextAdvantage = NumOps.Zero;
            
            // Calculate GAE in reverse order
            for (int i = _experienceBuffer.Count - 1; i >= 0; i--)
            {
                var (state, _, reward, nextState, done) = _experienceBuffer[i];
                
                // Calculate the current state value
                T currentValue = _critic.PredictValue(state);
                
                // Calculate the next state value (or 0 if terminal)
                if (!done)
                {
                    nextValue = (_useCriticTargetNetwork && _criticTarget != null)
                        ? _criticTarget.PredictValue(nextState)
                        : _critic.PredictValue(nextState);
                }
                else
                {
                    nextValue = NumOps.Zero;
                }
                
                // Calculate the TD error
                T tdError = NumOps.Subtract(NumOps.Add(reward, NumOps.Multiply(Gamma, nextValue)), currentValue);
                
                // Calculate the GAE advantage
                // A_t = δ_t + (γλ)A_{t+1}
                T gaePart = NumOps.Multiply(NumOps.Multiply(Gamma, _gaeParameter), done ? NumOps.Zero : nextAdvantage);
                advantages[i] = NumOps.Add(tdError, gaePart);
                
                // Update next advantage for the next iteration
                nextAdvantage = advantages[i];
            }
            
            return advantages;
        }

        /// <summary>
        /// Calculates Monte Carlo returns for the collected experiences.
        /// </summary>
        /// <returns>A vector of Monte Carlo returns.</returns>
        private Vector<T> CalculateReturns()
        {
            var returns = new Vector<T>(_experienceBuffer.Count);
            T discountedReturn = NumOps.Zero;

            // Calculate returns in reverse order
            for (int i = _experienceBuffer.Count - 1; i >= 0; i--)
            {
                var (_, _, reward, _, done) = _experienceBuffer[i];
                
                // Reset the return calculation if this is the start of a new episode
                if (done && i < _experienceBuffer.Count - 1)
                {
                    discountedReturn = NumOps.Zero;
                }
                
                // G_t = r_t + gamma * G_{t+1}
                discountedReturn = NumOps.Add(reward, NumOps.Multiply(Gamma, discountedReturn));
                returns[i] = discountedReturn;
            }

            return returns;
        }

        /// <summary>
        /// Updates the actor (policy) network using the calculated advantages.
        /// </summary>
        /// <param name="advantages">The advantages used for policy gradient estimation.</param>
        private void UpdateActor(Vector<T> advantages)
        {
            List<object> gradients = new List<object>();

            // Process each experience in the buffer
            for (int i = 0; i < _experienceBuffer.Count; i++)
            {
                var (state, action, _, _, _) = _experienceBuffer[i];
                T advantage = advantages[i];

                // Calculate log probability of the action
                T logProb = _actor.LogProbability(state, action);

                // Calculate policy gradient
                T gradientScale = advantage;

                // Add entropy term if enabled
                if (NumOps.GreaterThan(_entropyCoefficient, NumOps.Zero))
                {
                    T entropy = _actor.GetEntropy(state);
                    gradientScale = NumOps.Add(gradientScale, NumOps.Multiply(_entropyCoefficient, entropy));
                }

                // Store the gradient information
                gradients.Add((state, action, gradientScale, logProb));
            }

            // Apply gradients to update the policy
            _actor.UpdateParameters(gradients, _actorLearningRate);
        }

        /// <summary>
        /// Normalizes a vector by subtracting the mean and dividing by the standard deviation.
        /// </summary>
        /// <param name="vector">The vector to normalize.</param>
        private void NormalizeVector(Vector<T> vector)
        {
            // Calculate mean
            T sum = NumOps.Zero;
            for (int i = 0; i < vector.Length; i++)
            {
                sum = NumOps.Add(sum, vector[i]);
            }
            T mean = NumOps.Divide(sum, NumOps.FromDouble(vector.Length));

            // Calculate standard deviation
            T sumSquaredDiff = NumOps.Zero;
            for (int i = 0; i < vector.Length; i++)
            {
                T diff = NumOps.Subtract(vector[i], mean);
                sumSquaredDiff = NumOps.Add(sumSquaredDiff, NumOps.Multiply(diff, diff));
            }
            T stdDev = NumOps.Sqrt(NumOps.Divide(sumSquaredDiff, NumOps.FromDouble(vector.Length)));

            // Add small epsilon to avoid division by zero
            T epsilon = NumOps.FromDouble(1e-8);
            stdDev = MathHelper.Max(stdDev, epsilon);

            // Normalize
            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] = NumOps.Divide(NumOps.Subtract(vector[i], mean), stdDev);
            }
        }

        /// <summary>
        /// Gets the parameters of the agent (both actor and critic).
        /// </summary>
        /// <returns>A vector containing all parameters.</returns>
        public Vector<T> GetParameters()
        {
            var allParameters = new List<T>();
            
            // Get actor parameters
            if (_actor is DiscreteStochasticPolicy<T> discretePolicy)
            {
                var actorParams = discretePolicy.GetParameters();
                foreach (var paramVector in actorParams)
                {
                    for (int i = 0; i < paramVector.Length; i++)
                    {
                        allParameters.Add(paramVector[i]);
                    }
                }
            }
            else if (_actor is ContinuousStochasticPolicy<T> continuousPolicy)
            {
                var (commonParams, meanParams, stdDevParams) = continuousPolicy.GetParameters();
                
                // Add common layer parameters
                foreach (var paramVector in commonParams)
                {
                    for (int i = 0; i < paramVector.Length; i++)
                    {
                        allParameters.Add(paramVector[i]);
                    }
                }
                
                // Add mean layer parameters
                foreach (var paramVector in meanParams)
                {
                    for (int i = 0; i < paramVector.Length; i++)
                    {
                        allParameters.Add(paramVector[i]);
                    }
                }
                
                // Add std dev layer parameters
                foreach (var paramVector in stdDevParams)
                {
                    for (int i = 0; i < paramVector.Length; i++)
                    {
                        allParameters.Add(paramVector[i]);
                    }
                }
            }
            
            // Get critic parameters
            var criticParams = _critic.GetParameters();
            for (int i = 0; i < criticParams.Length; i++)
            {
                allParameters.Add(criticParams[i]);
            }
            
            // Get target critic parameters if applicable
            if (_useCriticTargetNetwork && _criticTarget != null)
            {
                var targetParams = _criticTarget.GetParameters();
                for (int i = 0; i < targetParams.Length; i++)
                {
                    allParameters.Add(targetParams[i]);
                }
            }
            
            return new Vector<T>([.. allParameters]);
        }

        /// <summary>
        /// Sets the parameters of the agent (both actor and critic).
        /// </summary>
        /// <param name="parameters">A vector containing all parameters.</param>
        public void SetParameters(Vector<T> parameters)
        {
            int index = 0;
            
            // Set actor parameters
            if (_actor is DiscreteStochasticPolicy<T> discretePolicy)
            {
                var actorParams = discretePolicy.GetParameters();
                var newActorParams = new List<Vector<T>>();
                
                foreach (var paramVector in actorParams)
                {
                    var newVector = new Vector<T>(paramVector.Length);
                    for (int i = 0; i < paramVector.Length; i++)
                    {
                        newVector[i] = parameters[index++];
                    }
                    newActorParams.Add(newVector);
                }
                
                discretePolicy.SetParameters(newActorParams);
            }
            else if (_actor is ContinuousStochasticPolicy<T> continuousPolicy)
            {
                var (commonParams, meanParams, stdDevParams) = continuousPolicy.GetParameters();
                var newCommonParams = new List<Vector<T>>();
                var newMeanParams = new List<Vector<T>>();
                var newStdDevParams = new List<Vector<T>>();
                
                // Extract common layer parameters
                foreach (var paramVector in commonParams)
                {
                    var newVector = new Vector<T>(paramVector.Length);
                    for (int i = 0; i < paramVector.Length; i++)
                    {
                        newVector[i] = parameters[index++];
                    }
                    newCommonParams.Add(newVector);
                }
                
                // Extract mean layer parameters
                foreach (var paramVector in meanParams)
                {
                    var newVector = new Vector<T>(paramVector.Length);
                    for (int i = 0; i < paramVector.Length; i++)
                    {
                        newVector[i] = parameters[index++];
                    }
                    newMeanParams.Add(newVector);
                }
                
                // Extract std dev layer parameters
                foreach (var paramVector in stdDevParams)
                {
                    var newVector = new Vector<T>(paramVector.Length);
                    for (int i = 0; i < paramVector.Length; i++)
                    {
                        newVector[i] = parameters[index++];
                    }
                    newStdDevParams.Add(newVector);
                }
                
                continuousPolicy.SetParameters((newCommonParams, newMeanParams, newStdDevParams));
            }
            
            // Set critic parameters
            var criticParams = _critic.GetParameters();
            var newCriticParams = new Vector<T>(criticParams.Length);
            for (int i = 0; i < criticParams.Length; i++)
            {
                newCriticParams[i] = parameters[index++];
            }
            _critic.SetParameters(newCriticParams);
            
            // Set target critic parameters if applicable
            if (_useCriticTargetNetwork && _criticTarget != null)
            {
                var targetParams = _criticTarget.GetParameters();
                var newTargetParams = new Vector<T>(targetParams.Length);
                for (int i = 0; i < targetParams.Length; i++)
                {
                    newTargetParams[i] = parameters[index++];
                }
                _criticTarget.SetParameters(newTargetParams);
            }
        }

        /// <summary>
        /// Saves the agent's state to a file.
        /// </summary>
        /// <param name="filePath">The path where the agent's state should be saved.</param>
        public override void Save(string filePath)
        {
            base.Save(filePath);
        }

        /// <summary>
        /// Loads the agent's state from a file.
        /// </summary>
        /// <param name="filePath">The path from which to load the agent's state.</param>
        public override void Load(string filePath)
        {
            base.Load(filePath);
        }
        
        /// <summary>
        /// Saves agent-specific state.
        /// </summary>
        /// <param name="writer">The binary writer to write state to.</param>
        protected override void SaveAgentSpecificState(BinaryWriter writer)
        {
            // Save A2C-specific parameters
            writer.Write(Convert.ToDouble(_actorLearningRate));
            writer.Write(Convert.ToDouble(_criticLearningRate));
            writer.Write(Convert.ToDouble(_entropyCoefficient));
            writer.Write(Convert.ToDouble(_valueLossCoefficient));
            writer.Write(_useGAE);
            writer.Write(Convert.ToDouble(_gaeParameter));
            writer.Write(_normalizeAdvantages);
            writer.Write(_standardizeRewards);
            writer.Write(_useCriticTargetNetwork);
            writer.Write(_stepsPerUpdate);
            writer.Write(_useNStepReturns);
            writer.Write(_nSteps);
            writer.Write(Convert.ToDouble(_maxGradientNorm));
            writer.Write(_stepsCollected);
            
            // Save experience buffer size (not the actual experiences as they're transient)
            writer.Write(_experienceBuffer.Count);
            
            // Save network parameters
            var parameters = GetParameters();
            writer.Write(parameters.Length);
            for (int i = 0; i < parameters.Length; i++)
            {
                writer.Write(Convert.ToDouble(parameters[i]));
            }
        }
        
        /// <summary>
        /// Loads agent-specific state.
        /// </summary>
        /// <param name="reader">The binary reader to read state from.</param>
        protected override void LoadAgentSpecificState(BinaryReader reader)
        {
            // Skip A2C-specific parameters (they are readonly and set in constructor)
            // We need to read them to maintain file format compatibility
            reader.ReadDouble(); // _actorLearningRate
            reader.ReadDouble(); // _criticLearningRate
            reader.ReadDouble(); // _entropyCoefficient
            reader.ReadDouble(); // _valueLossCoefficient
            reader.ReadBoolean(); // _useGAE
            reader.ReadDouble(); // _gaeParameter
            reader.ReadBoolean(); // _normalizeAdvantages
            reader.ReadBoolean(); // _standardizeRewards
            reader.ReadBoolean(); // _useCriticTargetNetwork
            reader.ReadInt32(); // _stepsPerUpdate
            reader.ReadBoolean(); // _useNStepReturns
            reader.ReadInt32(); // _nSteps
            reader.ReadDouble(); // _maxGradientNorm
            _stepsCollected = reader.ReadInt32();
            
            // Read experience buffer size (we don't restore the actual experiences)
            int bufferCount = reader.ReadInt32();
            _experienceBuffer.Clear();
            
            // Load network parameters
            int paramCount = reader.ReadInt32();
            var parameters = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                parameters[i] = NumOps.FromDouble(reader.ReadDouble());
            }
            SetParameters(parameters);
        }

        /// <summary>
        /// Trains the agent on a batch of experiences.
        /// </summary>
        /// <param name="states">The batch of states.</param>
        /// <param name="actions">The batch of actions.</param>
        /// <param name="rewards">The batch of rewards.</param>
        /// <param name="nextStates">The batch of next states.</param>
        /// <param name="dones">The batch of done flags.</param>
        /// <returns>The loss value from the training.</returns>
        public T Train(TState states, TAction[] actions, Vector<T> rewards, TState nextStates, Vector<T> dones)
        {
            if (!IsTraining)
                return NumOps.Zero;
                
            // Clear any existing experiences to avoid duplicates
            _experienceBuffer.Clear();
            
            // Process the batch
            int batchSize = rewards.Length;
            for (int i = 0; i < batchSize; i++)
            {
                bool isDone = NumOps.GreaterThan(dones[i], NumOps.Zero);
                
                // Add each experience to the buffer
                // For tensor states, we need to extract each individual state from the batch
                var state = GetStateFromBatch(states, i);
                var nextState = GetStateFromBatch(nextStates, i);
                var action = actions[i];
                
                _experienceBuffer.Add((state, action, rewards[i], nextState, isDone));
            }
            
            // Update networks
            UpdateNetworks();
            
            // Return the latest loss value
            return LastLoss;
        }
        
        /// <summary>
        /// Extracts an individual state from a batch of states.
        /// </summary>
        /// <param name="batchedStates">The batched states.</param>
        /// <param name="index">The index of the state to extract.</param>
        /// <returns>The individual state at the specified index.</returns>
        private TState GetStateFromBatch(TState batchedStates, int index)
        {
            // This implementation assumes TState is Tensor<T> and can handle slicing
            // We know TState is Tensor<T> from the constraint
            return (TState)(object)((Tensor<T>)(object)batchedStates).GetSlice(index);
        }

        /// <summary>
        /// Gets the actor (policy) network used by this agent.
        /// </summary>
        /// <returns>The actor (policy) network.</returns>
        /// <remarks>
        /// This method is provided to allow for component reuse by other algorithm implementations.
        /// </remarks>
        public IPolicy<TState, TAction, T> GetActor()
        {
            return _actor;
        }
        
        /// <summary>
        /// Gets the critic (value function) network used by this agent.
        /// </summary>
        /// <returns>The critic (value function) network.</returns>
        /// <remarks>
        /// This method is provided to allow for component reuse by other algorithm implementations.
        /// </remarks>
        public IValueFunction<TState, T> GetCritic()
        {
            return _critic;
        }
        
        /// <summary>
        /// Gets the latest loss value from training.
        /// </summary>
        /// <returns>The last computed loss value.</returns>
        public override T GetLatestLoss()
        {
            return LastLoss;
        }

        /// <summary>
        /// Private implementation of a value function network.
        /// </summary>
        /// <typeparam name="T">The numeric type.</typeparam>
        private class ValueNetwork : IValueFunction<TState, T>
        {
            /// <summary>
            /// Gets the numeric operations for type T.
            /// </summary>
            protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
            
            private readonly List<LayerBase<T>> _layers = default!;
            private readonly T _learningRate = default!;
            private readonly Random _random = default!;

            public ValueNetwork(int stateSize, int[] hiddenSizes, IActivationFunction<T> activation, int? seed = null)
            {
                _layers = new List<LayerBase<T>>();
                _learningRate = NumOps.FromDouble(0.001);
                _random = seed.HasValue ? new Random(seed.Value) : new Random();

                // Input layer to first hidden layer
                int inputSize = stateSize;
                for (int i = 0; i < hiddenSizes.Length; i++)
                {
                    _layers.Add(new DenseLayer<T>(inputSize, hiddenSizes[i], activation));
                    inputSize = hiddenSizes[i];
                }

                // Output layer (single value)
                _layers.Add(new DenseLayer<T>(inputSize, 1, new IdentityActivation<T>() as IActivationFunction<T>));
            }

            public T PredictValue(TState state)
            {
                Tensor<T> output = state;
                foreach (var layer in _layers)
                {
                    output = layer.Forward(output);
                }
                return output.ToVector()[0];
            }

            public Vector<T> PredictValues(TState[] states)
            {
                var values = new Vector<T>(states.Length);
                for (int i = 0; i < states.Length; i++)
                {
                    values[i] = PredictValue(states[i]);
                }
                return values;
            }

            public T Update(TState[] states, Vector<T> targets)
            {
                if (states.Length == 0)
                    return NumOps.Zero;

                T totalLoss = NumOps.Zero;

                // Shuffle indices for mini-batch training
                var indices = Enumerable.Range(0, states.Length).ToList();
                for (int i = indices.Count - 1; i > 0; i--)
                {
                    int j = _random.Next(i + 1);
                    (indices[i], indices[j]) = (indices[j], indices[i]);
                }

                // Calculate batch size (use full dataset if smaller than batch size)
                int batchSize = Math.Min(64, states.Length);
                
                // Process each mini-batch
                for (int batchStart = 0; batchStart < states.Length; batchStart += batchSize)
                {
                    int currentBatchSize = Math.Min(batchSize, states.Length - batchStart);
                    T batchLoss = NumOps.Zero;

                    // Process each sample in the mini-batch
                    for (int i = 0; i < currentBatchSize; i++)
                    {
                        int idx = indices[batchStart + i];
                        
                        // Forward pass
                        Tensor<T> output = states[idx];
                        foreach (var layer in _layers)
                        {
                            output = layer.Forward(output);
                        }

                        // Compute loss
                        T prediction = output.ToVector()[0];
                        T error = NumOps.Subtract(prediction, targets[idx]);
                        batchLoss = NumOps.Add(batchLoss, NumOps.Multiply(error, error));

                        // Backward pass (simple gradient for MSE loss)
                        Tensor<T> gradient = new Tensor<T>(new[] { 1 });
                        gradient[0] = NumOps.Multiply(NumOps.FromDouble(2.0), error);

                        foreach (var layer in _layers.AsEnumerable().Reverse())
                        {
                            if (layer is DenseLayer<T> denseLayer)
                            {
                                gradient = denseLayer.Backward(gradient);
                                denseLayer.UpdateParameters(_learningRate);
                            }
                        }
                    }

                    // Accumulate batch loss
                    totalLoss = NumOps.Add(totalLoss, NumOps.Divide(batchLoss, NumOps.FromDouble(currentBatchSize)));
                }

                // Return average loss across all batches
                return NumOps.Divide(totalLoss, NumOps.FromDouble((states.Length + batchSize - 1) / batchSize));
            }

            public Vector<T> GetParameters()
            {
                var parameters = new List<T>();

                foreach (var layer in _layers)
                {
                    if (layer is DenseLayer<T> denseLayer)
                    {
                        // Add weights
                        var layerParams = denseLayer.GetParameters();
                        for (int i = 0; i < layerParams.Count(); i++)
                        {
                            parameters.Add(layerParams[i]);
                        }
                    }
                }

                return new Vector<T>([.. parameters]);
            }

            public void SetParameters(Vector<T> parameters)
            {
                int index = 0;

                foreach (var layer in _layers)
                {
                    if (layer is DenseLayer<T> denseLayer)
                    {
                        // Get the number of parameters this layer has
                        int layerParamCount = denseLayer.ParameterCount;
                        
                        // Extract parameters for this layer
                        var layerParams = new Vector<T>(layerParamCount);
                        for (int i = 0; i < layerParamCount; i++)
                        {
                            layerParams[i] = parameters[index++];
                        }
                        
                        // Set the parameters
                        denseLayer.SetParameters(layerParams);
                    }
                }
            }

            public void CopyParametersFrom(IValueFunction<TState, T> source)
            {
                var parameters = source.GetParameters();
                SetParameters(parameters);
            }

            public void SoftUpdate(IValueFunction<TState, T> source, T tau)
            {
                // Get parameters from both networks
                var targetParams = GetParameters();
                var sourceParams = source.GetParameters();

                // Ensure the parameter vectors have the same length
                if (targetParams.Length != sourceParams.Length)
                {
                    throw new InvalidOperationException("Parameter vectors must have the same length for soft update");
                }

                // Apply soft update: targetParams = (1 - tau) * targetParams + tau * sourceParams
                for (int i = 0; i < targetParams.Length; i++)
                {
                    targetParams[i] = NumOps.Add(
                        NumOps.Multiply(NumOps.Subtract(NumOps.One, tau), targetParams[i]),
                        NumOps.Multiply(tau, sourceParams[i])
                    );
                }

                // Set the updated parameters
                SetParameters(targetParams);
            }
        }
    }
}