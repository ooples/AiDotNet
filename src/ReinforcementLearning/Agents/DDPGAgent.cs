global using AiDotNet.ReinforcementLearning.Exploration;
global using AiDotNet.Factories;

namespace AiDotNet.ReinforcementLearning.Agents;

/// <summary>
/// Implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm for continuous control.
/// </summary>
/// <typeparam name="TState">The type used to represent the environment state.</typeparam>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DDPG is an actor-critic, model-free algorithm for learning continuous actions.
/// It combines ideas from DPG (Deterministic Policy Gradient) and DQN (Deep Q-Network).
/// </para>
/// <para>
/// DDPG uses a deterministic policy gradient approach with four neural networks:
/// - Actor: Maps states to deterministic actions
/// - Critic: Maps state-action pairs to Q-values
/// - Target Actor: Slowly updated copy of the actor for stability
/// - Target Critic: Slowly updated copy of the critic for stability
/// </para>
/// <para>
/// DDPG includes several stabilizing techniques:
/// - Experience replay buffer
/// - Soft target updates
/// - Batch normalization
/// - Action noise for exploration
/// </para>
/// </remarks>
public class DDPGAgent<TState, T> : ActorCriticAgentBase<TState, Vector<T>, T, IDeterministicPolicy<TState, Vector<T>, T>, IActionValueFunction<TState, Vector<T>, T>>
    where TState : Tensor<T>
{
    private readonly DDPGOptions _options;
    private readonly bool _useSeparateTargetCriticForActorUpdate;

    /// <summary>
    /// Initializes a new instance of the <see cref="DDPGAgent{TState, T}"/> class.
    /// </summary>
    /// <param name="options">Options for the DDPG algorithm.</param>
    public DDPGAgent(DDPGOptions options)
        : base(
              // Actor and critic networks
              actor: new DeterministicPolicy<TState, T>(
                  options.StateSize,
                  options.ActionSize,
                  options.ActorNetworkArchitecture,
                  options.ActorActivationFunction,
                  options.ActorFinalActivationFunction,
                  options.UseLayerNormalization,
                  options.Seed),
              actorTarget: new DeterministicPolicy<TState, T>(
                  options.StateSize,
                  options.ActionSize,
                  options.ActorNetworkArchitecture,
                  options.ActorActivationFunction,
                  options.ActorFinalActivationFunction,
                  options.UseLayerNormalization,
                  options.Seed),
              critic: new QNetwork<TState, Vector<T>, T>(
                  options.StateSize,
                  options.ActionSize,
                  options.CriticNetworkArchitecture,
                  options.CriticActivationFunction,
                  options.UseLayerNormalization,
                  options.Seed),
              criticTarget: new QNetwork<TState, Vector<T>, T>(
                  options.StateSize,
                  options.ActionSize,
                  options.CriticNetworkArchitecture,
                  options.CriticActivationFunction,
                  options.UseLayerNormalization,
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
                  ? new GaussianNoiseStrategy<Vector<T>, T>(
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
              seed: options.Seed)
    {
        _options = options;
        _useSeparateTargetCriticForActorUpdate = options.UseSeparateTargetCriticForActorUpdate;
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
        if (isTraining && TotalSteps >= WarmUpSteps)
        {
            // Apply exploration noise using the interface
            action = ExplorationStrategy.ApplyExploration(action, TotalSteps);
            
            // Ensure actions are clipped to valid range (assuming [-1, 1])
            for (int i = 0; i < action.Length; i++)
            {
                action[i] = MathHelper.Clamp(action[i], NumOps.Negate(NumOps.One), NumOps.One);
            }
        }
        else if (isTraining && TotalSteps < WarmUpSteps)
        {
            // During warm-up phase, use random actions for better exploration
            action = GenerateRandomAction();
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
        // Update critic
        T criticLoss = UpdateCritic(states, actions, rewards, nextStates, dones, weights, indices);

        // Update actor (less frequently than critic)
        T actorLoss = NumOps.Zero;
        if (TotalSteps % 2 == 0)  // Update actor every 2 steps
        {
            actorLoss = UpdateActor(states);
        }
        
        // Store the combined loss
        LastLoss = NumOps.Add(criticLoss, actorLoss);
    }
    
    /// <summary>
    /// Updates the critic network.
    /// </summary>
    /// <param name="states">Batch of states.</param>
    /// <param name="actions">Batch of actions.</param>
    /// <param name="rewards">Batch of rewards.</param>
    /// <param name="nextStates">Batch of next states.</param>
    /// <param name="dones">Batch of episode termination flags.</param>
    /// <param name="weights">Importance sampling weights (for prioritized replay).</param>
    /// <param name="indices">Indices of the sampled experiences (for prioritized replay).</param>
    /// <returns>The critic loss.</returns>
    private T UpdateCritic(
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
        for (int i = 0; i < states.Length; i++)
        {
            // Get next action from target actor
            Vector<T> nextAction = ActorTarget.SelectAction(nextStates[i]);
            
            // Get Q-value from target critic
            T nextQValue = CriticTarget.PredictQValue(nextStates[i], nextAction);
            
            // If done, only consider immediate reward
            if (dones[i])
            {
                targetQValues[i] = rewards[i];
            }
            else
            {
                // Q-target = r + gamma * Q'(s', a')
                targetQValues[i] = NumOps.Add(rewards[i], NumOps.Multiply(Gamma, nextQValue));
            }
        }

        // Update critic
        T criticLoss = Critic.Update(states, actions, targetQValues, weights);
        
        // Update priorities if using prioritized replay
        if (ReplayBuffer is PrioritizedReplayBuffer<TState, Vector<T>, T> prioritizedBuffer)
        {
            // Compute TD errors for priority updates
            var tdErrors = new Vector<T>(states.Length);
            for (int i = 0; i < states.Length; i++)
            {
                T currentQValue = Critic.PredictQValue(states[i], actions[i]);
                tdErrors[i] = NumOps.Abs(NumOps.Subtract(targetQValues[i], currentQValue));
            }
            
            // Update priorities
            for (int i = 0; i < indices.Length; i++)
            {
                prioritizedBuffer.UpdatePriority(indices[i], tdErrors[i]);
            }
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
            
            // Compute action gradients from critic
            IActionValueFunction<TState, Vector<T>, T> criticToUse = 
                _useSeparateTargetCriticForActorUpdate ? CriticTarget : Critic;
                
            Vector<T> actionGradient = criticToUse.ActionGradients(states[i], action);
            
            // Negate gradient since we want to maximize Q (minimize -Q)
            for (int j = 0; j < actionGradient.Length; j++)
            {
                actionGradient[j] = NumOps.Negate(actionGradient[j]);
            }
            
            // Store gradients for actor update
            policyGradients.Add((states[i], actionGradient));
            
            // Compute loss (for monitoring)
            totalLoss = NumOps.Add(totalLoss, NumOps.Negate(criticToUse.PredictQValue(states[i], action)));
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
        CriticTarget.SoftUpdate(Critic, Tau);
    }
    
    /// <summary>
    /// Initializes the target networks by copying parameters from the main networks.
    /// </summary>
    protected override void InitializeTargetNetworks()
    {
        ActorTarget.CopyParametersFrom(Actor);
        CriticTarget.CopyParametersFrom(Critic);
    }
    
    /// <summary>
    /// Determines whether target networks should be updated in the current step.
    /// </summary>
    /// <returns>True if target networks should be updated, otherwise false.</returns>
    protected override bool ShouldUpdateTargets()
    {
        // Update target networks every 2 steps
        return TotalSteps % 2 == 0;
    }

    /// <summary>
    /// Saves the agent's state to a file.
    /// </summary>
    /// <param name="filePath">The path where the agent's state should be saved.</param>
    public override void Save(string filePath)
    {
        using (var stream = new FileStream(filePath, FileMode.Create))
        {
            using (var writer = new BinaryWriter(stream))
            {
                // Save network parameters
                SaveNetworkParameters(writer);
                
                // Save agent configuration
                writer.Write(_useSeparateTargetCriticForActorUpdate);
                writer.Write(TotalSteps);
                writer.Write(IsTraining);
                
                // Additional fields can be saved here as needed
            }
        }
    }
    
    /// <summary>
    /// Helper method to save network parameters.
    /// </summary>
    /// <param name="writer">The binary writer to use.</param>
    private void SaveNetworkParameters(BinaryWriter writer)
    {
        // TODO: Implement parameter extraction from actor and critic networks
        // The current policy interfaces don't expose GetParameters method
        // For now, save dummy data to maintain file format compatibility
        writer.Write(0); // Actor params count
        writer.Write(0); // Critic params count
    }

    /// <summary>
    /// Loads the agent's state from a file.
    /// </summary>
    /// <param name="filePath">The path from which to load the agent's state.</param>
    public override void Load(string filePath)
    {
        using (var stream = new FileStream(filePath, FileMode.Open))
        {
            using (var reader = new BinaryReader(stream))
            {
                // Load network parameters
                LoadNetworkParameters(reader);
                
                // Load agent configuration
                // Skip _useSeparateTargetCriticForActorUpdate as it's readonly
                reader.ReadBoolean(); // Discard
                // Skip TotalSteps as the setter is inaccessible
                reader.ReadInt32(); // Discard
                SetTrainingMode(reader.ReadBoolean());
                
                // Additional fields can be loaded here as needed
            }
        }
        
        // Update target networks after loading
        UpdateTargetNetworks();
    }
    
    /// <summary>
    /// Gets all parameters of the agent as a single vector.
    /// </summary>
    /// <returns>A vector containing all parameters.</returns>
    public Vector<T> GetParameters()
    {
        // TODO: Implement parameter extraction from actor and critic networks
        // The current policy interfaces don't expose GetParameters method
        return new Vector<T>(0);
    }
    
    /// <summary>
    /// Sets all parameters of the agent from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters.</param>
    public void SetParameters(Vector<T> parameters)
    {
        // TODO: Implement parameter setting for actor and critic networks
        // The current policy interfaces don't expose SetParameters method
        // For now, this is a no-op
    }
    
    /// <summary>
    /// Helper method to load network parameters.
    /// </summary>
    /// <param name="reader">The binary reader to use.</param>
    private void LoadNetworkParameters(BinaryReader reader)
    {
        // TODO: Implement parameter loading for actor and critic networks
        // The current policy interfaces don't expose SetParameters method
        // For now, read and discard dummy data to maintain file format compatibility
        int actorParamCount = reader.ReadInt32();
        for (int i = 0; i < actorParamCount; i++)
        {
            reader.ReadDouble(); // Discard
        }
        
        int criticParamCount = reader.ReadInt32();
        for (int i = 0; i < criticParamCount; i++)
        {
            reader.ReadDouble(); // Discard
        }
    }

    // SetTrainingMode is inherited from base class

    /// <summary>
    /// Gets the actor network.
    /// </summary>
    /// <returns>The actor network.</returns>
    public IDeterministicPolicy<TState, Vector<T>, T> GetActor()
    {
        return Actor;
    }

    /// <summary>
    /// Gets the critic network.
    /// </summary>
    /// <returns>The critic network.</returns>
    public IActionValueFunction<TState, Vector<T>, T> GetCritic()
    {
        return Critic;
    }
    
    /// <summary>
    /// Gets the target actor network.
    /// </summary>
    /// <returns>The target actor network.</returns>
    public IDeterministicPolicy<TState, Vector<T>, T> GetActorTarget()
    {
        return ActorTarget;
    }
    
    /// <summary>
    /// Gets the target critic network.
    /// </summary>
    /// <returns>The target critic network.</returns>
    public IActionValueFunction<TState, Vector<T>, T> GetCriticTarget()
    {
        return CriticTarget;
    }
    
    /// <summary>
    /// Trains the agent on a batch of experiences.
    /// </summary>
    /// <param name="states">Batch of states.</param>
    /// <param name="actions">Batch of actions.</param>
    /// <param name="rewards">Batch of rewards.</param>
    /// <param name="nextStates">Batch of next states.</param>
    /// <param name="dones">Batch of done flags.</param>
    /// <returns>The loss value from training.</returns>
    public T Train(Tensor<T> states, Vector<T>[] actions, Vector<T> rewards, Tensor<T> nextStates, Vector<T> dones)
    {
        // Convert tensor inputs to arrays
        var stateArray = new TState[states.Shape[0]];
        var nextStateArray = new TState[nextStates.Shape[0]];
        var rewardArray = new T[rewards.Length];
        var doneArray = new bool[dones.Length];
        
        // Extract states
        for (int i = 0; i < states.Shape[0]; i++)
        {
            stateArray[i] = (TState)states.GetSlice(i);
            nextStateArray[i] = (TState)nextStates.GetSlice(i);
            rewardArray[i] = rewards[i];
            doneArray[i] = NumOps.GreaterThan(dones[i], NumOps.FromDouble(0.5)); // Convert to boolean
        }
        
        // Create default weights for non-prioritized replay
        var weights = new T[stateArray.Length];
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = NumOps.One;
        }
        
        // No indices since we're not updating priorities
        var indices = new int[stateArray.Length];
        for (int i = 0; i < indices.Length; i++)
        {
            indices[i] = i;
        }
        
        // Update networks
        UpdateNetworks(stateArray, actions, rewardArray, nextStateArray, doneArray, weights, indices);
        
        // Update target networks if needed
        if (ShouldUpdateTargets())
        {
            UpdateTargetNetworks();
        }
        
        return LastLoss;
    }

    /// <summary>
    /// A deterministic policy implementation for DDPG's actor network.
    /// </summary>
    /// <typeparam name="TStateType">The type used to represent the environment state.</typeparam>
    /// <typeparam name="TNumeric">The numeric type used for calculations.</typeparam>
    public class DeterministicPolicy<TStateType, TNumeric> : IDeterministicPolicy<TStateType, Vector<TNumeric>, TNumeric>
        where TStateType : Tensor<TNumeric>
    {
        /// <summary>
        /// Gets the numeric operations for type TNumeric.
        /// </summary>
        protected INumericOperations<TNumeric> NumOps => MathHelper.GetNumericOperations<TNumeric>();
        private readonly List<LayerBase<TNumeric>> _layers;
        private readonly bool _useLayerNorm;
        private readonly TNumeric _learningRate;
        private readonly Random _random;
        
        public bool IsStochastic => false;
        public bool IsContinuous => true;

        public DeterministicPolicy(
            int stateSize,
            int actionSize,
            int[] hiddenSizes,
            ActivationFunction activation,
            ActivationFunction finalActivation,
            bool useLayerNorm,
            int? seed = null)
        {
            _layers = new List<LayerBase<TNumeric>>();
            _useLayerNorm = useLayerNorm;
            _learningRate = NumOps.FromDouble(0.0001);
            _random = seed.HasValue ? new Random(seed.Value) : new Random();
            
            // Input layer to first hidden layer
            int inputSize = stateSize;
            for (int i = 0; i < hiddenSizes.Length; i++)
            {
                _layers.Add(new DenseLayer<TNumeric>(inputSize, hiddenSizes[i], ActivationFunctionFactory<TNumeric>.CreateActivationFunction(activation)));
                
                if (_useLayerNorm)
                {
                    _layers.Add(new LayerNormalizationLayer<TNumeric>(hiddenSizes[i]));
                }
                
                inputSize = hiddenSizes[i];
            }
            
            // Output layer
            _layers.Add(new DenseLayer<TNumeric>(inputSize, actionSize, ActivationFunctionFactory<TNumeric>.CreateActivationFunction(finalActivation)));
        }
        
        public Vector<TNumeric> SelectAction(TStateType state)
        {
            Tensor<TNumeric> output = state;
            
            // Forward pass through all layers
            foreach (var layer in _layers)
            {
                output = layer.Forward(output);
            }
            
            return output.ToVector();
        }

        public void UpdateFromPolicyGradients(
            List<(TStateType state, Vector<TNumeric> actionGradient)> policyGradients,
            bool useGradientClipping,
            TNumeric maxGradientNorm)
        {
            if (policyGradients.Count == 0)
                return;
                
            // Process each sample
            foreach (var (state, actionGradient) in policyGradients)
            {
                // Forward pass
                Tensor<TNumeric> output = state;
                var layerOutputs = new List<Tensor<TNumeric>> { state };
                
                foreach (var layer in _layers)
                {
                    output = layer.Forward(output);
                    layerOutputs.Add(output);
                }
                
                // Initialize gradient for backward pass
                Tensor<TNumeric> gradient = Tensor<TNumeric>.FromVector(actionGradient);
                
                // Clip gradient if needed
                if (useGradientClipping)
                {
                    TNumeric gradNorm = NumOps.Zero;
                    for (int i = 0; i < actionGradient.Length; i++)
                    {
                        gradNorm = NumOps.Add(gradNorm, NumOps.Multiply(actionGradient[i], actionGradient[i]));
                    }
                    gradNorm = NumOps.Sqrt(gradNorm);
                    
                    if (NumOps.GreaterThan(gradNorm, maxGradientNorm))
                    {
                        TNumeric scale = NumOps.Divide(maxGradientNorm, NumOps.Add(gradNorm, NumOps.FromDouble(1e-8)));
                        for (int i = 0; i < actionGradient.Length; i++)
                        {
                            gradient[i] = NumOps.Multiply(gradient[i], scale);
                        }
                    }
                }
                
                // Backward pass
                for (int i = _layers.Count - 1; i >= 0; i--)
                {
                    if (_layers[i] is DenseLayer<TNumeric> denseLayer)
                    {
                        gradient = denseLayer.Backward(gradient);
                        denseLayer.UpdateParameters(_learningRate);
                    }
                    else if (_layers[i] is LayerNormalizationLayer<TNumeric> layerNormLayer)
                    {
                        gradient = layerNormLayer.Backward(gradient);
                        layerNormLayer.UpdateParameters(_learningRate);
                    }
                }
            }
        }

        public void CopyParametersFrom(IDeterministicPolicy<TStateType, Vector<TNumeric>, TNumeric> source)
        {
            if (source is DeterministicPolicy<TStateType, TNumeric> other)
            {
                // Copy parameters from each layer
                for (int i = 0; i < _layers.Count && i < other._layers.Count; i++)
                {
                    if (_layers[i] is DenseLayer<TNumeric> thisLayer && 
                        other._layers[i] is DenseLayer<TNumeric> otherLayer)
                    {
                        thisLayer.SetWeights(otherLayer.GetWeights());
                        thisLayer.SetBiases(otherLayer.GetBiases());
                    }
                    else if (_layers[i] is LayerNormalizationLayer<TNumeric> thisNormLayer && 
                             other._layers[i] is LayerNormalizationLayer<TNumeric> otherNormLayer)
                    {
                        thisNormLayer.CopyParameters(otherNormLayer);
                    }
                }
            }
        }

        public void SoftUpdate(IDeterministicPolicy<TStateType, Vector<TNumeric>, TNumeric> source, TNumeric tau)
        {
            if (source is DeterministicPolicy<TStateType, TNumeric> other)
            {
                // Soft update parameters from each layer
                for (int i = 0; i < _layers.Count && i < other._layers.Count; i++)
                {
                    if (_layers[i] is DenseLayer<TNumeric> thisLayer && 
                        other._layers[i] is DenseLayer<TNumeric> otherLayer)
                    {
                        // Get current weights and biases
                        var thisWeights = thisLayer.GetWeights();
                        var thisBiases = thisLayer.GetBiases();
                        var otherWeights = otherLayer.GetWeights();
                        var otherBiases = otherLayer.GetBiases();
                        
                        // Soft update weights: θ' = (1 - τ) * θ' + τ * θ
                        for (int row = 0; row < thisWeights.Shape[0]; row++)
                        {
                            for (int col = 0; col < thisWeights.Shape[1]; col++)
                            {
                                thisWeights[row, col] = NumOps.Add(
                                                         NumOps.Multiply(NumOps.Subtract(NumOps.One, tau), thisWeights[row, col]), 
                                                         NumOps.Multiply(tau, otherWeights[row, col]));
                            }
                        }
                        
                        // Soft update biases
                        for (int j = 0; j < thisBiases.Length; j++)
                        {
                            thisBiases[j] = NumOps.Add(
                                             NumOps.Multiply(NumOps.Subtract(NumOps.One, tau), thisBiases[j]),
                                             NumOps.Multiply(tau, otherBiases[j]));
                        }
                        
                        // Set updated weights and biases
                        thisLayer.SetWeights(thisWeights);
                        thisLayer.SetBiases(thisBiases);
                    }
                    else if (_layers[i] is LayerNormalizationLayer<TNumeric> thisNormLayer && 
                             other._layers[i] is LayerNormalizationLayer<TNumeric> otherNormLayer)
                    {
                        thisNormLayer.SoftUpdate(otherNormLayer, tau);
                    }
                }
            }
        }
        
        public object EvaluatePolicy(TStateType state)
        {
            // For deterministic policy, this just returns the action
            return SelectAction(state);
        }
        
        public TNumeric LogProbability(TStateType state, Vector<TNumeric> action)
        {
            // Deterministic policy doesn't have a probability distribution
            // This is only here to satisfy the interface
            return NumOps.Zero;
        }
        
        public void UpdateParameters(object gradients, TNumeric learningRate)
        {
            // This method is for stochastic policies and isn't used in DDPG
            throw new NotImplementedException("Deterministic policies don't use this update method.");
        }
        
        public TNumeric GetEntropy(TStateType state)
        {
            // Deterministic policy has zero entropy
            return NumOps.Zero;
        }
    }

    /// <summary>
    /// A Q-network implementation for DDPG's critic network.
    /// </summary>
    /// <typeparam name="TStateType">The type used to represent the environment state.</typeparam>
    /// <typeparam name="TActionType">The type used to represent actions.</typeparam>
    /// <typeparam name="TNumeric">The numeric type used for calculations.</typeparam>
    public class QNetwork<TStateType, TActionType, TNumeric> : IActionValueFunction<TStateType, TActionType, TNumeric>
        where TStateType : Tensor<TNumeric>
        where TActionType : Vector<TNumeric>
    {
        /// <summary>
        /// Gets the numeric operations for type TNumeric.
        /// </summary>
        protected INumericOperations<TNumeric> NumOps => MathHelper.GetNumericOperations<TNumeric>();
        private readonly List<LayerBase<TNumeric>> _layers;
        private readonly bool _useLayerNorm;
        private readonly TNumeric _learningRate;
        private readonly Random _random;
        private readonly int _actionSize;
        
        /// <summary>
        /// Gets the number of actions in the action space (for discrete action spaces)
        /// or the dimensionality of the action space (for continuous action spaces).
        /// </summary>
        public int ActionSize => _actionSize;

        /// <summary>
        /// Gets a value indicating whether the action space is continuous.
        /// </summary>
        public bool IsContinuous => true;
        
        public QNetwork(
            int stateSize,
            int actionSize,
            int[] hiddenSizes,
            ActivationFunction activation,
            bool useLayerNorm,
            int? seed = null)
        {
            _layers = new List<LayerBase<TNumeric>>();
            _useLayerNorm = useLayerNorm;
            _learningRate = NumOps.FromDouble(0.001);
            _random = seed.HasValue ? new Random(seed.Value) : new Random();
            _actionSize = actionSize;
            
            // First layer (state input)
            int inputSize = stateSize;
            _layers.Add(new DenseLayer<TNumeric>(inputSize, hiddenSizes[0], ActivationFunctionFactory<TNumeric>.CreateActivationFunction(activation)));
            
            if (_useLayerNorm)
            {
                _layers.Add(new LayerNormalizationLayer<TNumeric>(hiddenSizes[0]));
            }
            
            // Second layer (combines state features with action)
            // Input includes action after the first layer
            inputSize = hiddenSizes[0] + actionSize;
            
            for (int i = 1; i < hiddenSizes.Length; i++)
            {
                _layers.Add(new DenseLayer<TNumeric>(inputSize, hiddenSizes[i], ActivationFunctionFactory<TNumeric>.CreateActivationFunction(activation)));
                
                if (_useLayerNorm)
                {
                    _layers.Add(new LayerNormalizationLayer<TNumeric>(hiddenSizes[i]));
                }
                
                inputSize = hiddenSizes[i];
            }
            
            // Output layer (single Q-value)
            _layers.Add(new DenseLayer<TNumeric>(inputSize, 1, ActivationFunctionFactory<TNumeric>.CreateActivationFunction(ActivationFunction.Identity)));
        }
        
        public TNumeric PredictQValue(TStateType state, TActionType action)
        {
            // Forward pass
            var output = Forward(state, action);
            return output[0]; // Output has a single Q-value
        }
        
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
        /// Predicts Q-values for all possible actions in a given state.
        /// </summary>
        /// <param name="state">The state.</param>
        /// <returns>A vector of Q-values, one for each possible action.</returns>
        public Vector<TNumeric> PredictQValues(TStateType state)
        {
            // For continuous action spaces, this is not typically implemented directly
            // since there are infinite possible actions. Instead, we sample actions.
            throw new NotImplementedException(
                "PredictQValues for all actions is not supported for continuous action spaces. " +
                "Use PredictQValue with specific actions instead.");
        }
        
        /// <summary>
        /// Predicts Q-values for all possible actions for a batch of states.
        /// </summary>
        /// <param name="states">The batch of states.</param>
        /// <returns>A matrix of Q-values, where each row corresponds to a state and each column to an action.</returns>
        public Matrix<TNumeric> PredictQValuesBatch(TStateType[] states)
        {
            // For continuous action spaces, this is not typically implemented directly
            // since there are infinite possible actions. Instead, we sample actions.
            throw new NotImplementedException(
                "PredictQValuesBatch for all actions is not supported for continuous action spaces. " +
                "Use PredictQValues with specific state-action pairs instead.");
        }
        
        /// <summary>
        /// Gets the best action for a given state (the action with the highest Q-value).
        /// </summary>
        /// <param name="state">The state.</param>
        /// <returns>The best action.</returns>
        public TActionType GetBestAction(TStateType state)
        {
            // For continuous action spaces, finding the best action requires optimization
            // This is a simplified approach using random sampling
            int sampleCount = 100; // Number of samples to try
            TActionType? bestAction = null;
            TNumeric bestQValue = NumOps.MinValue;
            
            for (int i = 0; i < sampleCount; i++)
            {
                // Generate a random action
                var action = new Vector<TNumeric>(_actionSize);
                for (int j = 0; j < _actionSize; j++)
                {
                    // Random values in [-1, 1]
                    action[j] = NumOps.FromDouble(_random.NextDouble() * 2.0 - 1.0);
                }
                
                // Convert to TActionType (requires TActionType to be Vector<TNumeric>)
                TActionType typedAction;
                try {
                    typedAction = (TActionType)action;
                } catch {
                    // If direct casting fails, try creating a new instance
                    var instance = Activator.CreateInstance(typeof(TActionType), new object[] { action });
                    if (instance == null)
                    {
                        throw new InvalidOperationException($"Failed to create instance of {typeof(TActionType)}");
                    }
                    typedAction = (TActionType)instance;
                }
                
                // Evaluate Q-value
                TNumeric qValue = PredictQValue(state, typedAction);
                
                // Update best action if better
                if (bestAction == null || NumOps.GreaterThan(qValue, bestQValue))
                {
                    bestAction = typedAction;
                    bestQValue = qValue;
                }
            }
            
            return bestAction ?? throw new InvalidOperationException("Failed to find best action");
        }
        
        /// <summary>
        /// Updates the Q-function based on target Q-values for specific state-action pairs.
        /// </summary>
        /// <param name="states">The states.</param>
        /// <param name="actions">The actions taken in each state.</param>
        /// <param name="targets">The target Q-values for each state-action pair.</param>
        /// <returns>The loss value after the update.</returns>
        public TNumeric UpdateQ(TStateType[] states, TActionType[] actions, Vector<TNumeric> targets)
        {
            // Call the weighted update method with no weights
            return Update(states, actions, targets);
        }
        
        public TNumeric Update(TStateType[] states, TActionType[] actions, Vector<TNumeric> targets, TNumeric[]? weights = null)
        {
            if (states.Length != actions.Length || states.Length != targets.Length)
            {
                throw new ArgumentException("Batch sizes must match");
            }
            
            TNumeric totalLoss = NumOps.Zero;
            
            // If no weights are provided, use uniform weights
            if (weights == null)
            {
                weights = new TNumeric[states.Length];
                for (int i = 0; i < states.Length; i++)
                {
                    weights[i] = NumOps.One;
                }
            }
            
            // Process each sample
            for (int i = 0; i < states.Length; i++)
            {
                // Forward pass to get current Q-value
                var layerInputs = new List<Tensor<TNumeric>>();
                var layerOutputs = new List<Tensor<TNumeric>>();
                
                // Process the first layer separately (state only)
                Tensor<TNumeric> stateInput = states[i];
                layerInputs.Add(stateInput);
                
                Tensor<TNumeric> output = _layers[0].Forward(stateInput);
                layerOutputs.Add(output);
                
                int layerIndex = 1;
                if (_useLayerNorm)
                {
                    layerInputs.Add(output);
                    output = _layers[layerIndex].Forward(output);
                    layerOutputs.Add(output);
                    layerIndex++;
                }
                
                // Concatenate action with the output of the first layer
                Tensor<TNumeric> actionInput = Tensor<TNumeric>.FromVector(actions[i]);
                Tensor<TNumeric> combined = ConcatenateTensors(output, actionInput);
                
                // Process remaining layers
                for (; layerIndex < _layers.Count; layerIndex++)
                {
                    layerInputs.Add(combined);
                    output = _layers[layerIndex].Forward(combined);
                    layerOutputs.Add(output);
                    combined = output;
                }
                
                // Compute loss and gradient
                TNumeric prediction = output.ToVector()[0];
                TNumeric target = targets[i];
                TNumeric error = NumOps.Subtract(prediction, target);
                TNumeric loss = NumOps.Multiply(NumOps.Multiply(error, error), weights[i]);
                totalLoss = NumOps.Add(totalLoss, loss);
                
                // Initialize gradient for backward pass
                Tensor<TNumeric> gradient = new Tensor<TNumeric>(new[] { 1 });
                gradient[0] = NumOps.Multiply(NumOps.FromDouble(2.0), NumOps.Multiply(error, weights[i]));
                
                // Backward pass
                for (int j = _layers.Count - 1; j >= 0; j--)
                {
                    if (j == 0 || (j == 1 && _useLayerNorm))
                    {
                        // Skip action gradient computation for the first layer(s)
                        if (_layers[j] is DenseLayer<TNumeric> denseLayer)
                        {
                            gradient = denseLayer.Backward(gradient);
                            denseLayer.UpdateParameters(_learningRate);
                        }
                        else if (_layers[j] is LayerNormalizationLayer<TNumeric> layerNormLayer)
                        {
                            gradient = layerNormLayer.Backward(gradient);
                            layerNormLayer.UpdateParameters(_learningRate);
                        }
                    }
                    else
                    {
                        // Regular backward pass
                        if (_layers[j] is DenseLayer<TNumeric> denseLayer)
                        {
                            gradient = denseLayer.Backward(gradient);
                            denseLayer.UpdateParameters(_learningRate);
                        }
                        else if (_layers[j] is LayerNormalizationLayer<TNumeric> layerNormLayer)
                        {
                            gradient = layerNormLayer.Backward(gradient);
                            layerNormLayer.UpdateParameters(_learningRate);
                        }
                    }
                }
            }
            
            // Return average loss
            return NumOps.Divide(totalLoss, NumOps.FromDouble(states.Length));
        }
        
        public Vector<TNumeric> ActionGradients(TStateType state, TActionType action)
        {
            // Forward pass
            var layerInputs = new List<Tensor<TNumeric>>();
            var layerOutputs = new List<Tensor<TNumeric>>();
            
            // Process the first layer separately (state only)
            Tensor<TNumeric> stateInput = state;
            layerInputs.Add(stateInput);
            
            Tensor<TNumeric> output = _layers[0].Forward(stateInput);
            layerOutputs.Add(output);
            
            int layerIndex = 1;
            if (_useLayerNorm)
            {
                layerInputs.Add(output);
                output = _layers[layerIndex].Forward(output);
                layerOutputs.Add(output);
                layerIndex++;
            }
            
            // Store first layer output (before combining with action)
            Tensor<TNumeric> firstLayerOutput = output;
            
            // Concatenate action with the output of the first layer
            Tensor<TNumeric> actionInput = Tensor<TNumeric>.FromVector(action);
            Tensor<TNumeric> combined = ConcatenateTensors(output, actionInput);
            
            // Process remaining layers
            for (; layerIndex < _layers.Count; layerIndex++)
            {
                layerInputs.Add(combined);
                output = _layers[layerIndex].Forward(combined);
                layerOutputs.Add(output);
                combined = output;
            }
            
            // Initialize gradient for backward pass
            Tensor<TNumeric> gradient = new Tensor<TNumeric>(new[] { 1 });
            gradient[0] = NumOps.One; // Gradient of output w.r.t. itself is 1
            
            // Backward pass until we reach the action input
            for (int j = _layers.Count - 1; j > (_useLayerNorm ? 1 : 0); j--)
            {
                if (_layers[j] is DenseLayer<TNumeric> denseLayer)
                {
                    gradient = denseLayer.Backward(gradient);
                }
                else if (_layers[j] is LayerNormalizationLayer<TNumeric> layerNormLayer)
                {
                    gradient = layerNormLayer.Backward(gradient);
                }
            }
            
            // The gradient is now w.r.t. the combined state-action input
            // Extract the part corresponding to the action
            int firstLayerSize = firstLayerOutput.Shape[0];
            var actionGradient = new Vector<TNumeric>(_actionSize);
            for (int i = 0; i < _actionSize; i++)
            {
                actionGradient[i] = gradient[firstLayerSize + i];
            }
            
            return actionGradient;
        }
        
        public void CopyParametersFrom(IActionValueFunction<TStateType, TActionType, TNumeric> source)
        {
            if (source is QNetwork<TStateType, TActionType, TNumeric> other)
            {
                // Copy parameters from each layer
                for (int i = 0; i < _layers.Count && i < other._layers.Count; i++)
                {
                    if (_layers[i] is DenseLayer<TNumeric> thisLayer && 
                        other._layers[i] is DenseLayer<TNumeric> otherLayer)
                    {
                        thisLayer.SetWeights(otherLayer.GetWeights());
                        thisLayer.SetBiases(otherLayer.GetBiases());
                    }
                    else if (_layers[i] is LayerNormalizationLayer<TNumeric> thisNormLayer && 
                             other._layers[i] is LayerNormalizationLayer<TNumeric> otherNormLayer)
                    {
                        thisNormLayer.CopyParameters(otherNormLayer);
                    }
                }
            }
        }
        
        public void SoftUpdate(IActionValueFunction<TStateType, TActionType, TNumeric> source, TNumeric tau)
        {
            if (source is QNetwork<TStateType, TActionType, TNumeric> other)
            {
                // Soft update parameters from each layer
                for (int i = 0; i < _layers.Count && i < other._layers.Count; i++)
                {
                    if (_layers[i] is DenseLayer<TNumeric> thisLayer && 
                        other._layers[i] is DenseLayer<TNumeric> otherLayer)
                    {
                        // Get current weights and biases
                        var thisWeights = thisLayer.GetWeights();
                        var thisBiases = thisLayer.GetBiases();
                        var otherWeights = otherLayer.GetWeights();
                        var otherBiases = otherLayer.GetBiases();
                        
                        // Soft update weights: θ' = (1 - τ) * θ' + τ * θ
                        for (int row = 0; row < thisWeights.Shape[0]; row++)
                        {
                            for (int col = 0; col < thisWeights.Shape[1]; col++)
                            {
                                thisWeights[row, col] = NumOps.Add(
                                                         NumOps.Multiply(NumOps.Subtract(NumOps.One, tau), thisWeights[row, col]), 
                                                         NumOps.Multiply(tau, otherWeights[row, col]));
                            }
                        }
                        
                        // Soft update biases
                        for (int j = 0; j < thisBiases.Length; j++)
                        {
                            thisBiases[j] = NumOps.Add(
                                             NumOps.Multiply(NumOps.Subtract(NumOps.One, tau), thisBiases[j]),
                                             NumOps.Multiply(tau, otherBiases[j]));
                        }
                        
                        // Set updated weights and biases
                        thisLayer.SetWeights(thisWeights);
                        thisLayer.SetBiases(thisBiases);
                    }
                    else if (_layers[i] is LayerNormalizationLayer<TNumeric> thisNormLayer && 
                             other._layers[i] is LayerNormalizationLayer<TNumeric> otherNormLayer)
                    {
                        thisNormLayer.SoftUpdate(otherNormLayer, tau);
                    }
                }
            }
        }
        
        /// <summary>
        /// Predicts the value for a given state.
        /// </summary>
        /// <param name="state">The state for which to predict the value.</param>
        /// <returns>The predicted value.</returns>
        public TNumeric PredictValue(TStateType state)
        {
            // For Q-networks in actor-critic methods, this typically means the value
            // when taking the best action according to the current policy
            // Since we don't have direct access to the policy here, this is approximated
            return PredictQValue(state, GetBestAction(state));
        }
        
        /// <summary>
        /// Predicts values for a batch of states.
        /// </summary>
        /// <param name="states">The batch of states.</param>
        /// <returns>The predicted values for each state.</returns>
        public Vector<TNumeric> PredictValues(TStateType[] states)
        {
            var values = new Vector<TNumeric>(states.Length);
            for (int i = 0; i < states.Length; i++)
            {
                values[i] = PredictValue(states[i]);
            }
            return values;
        }
        
        /// <summary>
        /// Updates the value function based on target values.
        /// </summary>
        /// <param name="states">The states for which to update values.</param>
        /// <param name="targets">The target values for each state.</param>
        /// <returns>The loss value after the update.</returns>
        public TNumeric Update(TStateType[] states, Vector<TNumeric> targets)
        {
            // For Q-networks, we need actions to update
            // Get best actions for each state
            var actions = new TActionType[states.Length];
            for (int i = 0; i < states.Length; i++)
            {
                actions[i] = GetBestAction(states[i]);
            }
            
            // Use these state-action pairs for the update
            return Update(states, actions, targets);
        }
        
        /// <summary>
        /// Gets the parameters (weights and biases) of the value function.
        /// </summary>
        /// <returns>The parameters as a flat vector.</returns>
        public Vector<TNumeric> GetParameters()
        {
            // Collect all parameters from all layers
            var parameters = new List<TNumeric>();
            
            foreach (var layer in _layers)
            {
                if (layer is DenseLayer<TNumeric> denseLayer)
                {
                    var weights = denseLayer.GetWeights();
                    var biases = denseLayer.GetBiases();
                    
                    // Add weights
                    for (int i = 0; i < weights.Shape[0]; i++)
                    {
                        for (int j = 0; j < weights.Shape[1]; j++)
                        {
                            parameters.Add(weights[i, j]);
                        }
                    }
                    
                    // Add biases
                    for (int i = 0; i < biases.Length; i++)
                    {
                        parameters.Add(biases[i]);
                    }
                }
                else if (layer is LayerNormalizationLayer<TNumeric> layerNormLayer)
                {
                    // Add normalization parameters
                    var normParams = layerNormLayer.GetParameters();
                    foreach (var param in normParams)
                    {
                        parameters.Add(param);
                    }
                }
            }
            
            return new Vector<TNumeric>(parameters.ToArray());
        }
        
        /// <summary>
        /// Sets the parameters of the value function.
        /// </summary>
        /// <param name="parameters">The new parameter values.</param>
        public void SetParameters(Vector<TNumeric> parameters)
        {
            int paramIndex = 0;
            
            foreach (var layer in _layers)
            {
                if (layer is DenseLayer<TNumeric> denseLayer)
                {
                    var weights = denseLayer.GetWeights();
                    var biases = denseLayer.GetBiases();
                    
                    // Set weights
                    for (int i = 0; i < weights.Shape[0]; i++)
                    {
                        for (int j = 0; j < weights.Shape[1]; j++)
                        {
                            weights[i, j] = parameters[paramIndex++];
                        }
                    }
                    
                    // Set biases
                    for (int i = 0; i < biases.Length; i++)
                    {
                        biases[i] = parameters[paramIndex++];
                    }
                    
                    denseLayer.SetWeights(weights);
                    denseLayer.SetBiases(biases);
                }
                else if (layer is LayerNormalizationLayer<TNumeric> layerNormLayer)
                {
                    // Set normalization parameters
                    var normParamCount = layerNormLayer.ParameterCount;
                    var normParams = new Vector<TNumeric>(normParamCount);
                    
                    for (int i = 0; i < normParamCount; i++)
                    {
                        normParams[i] = parameters[paramIndex++];
                    }
                    
                    layerNormLayer.SetParameters(normParams);
                }
            }
        }
        
        /// <summary>
        /// Copies the parameters from another value function.
        /// </summary>
        /// <param name="source">The source value function from which to copy parameters.</param>
        public void CopyParametersFrom(IValueFunction<TStateType, TNumeric> source)
        {
            // Get parameters from source
            Vector<TNumeric> sourceParams = source.GetParameters();
            
            // Set parameters to this network
            SetParameters(sourceParams);
        }
        
        /// <summary>
        /// Performs a soft update of parameters from another value function.
        /// </summary>
        /// <param name="source">The source value function from which to update parameters.</param>
        /// <param name="tau">The soft update factor (between 0 and 1).</param>
        public void SoftUpdate(IValueFunction<TStateType, TNumeric> source, TNumeric tau)
        {
            // Get parameters from both networks
            Vector<TNumeric> sourceParams = source.GetParameters();
            Vector<TNumeric> targetParams = GetParameters();
            
            // Apply soft update formula: target = (1-tau)*target + tau*source
            for (int i = 0; i < targetParams.Length; i++)
            {
                targetParams[i] = NumOps.Add(
                    NumOps.Multiply(NumOps.Subtract(NumOps.One, tau), targetParams[i]),
                    NumOps.Multiply(tau, sourceParams[i]));
            }
            
            // Set updated parameters
            SetParameters(targetParams);
        }

        private Vector<TNumeric> Forward(TStateType state, TActionType action)
        {
            // Process the first layer separately (state only)
            Tensor<TNumeric> output = _layers[0].Forward(state);
            
            int layerIndex = 1;
            if (_useLayerNorm)
            {
                output = _layers[layerIndex].Forward(output);
                layerIndex++;
            }
            
            // Concatenate action with the output of the first layer
            Tensor<TNumeric> actionInput = Tensor<TNumeric>.FromVector(action);
            Tensor<TNumeric> combined = ConcatenateTensors(output, actionInput);
            
            // Process remaining layers
            for (; layerIndex < _layers.Count; layerIndex++)
            {
                output = _layers[layerIndex].Forward(combined);
                combined = output;
            }
            
            return output.ToVector();
        }
        
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
    /// Class for generating Gaussian noise for exploration.
    /// </summary>
    /// <typeparam name="TActionType">The type used to represent actions.</typeparam>
    /// <typeparam name="TNumeric">The numeric type used for calculations.</typeparam>
    public class GaussianNoiseStrategy<TActionType, TNumeric> : IExplorationStrategy<TActionType, TNumeric>
        where TActionType : Vector<TNumeric>
    {
        /// <summary>
        /// Gets the numeric operations for type TNumeric.
        /// </summary>
        protected INumericOperations<TNumeric> NumOps => MathHelper.GetNumericOperations<TNumeric>();
        private readonly TNumeric _stdDev;
        private readonly Random _random;
        private readonly long _decaySteps;
        private readonly TNumeric _decayRate;
        
        public bool IsContinuous => true;
        
        public TNumeric ExplorationRate { get; private set; }
        
        public GaussianNoiseStrategy(TNumeric stdDev, int? seed = null)
        {
            _stdDev = stdDev;
            _random = seed.HasValue ? new Random(seed.Value) : new Random();
            ExplorationRate = NumOps.One;
            _decayRate = NumOps.FromDouble(0.995); // Default decay rate
            _decaySteps = 1000000; // Default very long decay steps
        }
        
        public TActionType ApplyExploration(TActionType action, long step)
        {
            Decay(step);
            return ApplyNoise(action, ExplorationRate);
        }
        
        public TActionType ApplyNoise(TActionType action, TNumeric scale)
        {
            int actionDimension = action.Length;
            var noisyAction = new Vector<TNumeric>(actionDimension);
            
            for (int i = 0; i < actionDimension; i++)
            {
                // Generate Gaussian noise
                double u1 = 1.0 - _random.NextDouble();
                double u2 = 1.0 - _random.NextDouble();
                double randNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                
                TNumeric noise = NumOps.Multiply(NumOps.Multiply(NumOps.FromDouble(randNormal), _stdDev), scale);
                noisyAction[i] = NumOps.Add(action[i], noise);
            }
            
            return (TActionType)noisyAction;
        }
        
        public void Decay(long step)
        {
            if (step <= _decaySteps)
            {
                TNumeric decayFactor = NumOps.FromDouble(1.0 - (double)step / _decaySteps);
                ExplorationRate = MathHelper.Max(NumOps.FromDouble(0.01), decayFactor);
            }
            else
            {
                ExplorationRate = NumOps.FromDouble(0.01); // Minimum exploration rate
            }
        }
        
        public bool IsActive(long step)
        {
            return NumOps.GreaterThan(ExplorationRate, NumOps.FromDouble(0.001));
        }
        
        public void Reset()
        {
            ExplorationRate = NumOps.One;
        }
    }
}