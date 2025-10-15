namespace AiDotNet.ReinforcementLearning.Agents;

/// <summary>
/// Implements the Proximal Policy Optimization (PPO) algorithm for reinforcement learning.
/// </summary>
/// <typeparam name="TState">The type used to represent the environment state.</typeparam>
/// <typeparam name="TAction">The type used to represent actions.</typeparam>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PPO is a policy gradient method that uses a clipped surrogate objective to ensure 
/// that policy updates don't deviate too far from the previous policy, which improves 
/// stability and sample efficiency compared to standard policy gradient methods.
/// </para>
/// <para>
/// This implementation includes options for clipped value function loss, adaptive KL penalty,
/// mini-batch training, and early stopping based on excessive KL divergence.
/// </para>
/// </remarks>
public class PPOAgent<TState, TAction, T> : AgentBase<TState, TAction, T>
    where TState : Tensor<T>
{
    private readonly IPolicy<TState, TAction, T> _actor = default!;
    private readonly IValueFunction<TState, T> _critic = default!;
    private readonly T _actorLearningRate = default!;
    private readonly T _criticLearningRate = default!;
    private readonly T _entropyCoefficient = default!;
    private readonly T _valueLossCoefficient = default!;
    private readonly bool _useGAE;
    private readonly T _gaeParameter = default!;
    private readonly bool _normalizeAdvantages;
    private readonly bool _standardizeRewards;
    private readonly int _stepsPerUpdate;
    private readonly int _batchSize;
    private readonly int _minibatchSize;
    private readonly int _epochsPerBatch;
    private readonly T _clipParameter = default!;
    private readonly bool _useValueClipping;
    private readonly T _valueClipParameter = default!;
    private readonly bool _useKLPenalty;
    private readonly T _targetKL = default!;
    private readonly T _klCoefficient = default!;
    private readonly bool _useEarlyStoppingKL;
    private readonly T _earlyStoppingKLThreshold = default!;
    private readonly bool _useClipDecay;
    private readonly T _finalClipParameter = default!;
    private readonly int _clipDecayUpdates;
    private readonly bool _useAdaptiveLearningRate;
    private readonly T _learningRateDecreaseFactor = default!;
    private readonly T _klLearningRateThreshold = default!;

    private T _currentClipParameter = default!;
    private T _currentKLCoefficient = default!;
    private T _currentActorLearningRate = default!;
    private T _currentCriticLearningRate = default!;
    private int _updateCount;

    // Buffer to store experiences until it's time to update
    private readonly List<(TState state, TAction action, T reward, TState nextState, bool done)> _experienceBuffer;
    private readonly List<(TState state, TAction action, T logProb, T value, T advantage, T return_)> _optimizationBuffer;
    private int _stepsCollected;

    /// <summary>
    /// Initializes a new instance of the <see cref="PPOAgent{TState, TAction, T}"/> class.
    /// </summary>
    /// <param name="options">The options for the PPO algorithm.</param>
    public PPOAgent(PPOOptions<T> options)
        : base(options.Gamma, 0.0, options.BatchSize, options.Seed)
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
        _stepsPerUpdate = options.StepsPerUpdate;
        _batchSize = options.BatchSize;
        _minibatchSize = options.MinibatchSize;
        _epochsPerBatch = options.EpochsPerBatch;
        _clipParameter = NumOps.FromDouble(options.ClipParameter);
        _useValueClipping = options.UseValueClipping;
        _valueClipParameter = NumOps.FromDouble(options.ValueClipParameter);
        _useKLPenalty = options.UseKLPenalty;
        _targetKL = NumOps.FromDouble(options.TargetKL);
        _klCoefficient = NumOps.FromDouble(options.KLCoefficient);
        _useEarlyStoppingKL = options.UseEarlyStoppingKL;
        _earlyStoppingKLThreshold = NumOps.FromDouble(options.EarlyStoppingKLThreshold);
        _useClipDecay = options.UseClipDecay;
        _finalClipParameter = NumOps.FromDouble(options.FinalClipParameter);
        _clipDecayUpdates = options.ClipDecayUpdates;
        _useAdaptiveLearningRate = options.UseAdaptiveLearningRate;
        _learningRateDecreaseFactor = NumOps.FromDouble(options.LearningRateDecreaseFactor);
        _klLearningRateThreshold = NumOps.FromDouble(options.KLLearningRateThreshold);

        // Initialize current parameters
        _currentClipParameter = _clipParameter;
        _currentKLCoefficient = _klCoefficient;
        _currentActorLearningRate = _actorLearningRate;
        _currentCriticLearningRate = _criticLearningRate;
        _updateCount = 0;

        _experienceBuffer = new List<(TState, TAction, T, TState, bool)>();
        _optimizationBuffer = new List<(TState, TAction, T, T, T, T)>();
        _stepsCollected = 0;

        // Create actor and critic based on existing Actor-Critic implementation
        if (!options.IsContinuous)
        {
            if (typeof(TAction) != typeof(int))
            {
                throw new ArgumentException("For discrete action spaces, TAction must be int");
            }
        
            // Use the same actor and critic implementations from A2C
            var a2cAgent = new AdvantageActorCriticAgent<TState, TAction, T>(options);
            _actor = a2cAgent.GetActor();
            _critic = a2cAgent.GetCritic();
        }
        else
        {
            if (typeof(TAction) != typeof(Vector<T>))
            {
                throw new ArgumentException("For continuous action spaces, TAction must be Vector<T>");
            }
        
            // Use the same actor and critic implementations from A2C
            var a2cAgent = new AdvantageActorCriticAgent<TState, TAction, T>(options);
            _actor = a2cAgent.GetActor();
            _critic = a2cAgent.GetCritic();
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
        // Always use the policy to select actions during training
        // During evaluation, we might want to use the mode/mean instead
        if (isTraining)
        {
            return _actor.SelectAction(state);
        }
        else
        {
            // For evaluation, use the deterministic version (mode/mean)
            if (_actor.IsContinuous)
            {
                // For continuous actions, get the mean
                var policyOutput = _actor.EvaluatePolicy(state);
                if (policyOutput is ValueTuple<Vector<T>, Vector<T>> tuple)
                {
                    return (TAction)(object)tuple.Item1;
                }
            }
        
            // Default to stochastic selection
            return _actor.SelectAction(state);
        }
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

        // Check if it's time to update
        if (_stepsCollected >= _stepsPerUpdate)
        {
            UpdateNetworks();
            _stepsCollected = 0;
            _updateCount++;
        
            // Update parameters if needed
            if (_useClipDecay)
            {
                UpdateClipParameter();
            }
        }
    }

    /// <summary>
    /// Trains the agent on a batch of experiences.
    /// </summary>
    /// <param name="states">The states experienced.</param>
    /// <param name="actions">The actions taken.</param>
    /// <param name="rewards">The rewards received.</param>
    /// <param name="nextStates">The next states reached.</param>
    /// <param name="dones">Whether each transition led to a terminal state.</param>
    /// <returns>The loss value from training.</returns>
    public T Train(TState[] states, TAction[] actions, T[] rewards, TState[] nextStates, bool[] dones)
    {
        // Add all experiences to the buffer
        for (int i = 0; i < states.Length; i++)
        {
            Learn(states[i], actions[i], rewards[i], nextStates[i], dones[i]);
        }
    
        // Force an update if we have enough data
        if (_experienceBuffer.Count >= _minibatchSize)
        {
            UpdateNetworks();
            return GetLatestLoss();
        }
    
        return NumOps.Zero;
    }

    /// <summary>
    /// Gets the latest loss value from training.
    /// </summary>
    /// <returns>The latest loss value.</returns>
    public override T GetLatestLoss()
    {
        return LastLoss;
    }

    /// <summary>
    /// Updates the actor and critic networks using the PPO algorithm.
    /// </summary>
    private void UpdateNetworks()
    {
        if (_experienceBuffer.Count == 0)
            return;

        // Prepare the optimization buffer by calculating advantages and returns
        PrepareOptimizationBuffer();

        // Perform multiple epochs of updates on the collected data
        for (int epoch = 0; epoch < _epochsPerBatch; epoch++)
        {
            // Create mini-batches for training
            List<int> indices = Enumerable.Range(0, _optimizationBuffer.Count).ToList();
        
            // Shuffle the indices
            for (int i = indices.Count - 1; i > 0; i--)
            {
                int j = Random.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            // Train on mini-batches
            T totalActorLoss = NumOps.Zero;
            T totalCriticLoss = NumOps.Zero;
            T totalKL = NumOps.Zero;
            int batchCount = 0;

            for (int i = 0; i < indices.Count; i += _minibatchSize)
            {
                // Create a mini-batch
                var miniBatchIndices = indices.Skip(i).Take(_minibatchSize).ToList();
                if (miniBatchIndices.Count == 0)
                    continue;

                // Perform update on this mini-batch
                (T actorLoss, T criticLoss, T klDivergence) = UpdateOnMiniBatch(miniBatchIndices);
            
                totalActorLoss = NumOps.Add(totalActorLoss, actorLoss);
                totalCriticLoss = NumOps.Add(totalCriticLoss, criticLoss);
                totalKL = NumOps.Add(totalKL, klDivergence);
                batchCount++;

                // Check for early stopping based on KL divergence
                if (_useEarlyStoppingKL && NumOps.GreaterThan(klDivergence, _earlyStoppingKLThreshold))
                {
                    // Current update is causing too much policy change, stop early
                    break;
                }
            }

            // Calculate average losses and KL
            T avgActorLoss = batchCount > 0 ? NumOps.Divide(totalActorLoss, NumOps.FromDouble(batchCount)) : NumOps.Zero;
            T avgCriticLoss = batchCount > 0 ? NumOps.Divide(totalCriticLoss, NumOps.FromDouble(batchCount)) : NumOps.Zero;
            T avgKL = batchCount > 0 ? NumOps.Divide(totalKL, NumOps.FromDouble(batchCount)) : NumOps.Zero;

            // Adaptive learning rate based on KL divergence
            if (_useAdaptiveLearningRate && NumOps.GreaterThan(avgKL, _klLearningRateThreshold))
            {
                _currentActorLearningRate = NumOps.Multiply(_currentActorLearningRate, _learningRateDecreaseFactor);
                _currentCriticLearningRate = NumOps.Multiply(_currentCriticLearningRate, _learningRateDecreaseFactor);
            }

            // Adaptive KL penalty
            if (_useKLPenalty)
            {
                UpdateKLCoefficient(avgKL);
            }
        }

        // Clear the buffers after training
        _experienceBuffer.Clear();
        _optimizationBuffer.Clear();
    }

    /// <summary>
    /// Prepares the optimization buffer by calculating advantages and returns.
    /// </summary>
    private void PrepareOptimizationBuffer()
    {
        // Calculate advantages and returns
        (Vector<T> advantages, Vector<T> returns) = CalculateAdvantagesAndReturns();

        // Store state, action, old log prob, value, advantage, and return
        _optimizationBuffer.Clear();
        for (int i = 0; i < _experienceBuffer.Count; i++)
        {
            var (state, action, _, _, _) = _experienceBuffer[i];
        
            // Calculate log probability of the action under current policy
            T logProb = _actor.LogProbability(state, action);
        
            // Get value estimate for the state
            T value = _critic.PredictValue(state);
        
            // Store for optimization
            _optimizationBuffer.Add((state, action, logProb, value, advantages[i], returns[i]));
        }
    }

    /// <summary>
    /// Updates the networks on a single mini-batch.
    /// </summary>
    /// <param name="indices">The indices in the optimization buffer to use for this mini-batch.</param>
    /// <returns>A tuple containing (actorLoss, criticLoss, klDivergence).</returns>
    private (T actorLoss, T criticLoss, T klDivergence) UpdateOnMiniBatch(List<int> indices)
    {
        // Extract mini-batch data
        List<TState> states = new List<TState>();
        List<TAction> actions = new List<TAction>();
        List<T> oldLogProbs = new List<T>();
        List<T> oldValues = new List<T>();
        List<T> advantages = new List<T>();
        List<T> returns = new List<T>();

        foreach (int idx in indices)
        {
            var (state, action, logProb, value, advantage, return_) = _optimizationBuffer[idx];
            states.Add(state);
            actions.Add(action);
            oldLogProbs.Add(logProb);
            oldValues.Add(value);
            advantages.Add(advantage);
            returns.Add(return_);
        }

        // Update critic
        T criticLoss = UpdateCritic(states.ToArray(), returns.ToArray(), oldValues.ToArray());

        // Update actor
        (T actorLoss, T klDivergence) = UpdateActor(states.ToArray(), actions.ToArray(), advantages.ToArray(), oldLogProbs.ToArray());

        return (actorLoss, criticLoss, klDivergence);
    }

    /// <summary>
    /// Updates the critic (value function) network.
    /// </summary>
    /// <param name="states">The batch of states.</param>
    /// <param name="returns">The returns for each state.</param>
    /// <param name="oldValues">The old value estimates for each state.</param>
    /// <returns>The critic loss.</returns>
    private T UpdateCritic(TState[] states, T[] returns, T[] oldValues)
    {
        if (states.Length == 0)
            return NumOps.Zero;
    
        // Convert arrays to vectors for easier manipulation
        var returnsVector = new Vector<T>(returns);
    
        if (_useValueClipping)
        {
            // Create a vector to store the final targets
            var finalTargets = new Vector<T>(returns.Length);
        
            // Get current value predictions
            var currentValues = _critic.PredictValues(states);
        
            // Create a vector of old values
            var oldValuesVector = new Vector<T>(oldValues);
        
            // Calculate clipped values
            for (int i = 0; i < returns.Length; i++)
            {
                // Calculate value target using clipping
                // vf_clipped = old_value + clip(current_value - old_value, -clip_param, clip_param)
                T valueDiff = NumOps.Subtract(currentValues[i], oldValuesVector[i]);
                T clippedValueDiff = MathHelper.Clamp(valueDiff, NumOps.Negate(_valueClipParameter), _valueClipParameter);
                T clippedValue = NumOps.Add(oldValuesVector[i], clippedValueDiff);
            
                // Calculate losses for both unclipped and clipped values
                T unclippedDiff = NumOps.Subtract(currentValues[i], returnsVector[i]);
                T unclippedLoss = NumOps.Multiply(unclippedDiff, unclippedDiff);
            
                T clippedDiff = NumOps.Subtract(clippedValue, returnsVector[i]);
                T clippedLoss = NumOps.Multiply(clippedDiff, clippedDiff);
            
                // Use max of clipped and unclipped loss (pessimistic)
                T valueSign;
                T sqrtLoss;
            
                if (NumOps.GreaterThan(unclippedLoss, clippedLoss)) {
                    sqrtLoss = NumOps.Sqrt(unclippedLoss);
                    valueSign = NumOps.SignOrZero(NumOps.Subtract(currentValues[i], returnsVector[i]));
                } else {
                    sqrtLoss = NumOps.Sqrt(clippedLoss);
                    valueSign = NumOps.SignOrZero(NumOps.Subtract(clippedValue, returnsVector[i]));
                }
            
                finalTargets[i] = NumOps.Add(returnsVector[i], NumOps.Multiply(sqrtLoss, valueSign));
            }
        
            // Update critic with the calculated targets
            return _critic.Update(states, finalTargets);
        }
        else
        {
            // Simple MSE update without clipping
            return _critic.Update(states, returnsVector);
        }
    }

    /// <summary>
    /// Updates the actor (policy) network.
    /// </summary>
    /// <param name="states">The batch of states.</param>
    /// <param name="actions">The actions taken in each state.</param>
    /// <param name="advantages">The advantages for each action.</param>
    /// <param name="oldLogProbs">The log probabilities of each action under the old policy.</param>
    /// <returns>A tuple containing (actorLoss, klDivergence).</returns>
    private (T actorLoss, T klDivergence) UpdateActor(TState[] states, TAction[] actions, T[] advantages, T[] oldLogProbs)
    {
        if (states.Length == 0)
            return (NumOps.Zero, NumOps.Zero);

        // Calculate policy gradient loss for each state-action pair
        List<object> gradients = new List<object>();
        T totalLoss = NumOps.Zero;
        T totalKL = NumOps.Zero;

        for (int i = 0; i < states.Length; i++)
        {
            // Get current log probability
            T logProb = _actor.LogProbability(states[i], actions[i]);
        
            // Calculate probability ratio
            // r = exp(logProb - oldLogProb)
            T logProbDiff = NumOps.Subtract(logProb, oldLogProbs[i]);
            T probRatio = NumOps.Exp(logProbDiff);
        
            // Calculate surrogate loss
            T advantage = advantages[i];
            T surrogateLoss1 = NumOps.Multiply(probRatio, advantage);
        
            T clipLowerBound = NumOps.Subtract(NumOps.One, _currentClipParameter);
            T clipUpperBound = NumOps.Add(NumOps.One, _currentClipParameter);
            T clippedProbRatio = MathHelper.Clamp(probRatio, clipLowerBound, clipUpperBound);
            T surrogateLoss2 = NumOps.Multiply(clippedProbRatio, advantage);
        
            // PPO's clipped objective (negative because we're minimizing)
            T objectiveLoss = NumOps.Negate(MathHelper.Min(surrogateLoss1, surrogateLoss2));
        
            // Add entropy bonus (negative because we want to maximize entropy)
            T entropy = _actor.GetEntropy(states[i]);
            T entropyBonus = NumOps.Negate(NumOps.Multiply(_entropyCoefficient, entropy));
        
            // Add KL penalty if enabled
            T klPenalty = NumOps.Zero;
            if (_useKLPenalty)
            {
                // Approximate KL divergence: KL ≈ (logProb - oldLogProb)^2 / 2
                T approxKL = NumOps.Multiply(NumOps.Multiply(logProbDiff, logProbDiff), NumOps.FromDouble(0.5));
                klPenalty = NumOps.Multiply(_currentKLCoefficient, approxKL);
                totalKL = NumOps.Add(totalKL, approxKL);
            }
        
            // Total loss
            T totalActorLoss = NumOps.Add(NumOps.Add(objectiveLoss, entropyBonus), klPenalty);
            totalLoss = NumOps.Add(totalLoss, totalActorLoss);
        
            // Calculate gradient scale (using the negative loss since we're doing gradient descent)
            // and store gradient information for policy update
            gradients.Add((states[i], actions[i], NumOps.Negate(totalActorLoss), logProb));
        }
    
        // Apply gradients to update the policy
        _actor.UpdateParameters(gradients, _currentActorLearningRate);
    
        // Calculate average KL divergence across the batch
        T avgKL = states.Length > 0 ? NumOps.Divide(totalKL, NumOps.FromDouble(states.Length)) : NumOps.Zero;
    
        // Calculate average loss across the batch
        T avgLoss = states.Length > 0 ? NumOps.Divide(totalLoss, NumOps.FromDouble(states.Length)) : NumOps.Zero;
    
        return (avgLoss, avgKL);
    }

    /// <summary>
    /// Calculates advantages and returns for the collected experiences.
    /// </summary>
    /// <returns>A tuple containing the advantages and returns vectors.</returns>
    private (Vector<T> advantages, Vector<T> returns) CalculateAdvantagesAndReturns()
    {
        // Reuse the same logic from A2C
        var helper = new A2CHelper<TState, TAction, T>(
            _experienceBuffer,
            _critic,
            null, // No target critic for PPO
            Gamma,
            _useGAE,
            _gaeParameter,
            true, // Always use n-step returns in PPO
            Math.Min(_stepsPerUpdate, 10) // Use a reasonable n-step value
        );
    
        var (advantages, returns) = helper.CalculateAdvantagesAndReturns();
    
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
    /// Updates the clip parameter based on decay settings.
    /// </summary>
    private void UpdateClipParameter()
    {
        if (_clipDecayUpdates <= 0)
            return;
    
        // Linear decay from initial to final clip parameter
        T progress = MathHelper.Min(NumOps.One, NumOps.Divide(NumOps.FromDouble(_updateCount), NumOps.FromDouble(_clipDecayUpdates)));
        _currentClipParameter = NumOps.Add(_clipParameter, NumOps.Multiply(progress, NumOps.Subtract(_finalClipParameter, _clipParameter)));
    }

    /// <summary>
    /// Updates the KL coefficient based on the most recent KL divergence.
    /// </summary>
    /// <param name="kl">The KL divergence from the last update.</param>
    private void UpdateKLCoefficient(T kl)
    {
        T halfTargetKL = NumOps.Multiply(_targetKL, NumOps.FromDouble(0.5));
        if (NumOps.LessThan(kl, halfTargetKL))
        {
            // KL too low, reduce penalty
            _currentKLCoefficient = NumOps.Divide(_currentKLCoefficient, NumOps.FromDouble(1.5));
        }
        else 
        {
            T oneAndHalfTargetKL = NumOps.Multiply(_targetKL, NumOps.FromDouble(1.5));
            if (NumOps.GreaterThan(kl, oneAndHalfTargetKL))
            {
                // KL too high, increase penalty
                _currentKLCoefficient = NumOps.Multiply(_currentKLCoefficient, NumOps.FromDouble(1.5));
            }
        }
    
        // Ensure coefficient doesn't get too extreme
        T minCoef = NumOps.FromDouble(1e-4);
        T maxCoef = NumOps.FromDouble(10.0);
        _currentKLCoefficient = MathHelper.Clamp(_currentKLCoefficient, minCoef, maxCoef);
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
    /// Saves the agent's state to a file.
    /// </summary>
    /// <param name="filePath">The path where the agent's state should be saved.</param>
    public override void Save(string filePath)
    {
        // Save basic state information that can help with resuming training
        var lines = new List<string>
        {
            "PPOAgent",
            $"UpdateCount:{_updateCount}",
            $"StepsCollected:{_stepsCollected}",
            $"CurrentClipParameter:{_currentClipParameter}",
            $"CurrentKLCoefficient:{_currentKLCoefficient}",
            $"CurrentActorLearningRate:{_currentActorLearningRate}",
            $"CurrentCriticLearningRate:{_currentCriticLearningRate}",
            $"IsTraining:{IsTraining}",
            $"TotalSteps:{TotalSteps}",
            $"LastLoss:{LastLoss}"
        };
        
        System.IO.File.WriteAllLines(filePath, lines);
        
        // TODO: In a complete implementation, also save:
        // - Actor and critic network states (requires network serialization support)
        // - Experience buffer state for resuming mid-update
        // - All PPO hyperparameters for validation
    }

    /// <summary>
    /// Loads the agent's state from a file.
    /// </summary>
    /// <param name="filePath">The path from which to load the agent's state.</param>
    public override void Load(string filePath)
    {
        if (!System.IO.File.Exists(filePath))
        {
            throw new System.IO.FileNotFoundException($"Agent state file not found: {filePath}");
        }
        
        var lines = System.IO.File.ReadAllLines(filePath);
        if (lines.Length == 0 || lines[0] != "PPOAgent")
        {
            throw new InvalidOperationException("Invalid PPO agent state file");
        }
        
        // Parse saved state
        foreach (var line in lines.Skip(1))
        {
            var parts = line.Split(':');
            if (parts.Length != 2) continue;
            
            var key = parts[0];
            var value = parts[1];
            
            switch (key)
            {
                case "UpdateCount":
                    _updateCount = int.Parse(value);
                    break;
                case "StepsCollected":
                    _stepsCollected = int.Parse(value);
                    break;
                case "CurrentClipParameter":
                    _currentClipParameter = NumOps.FromDouble(double.Parse(value));
                    break;
                case "CurrentKLCoefficient":
                    _currentKLCoefficient = NumOps.FromDouble(double.Parse(value));
                    break;
                case "CurrentActorLearningRate":
                    _currentActorLearningRate = NumOps.FromDouble(double.Parse(value));
                    break;
                case "CurrentCriticLearningRate":
                    _currentCriticLearningRate = NumOps.FromDouble(double.Parse(value));
                    break;
                case "IsTraining":
                    SetTrainingMode(bool.Parse(value));
                    break;
                case "LastLoss":
                    LastLoss = NumOps.FromDouble(double.Parse(value));
                    break;
            }
        }
        
        // Clear buffers for a fresh start
        _experienceBuffer.Clear();
        _optimizationBuffer.Clear();
        
        // TODO: In a complete implementation, also load:
        // - Actor and critic network states
        // - Any other state needed for seamless resumption
    }

    /// <summary>
    /// Sets the agent's training mode.
    /// </summary>
    /// <param name="isTraining">A flag indicating whether the agent should be in training mode.</param>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
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
    }

    /// <summary>
    /// Helper class to calculate advantages and returns, reusing logic from A2C.
    /// </summary>
    /// <typeparam name="TStateType">The type used to represent states.</typeparam>
    /// <typeparam name="TActionType">The type used to represent actions.</typeparam>
    /// <typeparam name="TNumeric">The numeric type used for calculations.</typeparam>
    private class A2CHelper<TStateType, TActionType, TNumeric>
        where TStateType : Tensor<TNumeric>
    {
        /// <summary>
        /// Gets the numeric operations for type TNumeric.
        /// </summary>
        protected INumericOperations<TNumeric> NumOps => MathHelper.GetNumericOperations<TNumeric>();
        private readonly List<(TStateType state, TActionType action, TNumeric reward, TStateType nextState, bool done)> _buffer;
        private readonly IValueFunction<TStateType, TNumeric> _critic = default!;
        private readonly IValueFunction<TStateType, TNumeric>? _criticTarget;
        private readonly TNumeric Gamma;
        private readonly bool _useGAE;
        private readonly TNumeric _gaeParameter = default!;
        private readonly bool _useNStepReturns;
        private readonly int _nSteps;

        public A2CHelper(
            List<(TStateType, TActionType, TNumeric, TStateType, bool)> buffer,
            IValueFunction<TStateType, TNumeric> critic,
            IValueFunction<TStateType, TNumeric>? criticTarget,
            TNumeric gamma,
            bool useGAE,
            TNumeric gaeParameter,
            bool useNStepReturns,
            int nSteps)
        {
            _buffer = buffer;
            _critic = critic;
            _criticTarget = criticTarget;
            Gamma = gamma;
            _useGAE = useGAE;
            _gaeParameter = gaeParameter;
            _useNStepReturns = useNStepReturns;
            _nSteps = nSteps;
        }

        public (Vector<TNumeric> advantages, Vector<TNumeric> returns) CalculateAdvantagesAndReturns()
        {
            var advantages = new Vector<TNumeric>(_buffer.Count);
            var returns = new Vector<TNumeric>(_buffer.Count);

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
                for (int i = 0; i < _buffer.Count; i++)
                {
                    TNumeric baselineValue = _critic.PredictValue(_buffer[i].state);
                    advantages[i] = NumOps.Subtract(returns[i], baselineValue);
                }
            }
            else
            {
                // Simple one-step TD error as advantage
                for (int i = 0; i < _buffer.Count; i++)
                {
                    var (state, _, reward, nextState, done) = _buffer[i];
                
                    // Calculate the current state value
                    TNumeric currentValue = _critic.PredictValue(state);
                
                    // Calculate the next state value (or 0 if terminal)
                    TNumeric nextValue = done ? NumOps.Zero : (_criticTarget != null) 
                        ? _criticTarget.PredictValue(nextState) 
                        : _critic.PredictValue(nextState);
                
                    // Calculate the TD error (advantage)
                    advantages[i] = NumOps.Subtract(
                        NumOps.Add(reward, NumOps.Multiply(Gamma, nextValue)), 
                        currentValue);
                
                    // Return is the TD target
                    returns[i] = NumOps.Add(reward, NumOps.Multiply(Gamma, nextValue));
                }
            }

            return (advantages, returns);
        }

        private Vector<TNumeric> CalculateNStepReturns()
        {
            var returns = new Vector<TNumeric>(_buffer.Count);

            for (int i = 0; i < _buffer.Count; i++)
            {
                // Start with the immediate reward
                TNumeric nStepReturn = _buffer[i].reward;
            
                // Add discounted future rewards up to n steps
                TNumeric discountFactor = Gamma;
                for (int j = 1; j < _nSteps && (i + j) < _buffer.Count; j++)
                {
                    nStepReturn = NumOps.Add(nStepReturn, NumOps.Multiply(discountFactor, _buffer[i + j].reward));
                    discountFactor = NumOps.Multiply(discountFactor, Gamma);

                    // If we reach a terminal state, stop adding future rewards
                    if (_buffer[i + j].done)
                        break;
                }

                // Add the bootstrapped value if we haven't reached a terminal state or the end of buffer
                if ((i + _nSteps) < _buffer.Count && !_buffer[i + _nSteps - 1].done)
                {
                    TStateType nStepState = _buffer[i + _nSteps].state;
                    TNumeric nStepValue = (_criticTarget != null)
                        ? _criticTarget.PredictValue(nStepState)
                        : _critic.PredictValue(nStepState);
                    nStepReturn = NumOps.Add(nStepReturn, NumOps.Multiply(discountFactor, nStepValue));
                }

                returns[i] = nStepReturn;
            }

            return returns;
        }

        private Vector<TNumeric> CalculateGAEAdvantages()
        {
            var advantages = new Vector<TNumeric>(_buffer.Count);
        
            // Initialize the next state value and advantage for bootstrapping
            TNumeric nextValue = NumOps.Zero;
            TNumeric nextAdvantage = NumOps.Zero;
        
            // Calculate GAE in reverse order
            for (int i = _buffer.Count - 1; i >= 0; i--)
            {
                var (state, _, reward, nextState, done) = _buffer[i];
            
                // Calculate the current state value
                TNumeric currentValue = _critic.PredictValue(state);
            
                // Calculate the next state value (or 0 if terminal)
                if (!done)
                {
                    nextValue = (_criticTarget != null)
                        ? _criticTarget.PredictValue(nextState)
                        : _critic.PredictValue(nextState);
                }
                else
                {
                    nextValue = NumOps.Zero;
                }
            
                // Calculate the TD error
                TNumeric tdError = NumOps.Subtract(
                    NumOps.Add(reward, NumOps.Multiply(Gamma, nextValue)), 
                    currentValue);
            
                // Calculate the GAE advantage
                // A_t = δ_t + (γλ)A_{t+1}
                TNumeric gaeDiscount = NumOps.Multiply(Gamma, _gaeParameter);
                TNumeric futureAdvantage = done ? NumOps.Zero : nextAdvantage;
                advantages[i] = NumOps.Add(tdError, NumOps.Multiply(gaeDiscount, futureAdvantage));
            
                // Update next advantage for the next iteration
                nextAdvantage = advantages[i];
            }
        
            return advantages;
        }

        private Vector<TNumeric> CalculateReturns()
        {
            var returns = new Vector<TNumeric>(_buffer.Count);
            TNumeric discountedReturn = NumOps.Zero;

            // Calculate returns in reverse order
            for (int i = _buffer.Count - 1; i >= 0; i--)
            {
                var (_, _, reward, _, done) = _buffer[i];
            
                // Reset the return calculation if this is the start of a new episode
                if (done && i < _buffer.Count - 1)
                {
                    discountedReturn = NumOps.Zero;
                }
            
                // G_t = r_t + gamma * G_{t+1}
                discountedReturn = NumOps.Add(reward, NumOps.Multiply(Gamma, discountedReturn));
                returns[i] = discountedReturn;
            }

            return returns;
        }
    }
}