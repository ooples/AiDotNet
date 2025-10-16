using AiDotNet.ReinforcementLearning.Agents.Networks;
using AiDotNet.ReinforcementLearning.Models.Options;

namespace AiDotNet.ReinforcementLearning.Agents;

/// <summary>
/// Agent implementing the Model-Based Policy Optimization (MBPO) algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class MBPOAgent<T> : AgentBase<Tensor<T>, Vector<T>, T>
{
    private readonly MBPOOptions _options = default!;
    
    // Neural network components
    private readonly DynamicsModel<T>[] _dynamicsModels; // Ensemble of dynamics models
    private readonly PolicyNetwork<T> _policyNetwork;     // Policy network (actor)
    private readonly ValueNetwork<T>[] _valueNetworks;    // Ensemble of value networks (critics)
    
    // Replay buffers
    private readonly IReplayBuffer<Tensor<T>, Vector<T>, T> _realExperienceBuffer;  // Real experiences
    private readonly IReplayBuffer<Tensor<T>, Vector<T>, T> _modelExperienceBuffer; // Model-generated experiences
    
    // Temperature parameter for entropy regularization
    private T _temperature = default!;
    private T _targetEntropy = default!;
    private T _temperatureOptimizer = default!;
    
    // Loss tracking
    private T _lastPolicyLoss = default!;
    private T _lastValueLoss = default!;
    private T _lastModelLoss = default!;

    /// <summary>
    /// Initializes a new instance of the <see cref="MBPOAgent{T}"/> class.
    /// </summary>
    /// <param name="options">The options for configuring the MBPO algorithm.</param>
    public MBPOAgent(MBPOOptions options)
        : base(options.Gamma, options.Tau, options.BatchSize, options.Seed)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        
        // Initialize loss tracking
        _lastPolicyLoss = NumOps.Zero;
        _lastValueLoss = NumOps.Zero;
        _lastModelLoss = NumOps.Zero;
        
        // Initialize temperature parameter
        _temperature = NumOps.FromDouble(options.InitialTemperature);
        // Target entropy is -dimension of action space
        _targetEntropy = NumOps.FromDouble(-options.ActionSize);
        _temperatureOptimizer = NumOps.Zero;
        
        // Create dynamics model ensemble
        _dynamicsModels = new DynamicsModel<T>[options.EnsembleSize];
        for (int i = 0; i < options.EnsembleSize; i++)
        {
            _dynamicsModels[i] = new DynamicsModel<T>(
                options.StateSize,
                options.ActionSize,
                options.ModelHiddenSizes,
                options.ModelLearningRate,
                options.ProbabilisticModel);
        }
        
        // Create policy network
        _policyNetwork = new PolicyNetwork<T>(
            options.StateSize,
            options.ActionSize,
            options.PolicyHiddenSizes,
            options.PolicyLearningRate,
            options.IsContinuous);
        
        // Create value network ensemble (usually 2 for soft actor-critic)
        int valueEnsembleSize = 2;
        _valueNetworks = new ValueNetwork<T>[valueEnsembleSize];
        for (int i = 0; i < valueEnsembleSize; i++)
        {
            _valueNetworks[i] = new ValueNetwork<T>(
                options.StateSize,
                options.ActionSize,
                options.ValueHiddenSizes,
                options.ValueLearningRate,
                options.IsContinuous);
        }
        
        // Initialize replay buffers
        _realExperienceBuffer = new StandardReplayBuffer<Tensor<T>, Vector<T>, T>(
            options.ReplayBufferCapacity);
            
        _modelExperienceBuffer = new StandardReplayBuffer<Tensor<T>, Vector<T>, T>(
            options.ReplayBufferCapacity);
    }
    
    /// <summary>
    /// Selects an action for the given state.
    /// </summary>
    /// <param name="state">The current state observation.</param>
    /// <param name="isTraining">Whether the agent is in training mode.</param>
    /// <returns>The selected action.</returns>
    public override Vector<T> SelectAction(Tensor<T> state, bool isTraining = true)
    {
        // Get action from policy network with appropriate noise level
        return _policyNetwork.GetAction(state, isTraining);
    }
    
    /// <summary>
    /// Adds a real experience to the buffer.
    /// </summary>
    /// <param name="state">The state observation.</param>
    /// <param name="action">The action taken.</param>
    /// <param name="reward">The reward received.</param>
    /// <param name="nextState">The next state observation.</param>
    /// <param name="done">Whether the episode is done.</param>
    public void AddRealExperience(Tensor<T> state, Vector<T> action, T reward, Tensor<T> nextState, bool done)
    {
        _realExperienceBuffer.Add(state, action, reward, nextState, done);
    }
    
    /// <summary>
    /// Updates the agent's knowledge based on an experience tuple.
    /// </summary>
    /// <param name="state">The state before the action was taken.</param>
    /// <param name="action">The action that was taken.</param>
    /// <param name="reward">The reward received after taking the action.</param>
    /// <param name="nextState">The state after the action was taken.</param>
    /// <param name="done">A flag indicating whether the episode ended after this action.</param>
    public override void Learn(Tensor<T> state, Vector<T> action, T reward, Tensor<T> nextState, bool done)
    {
        // Add to real experience buffer
        AddRealExperience(state, action, reward, nextState, done);
        
        // Increment step counter
        IncrementStepCounter();
        
        // Check if we have enough samples to start learning
        if (_realExperienceBuffer.Size < BatchSize)
        {
            return;
        }
        
        // Train dynamics model
        TrainDynamicsModel();
        
        // Generate synthetic experiences if needed
        if (TotalSteps % _options.ModelRolloutFrequency == 0)
        {
            GenerateSyntheticExperiences(_options.NumSyntheticExperiences);
        }
        
        // Update policy and value networks
        if (_modelExperienceBuffer.Size >= BatchSize)
        {
            UpdatePolicyFromAllData();
        }
        else
        {
            UpdatePolicyFromRealData();
        }
        
        // Store loss for reporting
        LastLoss = _lastPolicyLoss;
    }
    
    /// <summary>
    /// Trains the dynamics model on the real experience data.
    /// </summary>
    /// <returns>The average loss across the model ensemble.</returns>
    public T TrainDynamicsModel()
    {
        // Check if we have enough data
        if (_realExperienceBuffer.Size < BatchSize)
        {
            return NumOps.Zero;
        }
        
        T totalLoss = NumOps.Zero;
        
        // Train each model in the ensemble
        for (int m = 0; m < _dynamicsModels.Length; m++)
        {
            T modelLoss = NumOps.Zero;
            
            // Train for multiple epochs
            for (int epoch = 0; epoch < _options.ModelEpochs; epoch++)
            {
                // Sample batch of real experiences
                var batch = _realExperienceBuffer.SampleBatch(_options.ModelBatchSize);
                
                // Train the model on this batch
                modelLoss = _dynamicsModels[m].Train(
                    batch.States, batch.Actions, batch.Rewards, batch.NextStates, batch.Dones);
            }
            
            totalLoss = NumOps.Add(totalLoss, modelLoss);
        }
        
        // Calculate average loss
        _lastModelLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(_dynamicsModels.Length));
        
        return _lastModelLoss;
    }
    
    /// <summary>
    /// Generates synthetic experiences using the dynamics model.
    /// </summary>
    /// <param name="numExperiences">The number of synthetic experiences to generate.</param>
    public void GenerateSyntheticExperiences(int numExperiences)
    {
        // Clear the model experience buffer if it's getting too full
        if (_modelExperienceBuffer.Size + numExperiences > _options.ReplayBufferCapacity)
        {
            _modelExperienceBuffer.Clear();
        }
        
        // Sample initial states from real buffer
        var initialBatch = _realExperienceBuffer.SampleBatch(numExperiences);
        
        // Generate rollouts from each initial state
        for (int i = 0; i < numExperiences; i++)
        {
            Tensor<T> currentState = initialBatch.States[i];
            
            // Start rollout chains
            if (_options.BranchingRollouts)
            {
                // Branching rollouts - create multiple possible futures
                GenerateBranchingRollout(currentState, _options.RolloutHorizon, _options.NumBranches);
            }
            else
            {
                // Standard rollouts - one chain per initial state
                GenerateRollout(currentState, _options.RolloutHorizon);
            }
        }
    }
    
    /// <summary>
    /// Generates a rollout of synthetic experiences starting from the given state.
    /// </summary>
    /// <param name="initialState">The initial state for the rollout.</param>
    /// <param name="horizon">The number of steps to roll out.</param>
    private void GenerateRollout(Tensor<T> initialState, int horizon)
    {
        Tensor<T> currentState = initialState;
        
        for (int h = 0; h < horizon; h++)
        {
            // Get action from policy
            Vector<T> action = _policyNetwork.GetAction(currentState, true);
            
            // Select a random dynamics model from the ensemble
            int modelIndex = Random.Next(_dynamicsModels.Length);
            var selectedModel = _dynamicsModels[modelIndex];
            
            // Predict next state and reward
            var (nextState, reward, done) = selectedModel.Forward(currentState, action);
            
            // Add to model experience buffer
            _modelExperienceBuffer.Add(currentState, action, reward, nextState, done);
            
            // Stop rollout if episode ended
            if (done)
            {
                break;
            }
            
            // Continue rollout from next state
            currentState = nextState;
        }
    }
    
    /// <summary>
    /// Generates a branching rollout of synthetic experiences starting from the given state.
    /// </summary>
    /// <param name="initialState">The initial state for the rollout.</param>
    /// <param name="horizon">The number of steps to roll out.</param>
    /// <param name="numBranches">The number of branches to create at each step.</param>
    private void GenerateBranchingRollout(Tensor<T> initialState, int horizon, int numBranches)
    {
        // Start with a single state
        List<Tensor<T>> currentStates = new List<Tensor<T>> { initialState };
        
        // For each step in the horizon
        for (int h = 0; h < horizon; h++)
        {
            List<Tensor<T>> nextStates = new List<Tensor<T>>();
            
            // For each current state
            foreach (var state in currentStates)
            {
                // Create branches
                for (int b = 0; b < numBranches; b++)
                {
                    // Get action from policy (with noise for diversity)
                    Vector<T> action = _policyNetwork.GetAction(state, true);
                    
                    // Select a random dynamics model from the ensemble
                    int modelIndex = Random.Next(_dynamicsModels.Length);
                    var selectedModel = _dynamicsModels[modelIndex];
                    
                    // Predict next state and reward
                    var (nextState, reward, done) = selectedModel.Forward(state, action);
                    
                    // Add to model experience buffer
                    _modelExperienceBuffer.Add(state, action, reward, nextState, done);
                    
                    // Add to next states if episode didn't end
                    if (!done)
                    {
                        nextStates.Add(nextState);
                    }
                }
            }
            
            // Stop if no valid next states
            if (nextStates.Count == 0)
            {
                break;
            }
            
            // Limit the number of branches for computational efficiency
            int maxBranches = 10; // Arbitrary limit
            if (nextStates.Count > maxBranches)
            {
                nextStates = nextStates.Take(maxBranches).ToList();
            }
            
            // Update current states for next iteration
            currentStates = nextStates;
        }
    }
    
    /// <summary>
    /// Updates the policy network using only real experiences.
    /// </summary>
    /// <returns>The policy loss from the update.</returns>
    public T UpdatePolicyFromRealData()
    {
        // Check if we have enough data
        if (_realExperienceBuffer.Size < BatchSize)
        {
            return NumOps.Zero;
        }
        
        T totalPolicyLoss = NumOps.Zero;
        T totalValueLoss = NumOps.Zero;
        
        // Train for multiple epochs
        for (int epoch = 0; epoch < _options.PolicyEpochs; epoch++)
        {
            // Sample batch of real experiences
            var batch = _realExperienceBuffer.SampleBatch(BatchSize);
            
            // Update value networks
            T valueLoss = UpdateValueNetworks(batch.States, batch.Actions, batch.Rewards, batch.NextStates, batch.Dones);
            
            // Update policy network
            T policyLoss = UpdatePolicyNetwork(batch.States);
            
            // Update temperature parameter if auto-tuning is enabled
            if (_options.AutoTuneEntropy)
            {
                UpdateTemperature(batch.States);
            }
            
            totalPolicyLoss = NumOps.Add(totalPolicyLoss, policyLoss);
            totalValueLoss = NumOps.Add(totalValueLoss, valueLoss);
        }
        
        // Calculate average losses
        _lastPolicyLoss = NumOps.Divide(totalPolicyLoss, NumOps.FromDouble(_options.PolicyEpochs));
        _lastValueLoss = NumOps.Divide(totalValueLoss, NumOps.FromDouble(_options.PolicyEpochs));
        
        return _lastPolicyLoss;
    }
    
    /// <summary>
    /// Updates the policy and value networks using both real and model-generated experiences.
    /// </summary>
    /// <returns>The policy loss from the update.</returns>
    public T UpdatePolicyFromAllData()
    {
        // Check if we have enough data
        if (_realExperienceBuffer.Size < BatchSize || _modelExperienceBuffer.Size < BatchSize)
        {
            return UpdatePolicyFromRealData(); // Fall back to real data only
        }
        
        T totalPolicyLoss = NumOps.Zero;
        T totalValueLoss = NumOps.Zero;
        
        // Train for multiple epochs
        for (int epoch = 0; epoch < _options.PolicyEpochs; epoch++)
        {
            // Sample batch from real experiences
            var realBatch = _realExperienceBuffer.SampleBatch(BatchSize);
            
            // Sample batch from model-generated experiences
            var modelBatch = _modelExperienceBuffer.SampleBatch(BatchSize);
            
            // Combine batches
            var combinedStates = CombineTensors(realBatch.States, modelBatch.States);
            var combinedActions = CombineVectors(realBatch.Actions, modelBatch.Actions);
            var combinedRewards = CombineVectors(realBatch.Rewards, modelBatch.Rewards);
            var combinedNextStates = CombineTensors(realBatch.NextStates, modelBatch.NextStates);
            var combinedDones = CombineVectors(realBatch.Dones, modelBatch.Dones);
            
            // Update value networks
            T valueLoss = UpdateValueNetworks(
                combinedStates, combinedActions, combinedRewards, combinedNextStates, combinedDones);
            
            // Update policy network
            T policyLoss = UpdatePolicyNetwork(combinedStates);
            
            // Update temperature parameter if auto-tuning is enabled
            if (_options.AutoTuneEntropy)
            {
                UpdateTemperature(combinedStates);
            }
            
            totalPolicyLoss = NumOps.Add(totalPolicyLoss, policyLoss);
            totalValueLoss = NumOps.Add(totalValueLoss, valueLoss);
        }
        
        // Calculate average losses
        _lastPolicyLoss = NumOps.Divide(totalPolicyLoss, NumOps.FromDouble(_options.PolicyEpochs));
        _lastValueLoss = NumOps.Divide(totalValueLoss, NumOps.FromDouble(_options.PolicyEpochs));
        
        return _lastPolicyLoss;
    }
    
    /// <summary>
    /// Updates the value networks based on the current policy.
    /// </summary>
    /// <param name="states">Batch of states.</param>
    /// <param name="actions">Batch of actions.</param>
    /// <param name="rewards">Batch of rewards.</param>
    /// <param name="nextStates">Batch of next states.</param>
    /// <param name="dones">Batch of done flags.</param>
    /// <returns>The average value loss.</returns>
    private T UpdateValueNetworks(
        Tensor<T>[] states,
        Vector<T>[] actions,
        T[] rewards,
        Tensor<T>[] nextStates,
        bool[] dones)
    {
        int batchSize = states.Length;
        T totalLoss = NumOps.Zero;
        
        // Get next actions and log probs from current policy
        var (nextActions, nextLogProbs) = _policyNetwork.GetActionAndLogProb(nextStates);
        
        // Calculate target Q-values
        var targetQValues = new T[batchSize];
        
        for (int i = 0; i < batchSize; i++)
        {
            // Skip if done
            if (dones[i])
            {
                targetQValues[i] = rewards[i];
                continue;
            }
            
            // Get Q-values from both target networks
            T q1 = _valueNetworks[0].GetValue(nextStates[i], nextActions[i]);
            T q2 = _valueNetworks[1].GetValue(nextStates[i], nextActions[i]);
            
            // Take minimum Q-value for robustness
            T minQ = NumOps.LessThan(q1, q2) ? q1 : q2;
            
            // Subtract entropy term (from log_prob)
            T entropyBonus = NumOps.Multiply(_temperature, nextLogProbs[i]);
            T valueWithEntropy = NumOps.Subtract(minQ, entropyBonus);
            
            // Calculate target using Bellman equation
            T discountedValue = NumOps.Multiply(Gamma, valueWithEntropy);
            targetQValues[i] = NumOps.Add(rewards[i], discountedValue);
        }
        
        // Update each value network
        for (int v = 0; v < _valueNetworks.Length; v++)
        {
            T loss = _valueNetworks[v].Update(states, actions, targetQValues);
            totalLoss = NumOps.Add(totalLoss, loss);
        }
        
        // Return average loss
        return NumOps.Divide(totalLoss, NumOps.FromDouble(_valueNetworks.Length));
    }
    
    /// <summary>
    /// Updates the policy network based on the current value networks.
    /// </summary>
    /// <param name="states">Batch of states.</param>
    /// <returns>The policy loss.</returns>
    private T UpdatePolicyNetwork(Tensor<T>[] states)
    {
        int batchSize = states.Length;
        
        // Get actions and log probs from current policy
        var (actions, logProbs) = _policyNetwork.GetActionAndLogProb(states);
        
        // Calculate Q-values for these actions
        var qValues = new T[batchSize];
        
        for (int i = 0; i < batchSize; i++)
        {
            // Get Q-values from both networks
            T q1 = _valueNetworks[0].GetValue(states[i], actions[i]);
            T q2 = _valueNetworks[1].GetValue(states[i], actions[i]);
            
            // Take minimum Q-value for robustness
            qValues[i] = NumOps.LessThan(q1, q2) ? q1 : q2;
        }
        
        // Update policy to maximize Q-value minus entropy term
        return _policyNetwork.Update(states, actions, logProbs, qValues, _temperature);
    }
    
    /// <summary>
    /// Updates the temperature parameter for entropy regularization.
    /// </summary>
    /// <param name="states">Batch of states.</param>
    private void UpdateTemperature(Tensor<T>[] states)
    {
        // Get actions and log probs from current policy
        var (_, logProbs) = _policyNetwork.GetActionAndLogProb(states);
        
        // Calculate average log prob
        T sumLogProb = NumOps.Zero;
        for (int i = 0; i < logProbs.Length; i++)
        {
            sumLogProb = NumOps.Add(sumLogProb, logProbs[i]);
        }
        T avgLogProb = NumOps.Divide(sumLogProb, NumOps.FromDouble(logProbs.Length));
        
        // Calculate current entropy
        T negAvgLogProb = NumOps.Negate(avgLogProb);
        
        // Calculate difference from target entropy
        T entropyDiff = NumOps.Subtract(negAvgLogProb, _targetEntropy);
        
        // Calculate temperature loss
        T temperatureLoss = NumOps.Multiply(_temperature, entropyDiff);
        
        // Update temperature using gradient descent
        T learningRate = NumOps.FromDouble(0.001);
        T gradientStep = NumOps.Multiply(learningRate, temperatureLoss);
        _temperature = NumOps.Subtract(_temperature, gradientStep);
        
        // Ensure temperature stays positive
        if (NumOps.LessThan(_temperature, NumOps.FromDouble(0.001)))
        {
            _temperature = NumOps.FromDouble(0.001);
        }
    }
    
    /// <summary>
    /// Predicts future states using the dynamics model.
    /// </summary>
    /// <param name="state">The starting state.</param>
    /// <param name="action">The action to take.</param>
    /// <param name="numSteps">The number of steps to predict.</param>
    /// <returns>A list of predicted states.</returns>
    public List<Tensor<T>> PredictFutureStates(Tensor<T> state, Vector<T> action, int numSteps)
    {
        var predictions = new List<Tensor<T>> { state };
        Tensor<T> currentState = state;
        
        for (int i = 0; i < numSteps; i++)
        {
            // Use ensemble mean prediction
            Tensor<T>? nextStateSum = null;
            
            for (int m = 0; m < _dynamicsModels.Length; m++)
            {
                var (nextState, _, _) = _dynamicsModels[m].Forward(currentState, action);
                
                if (nextStateSum == null)
                {
                    nextStateSum = nextState.Clone();
                }
                else
                {
                    // Add to sum
                    for (int j = 0; j < nextState.Length; j++)
                    {
                        var index = nextState.GetIndexFromFlat(j);
                        nextStateSum[index] = NumOps.Add(nextStateSum[index], nextState[index]);
                    }
                }
            }

            if (nextStateSum == null)
            {
                throw new InvalidOperationException("Dynamics model predictions failed.");
            }

            // Calculate mean
            for (int j = 0; j < nextStateSum.Length; j++)
            {
                var index = nextStateSum.GetIndexFromFlat(j);
                nextStateSum[index] = NumOps.Divide(nextStateSum[index], 
                    NumOps.FromDouble(_dynamicsModels.Length));
            }
            
            // Add to predictions
            predictions.Add(nextStateSum);
            
            // Get next action from policy
            action = _policyNetwork.GetAction(nextStateSum, false);
            
            // Continue from predicted state
            currentState = nextStateSum;
        }
        
        return predictions;
    }
    
    /// <summary>
    /// Gets the uncertainty in predictions from the dynamics model ensemble.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <param name="action">The action to evaluate.</param>
    /// <returns>A measure of prediction uncertainty.</returns>
    public T GetPredictionUncertainty(Tensor<T> state, Vector<T> action)
    {
        // Get predictions from all models
        var predictions = new List<Tensor<T>>();
        
        for (int m = 0; m < _dynamicsModels.Length; m++)
        {
            var (nextState, _, _) = _dynamicsModels[m].Forward(state, action);
            predictions.Add(nextState);
        }
        
        // Calculate mean prediction
        var mean = predictions[0].Clone();
        
        for (int i = 1; i < predictions.Count; i++)
        {
            for (int j = 0; j < mean.Length; j++)
            {
                var index = mean.GetIndexFromFlat(j);
                mean[index] = NumOps.Add(mean[index], predictions[i][index]);
            }
        }
        
        for (int j = 0; j < mean.Length; j++)
        {
            var index = mean.GetIndexFromFlat(j);
            mean[index] = NumOps.Divide(mean[index], NumOps.FromDouble(predictions.Count));
        }
        
        // Calculate variance
        T totalVariance = NumOps.Zero;
        
        for (int i = 0; i < predictions.Count; i++)
        {
            for (int j = 0; j < mean.Length; j++)
            {
                var index = mean.GetIndexFromFlat(j);
                T diff = NumOps.Subtract(predictions[i][index], mean[index]);
                T squaredDiff = NumOps.Multiply(diff, diff);
                totalVariance = NumOps.Add(totalVariance, squaredDiff);
            }
        }
        
        // Normalize by number of elements
        totalVariance = NumOps.Divide(totalVariance, 
            NumOps.FromDouble(predictions.Count * mean.Length));
        
        // Return standard deviation as uncertainty measure
        return NumOps.Sqrt(totalVariance);
    }
    
    /// <summary>
    /// Gets the agent's statistics.
    /// </summary>
    /// <returns>A dictionary of statistics.</returns>
    public Dictionary<string, double> GetStats()
    {
        var stats = new Dictionary<string, double>
        {
            { "PolicyLoss", Convert.ToDouble(_lastPolicyLoss) },
            { "ValueLoss", Convert.ToDouble(_lastValueLoss) },
            { "ModelLoss", Convert.ToDouble(_lastModelLoss) },
            { "Temperature", Convert.ToDouble(_temperature) },
            { "RealBufferSize", _realExperienceBuffer.Size },
            { "ModelBufferSize", _modelExperienceBuffer.Size },
            { "TotalSteps", TotalSteps }
        };
        
        return stats;
    }
    
    /// <summary>
    /// Saves the agent's state to a file.
    /// </summary>
    /// <param name="filePath">The path where the agent's state should be saved.</param>
    public override void Save(string filePath)
    {
        // Save policy network
        _policyNetwork.SaveToFile($"{filePath}_policy");
        
        // Save value networks
        for (int i = 0; i < _valueNetworks.Length; i++)
        {
            _valueNetworks[i].Save($"{filePath}_value_{i}");
        }
        
        // Save dynamics models
        for (int i = 0; i < _dynamicsModels.Length; i++)
        {
            _dynamicsModels[i].Save($"{filePath}_dynamics_{i}");
        }
    }

    /// <summary>
    /// Loads the agent's state from a file.
    /// </summary>
    /// <param name="filePath">The path from which to load the agent's state.</param>
    public override void Load(string filePath)
    {
        // Load policy network
        _policyNetwork.LoadFromFile($"{filePath}_policy");
        
        // Load value networks
        for (int i = 0; i < _valueNetworks.Length; i++)
        {
            _valueNetworks[i].Load($"{filePath}_value_{i}");
        }
        
        // Load dynamics models
        for (int i = 0; i < _dynamicsModels.Length; i++)
        {
            _dynamicsModels[i].Load($"{filePath}_dynamics_{i}");
        }
    }
    
    /// <summary>
    /// Gets the agent's parameters as a single vector.
    /// </summary>
    /// <returns>A vector containing all parameters of the agent.</returns>
    public Vector<T> GetParameters()
    {
        var allParameters = new List<Vector<T>>();
        
        // Add dynamics model parameters
        for (int m = 0; m < _dynamicsModels.Length; m++)
        {
            allParameters.Add(_dynamicsModels[m].GetParameters());
        }
        
        // Add policy network parameters
        allParameters.Add(_policyNetwork.GetParameters());
        
        // Add value network parameters
        for (int v = 0; v < _valueNetworks.Length; v++)
        {
            allParameters.Add(_valueNetworks[v].GetParameters());
        }
        
        // Combine all parameters
        return ConcatenateVectors(allParameters);
    }
    
    /// <summary>
    /// Sets the agent's parameters from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    public void SetParameters(Vector<T> parameters)
    {
        int index = 0;
        
        // Set dynamics model parameters
        for (int m = 0; m < _dynamicsModels.Length; m++)
        {
            int modelParamSize = _dynamicsModels[m].GetParameters().Length;
            var modelParams = ExtractVector(parameters, index, modelParamSize);
            _dynamicsModels[m].SetParameters(modelParams);
            index += modelParamSize;
        }
        
        // Set policy network parameters
        int policyParamSize = _policyNetwork.GetParameters().Length;
        var policyParams = ExtractVector(parameters, index, policyParamSize);
        _policyNetwork.SetParameters(policyParams);
        index += policyParamSize;
        
        // Set value network parameters
        for (int v = 0; v < _valueNetworks.Length; v++)
        {
            int valueParamSize = _valueNetworks[v].GetParameters().Length;
            var valueParams = ExtractVector(parameters, index, valueParamSize);
            _valueNetworks[v].SetParameters(valueParams);
            index += valueParamSize;
        }
    }
    
    /// <summary>
    /// Combines two arrays of tensors into a single array.
    /// </summary>
    private Tensor<T>[] CombineTensors(Tensor<T>[] a, Tensor<T>[] b)
    {
        var result = new Tensor<T>[a.Length + b.Length];
        Array.Copy(a, result, a.Length);
        Array.Copy(b, 0, result, a.Length, b.Length);
        return result;
    }
    
    /// <summary>
    /// Combines two arrays of vectors into a single array.
    /// </summary>
    private Vector<T>[] CombineVectors(Vector<T>[] a, Vector<T>[] b)
    {
        var result = new Vector<T>[a.Length + b.Length];
        Array.Copy(a, result, a.Length);
        Array.Copy(b, 0, result, a.Length, b.Length);
        return result;
    }
    
    /// <summary>
    /// Combines two arrays of values into a single array.
    /// </summary>
    private T[] CombineVectors(T[] a, T[] b)
    {
        var result = new T[a.Length + b.Length];
        Array.Copy(a, result, a.Length);
        Array.Copy(b, 0, result, a.Length, b.Length);
        return result;
    }
    
    /// <summary>
    /// Combines two arrays of flags into a single array.
    /// </summary>
    private bool[] CombineVectors(bool[] a, bool[] b)
    {
        var result = new bool[a.Length + b.Length];
        Array.Copy(a, result, a.Length);
        Array.Copy(b, 0, result, a.Length, b.Length);
        return result;
    }
    
    /// <summary>
    /// Concatenates a list of vectors into a single vector.
    /// </summary>
    private Vector<T> ConcatenateVectors(List<Vector<T>> vectors)
    {
        // Calculate total length
        int totalLength = 0;
        foreach (var vector in vectors)
        {
            totalLength += vector.Length;
        }
        
        // Create result vector
        var result = new Vector<T>(totalLength);
        
        // Copy values
        int index = 0;
        foreach (var vector in vectors)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                result[index++] = vector[i];
            }
        }
        
        return result;
    }
    
    /// <summary>
    /// Extracts a portion of a vector.
    /// </summary>
    private Vector<T> ExtractVector(Vector<T> source, int startIndex, int length)
    {
        var result = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            result[i] = source[startIndex + i];
        }
        return result;
    }
}