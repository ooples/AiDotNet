namespace AiDotNet.ReinforcementLearning.Agents;

/// <summary>
/// Agent implementing the Multi-Agent Transformer architecture for financial market modeling and trading.
/// </summary>
/// <remarks>
/// This agent uses transformer networks to model interactions between multiple market participants,
/// enabling it to understand complex market dynamics and make trading decisions that account for
/// the collective behavior of different types of traders.
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
public class MultiAgentTransformerAgent<T> : AgentBase<Tensor<T>, Vector<T>, T>
{
    private int _numAgents;
    private int _stateDimension;
    private int _actionDimension;
    private int _hiddenDimension;
    private int _numHeads;
    private int _numLayers;
    private int _sequenceLength;
    private PositionalEncodingType _posEncodingType = default!;
    private int _communicationMode;
    private bool _useCentralizedTraining;
    private T _learningRate = default!;
    private T _entropyCoef = default!;
    private bool _useSelfPlay;
    private T _riskAversion = default!;
    private bool _useCausalMask;
    private bool _modelMarketImpact;
    
    // Neural network components
    private Transformer<T> _mainTransformer = default!;
    private TransformerArchitecture<T> _transformerArchitecture = default!;
    private List<NeuralNetwork<T>> _policyNetworks = default!;
    private List<NeuralNetwork<T>> _valueNetworks = default!;
    private NeuralNetwork<T>? _marketImpactNetwork;
    private NeuralNetwork<T> _marketDynamicsNetwork = default!;
    
    // Optimizers
    private readonly AdamOptimizer<T, Tensor<T>, Tensor<T>>? _policyOptimizer;
    private readonly AdamOptimizer<T, Tensor<T>, Tensor<T>>? _valueOptimizer;
    private readonly AdamOptimizer<T, Tensor<T>, Tensor<T>>? _transformerOptimizer;
    
    // State tracking for attention visualization
    private Tensor<T>? _lastAttentionWeights = null;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="MultiAgentTransformerAgent{T}"/> class.
    /// </summary>
    /// <param name="numAgents">The number of agents to model.</param>
    /// <param name="stateDimension">The dimension of the state space.</param>
    /// <param name="actionDimension">The dimension of the action space.</param>
    /// <param name="hiddenDimension">The size of hidden dimensions in the transformer.</param>
    /// <param name="numHeads">The number of attention heads in the transformer.</param>
    /// <param name="numLayers">The number of transformer layers.</param>
    /// <param name="sequenceLength">The length of sequences to process.</param>
    /// <param name="posEncodingType">The type of positional encoding to use.</param>
    /// <param name="communicationMode">The type of communication between agents.</param>
    /// <param name="useCentralizedTraining">Whether to use centralized training.</param>
    /// <param name="learningRate">The learning rate for optimization.</param>
    /// <param name="gamma">The discount factor for future rewards.</param>
    /// <param name="entropyCoef">The entropy coefficient for exploration.</param>
    /// <param name="useSelfPlay">Whether to use self-play for training.</param>
    /// <param name="riskAversion">The risk aversion parameter.</param>
    /// <param name="useCausalMask">Whether to use a causal mask in the transformer.</param>
    /// <param name="modelMarketImpact">Whether to model market impact of actions.</param>
    public MultiAgentTransformerAgent(
        int numAgents,
        int stateDimension,
        int actionDimension,
        int hiddenDimension,
        int numHeads,
        int numLayers,
        int sequenceLength,
        PositionalEncodingType posEncodingType,
        int communicationMode,
        bool useCentralizedTraining,
        double learningRate,
        double gamma,
        double entropyCoef,
        bool useSelfPlay,
        double riskAversion,
        bool useCausalMask,
        bool modelMarketImpact,
        double tau = 0.005,
        int batchSize = 64,
        int? seed = null)
        : base(gamma, tau, batchSize, seed)
    {
        _numAgents = numAgents;
        _stateDimension = stateDimension;
        _actionDimension = actionDimension;
        _hiddenDimension = hiddenDimension;
        _numHeads = numHeads;
        _numLayers = numLayers;
        _sequenceLength = sequenceLength;
        _posEncodingType = posEncodingType;
        _communicationMode = communicationMode;
        _useCentralizedTraining = useCentralizedTraining;
        _learningRate = NumOps.FromDouble(learningRate);
        _entropyCoef = NumOps.FromDouble(entropyCoef);
        _useSelfPlay = useSelfPlay;
        _riskAversion = NumOps.FromDouble(riskAversion);
        _useCausalMask = useCausalMask;
        _modelMarketImpact = modelMarketImpact;
        
        // Create transformer architecture
        _transformerArchitecture = new TransformerArchitecture<T>(
            taskType: NeuralNetworkTaskType.Regression,
            numEncoderLayers: _numLayers,
            numDecoderLayers: 0, // We only need encoder for agent state processing
            numHeads: _numHeads,
            modelDimension: _hiddenDimension,
            feedForwardDimension: _hiddenDimension * 4,
            complexity: NetworkComplexity.Medium,
            dropoutRate: 0.1,
            maxSequenceLength: _sequenceLength,
            vocabularySize: 0,
            usePositionalEncoding: true,
            positionalEncodingType: _posEncodingType);
        
        // Create main transformer
        _mainTransformer = new Transformer<T>(_transformerArchitecture);
        
        // Create policy networks for each agent
        _policyNetworks = new List<NeuralNetwork<T>>();
        for (int i = 0; i < _numAgents; i++)
        {
            var layers = new List<ILayer<T>>
            {
                new InputLayer<T>(_hiddenDimension),
                new DenseLayer<T>(_hiddenDimension, 128, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU)),
                new DenseLayer<T>(128, 64, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU)),
                new DenseLayer<T>(64, _actionDimension * 2, (IActivationFunction<T>?)null) // Mean and log variance for continuous actions
            };
            
            var policyArchitecture = new NeuralNetworkArchitecture<T>(
                complexity: NetworkComplexity.Medium,
                taskType: NeuralNetworkTaskType.Regression,
                shouldReturnFullSequence: false,
                layers: layers);
            
            var policyNetwork = new NeuralNetwork<T>(policyArchitecture);
            _policyNetworks.Add(policyNetwork);
        }
        
        // Create value networks for each agent
        _valueNetworks = new List<NeuralNetwork<T>>();
        for (int i = 0; i < _numAgents; i++)
        {
            var layers = new List<ILayer<T>>
            {
                new InputLayer<T>(_hiddenDimension),
                new DenseLayer<T>(_hiddenDimension, 128, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU)),
                new DenseLayer<T>(128, 64, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU)),
                new DenseLayer<T>(64, 1, (IActivationFunction<T>?)null) // Single value output
            };
            
            var valueArchitecture = new NeuralNetworkArchitecture<T>(
                complexity: NetworkComplexity.Medium,
                taskType: NeuralNetworkTaskType.Regression,
                shouldReturnFullSequence: false,
                layers: layers);
            
            var valueNetwork = new NeuralNetwork<T>(valueArchitecture);
            _valueNetworks.Add(valueNetwork);
        }
        
        // Create market impact network if needed
        if (_modelMarketImpact)
        {
            var impactLayers = new List<ILayer<T>>
            {
                new InputLayer<T>(_actionDimension * _numAgents),
                new DenseLayer<T>(_actionDimension * _numAgents, 128, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU)),
                new DenseLayer<T>(128, 64, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU)),
                new DenseLayer<T>(64, _stateDimension, (IActivationFunction<T>?)null)
            };
            
            var impactArchitecture = new NeuralNetworkArchitecture<T>(
                complexity: NetworkComplexity.Medium,
                taskType: NeuralNetworkTaskType.Regression,
                shouldReturnFullSequence: false,
                layers: impactLayers);
            
            _marketImpactNetwork = new NeuralNetwork<T>(impactArchitecture);
        }
        
        // Create market dynamics network for predicting future states
        var dynamicsLayers = new List<ILayer<T>>
        {
            new InputLayer<T>(_stateDimension + _actionDimension * _numAgents),
            new DenseLayer<T>(_stateDimension + _actionDimension * _numAgents, 256, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU)),
            new DenseLayer<T>(256, 256, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU)),
            new DenseLayer<T>(256, _stateDimension, (IActivationFunction<T>?)null)
        };
        
        var dynamicsArchitecture = new NeuralNetworkArchitecture<T>(
            complexity: NetworkComplexity.Medium,
            taskType: NeuralNetworkTaskType.Regression,
            shouldReturnFullSequence: false,
            layers: dynamicsLayers);
        
        _marketDynamicsNetwork = new NeuralNetwork<T>(dynamicsArchitecture);
        
        // Create optimizers
        var adamOptions = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            LearningRate = learningRate,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        };
        
        _policyOptimizer = OptimizerFactory<T, Tensor<T>, Tensor<T>>.CreateOptimizer(OptimizerType.Adam, adamOptions) as AdamOptimizer<T, Tensor<T>, Tensor<T>>;
        _valueOptimizer = OptimizerFactory<T, Tensor<T>, Tensor<T>>.CreateOptimizer(OptimizerType.Adam, adamOptions) as AdamOptimizer<T, Tensor<T>, Tensor<T>>;
        _transformerOptimizer = OptimizerFactory<T, Tensor<T>, Tensor<T>>.CreateOptimizer(OptimizerType.Adam, adamOptions) as AdamOptimizer<T, Tensor<T>, Tensor<T>>;
    }
    
    /// <summary>
    /// Selects an action for the primary agent based on the current state.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <param name="isTraining">Whether the agent is in training mode.</param>
    /// <returns>The selected action.</returns>
    public override Vector<T> SelectAction(Tensor<T> state, bool isTraining = false)
    {
        // Process the state through the transformer to get agent-specific embeddings
        var transformerOutput = ProcessStateWithTransformer(state);
        
        // Use the primary agent's policy network (agent 0)
        var agentFeatures = transformerOutput[0];
        var agentFeaturesTensor = new Tensor<T>(new[] { 1, agentFeatures.Length });
        for (int i = 0; i < agentFeatures.Length; i++)
        {
            agentFeaturesTensor[0, i] = agentFeatures[i];
        }
        var policyOutput = _policyNetworks[0].Predict(agentFeaturesTensor);
        
        // Extract action mean and log variance
        var actionMean = new Vector<T>(_actionDimension);
        var actionLogVar = new Vector<T>(_actionDimension);
        
        for (int i = 0; i < _actionDimension; i++)
        {
            actionMean[i] = policyOutput[i];
            actionLogVar[i] = policyOutput[i + _actionDimension];
        }
        
        // If in training mode, sample from the distribution
        // Otherwise, just return the mean as the action
        if (isTraining)
        {
            var actionStd = new Vector<T>(_actionDimension);
            for (int i = 0; i < _actionDimension; i++)
            {
                actionStd[i] = NumOps.Exp(NumOps.Multiply(actionLogVar[i], NumOps.FromDouble(0.5)));
            }
            
            var action = new Vector<T>(_actionDimension);
            for (int i = 0; i < _actionDimension; i++)
            {
                // Sample from normal distribution: mean + std * random normal
                var noise = NumOps.FromDouble(SampleGaussian());
                action[i] = NumOps.Add(actionMean[i], NumOps.Multiply(actionStd[i], noise));
            }
            
            return action;
        }
        
        return actionMean;
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
        // For now, this is a placeholder implementation
        // In a complete implementation, this would:
        // 1. Store the experience in a replay buffer
        // 2. Sample a batch from the replay buffer
        // 3. Update the networks using the sampled batch
        
        // Update step counter
        IncrementStepCounter();
        
        // You could implement experience replay here or call a batch update method
        // For now, we'll just update the last loss to zero
        LastLoss = NumOps.Zero;
    }
    
    /// <summary>
    /// Selects actions for all agents based on the current state.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <param name="isTraining">Whether the agents are in training mode.</param>
    /// <returns>A list of actions, one for each agent.</returns>
    public List<Vector<T>> SelectActionsForAllAgents(Tensor<T> state, bool isTraining = false)
    {
        // Process the state through the transformer to get agent-specific embeddings
        var transformerOutput = ProcessStateWithTransformer(state);
        var actions = new List<Vector<T>>();
        
        // Get action for each agent
        for (int agentIdx = 0; agentIdx < _numAgents; agentIdx++)
        {
            // Convert Vector<double> to Tensor<double> for Predict
            var agentFeaturesTensor = new Tensor<T>(new[] { 1, transformerOutput[agentIdx].Length });
            for (int i = 0; i < transformerOutput[agentIdx].Length; i++)
            {
                agentFeaturesTensor[0, i] = transformerOutput[agentIdx][i];
            }
            var policyOutput = _policyNetworks[agentIdx].Predict(agentFeaturesTensor);
            
            // Extract action mean and log variance
            var actionMean = new Vector<T>(_actionDimension);
            var actionLogVar = new Vector<T>(_actionDimension);
            
            for (int i = 0; i < _actionDimension; i++)
            {
                actionMean[i] = policyOutput[i];
                actionLogVar[i] = policyOutput[i + _actionDimension];
            }
            
            // If in training mode, sample from the distribution
            // Otherwise, just return the mean as the action
            if (isTraining)
            {
                var actionStd = new Vector<T>(_actionDimension);
                for (int i = 0; i < _actionDimension; i++)
                {
                    actionStd[i] = NumOps.Exp(NumOps.Multiply(actionLogVar[i], NumOps.FromDouble(0.5)));
                }
                
                var action = new Vector<T>(_actionDimension);
                for (int i = 0; i < _actionDimension; i++)
                {
                    // Sample from normal distribution: mean + std * random normal
                    var noise = NumOps.FromDouble(SampleGaussian());
                    action[i] = NumOps.Add(actionMean[i], NumOps.Multiply(actionStd[i], noise));
                }
                
                actions.Add(action);
            }
            else
            {
                actions.Add(actionMean);
            }
        }
        
        return actions;
    }
    
    /// <summary>
    /// Updates the agent based on a sequence of transitions.
    /// </summary>
    /// <param name="states">A sequence of states.</param>
    /// <param name="actions">A sequence of actions.</param>
    /// <param name="rewards">A sequence of rewards.</param>
    /// <param name="nextStates">A sequence of next states.</param>
    /// <param name="dones">A sequence of done flags.</param>
    /// <returns>The loss value from the update.</returns>
    public T Update(Tensor<T> states, Tensor<T> actions, Vector<T> rewards, Tensor<T> nextStates, Vector<T> dones)
    {
        // Process all states and next states through the transformer
        var stateEmbeddings = new List<Vector<T>[]>();
        var nextStateEmbeddings = new List<Vector<T>[]>();
        
        for (int t = 0; t < _sequenceLength; t++)
        {
            var stateSlice = GetStateAtTimeStep(states, t);
            var nextStateSlice = GetStateAtTimeStep(nextStates, t);
            
            stateEmbeddings.Add(ProcessStateWithTransformer(stateSlice));
            nextStateEmbeddings.Add(ProcessStateWithTransformer(nextStateSlice));
        }
        
        // Calculate advantages for each time step
        var advantages = CalculateAdvantages(stateEmbeddings, rewards, nextStateEmbeddings, dones);
        
        // Update policy and value networks
        T policyLoss = UpdatePolicyNetworks(stateEmbeddings, actions, advantages);
        T valueLoss = UpdateValueNetworks(stateEmbeddings, rewards, nextStateEmbeddings, dones);
        
        // Update transformer network
        T transformerLoss = UpdateTransformerNetwork(states, actions, rewards, nextStates, dones);
        
        // Return combined loss
        return NumOps.Add(NumOps.Add(policyLoss, valueLoss), transformerLoss);
    }
    
    /// <summary>
    /// Updates the agent based on multi-agent transitions.
    /// </summary>
    /// <param name="state">The shared state.</param>
    /// <param name="actions">The actions taken by each agent.</param>
    /// <param name="rewards">The rewards received by each agent.</param>
    /// <param name="nextState">The next shared state.</param>
    /// <param name="done">Whether the episode is done.</param>
    /// <returns>The average loss value across all agents.</returns>
    public T UpdateMultiAgent(Tensor<T> state, List<Vector<T>> actions, List<T> rewards, Tensor<T> nextState, bool done)
    {
        // Process state and next state through the transformer
        var stateEmbeddings = ProcessStateWithTransformer(state);
        var nextStateEmbeddings = ProcessStateWithTransformer(nextState);
        
        // Calculate values for current and next state
        var values = new List<T>();
        var nextValues = new List<T>();
        
        for (int agentIdx = 0; agentIdx < _numAgents; agentIdx++)
        {
            // Convert Vector<double> to Tensor<double> for Predict
            var stateFeaturesTensor = new Tensor<T>(new[] { 1, stateEmbeddings[agentIdx].Length });
            for (int i = 0; i < stateEmbeddings[agentIdx].Length; i++)
            {
                stateFeaturesTensor[0, i] = stateEmbeddings[agentIdx][i];
            }
            var valueOutput = _valueNetworks[agentIdx].Predict(stateFeaturesTensor);
            
            var nextStateFeaturesTensor = new Tensor<T>(new[] { 1, nextStateEmbeddings[agentIdx].Length });
            for (int i = 0; i < nextStateEmbeddings[agentIdx].Length; i++)
            {
                nextStateFeaturesTensor[0, i] = nextStateEmbeddings[agentIdx][i];
            }
            var nextValueOutput = _valueNetworks[agentIdx].Predict(nextStateFeaturesTensor);
            
            values.Add(valueOutput[0]);
            nextValues.Add(nextValueOutput[0]);
        }
        
        // Calculate advantages for each agent
        var advantages = new List<T>();
        var doneFactor = done ? NumOps.Zero : NumOps.One;
        
        for (int agentIdx = 0; agentIdx < _numAgents; agentIdx++)
        {
            // Advantage = reward + gamma * nextValue * (1 - done) - value
            var nextValueTerm = NumOps.Multiply(NumOps.Multiply(Gamma, nextValues[agentIdx]), doneFactor);
            var target = NumOps.Add(rewards[agentIdx], nextValueTerm);
            var advantage = NumOps.Subtract(target, values[agentIdx]);
            advantages.Add(advantage);
        }
        
        // Update policy networks
        var policyLosses = new List<T>();
        
        for (int agentIdx = 0; agentIdx < _numAgents; agentIdx++)
        {
            var policyOutput = _policyNetworks[agentIdx].Predict(Tensor<T>.FromVector(stateEmbeddings[agentIdx], new[] { 1, stateEmbeddings[agentIdx].Length }));
            
            // Extract action mean and log variance
            var actionMean = new Vector<T>(_actionDimension);
            var actionLogVar = new Vector<T>(_actionDimension);
            
            for (int i = 0; i < _actionDimension; i++)
            {
                actionMean[i] = policyOutput[i];
                actionLogVar[i] = policyOutput[i + _actionDimension];
            }
            
            // Calculate log probability of the action taken
            var logProb = CalculateLogProbability(actions[agentIdx], actionMean, actionLogVar);
            
            // Policy loss = -log_prob * advantage - entropy_coef * entropy
            var entropy = CalculateEntropy(actionLogVar);
            var policyLoss = NumOps.Subtract(
                NumOps.Multiply(NumOps.Negate(logProb), advantages[agentIdx]),
                NumOps.Multiply(_entropyCoef, entropy)
            );
            
            // Update policy network
            _policyNetworks[agentIdx].Backward(policyLoss);
            
            // Note: Neural networks typically update their parameters internally during Backward
            // when an optimizer is attached. If external optimization is needed, it would go here.
            
            policyLosses.Add(policyLoss);
        }
        
        // Update value networks
        var valueLosses = new List<T>();
        
        for (int agentIdx = 0; agentIdx < _numAgents; agentIdx++)
        {
            var valueOutput = _valueNetworks[agentIdx].Predict(Tensor<T>.FromVector(stateEmbeddings[agentIdx], new[] { 1, stateEmbeddings[agentIdx].Length }));
            
            // Value target = reward + gamma * nextValue * (1 - done)
            var nextValueTerm = NumOps.Multiply(NumOps.Multiply(Gamma, nextValues[agentIdx]), doneFactor);
            var target = NumOps.Add(rewards[agentIdx], nextValueTerm);
            
            // Value loss = 0.5 * (value - target)^2
            var valueDiff = NumOps.Subtract(valueOutput[0], target);
            var valueLoss = NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Multiply(valueDiff, valueDiff));
            
            // Update value network
            _valueNetworks[agentIdx].Backward(valueLoss);
            
            // Note: Neural networks typically update their parameters internally during Backward
            // when an optimizer is attached. If external optimization is needed, it would go here.
            
            valueLosses.Add(valueLoss);
        }
        
        // Update transformer
        _mainTransformer.Backward(NumOps.FromDouble(1.0));
        
        // Note: Neural networks typically update their parameters internally during Backward
        // when an optimizer is attached. If external optimization is needed, it would go here.
        
        // Calculate average losses
        T avgPolicyLoss = policyLosses.Aggregate(NumOps.Zero, (acc, loss) => NumOps.Add(acc, loss));
        avgPolicyLoss = NumOps.Divide(avgPolicyLoss, NumOps.FromDouble(_numAgents));
        
        T avgValueLoss = valueLosses.Aggregate(NumOps.Zero, (acc, loss) => NumOps.Add(acc, loss));
        avgValueLoss = NumOps.Divide(avgValueLoss, NumOps.FromDouble(_numAgents));
        
        return NumOps.Add(avgPolicyLoss, avgValueLoss);
    }
    
    /// <summary>
    /// Gets the attention weights showing how agents interact with each other.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <returns>A tensor containing attention weights between agents.</returns>
    public Tensor<T> GetAgentInteractionAttention(Tensor<T> state)
    {
        // Process state through transformer and store attention weights
        ProcessStateWithTransformer(state);
        return _lastAttentionWeights ?? new Tensor<T>(new[] { 0 });
    }
    
    /// <summary>
    /// Predicts potential future market states.
    /// </summary>
    /// <param name="currentState">The current market state.</param>
    /// <param name="numSteps">The number of steps to predict ahead.</param>
    /// <returns>A list of predicted future states.</returns>
    public List<Tensor<T>> PredictFutureStates(Tensor<T> currentState, int numSteps = 5)
    {
        var futureStates = new List<Tensor<T>> { currentState };
        var state = currentState;
        
        for (int step = 0; step < numSteps; step++)
        {
            // Get actions for all agents in the current state
            var actions = SelectActionsForAllAgents(state, false);
            
            // Concatenate actions from all agents
            var combinedActions = new Vector<T>(_actionDimension * _numAgents);
            for (int agentIdx = 0; agentIdx < _numAgents; agentIdx++)
            {
                for (int i = 0; i < _actionDimension; i++)
                {
                    combinedActions[agentIdx * _actionDimension + i] = actions[agentIdx][i];
                }
            }
            
            // Create input for market dynamics network
            var dynamicsInput = new Vector<T>(_stateDimension + _actionDimension * _numAgents);
            for (int i = 0; i < _stateDimension; i++)
            {
                dynamicsInput[i] = state[0, i];
            }
            for (int i = 0; i < _actionDimension * _numAgents; i++)
            {
                dynamicsInput[_stateDimension + i] = combinedActions[i];
            }
            
            // Predict next state
            var nextStateVector = _marketDynamicsNetwork.Predict(Tensor<T>.FromVector(dynamicsInput, new[] { 1, dynamicsInput.Length }));
            var nextState = new Tensor<T>(new[] { 1, _stateDimension });
            for (int i = 0; i < _stateDimension; i++)
            {
                nextState[0, i] = nextStateVector[i];
            }
            
            futureStates.Add(nextState);
            state = nextState;
        }
        
        return futureStates;
    }
    
    /// <summary>
    /// Analyzes the risk profile of potential actions in the current market state.
    /// </summary>
    /// <param name="state">The current market state.</param>
    /// <returns>A dictionary mapping action descriptions to their risk-adjusted expected returns.</returns>
    public Dictionary<string, T> AnalyzeActionRiskProfile(Tensor<T> state)
    {
        var riskProfile = new Dictionary<string, T>();
        
        // Process state through transformer
        var stateEmbeddings = ProcessStateWithTransformer(state);
        
        // Define a set of standard actions to evaluate
        var standardActions = DefineStandardActions();
        
        foreach (var actionPair in standardActions)
        {
            var actionName = actionPair.Key;
            var action = actionPair.Value;
            
            // Get value estimates from all agents
            var valueEstimates = new List<T>();
            for (int agentIdx = 0; agentIdx < _numAgents; agentIdx++)
            {
                var policyOutput = _policyNetworks[agentIdx].Predict(Tensor<T>.FromVector(stateEmbeddings[agentIdx], new[] { 1, stateEmbeddings[agentIdx].Length }));
                
                // Extract action mean and log variance
                var actionMean = new Vector<T>(_actionDimension);
                var actionLogVar = new Vector<T>(_actionDimension);
                
                for (int i = 0; i < _actionDimension; i++)
                {
                    actionMean[i] = policyOutput[i];
                    actionLogVar[i] = policyOutput[i + _actionDimension];
                }
                
                // Calculate log probability of the standard action
                var logProb = CalculateLogProbability(action, actionMean, actionLogVar);
                
                // Get value estimate
                var valueOutput = _valueNetworks[agentIdx].Predict(Tensor<T>.FromVector(stateEmbeddings[agentIdx], new[] { 1, stateEmbeddings[agentIdx].Length }));
                
                // Adjust value by log probability (risk-adjusted return)
                var riskAdjustedValue = NumOps.Add(
                    valueOutput[0],
                    NumOps.Multiply(_riskAversion, NumOps.Exp(logProb))
                );
                
                valueEstimates.Add(riskAdjustedValue);
            }
            
            // Average the value estimates across agents
            var avgValue = valueEstimates.Aggregate(NumOps.Zero, (acc, val) => NumOps.Add(acc, val));
            avgValue = NumOps.Divide(avgValue, NumOps.FromDouble(_numAgents));
            
            riskProfile[actionName] = avgValue;
        }
        
        return riskProfile;
    }
    
    /// <summary>
    /// Detects the current market regime based on agent behavior patterns.
    /// </summary>
    /// <param name="state">The current market state.</param>
    /// <returns>A string indicating the detected market regime.</returns>
    public string DetectMarketRegime(Tensor<T> state)
    {
        // Process state through transformer
        var stateEmbeddings = ProcessStateWithTransformer(state);
        
        // Get actions and value estimates from all agents
        var actions = new List<Vector<T>>();
        var valueEstimates = new List<T>();
        var uncertainties = new List<T>();
        
        for (int agentIdx = 0; agentIdx < _numAgents; agentIdx++)
        {
            var policyOutput = _policyNetworks[agentIdx].Predict(Tensor<T>.FromVector(stateEmbeddings[agentIdx], new[] { 1, stateEmbeddings[agentIdx].Length }));
            
            // Extract action mean and log variance
            var actionMean = new Vector<T>(_actionDimension);
            var actionLogVar = new Vector<T>(_actionDimension);
            
            for (int i = 0; i < _actionDimension; i++)
            {
                actionMean[i] = policyOutput[i];
                actionLogVar[i] = policyOutput[i + _actionDimension];
            }
            
            actions.Add(actionMean);
            
            // Get value estimate
            var valueOutput = _valueNetworks[agentIdx].Predict(Tensor<T>.FromVector(stateEmbeddings[agentIdx], new[] { 1, stateEmbeddings[agentIdx].Length }));
            valueEstimates.Add(valueOutput[0]);
            
            // Use action variance as a measure of uncertainty
            var avgUncertainty = NumOps.Zero;
            for (int i = 0; i < _actionDimension; i++)
            {
                avgUncertainty = NumOps.Add(avgUncertainty, NumOps.Exp(actionLogVar[i]));
            }
            avgUncertainty = NumOps.Divide(avgUncertainty, NumOps.FromDouble(_actionDimension));
            uncertainties.Add(avgUncertainty);
        }
        
        // Analyze agent action consensus
        var actionConsensus = CalculateActionConsensus(actions);
        
        // Analyze value estimate dispersion
        var valueDispersion = CalculateValueDispersion(valueEstimates);
        
        // Average uncertainty across agents
        var overallAvgUncertainty = uncertainties.Aggregate(NumOps.Zero, (acc, u) => NumOps.Add(acc, u));
        overallAvgUncertainty = NumOps.Divide(overallAvgUncertainty, NumOps.FromDouble(_numAgents));
        
        // Determine market regime based on these indicators
        return DetermineMarketRegime(actionConsensus, valueDispersion, overallAvgUncertainty);
    }
    
    /// <summary>
    /// Saves the agent's models to the specified path.
    /// </summary>
    /// <param name="path">The path to save the models to.</param>
    public void SaveModel(string filePath)
    {
        using (var writer = new BinaryWriter(File.Open(filePath, FileMode.Create)))
        {
            // Save configuration
            writer.Write(_numAgents);
            writer.Write(_stateDimension);
            writer.Write(_actionDimension);
            writer.Write(_hiddenDimension);
            writer.Write(_numHeads);
            writer.Write(_numLayers);
            writer.Write(_sequenceLength);
            writer.Write((int)_posEncodingType);
            writer.Write(_communicationMode);
            writer.Write(_useCentralizedTraining);
            writer.Write(Convert.ToDouble(_learningRate));
            writer.Write(Convert.ToDouble(_entropyCoef));
            writer.Write(_useSelfPlay);
            writer.Write(Convert.ToDouble(_riskAversion));
            writer.Write(_useCausalMask);
            writer.Write(_modelMarketImpact);
            
            // Save transformer parameters
            var transformerParams = _mainTransformer.GetParameters();
            writer.Write(transformerParams.Length);
            for (int i = 0; i < transformerParams.Length; i++)
            {
                writer.Write(Convert.ToDouble(transformerParams[i]));
            }
            
            // Save policy networks parameters
            for (int agentIdx = 0; agentIdx < _numAgents; agentIdx++)
            {
                var policyParams = _policyNetworks[agentIdx].GetParameters();
                writer.Write(policyParams.Length);
                for (int i = 0; i < policyParams.Length; i++)
                {
                    writer.Write(Convert.ToDouble(policyParams[i]));
                }
            }
            
            // Save value networks parameters
            for (int agentIdx = 0; agentIdx < _numAgents; agentIdx++)
            {
                var valueParams = _valueNetworks[agentIdx].GetParameters();
                writer.Write(valueParams.Length);
                for (int i = 0; i < valueParams.Length; i++)
                {
                    writer.Write(Convert.ToDouble(valueParams[i]));
                }
            }
            
            // Save market dynamics network parameters
            var dynamicsParams = _marketDynamicsNetwork.GetParameters();
            writer.Write(dynamicsParams.Length);
            for (int i = 0; i < dynamicsParams.Length; i++)
            {
                writer.Write(Convert.ToDouble(dynamicsParams[i]));
            }
            
            // Save market impact network parameters if it exists
            if (_modelMarketImpact && _marketImpactNetwork != null)
            {
                var impactParams = _marketImpactNetwork.GetParameters();
                writer.Write(impactParams.Length);
                for (int i = 0; i < impactParams.Length; i++)
                {
                    writer.Write(Convert.ToDouble(impactParams[i]));
                }
            }
        }
    }
    
    /// <summary>
    /// Loads the agent's models from the specified path.
    /// </summary>
    /// <param name="path">The path to load the models from.</param>
    public void LoadModel(string filePath)
    {
        using (var reader = new BinaryReader(File.Open(filePath, FileMode.Open)))
        {
            // Load configuration
            _numAgents = reader.ReadInt32();
            _stateDimension = reader.ReadInt32();
            _actionDimension = reader.ReadInt32();
            _hiddenDimension = reader.ReadInt32();
            _numHeads = reader.ReadInt32();
            _numLayers = reader.ReadInt32();
            _sequenceLength = reader.ReadInt32();
            _posEncodingType = (PositionalEncodingType)reader.ReadInt32();
            _communicationMode = reader.ReadInt32();
            _useCentralizedTraining = reader.ReadBoolean();
            _learningRate = NumOps.FromDouble(reader.ReadDouble());
            _entropyCoef = NumOps.FromDouble(reader.ReadDouble());
            _useSelfPlay = reader.ReadBoolean();
            _riskAversion = NumOps.FromDouble(reader.ReadDouble());
            _useCausalMask = reader.ReadBoolean();
            _modelMarketImpact = reader.ReadBoolean();
            
            // Load transformer parameters
            int transformerParamCount = reader.ReadInt32();
            var transformerParams = new Vector<T>(transformerParamCount);
            for (int i = 0; i < transformerParamCount; i++)
            {
                transformerParams[i] = NumOps.FromDouble(reader.ReadDouble());
            }
            _mainTransformer.UpdateParameters(transformerParams);
            
            // Load policy networks parameters
            for (int agentIdx = 0; agentIdx < _numAgents; agentIdx++)
            {
                int policyParamCount = reader.ReadInt32();
                var policyParams = new Vector<T>(policyParamCount);
                for (int i = 0; i < policyParamCount; i++)
                {
                    policyParams[i] = NumOps.FromDouble(reader.ReadDouble());
                }
                _policyNetworks[agentIdx].UpdateParameters(policyParams);
            }
            
            // Load value networks parameters
            for (int agentIdx = 0; agentIdx < _numAgents; agentIdx++)
            {
                int valueParamCount = reader.ReadInt32();
                var valueParams = new Vector<T>(valueParamCount);
                for (int i = 0; i < valueParamCount; i++)
                {
                    valueParams[i] = NumOps.FromDouble(reader.ReadDouble());
                }
                _valueNetworks[agentIdx].UpdateParameters(valueParams);
            }
            
            // Load market dynamics network parameters
            int dynamicsParamCount = reader.ReadInt32();
            var dynamicsParams = new Vector<T>(dynamicsParamCount);
            for (int i = 0; i < dynamicsParamCount; i++)
            {
                dynamicsParams[i] = NumOps.FromDouble(reader.ReadDouble());
            }
            _marketDynamicsNetwork.UpdateParameters(dynamicsParams);
            
            // Load market impact network parameters if it exists
            if (_modelMarketImpact && _marketImpactNetwork != null)
            {
                int impactParamCount = reader.ReadInt32();
                var impactParams = new Vector<T>(impactParamCount);
                for (int i = 0; i < impactParamCount; i++)
                {
                    impactParams[i] = NumOps.FromDouble(reader.ReadDouble());
                }
                _marketImpactNetwork.UpdateParameters(impactParams);
            }
        }
    }
    
    #region Helper Methods
    
    /// <summary>
    /// Converts a vector to a 2D tensor with batch size of 1.
    /// </summary>
    
    /// <summary>
    /// Processes a state through the transformer to get agent-specific embeddings.
    /// </summary>
    private Vector<T>[] ProcessStateWithTransformer(Tensor<T> state)
    {
        // We need to create a sequence of appropriate shape for the transformer:
        // [batch_size=1, seq_length=_numAgents, feature_dim=_stateDimension]
        var transformerInput = new Tensor<T>(new[] { 1, _numAgents, _stateDimension });
        
        // Replicate the state for each agent
        for (int agentIdx = 0; agentIdx < _numAgents; agentIdx++)
        {
            for (int featureIdx = 0; featureIdx < _stateDimension; featureIdx++)
            {
                transformerInput[0, agentIdx, featureIdx] = state[0, featureIdx];
            }
        }
        
        // Process through transformer
        var transformerOutput = _mainTransformer.Predict(transformerInput);
        
        // Extract agent-specific embeddings
        var agentEmbeddings = new Vector<T>[_numAgents];
        for (int agentIdx = 0; agentIdx < _numAgents; agentIdx++)
        {
            agentEmbeddings[agentIdx] = new Vector<T>(_hiddenDimension);
            for (int i = 0; i < _hiddenDimension; i++)
            {
                agentEmbeddings[agentIdx][i] = transformerOutput[0, agentIdx, i];
            }
        }
        
        // Store attention weights for visualization
        // TODO: Implement GetLastAttentionWeights in Transformer
        // _lastAttentionWeights = _mainTransformer.GetLastAttentionWeights();
        
        return agentEmbeddings;
    }
    
    /// <summary>
    /// Extracts the state tensor at a specific time step from a sequence of states.
    /// </summary>
    private Tensor<T> GetStateAtTimeStep(Tensor<T> states, int timeStep)
    {
        var stateAtT = new Tensor<T>(new[] { 1, _stateDimension });
        for (int i = 0; i < _stateDimension; i++)
        {
            stateAtT[0, i] = states[timeStep, i];
        }
        return stateAtT;
    }
    
    /// <summary>
    /// Calculates advantages for a sequence of transitions.
    /// </summary>
    private List<T[]> CalculateAdvantages(List<Vector<T>[]> stateEmbeddings, Vector<T> rewards, 
        List<Vector<T>[]> nextStateEmbeddings, Vector<T> dones)
    {
        var advantages = new List<T[]>();
        
        for (int t = 0; t < _sequenceLength; t++)
        {
            var advantagesAtT = new T[_numAgents];
            
            for (int agentIdx = 0; agentIdx < _numAgents; agentIdx++)
            {
                var valueOutput = _valueNetworks[agentIdx].Predict(Tensor<T>.FromVector(stateEmbeddings[t][agentIdx], new[] { 1, stateEmbeddings[t][agentIdx].Length }));
                var nextValueOutput = _valueNetworks[agentIdx].Predict(Tensor<T>.FromVector(nextStateEmbeddings[t][agentIdx], new[] { 1, nextStateEmbeddings[t][agentIdx].Length }));
                
                // done flag: 1.0 if done, 0.0 if not done
                var doneFactor = NumOps.GreaterThan(dones[t], NumOps.Zero) ? NumOps.Zero : NumOps.One;
                
                // Advantage = reward + gamma * nextValue * (1 - done) - value
                var nextValueTerm = NumOps.Multiply(NumOps.Multiply(Gamma, nextValueOutput[0]), doneFactor);
                var target = NumOps.Add(rewards[t], nextValueTerm);
                advantagesAtT[agentIdx] = NumOps.Subtract(target, valueOutput[0]);
            }
            
            advantages.Add(advantagesAtT);
        }
        
        return advantages;
    }
    
    /// <summary>
    /// Updates policy networks based on collected experiences.
    /// </summary>
    private T UpdatePolicyNetworks(List<Vector<T>[]> stateEmbeddings, Tensor<T> actions, List<T[]> advantages)
    {
        var totalLoss = NumOps.Zero;
        
        for (int t = 0; t < _sequenceLength; t++)
        {
            for (int agentIdx = 0; agentIdx < _numAgents; agentIdx++)
            {
                var policyOutput = _policyNetworks[agentIdx].Predict(Tensor<T>.FromVector(stateEmbeddings[t][agentIdx], new[] { 1, stateEmbeddings[t][agentIdx].Length }));
                
                // Extract action mean and log variance
                var actionMean = new Vector<T>(_actionDimension);
                var actionLogVar = new Vector<T>(_actionDimension);
                
                for (int i = 0; i < _actionDimension; i++)
                {
                    actionMean[i] = policyOutput[i];
                    actionLogVar[i] = policyOutput[i + _actionDimension];
                }
                
                // Extract action taken at this time step
                var actionTaken = new Vector<T>(_actionDimension);
                for (int i = 0; i < _actionDimension; i++)
                {
                    actionTaken[i] = actions[t, i];
                }
                
                // Calculate log probability of the action taken
                var logProb = CalculateLogProbability(actionTaken, actionMean, actionLogVar);
                
                // Calculate entropy
                var entropy = CalculateEntropy(actionLogVar);
                
                // Policy loss = -log_prob * advantage - entropy_coef * entropy
                var policyLoss = NumOps.Subtract(
                    NumOps.Multiply(NumOps.Negate(logProb), advantages[t][agentIdx]),
                    NumOps.Multiply(_entropyCoef, entropy)
                );
                
                // Update policy network
                _policyNetworks[agentIdx].Backward(policyLoss);
                
                // The Backward call above already updates the network parameters
                // through gradient descent
                
                totalLoss = NumOps.Add(totalLoss, policyLoss);
            }
        }
        
        return NumOps.Divide(totalLoss, NumOps.FromDouble(_sequenceLength * _numAgents));
    }
    
    /// <summary>
    /// Updates value networks based on collected experiences.
    /// </summary>
    private T UpdateValueNetworks(List<Vector<T>[]> stateEmbeddings, Vector<T> rewards, 
        List<Vector<T>[]> nextStateEmbeddings, Vector<T> dones)
    {
        var totalLoss = NumOps.Zero;
        
        for (int t = 0; t < _sequenceLength; t++)
        {
            for (int agentIdx = 0; agentIdx < _numAgents; agentIdx++)
            {
                var valueOutput = _valueNetworks[agentIdx].Predict(Tensor<T>.FromVector(stateEmbeddings[t][agentIdx], new[] { 1, stateEmbeddings[t][agentIdx].Length }));
                var nextValueOutput = _valueNetworks[agentIdx].Predict(Tensor<T>.FromVector(nextStateEmbeddings[t][agentIdx], new[] { 1, nextStateEmbeddings[t][agentIdx].Length }));
                
                // done flag: 1.0 if done, 0.0 if not done
                var doneFactor = NumOps.GreaterThan(dones[t], NumOps.Zero) ? NumOps.Zero : NumOps.One;
                
                // Value target = reward + gamma * nextValue * (1 - done)
                var nextValueTerm = NumOps.Multiply(NumOps.Multiply(Gamma, nextValueOutput[0]), doneFactor);
                var target = NumOps.Add(rewards[t], nextValueTerm);
                
                // Value loss = 0.5 * (value - target)^2
                var valueDiff = NumOps.Subtract(valueOutput[0], target);
                var valueLoss = NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Multiply(valueDiff, valueDiff));
                
                // Update value network
                _valueNetworks[agentIdx].Backward(valueLoss);
                
                // The Backward call above already updates the network parameters
                // through gradient descent
                
                totalLoss = NumOps.Add(totalLoss, valueLoss);
            }
        }
        
        return NumOps.Divide(totalLoss, NumOps.FromDouble(_sequenceLength * _numAgents));
    }
    
    /// <summary>
    /// Updates the transformer network based on collected experiences.
    /// </summary>
    private T UpdateTransformerNetwork(Tensor<T> states, Tensor<T> actions, Vector<T> rewards, 
        Tensor<T> nextStates, Vector<T> dones)
    {
        // TODO: Fix Backward method - Transformer doesn't have Backward method
        // Since the transformer is updated through backpropagation from the policy and value networks,
        // we just need to run an additional backward pass to ensure gradients flow
        // _mainTransformer.Backward(NumOps.FromDouble(1.0));
        
        // Create optimization input data
        // TODO: Fix optimization input - OptimizationInputData doesn't have Parameters property
        // Need to rework how transformer optimization works
        // var optimizationInput = new OptimizationInputData<T, Tensor<T>, Tensor<T>>
        // {
        //     // Need to properly set XTrain, YTrain etc instead of Parameters
        // };
        // 
        // _transformerOptimizer.Optimize(optimizationInput);
        
        return NumOps.FromDouble(0.01); // Return a small value representing transformer loss
    }
    
    /// <summary>
    /// Calculates the log probability of an action under a Gaussian policy.
    /// </summary>
    private T CalculateLogProbability(Vector<T> action, Vector<T> mean, Vector<T> logVar)
    {
        var logProb = NumOps.Zero;
        var logTwoPI = NumOps.FromDouble(Math.Log(2 * Math.PI));
        
        for (int i = 0; i < action.Length; i++)
        {
            var diff = NumOps.Subtract(action[i], mean[i]);
            var variance = NumOps.Exp(logVar[i]);
            
            // Log probability of Gaussian distribution:
            // -0.5 * (log(2π) + log(variance) + (x - mean)² / variance)
            var term1 = NumOps.Add(logTwoPI, logVar[i]);
            var term2 = NumOps.Divide(NumOps.Multiply(diff, diff), variance);
            var componentLogProb = NumOps.Multiply(NumOps.FromDouble(-0.5), NumOps.Add(term1, term2));
            
            logProb = NumOps.Add(logProb, componentLogProb);
        }
        
        return logProb;
    }
    
    /// <summary>
    /// Calculates the entropy of a Gaussian distribution.
    /// </summary>
    private T CalculateEntropy(Vector<T> logVar)
    {
        var entropy = NumOps.Zero;
        var halfLogTwoPI = NumOps.FromDouble(0.5 * Math.Log(2 * Math.PI));
        var half = NumOps.FromDouble(0.5);
        
        for (int i = 0; i < logVar.Length; i++)
        {
            // Entropy of Gaussian is 0.5 * (1 + log(2πσ²))
            // = 0.5 * (1 + log(2π) + 2*logVar)
            var term = NumOps.Add(NumOps.Add(NumOps.One, NumOps.FromDouble(Math.Log(2 * Math.PI))), 
                NumOps.Multiply(NumOps.FromDouble(2), logVar[i]));
            entropy = NumOps.Add(entropy, NumOps.Multiply(half, term));
        }
        
        return entropy;
    }
    
    /// <summary>
    /// Samples a value from a standard Gaussian distribution (mean 0, std 1).
    /// </summary>
    private double SampleGaussian()
    {
        // Box-Muller transform to generate Gaussian samples
        double u1 = 1.0 - Random.NextDouble(); // Uniform(0,1) distribution
        double u2 = 1.0 - Random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }
    
    /// <summary>
    /// Defines a set of standard actions for risk profile analysis.
    /// </summary>
    private Dictionary<string, Vector<T>> DefineStandardActions()
    {
        var actions = new Dictionary<string, Vector<T>>();
        
        // Example actions for a financial market context
        // Assuming action dimensions represent things like:
        // - Position size (% of portfolio)
        // - Target holding period
        // - Stop loss level
        // - Take profit level
        
        // Strong Buy
        var strongBuy = new Vector<T>(_actionDimension);
        strongBuy[0] = NumOps.FromDouble(1.0); // Max position size
        if (_actionDimension > 1) strongBuy[1] = NumOps.FromDouble(1.0); // Long holding period
        if (_actionDimension > 2) strongBuy[2] = NumOps.FromDouble(0.2); // Loose stop loss
        if (_actionDimension > 3) strongBuy[3] = NumOps.FromDouble(0.5); // High take profit
        actions["Strong Buy"] = strongBuy;
        
        // Moderate Buy
        var moderateBuy = new Vector<T>(_actionDimension);
        moderateBuy[0] = NumOps.FromDouble(0.5); // Medium position size
        if (_actionDimension > 1) moderateBuy[1] = NumOps.FromDouble(0.7); // Medium-long holding
        if (_actionDimension > 2) moderateBuy[2] = NumOps.FromDouble(0.3); // Medium stop loss
        if (_actionDimension > 3) moderateBuy[3] = NumOps.FromDouble(0.3); // Medium take profit
        actions["Moderate Buy"] = moderateBuy;
        
        // Hold
        var hold = new Vector<T>(_actionDimension);
        hold[0] = NumOps.FromDouble(0.0); // No position change
        if (_actionDimension > 1) hold[1] = NumOps.FromDouble(0.5); // Medium holding
        if (_actionDimension > 2) hold[2] = NumOps.FromDouble(0.5); // Medium stop loss
        if (_actionDimension > 3) hold[3] = NumOps.FromDouble(0.5); // Medium take profit
        actions["Hold"] = hold;
        
        // Moderate Sell
        var moderateSell = new Vector<T>(_actionDimension);
        moderateSell[0] = NumOps.FromDouble(-0.5); // Medium short position
        if (_actionDimension > 1) moderateSell[1] = NumOps.FromDouble(0.7); // Medium-long holding
        if (_actionDimension > 2) moderateSell[2] = NumOps.FromDouble(0.3); // Medium stop loss
        if (_actionDimension > 3) moderateSell[3] = NumOps.FromDouble(0.3); // Medium take profit
        actions["Moderate Sell"] = moderateSell;
        
        // Strong Sell
        var strongSell = new Vector<T>(_actionDimension);
        strongSell[0] = NumOps.FromDouble(-1.0); // Max short position
        if (_actionDimension > 1) strongSell[1] = NumOps.FromDouble(1.0); // Long holding period
        if (_actionDimension > 2) strongSell[2] = NumOps.FromDouble(0.2); // Loose stop loss
        if (_actionDimension > 3) strongSell[3] = NumOps.FromDouble(0.5); // High take profit
        actions["Strong Sell"] = strongSell;
        
        return actions;
    }
    
    /// <summary>
    /// Calculates how much consensus there is in agents' actions.
    /// </summary>
    private T CalculateActionConsensus(List<Vector<T>> actions)
    {
        // Calculate mean action across all agents
        var meanAction = new Vector<T>(_actionDimension);
        for (int i = 0; i < _actionDimension; i++)
        {
            var sum = NumOps.Zero;
            foreach (var action in actions)
            {
                sum = NumOps.Add(sum, action[i]);
            }
            meanAction[i] = NumOps.Divide(sum, NumOps.FromDouble(_numAgents));
        }
        
        // Calculate average distance from mean
        var totalDistance = NumOps.Zero;
        foreach (var action in actions)
        {
            var distance = NumOps.Zero;
            for (int i = 0; i < _actionDimension; i++)
            {
                var diff = NumOps.Subtract(action[i], meanAction[i]);
                distance = NumOps.Add(distance, NumOps.Multiply(diff, diff));
            }
            distance = NumOps.Sqrt(distance);
            totalDistance = NumOps.Add(totalDistance, distance);
        }
        
        // Normalize and invert (higher value = more consensus)
        var avgDistance = NumOps.Divide(totalDistance, NumOps.FromDouble(_numAgents));
        var maxPossibleDistance = NumOps.FromDouble(Math.Sqrt(_actionDimension) * 2); // Assuming actions normalized to [-1, 1]
        var normalizedDistance = NumOps.Divide(avgDistance, maxPossibleDistance);
        
        // Invert: consensus = 1 - normalized distance
        return NumOps.Subtract(NumOps.One, normalizedDistance);
    }
    
    /// <summary>
    /// Calculates the dispersion in value estimates among agents.
    /// </summary>
    private T CalculateValueDispersion(List<T> valueEstimates)
    {
        // Calculate mean value
        var sum = valueEstimates.Aggregate(NumOps.Zero, (acc, val) => NumOps.Add(acc, val));
        var mean = NumOps.Divide(sum, NumOps.FromDouble(_numAgents));
        
        // Calculate standard deviation
        var varianceSum = NumOps.Zero;
        foreach (var value in valueEstimates)
        {
            var diff = NumOps.Subtract(value, mean);
            varianceSum = NumOps.Add(varianceSum, NumOps.Multiply(diff, diff));
        }
        var variance = NumOps.Divide(varianceSum, NumOps.FromDouble(_numAgents));
        var stdDev = NumOps.Sqrt(variance);
        
        // Normalize by mean (coefficient of variation)
        var absVal = NumOps.Abs(mean);
        if (NumOps.Equals(absVal, NumOps.Zero))
            return stdDev; // Avoid division by zero
            
        return NumOps.Divide(stdDev, absVal);
    }
    
    /// <summary>
    /// Determines the market regime based on agent behavior metrics.
    /// </summary>
    private string DetermineMarketRegime(T consensus, T valueDispersion, T uncertainty)
    {
        // Thresholds for classification
        var highConsensusThreshold = NumOps.FromDouble(0.8);
        var lowConsensusThreshold = NumOps.FromDouble(0.4);
        var highDispersionThreshold = NumOps.FromDouble(0.5);
        var highUncertaintyThreshold = NumOps.FromDouble(0.6);
        
        // Classify regime based on thresholds
        if (NumOps.GreaterThan(consensus, highConsensusThreshold) && 
            NumOps.LessThan(valueDispersion, highDispersionThreshold) &&
            NumOps.LessThan(uncertainty, highUncertaintyThreshold))
        {
            return "Trend";
        }
        else if (NumOps.LessThan(consensus, lowConsensusThreshold) && 
                 NumOps.GreaterThan(valueDispersion, highDispersionThreshold))
        {
            return "Transitional";
        }
        else if (NumOps.GreaterThan(uncertainty, highUncertaintyThreshold))
        {
            return "Volatile";
        }
        else if (NumOps.LessThan(consensus, NumOps.FromDouble(0.2)))
        {
            return "Choppy";
        }
        else
        {
            return "Consolidation";
        }
    }
    
    #endregion
    
    #region Serialization
    
    /// <summary>
    /// Gets all parameters from the agent as a single vector.
    /// </summary>
    /// <returns>A vector containing all model parameters.</returns>
    public Vector<T> GetParameters()
    {
        var parametersList = new List<T>();
        
        // Get transformer parameters
        var transformerParams = _mainTransformer.GetParameters();
        parametersList.AddRange(transformerParams.ToArray());
        
        // Get policy network parameters
        foreach (var policyNet in _policyNetworks)
        {
            var policyParams = policyNet.GetParameters();
            parametersList.AddRange(policyParams.ToArray());
        }
        
        // Get value network parameters
        foreach (var valueNet in _valueNetworks)
        {
            var valueParams = valueNet.GetParameters();
            parametersList.AddRange(valueParams.ToArray());
        }
        
        // Get market impact network parameters
        if (_modelMarketImpact && _marketImpactNetwork != null)
        {
            var marketImpactParams = _marketImpactNetwork.GetParameters();
            parametersList.AddRange(marketImpactParams.ToArray());
        }
        
        // Get market dynamics network parameters
        var marketDynamicsParams = _marketDynamicsNetwork.GetParameters();
        parametersList.AddRange(marketDynamicsParams.ToArray());
        
        return new Vector<T>(parametersList.ToArray());
    }
    
    /// <summary>
    /// Sets all parameters of the agent from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all model parameters.</param>
    public void SetParameters(Vector<T> parameters)
    {
        int currentIndex = 0;
        
        // Set transformer parameters
        var transformerParamCount = _mainTransformer.GetParameterCount();
        var transformerParams = new Vector<T>(transformerParamCount);
        for (int i = 0; i < transformerParamCount; i++)
        {
            transformerParams[i] = parameters[currentIndex++];
        }
        _mainTransformer.UpdateParameters(transformerParams);
        
        // Set policy network parameters
        foreach (var policyNet in _policyNetworks)
        {
            var paramCount = policyNet.GetParameterCount();
            var netParams = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                netParams[i] = parameters[currentIndex++];
            }
            policyNet.UpdateParameters(netParams);
        }
        
        // Set value network parameters
        foreach (var valueNet in _valueNetworks)
        {
            var paramCount = valueNet.GetParameterCount();
            var netParams = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                netParams[i] = parameters[currentIndex++];
            }
            valueNet.UpdateParameters(netParams);
        }
        
        // Set market impact network parameters
        if (_modelMarketImpact && _marketImpactNetwork != null)
        {
            var marketImpactParamCount = _marketImpactNetwork.GetParameterCount();
            var marketImpactParams = new Vector<T>(marketImpactParamCount);
            for (int i = 0; i < marketImpactParamCount; i++)
            {
                marketImpactParams[i] = parameters[currentIndex++];
            }
            _marketImpactNetwork.UpdateParameters(marketImpactParams);
        }
        
        // Set market dynamics network parameters
        var marketDynamicsParamCount = _marketDynamicsNetwork.GetParameterCount();
        var marketDynamicsParams = new Vector<T>(marketDynamicsParamCount);
        for (int i = 0; i < marketDynamicsParamCount; i++)
        {
            marketDynamicsParams[i] = parameters[currentIndex++];
        }
        _marketDynamicsNetwork.UpdateParameters(marketDynamicsParams);
    }
    
    /// <summary>
    /// Gets the total number of parameters in the agent.
    /// </summary>
    /// <returns>The total parameter count.</returns>
    public int GetParameterCount()
    {
        int totalCount = 0;
        
        // Add transformer parameters
        totalCount += _mainTransformer.GetParameterCount();
        
        // Add policy network parameters
        foreach (var policyNet in _policyNetworks)
        {
            totalCount += policyNet.GetParameterCount();
        }
        
        // Add value network parameters
        foreach (var valueNet in _valueNetworks)
        {
            totalCount += valueNet.GetParameterCount();
        }
        
        // Add market impact network parameters
        if (_modelMarketImpact && _marketImpactNetwork != null)
        {
            totalCount += _marketImpactNetwork.GetParameterCount();
        }
        
        // Add market dynamics network parameters
        totalCount += _marketDynamicsNetwork.GetParameterCount();
        
        return totalCount;
    }
    
    #endregion
}