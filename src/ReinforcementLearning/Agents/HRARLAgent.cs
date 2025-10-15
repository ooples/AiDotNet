namespace AiDotNet.ReinforcementLearning.Agents;

/// <summary>
/// Agent implementing the Hierarchical Risk-Aware Reinforcement Learning (HRARL) architecture.
/// </summary>
/// <remarks>
/// This agent uses a hierarchical structure where a high-level policy makes strategic decisions
/// and a low-level policy makes tactical decisions, both while explicitly considering financial risk.
/// This approach is particularly effective for financial markets where managing risk is as important
/// as maximizing returns.
/// </remarks>
/// <typeparam name="TState">The type used to represent the environment state.</typeparam>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
public class HRARLAgent<TState, T> : AgentBase<TState, Vector<T>, T>
    where TState : Tensor<T>
{
    private readonly int _stateDimension;
    private readonly int _actionDimension;
    private readonly int _numHierarchicalLevels;
    private readonly int _highLevelHiddenDimension;
    private readonly int _lowLevelHiddenDimension;
    private readonly int _highLevelTimeHorizon;
    private readonly int _lowLevelTimeHorizon;
    private readonly int _goalDimension;
    private readonly bool _useRecurrentHighLevelPolicy;
    private readonly bool _useIntrinsicRewards;
    private readonly T _intrinsicRewardScale = default!;
    private readonly bool _useTargetNetwork;
    private readonly int _targetUpdateFrequency;
    private readonly bool _useHindsightExperienceReplay;
    private readonly int _riskMetricType; // 0 = Variance, 1 = VaR, 2 = CVaR
    private readonly T _confidenceLevel = default!;
    
    // Discount factors
    private readonly T _highLevelGamma = default!;
    private readonly T _lowLevelGamma = default!;
    
    // Risk aversion parameter
    private T _riskAversionParameter = default!;
    
    // Neural network components
    private readonly NeuralNetwork<T> _highLevelPolicy = default!;
    private readonly NeuralNetwork<T> _highLevelValue = default!;
    private readonly NeuralNetwork<T> _lowLevelPolicy = default!;
    private readonly NeuralNetwork<T> _lowLevelValue = default!;
    private readonly NeuralNetwork<T> _riskAssessmentNetwork = default!;
    private readonly NeuralNetwork<T> _distributionalValueNetwork = default!;
    
    // Target networks for stable learning
    private NeuralNetwork<T>? _highLevelValueTarget;
    private NeuralNetwork<T>? _lowLevelValueTarget;
    
    // Experience memories
    private readonly List<(TState state, Vector<T> goal, T reward)> _highLevelMemory;
    private readonly List<(TState state, Vector<T> goal, Vector<T> action, T reward, TState nextState, bool done)> _lowLevelMemory;
    
    // Optimizers
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _highLevelPolicyOptimizer = default!;
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _highLevelValueOptimizer = default!;
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _lowLevelPolicyOptimizer = default!;
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _lowLevelValueOptimizer = default!;
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _distributionalValueOptimizer = default!;
    
    // State tracking
    private int _updateCounter;
    private Vector<T> _currentGoal = default!;
    private TState? _lastHighLevelState;
    private int _lowLevelStepsCounter;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="HRARLAgent{TState, T}"/> class.
    /// </summary>
    /// <param name="stateDimension">The dimension of the state space.</param>
    /// <param name="actionDimension">The dimension of the action space.</param>
    /// <param name="numHierarchicalLevels">The number of hierarchical levels.</param>
    /// <param name="highLevelHiddenDimension">The hidden dimension size for high-level networks.</param>
    /// <param name="lowLevelHiddenDimension">The hidden dimension size for low-level networks.</param>
    /// <param name="highLevelTimeHorizon">The time horizon for high-level decisions.</param>
    /// <param name="lowLevelTimeHorizon">The time horizon for low-level decisions.</param>
    /// <param name="highLevelGamma">The discount factor for high-level rewards.</param>
    /// <param name="lowLevelGamma">The discount factor for low-level rewards.</param>
    /// <param name="highLevelLearningRate">The learning rate for high-level networks.</param>
    /// <param name="lowLevelLearningRate">The learning rate for low-level networks.</param>
    /// <param name="highLevelEntropyCoef">The entropy coefficient for high-level exploration.</param>
    /// <param name="lowLevelEntropyCoef">The entropy coefficient for low-level exploration.</param>
    /// <param name="riskMetricType">The type of risk metric to use.</param>
    /// <param name="confidenceLevel">The confidence level for VaR/CVaR calculations.</param>
    /// <param name="useRecurrentHighLevelPolicy">Whether to use a recurrent network for high-level policy.</param>
    /// <param name="useIntrinsicRewards">Whether to use intrinsic rewards for exploration.</param>
    /// <param name="intrinsicRewardScale">The scale factor for intrinsic rewards.</param>
    /// <param name="useTargetNetwork">Whether to use target networks for stable learning.</param>
    /// <param name="targetUpdateFrequency">How often to update target networks.</param>
    /// <param name="useHindsightExperienceReplay">Whether to use hindsight experience replay.</param>
    /// <param name="batchSize">The batch size to use for training.</param>
    /// <param name="seed">Optional seed for the random number generator.</param>
    public HRARLAgent(
        int stateDimension,
        int actionDimension,
        int numHierarchicalLevels,
        int highLevelHiddenDimension,
        int lowLevelHiddenDimension,
        int highLevelTimeHorizon,
        int lowLevelTimeHorizon,
        double highLevelGamma,
        double lowLevelGamma,
        double highLevelLearningRate,
        double lowLevelLearningRate,
        double highLevelEntropyCoef,
        double lowLevelEntropyCoef,
        int riskMetricType,
        double confidenceLevel,
        bool useRecurrentHighLevelPolicy,
        bool useIntrinsicRewards,
        double intrinsicRewardScale,
        bool useTargetNetwork,
        int targetUpdateFrequency,
        bool useHindsightExperienceReplay,
        int batchSize = 32,
        int? seed = null)
        : base(lowLevelGamma, 0.001, batchSize, seed)
    {
        _stateDimension = stateDimension;
        _actionDimension = actionDimension;
        _numHierarchicalLevels = numHierarchicalLevels;
        _highLevelHiddenDimension = highLevelHiddenDimension;
        _lowLevelHiddenDimension = lowLevelHiddenDimension;
        _highLevelTimeHorizon = highLevelTimeHorizon;
        _lowLevelTimeHorizon = lowLevelTimeHorizon;
        _highLevelGamma = NumOps.FromDouble(highLevelGamma);
        _lowLevelGamma = NumOps.FromDouble(lowLevelGamma);
        _useRecurrentHighLevelPolicy = useRecurrentHighLevelPolicy;
        _useIntrinsicRewards = useIntrinsicRewards;
        _intrinsicRewardScale = NumOps.FromDouble(intrinsicRewardScale);
        _useTargetNetwork = useTargetNetwork;
        _targetUpdateFrequency = targetUpdateFrequency;
        _useHindsightExperienceReplay = useHindsightExperienceReplay;
        _riskMetricType = riskMetricType;
        _confidenceLevel = NumOps.FromDouble(confidenceLevel);
        
        // Default risk aversion parameter
        _riskAversionParameter = NumOps.FromDouble(0.5);
        
        // Define the goal dimension - this is what the high-level policy outputs
        // For financial applications, this might include:
        // - Target asset allocation
        // - Risk budget
        // - Time horizon for investment
        _goalDimension = Math.Max(2, _actionDimension);
        
        // Initialize memories
        _highLevelMemory = new List<(TState, Vector<T>, T)>();
        _lowLevelMemory = new List<(TState, Vector<T>, Vector<T>, T, TState, bool)>();
        
        // Create high-level policy network
        _highLevelPolicy = CreateHighLevelPolicyNetwork();
        
        // Create high-level value network
        _highLevelValue = CreateHighLevelValueNetwork();
        
        // Create low-level policy network
        _lowLevelPolicy = CreateLowLevelPolicyNetwork();
        
        // Create low-level value network
        _lowLevelValue = CreateLowLevelValueNetwork();
        
        // Create risk assessment network
        _riskAssessmentNetwork = CreateRiskAssessmentNetwork();
        
        // Create distributional value network for risk-aware decision making
        _distributionalValueNetwork = CreateDistributionalValueNetwork();
        
        // Create target networks if enabled
        if (_useTargetNetwork)
        {
            _highLevelValueTarget = (NeuralNetwork<T>)_highLevelValue.Clone()!;
            _lowLevelValueTarget = (NeuralNetwork<T>)_lowLevelValue.Clone()!;
        }
        
        // Create optimizers
        var highLevelPolicyOptions = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            LearningRate = highLevelLearningRate,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        };
        
        var highLevelValueOptions = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            LearningRate = highLevelLearningRate,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        };
        
        var lowLevelPolicyOptions = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            LearningRate = lowLevelLearningRate,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        };
        
        var lowLevelValueOptions = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            LearningRate = lowLevelLearningRate,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        };
        
        var distributionalValueOptions = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            LearningRate = lowLevelLearningRate,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        };
        
        _highLevelPolicyOptimizer = OptimizerFactory<T, Tensor<T>, Tensor<T>>.CreateOptimizer(OptimizerType.Adam, highLevelPolicyOptions);
        _highLevelValueOptimizer = OptimizerFactory<T, Tensor<T>, Tensor<T>>.CreateOptimizer(OptimizerType.Adam, highLevelValueOptions);
        _lowLevelPolicyOptimizer = OptimizerFactory<T, Tensor<T>, Tensor<T>>.CreateOptimizer(OptimizerType.Adam, lowLevelPolicyOptions);
        _lowLevelValueOptimizer = OptimizerFactory<T, Tensor<T>, Tensor<T>>.CreateOptimizer(OptimizerType.Adam, lowLevelValueOptions);
        _distributionalValueOptimizer = OptimizerFactory<T, Tensor<T>, Tensor<T>>.CreateOptimizer(OptimizerType.Adam, distributionalValueOptions);
        
        // Initialize state tracking variables
        _updateCounter = 0;
        _lowLevelStepsCounter = 0;
        
        // Initialize the current goal with zeros
        _currentGoal = new Vector<T>(_goalDimension);
        
        // Set the last high-level state to a default value
        _lastHighLevelState = default;
        
        // Initialize LastLoss
        LastLoss = NumOps.Zero;
    }
    
    /// <summary>
    /// Sets the risk aversion parameter.
    /// </summary>
    /// <param name="riskAversion">The risk aversion parameter (0-1 scale).</param>
    public void SetRiskAversionParameter(T riskAversion)
    {
        _riskAversionParameter = riskAversion;
    }
    
    /// <summary>
    /// Selects an action using the hierarchical policy structure.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <param name="isTraining">Whether the agent is in training mode.</param>
    /// <returns>The selected action.</returns>
    public override Vector<T> SelectAction(TState state, bool isTraining = true)
    {
        // Update goal from high-level policy when needed
        if (_lowLevelStepsCounter % _lowLevelTimeHorizon == 0 || _lastHighLevelState == default)
        {
            _currentGoal = GetHighLevelGoal(state);
            _lastHighLevelState = state;
        }
        
        // Increment low-level steps counter
        _lowLevelStepsCounter++;
        IncrementStepCounter();
        
        // Combine state and goal for low-level policy input
        var lowLevelInput = CombineStateAndGoal(state, _currentGoal);
        
        // Get action distribution from low-level policy
        var lowLevelInputTensor = new Tensor<T>(new[] { 1, lowLevelInput.Length });
        for (int i = 0; i < lowLevelInput.Length; i++)
        {
            lowLevelInputTensor[0, i] = lowLevelInput[i];
        }
        var lowLevelPolicyOutput = _lowLevelPolicy.Predict(lowLevelInputTensor);
        
        // Extract action mean and log variance (assuming Gaussian policy)
        var actionMean = new Vector<T>(_actionDimension);
        var actionLogVar = new Vector<T>(_actionDimension);
        
        for (int i = 0; i < _actionDimension; i++)
        {
            actionMean[i] = lowLevelPolicyOutput[i];
            actionLogVar[i] = lowLevelPolicyOutput[i + _actionDimension];
        }
        
        // Sample action from distribution if training, otherwise use mean
        if (isTraining)
        {
            return SampleAction(actionMean, actionLogVar);
        }
        else
        {
            return actionMean;
        }
    }
    
    /// <summary>
    /// Gets the high-level goal for the current state.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <returns>The high-level goal vector.</returns>
    public Vector<T> GetHighLevelGoal(TState state)
    {
        // Process state through high-level policy
        var highLevelInput = CreateHighLevelInput(state);
        var highLevelInputTensor = new Tensor<T>(new[] { 1, highLevelInput.Length });
        for (int i = 0; i < highLevelInput.Length; i++)
        {
            highLevelInputTensor[0, i] = highLevelInput[i];
        }
        var highLevelPolicyOutput = _highLevelPolicy.Predict(highLevelInputTensor);
        
        // Extract goal mean and log variance
        var goalMean = new Vector<T>(_goalDimension);
        var goalLogVar = new Vector<T>(_goalDimension);
        
        for (int i = 0; i < _goalDimension; i++)
        {
            goalMean[i] = highLevelPolicyOutput[i];
            goalLogVar[i] = highLevelPolicyOutput[i + _goalDimension];
        }
        
        // Sample goal from distribution during training, use mean otherwise
        return goalMean;
    }
    
    /// <summary>
    /// Updates the agent based on the observed transition.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <param name="action">The action taken.</param>
    /// <param name="reward">The reward received.</param>
    /// <param name="nextState">The next state.</param>
    /// <param name="done">Whether the episode is done.</param>
    public override void Learn(TState state, Vector<T> action, T reward, TState nextState, bool done)
    {
        // Add experience to low-level memory
        _lowLevelMemory.Add((state, _currentGoal, action, reward, nextState, done));
        
        // Update high-level memory if it's time for a high-level update
        if (_lowLevelStepsCounter % _highLevelTimeHorizon == 0 || done)
        {
            // Accumulate rewards over high-level time horizon
            T accumulatedReward = AccumulateHighLevelReward();
            
            // Add to high-level memory
            if (_lastHighLevelState != null)
            {
                _highLevelMemory.Add((_lastHighLevelState, _currentGoal, accumulatedReward));
            }
            
            // Reset accumulated reward
            if (done)
            {
                _lowLevelStepsCounter = 0;
            }
        }
        
        // Update networks
        T lowLevelLoss = UpdateLowLevelNetworks();
        T highLevelLoss = UpdateHighLevelNetworks();
        T distributionalLoss = UpdateDistributionalValueNetwork();
        
        // Update target networks if enabled
        if (_useTargetNetwork && _updateCounter % _targetUpdateFrequency == 0)
        {
            UpdateTargetNetworks();
        }
        
        // Increment update counter
        _updateCounter++;
        
        // Set combined loss
        LastLoss = NumOps.Add(NumOps.Add(lowLevelLoss, highLevelLoss), distributionalLoss);
    }
    
    /// <summary>
    /// Gets the risk-adjusted expected value for a state and action.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <param name="action">The action to evaluate.</param>
    /// <returns>The risk-adjusted expected value.</returns>
    public T GetRiskAdjustedValue(TState state, Vector<T> action)
    {
        var expectedValue = GetExpectedValue(state, action);
        var riskMeasure = GetRiskMeasure(state, action);
        
        // Risk-adjusted value = expectedValue - riskAversionParameter * riskMeasure
        return NumOps.Subtract(
            expectedValue,
            NumOps.Multiply(_riskAversionParameter, riskMeasure));
    }
    
    /// <summary>
    /// Gets the expected value for a state and action.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <param name="action">The action to evaluate.</param>
    /// <returns>The expected value.</returns>
    public T GetExpectedValue(TState state, Vector<T> action)
    {
        // Combine state and action
        var stateActionInput = CombineStateAndAction(state, action);
        
        // Get value from low-level value network
        var stateActionTensor = new Tensor<T>(new[] { 1, stateActionInput.Length });
        for (int i = 0; i < stateActionInput.Length; i++)
        {
            stateActionTensor[0, i] = stateActionInput[i];
        }
        var valueOutput = _lowLevelValue.Predict(stateActionTensor);
        return valueOutput[0, 0];
    }
    
    /// <summary>
    /// Gets the distribution of possible values for a state and action.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <param name="action">The action to evaluate (if default, uses best action).</param>
    /// <returns>A vector representing the distribution of possible returns.</returns>
    public Vector<T> GetValueDistribution(TState state, Vector<T>? action = null)
    {
        // If no action provided, use the best action
        if (action == null)
        {
            action = SelectAction(state, false);
        }
        
        // Combine state and action
        var stateActionInput = CombineStateAndAction(state, action);
        
        // Get distribution from distributional value network
        var stateActionTensor = new Tensor<T>(new[] { 1, stateActionInput.Length });
        for (int i = 0; i < stateActionInput.Length; i++)
        {
            stateActionTensor[0, i] = stateActionInput[i];
        }
        var output = _distributionalValueNetwork.Predict(stateActionTensor);
        // Convert tensor output to vector
        var result = new Vector<T>(output.Shape[output.Shape.Length - 1]);
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = output.Shape.Length == 1 ? output[i] : output[0, i];
        }
        return result;
    }
    
    /// <summary>
    /// Gets the variance of returns for a state and action.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <param name="action">The action to evaluate.</param>
    /// <returns>The variance of returns.</returns>
    public T GetReturnVariance(TState state, Vector<T> action)
    {
        var distribution = GetValueDistribution(state, action);
        var mean = CalculateDistributionMean(distribution);
        
        // Calculate variance
        T variance = NumOps.Zero;
        for (int i = 0; i < distribution.Length; i++)
        {
            // The distribution values are assumed to be probabilities of discrete return values
            // For simplicity, we use the index as the return value (scaled appropriately)
            T value = NumOps.FromDouble((double)i / distribution.Length * 2 - 1); // Scale to [-1, 1]
            T diff = NumOps.Subtract(value, mean);
            variance = NumOps.Add(variance, NumOps.Multiply(distribution[i], NumOps.Multiply(diff, diff)));
        }
        
        return variance;
    }
    
    /// <summary>
    /// Gets the Value at Risk (VaR) for a state and action.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <param name="action">The action to evaluate.</param>
    /// <param name="confidenceLevel">The confidence level for VaR calculation.</param>
    /// <returns>The Value at Risk estimate.</returns>
    public T GetValueAtRisk(TState state, Vector<T> action, T confidenceLevel)
    {
        var distribution = GetValueDistribution(state, action);
        
        // We need to convert the distribution into a cumulative distribution
        var cumulativeDistribution = new Vector<T>(distribution.Length);
        T sum = NumOps.Zero;
        
        for (int i = 0; i < distribution.Length; i++)
        {
            sum = NumOps.Add(sum, distribution[i]);
            cumulativeDistribution[i] = sum;
        }
        
        // Find the VaR threshold - the return value at which the cumulative probability
        // crosses the confidence level
        for (int i = 0; i < cumulativeDistribution.Length; i++)
        {
            if (NumOps.GreaterThanOrEquals(cumulativeDistribution[i], confidenceLevel))
            {
                // Convert index to a return value scaled to [-1, 1]
                return NumOps.FromDouble((double)i / distribution.Length * 2 - 1);
            }
        }
        
        // If we didn't find a threshold (should be rare), return the worst possible outcome
        return NumOps.FromDouble(-1.0);
    }
    
    /// <summary>
    /// Gets the Conditional Value at Risk (CVaR) for a state and action.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <param name="action">The action to evaluate.</param>
    /// <param name="confidenceLevel">The confidence level for CVaR calculation.</param>
    /// <returns>The Conditional Value at Risk estimate.</returns>
    public T GetConditionalValueAtRisk(TState state, Vector<T> action, T confidenceLevel)
    {
        var distribution = GetValueDistribution(state, action);
        var var = GetValueAtRisk(state, action, confidenceLevel);
        
        // Find index corresponding to VaR
        int varIndex = (int)(Convert.ToDouble(NumOps.Divide(NumOps.Add(var, NumOps.FromDouble(1.0)), NumOps.FromDouble(2.0))) * distribution.Length);
        
        // Calculate CVaR as the expected value in the tail (beyond VaR)
        T tailSum = NumOps.Zero;
        T tailProbSum = NumOps.Zero;
        
        for (int i = 0; i < varIndex; i++)
        {
            // Convert index to a return value scaled to [-1, 1]
            T value = NumOps.FromDouble((double)i / distribution.Length * 2 - 1);
            tailSum = NumOps.Add(tailSum, NumOps.Multiply(value, distribution[i]));
            tailProbSum = NumOps.Add(tailProbSum, distribution[i]);
        }
        
        // Avoid division by zero
        if (NumOps.Equals(tailProbSum, NumOps.Zero))
        {
            return var;
        }
        
        return NumOps.Divide(tailSum, tailProbSum);
    }
    
    /// <summary>
    /// Gets the probability of achieving a positive return with a given action.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <param name="action">The action to evaluate.</param>
    /// <returns>The probability of a positive return.</returns>
    public T GetProbabilityOfPositiveReturn(TState state, Vector<T> action)
    {
        var distribution = GetValueDistribution(state, action);
        
        // Sum probability mass for positive returns
        T probPositive = NumOps.Zero;
        for (int i = distribution.Length / 2; i < distribution.Length; i++)
        {
            probPositive = NumOps.Add(probPositive, distribution[i]);
        }
        
        return probPositive;
    }
    
    /// <summary>
    /// Assesses the current market risk level based on the state.
    /// </summary>
    /// <param name="state">The current market state.</param>
    /// <returns>The risk assessment (0-1 scale, higher = more risky).</returns>
    public T AssessMarketRisk(TState state)
    {
        // Use the risk assessment network to estimate current market risk
        var stateVector = CreateStateVector(state);
        var stateTensor = new Tensor<T>(new[] { 1, stateVector.Length });
        for (int i = 0; i < stateVector.Length; i++)
        {
            stateTensor[0, i] = stateVector[i];
        }
        var riskOutput = _riskAssessmentNetwork.Predict(stateTensor);
        
        // Ensure the risk is in the [0, 1] range
        return ClampToRange(riskOutput[0], NumOps.Zero, NumOps.One);
    }
    
    /// <summary>
    /// Simulates the agent's strategy over multiple time steps with Monte Carlo sampling.
    /// </summary>
    /// <param name="initialState">The initial state.</param>
    /// <param name="numSteps">The number of steps to simulate.</param>
    /// <param name="numSimulations">The number of Monte Carlo simulations.</param>
    /// <returns>A list of return trajectories from the simulations.</returns>
    public List<Vector<T>> SimulateStrategy(TState initialState, int numSteps, int numSimulations)
    {
        var simulationResults = new List<Vector<T>>();
        
        // Run multiple simulations
        for (int sim = 0; sim < numSimulations; sim++)
        {
            var returnTrajectory = new Vector<T>(numSteps);
            T cumulativeReturn = NumOps.Zero;
            TState state = initialState;
            
            // Reset the goal and low-level steps counter for this simulation
            _currentGoal = GetHighLevelGoal(state);
            _lowLevelStepsCounter = 0;
            
            // Simulate the strategy over numSteps
            for (int step = 0; step < numSteps; step++)
            {
                // Select action using the hierarchy
                var action = SelectAction(state, false);
                
                // Get the expected reward for this action
                var expectedValue = GetExpectedValue(state, action);
                
                // Add some randomness to simulate real-world uncertainty
                var noise = NumOps.FromDouble(SampleGaussian() * 0.2); // 20% noise
                var reward = NumOps.Add(expectedValue, noise);
                
                // Update cumulative return
                cumulativeReturn = NumOps.Add(cumulativeReturn, reward);
                returnTrajectory[step] = cumulativeReturn;
                
                // Guess the next state (simplified for simulation purposes)
                state = SimulateNextState(state, action, reward);
                
                // Update goal if needed
                if (_lowLevelStepsCounter % _lowLevelTimeHorizon == 0)
                {
                    _currentGoal = GetHighLevelGoal(state);
                }
                
                // Increment steps counter
                _lowLevelStepsCounter++;
            }
            
            simulationResults.Add(returnTrajectory);
        }
        
        return simulationResults;
    }
    
    /// <summary>
    /// Gets the risk measure based on the configured risk metric type.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <param name="action">The action to evaluate.</param>
    /// <returns>The risk measure value.</returns>
    private T GetRiskMeasure(TState state, Vector<T> action)
    {
        switch (_riskMetricType)
        {
            case 0: // Variance
                return GetReturnVariance(state, action);
            case 1: // Value at Risk (VaR)
                return NumOps.Negate(GetValueAtRisk(state, action, _confidenceLevel));
            case 2: // Conditional Value at Risk (CVaR)
                return NumOps.Negate(GetConditionalValueAtRisk(state, action, _confidenceLevel));
            default:
                return GetReturnVariance(state, action);
        }
    }
    
    /// <summary>
    /// Saves the agent's models to the specified path.
    /// </summary>
    /// <param name="filePath">The path to save the models to.</param>
    public override void Save(string filePath)
    {
        _highLevelPolicy.SaveModel($"{filePath}_high_level_policy");
        _highLevelValue.SaveModel($"{filePath}_high_level_value");
        _lowLevelPolicy.SaveModel($"{filePath}_low_level_policy");
        _lowLevelValue.SaveModel($"{filePath}_low_level_value");
        _riskAssessmentNetwork.SaveModel($"{filePath}_risk_assessment");
        _distributionalValueNetwork.SaveModel($"{filePath}_distributional_value");
    }
    
    /// <summary>
    /// Loads the agent's models from the specified path.
    /// </summary>
    /// <param name="filePath">The path to load the models from.</param>
    public override void Load(string filePath)
    {
        _highLevelPolicy.LoadModel($"{filePath}_high_level_policy");
        _highLevelValue.LoadModel($"{filePath}_high_level_value");
        _lowLevelPolicy.LoadModel($"{filePath}_low_level_policy");
        _lowLevelValue.LoadModel($"{filePath}_low_level_value");
        _riskAssessmentNetwork.LoadModel($"{filePath}_risk_assessment");
        _distributionalValueNetwork.LoadModel($"{filePath}_distributional_value");
        
        if (_useTargetNetwork)
        {
            _highLevelValueTarget = (NeuralNetwork<T>)_highLevelValue.Clone()!;
            _lowLevelValueTarget = (NeuralNetwork<T>)_lowLevelValue.Clone()!;
        }
    }
    
    /// <summary>
    /// Sets the agent's training mode.
    /// </summary>
    /// <param name="isTraining">Whether the agent should be in training mode.</param>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        
        // Set neural networks to training or evaluation mode
        _highLevelPolicy.SetTrainingMode(isTraining);
        _highLevelValue.SetTrainingMode(isTraining);
        _lowLevelPolicy.SetTrainingMode(isTraining);
        _lowLevelValue.SetTrainingMode(isTraining);
        _riskAssessmentNetwork.SetTrainingMode(isTraining);
        _distributionalValueNetwork.SetTrainingMode(isTraining);
        
        // Target networks are always in evaluation mode
        if (_useTargetNetwork)
        {
            _highLevelValueTarget?.SetTrainingMode(false);
            _lowLevelValueTarget?.SetTrainingMode(false);
        }
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
    /// Trains the agent on a batch of experiences.
    /// </summary>
    /// <param name="states">Batch of states.</param>
    /// <param name="actions">Batch of actions.</param>
    /// <param name="rewards">Batch of rewards.</param>
    /// <param name="nextStates">Batch of next states.</param>
    /// <param name="dones">Batch of done flags.</param>
    public void TrainOnBatch(TState states, Vector<T>[] actions, Vector<T> rewards, TState nextStates, Vector<T> dones)
    {
        // Process each state-action pair in the batch
        int batchSize = states.Shape[0];
        T totalLoss = NumOps.Zero;
        
        for (int i = 0; i < batchSize; i++)
        {
            // Extract individual state, action, reward, next state, and done flag
            var state = (TState)states.GetSlice(i);
            var action = actions[i];
            var reward = rewards[i];
            var nextState = (TState)nextStates.GetSlice(i);
            var done = NumOps.GreaterThan(dones[i], NumOps.FromDouble(0.5)); // Convert to boolean
            
            // Use the learn method to update networks
            Learn(state, action, reward, nextState, done);
            
            // Accumulate loss
            totalLoss = NumOps.Add(totalLoss, LastLoss);
        }
        
        // Average the loss across the batch
        if (batchSize > 0)
        {
            LastLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
        }
    }
    
    #region Helper Methods
    
    /// <summary>
    /// Creates the high-level policy network.
    /// </summary>
    private NeuralNetwork<T> CreateHighLevelPolicyNetwork()
    {
        // Create layers list
        var layers = new List<ILayer<T>>();
        
        if (_useRecurrentHighLevelPolicy)
        {
            // For recurrent policy, use LSTM
            layers.Add(new InputLayer<T>(_stateDimension));
            layers.Add(new DenseLayer<T>(_stateDimension, _highLevelHiddenDimension, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.Tanh)));
            layers.Add(new LSTMLayer<T>(_highLevelHiddenDimension, _highLevelHiddenDimension, new[] { _highLevelHiddenDimension }, 
                (IActivationFunction<T>?)null, (IActivationFunction<T>?)null));
            layers.Add(new DenseLayer<T>(_highLevelHiddenDimension, _highLevelHiddenDimension / 2, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.Tanh)));
            layers.Add(new DenseLayer<T>(_highLevelHiddenDimension / 2, _goalDimension * 2, (IActivationFunction<T>?)null)); // Mean and log var
        }
        else
        {
            // For feedforward policy
            layers.Add(new InputLayer<T>(_stateDimension));
            layers.Add(new DenseLayer<T>(_stateDimension, _highLevelHiddenDimension, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU)));
            layers.Add(new DenseLayer<T>(_highLevelHiddenDimension, _highLevelHiddenDimension, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU)));
            layers.Add(new DenseLayer<T>(_highLevelHiddenDimension, _highLevelHiddenDimension / 2, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU)));
            layers.Add(new DenseLayer<T>(_highLevelHiddenDimension / 2, _goalDimension * 2, (IActivationFunction<T>?)null)); // Mean and log var
        }
        
        // Create neural network architecture
        var architecture = new NeuralNetworkArchitecture<T>(
            complexity: NetworkComplexity.Medium,
            taskType: NeuralNetworkTaskType.Regression,
            shouldReturnFullSequence: false,
            layers: layers,
            isDynamicSampleCount: true,
            isPlaceholder: false);
        
        // Create and return the network
        return new NeuralNetwork<T>(architecture);
    }
    
    /// <summary>
    /// Creates the high-level value network.
    /// </summary>
    private NeuralNetwork<T> CreateHighLevelValueNetwork()
    {
        // Create layers list
        var layers = new List<ILayer<T>>();
        
        // Input: state + goal
        layers.Add(new InputLayer<T>(_stateDimension + _goalDimension));
        layers.Add(new DenseLayer<T>(_stateDimension + _goalDimension, _highLevelHiddenDimension, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU)));
        layers.Add(new DenseLayer<T>(_highLevelHiddenDimension, _highLevelHiddenDimension / 2, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU)));
        layers.Add(new DenseLayer<T>(_highLevelHiddenDimension / 2, 1, (IActivationFunction<T>?)null));
        
        // Create neural network architecture
        var architecture = new NeuralNetworkArchitecture<T>(
            complexity: NetworkComplexity.Medium,
            taskType: NeuralNetworkTaskType.Regression,
            shouldReturnFullSequence: false,
            layers: layers,
            isDynamicSampleCount: true,
            isPlaceholder: false);
        
        // Create and return the network
        return new NeuralNetwork<T>(architecture);
    }
    
    /// <summary>
    /// Creates the low-level policy network.
    /// </summary>
    private NeuralNetwork<T> CreateLowLevelPolicyNetwork()
    {
        // Create layers list
        var layers = new List<ILayer<T>>();
        
        // Input: state + goal
        layers.Add(new InputLayer<T>(_stateDimension + _goalDimension));
        layers.Add(new DenseLayer<T>(_stateDimension + _goalDimension, _lowLevelHiddenDimension, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU)));
        layers.Add(new DenseLayer<T>(_lowLevelHiddenDimension, _lowLevelHiddenDimension, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU)));
        layers.Add(new DenseLayer<T>(_lowLevelHiddenDimension, _lowLevelHiddenDimension / 2, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU)));
        layers.Add(new DenseLayer<T>(_lowLevelHiddenDimension / 2, _actionDimension * 2, (IActivationFunction<T>?)null)); // Mean and log var
        
        // Create neural network architecture
        var architecture = new NeuralNetworkArchitecture<T>(
            complexity: NetworkComplexity.Medium,
            taskType: NeuralNetworkTaskType.Regression,
            shouldReturnFullSequence: false,
            layers: layers,
            isDynamicSampleCount: true,
            isPlaceholder: false);
        
        // Create and return the network
        return new NeuralNetwork<T>(architecture);
    }
    
    /// <summary>
    /// Creates the low-level value network.
    /// </summary>
    private NeuralNetwork<T> CreateLowLevelValueNetwork()
    {
        // Create layers list
        var layers = new List<ILayer<T>>();
        
        // Input: state + action
        layers.Add(new InputLayer<T>(_stateDimension + _actionDimension));
        layers.Add(new DenseLayer<T>(_stateDimension + _actionDimension, _lowLevelHiddenDimension, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU)));
        layers.Add(new DenseLayer<T>(_lowLevelHiddenDimension, _lowLevelHiddenDimension / 2, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU)));
        layers.Add(new DenseLayer<T>(_lowLevelHiddenDimension / 2, 1, (IActivationFunction<T>?)null));
        
        // Create neural network architecture
        var architecture = new NeuralNetworkArchitecture<T>(
            complexity: NetworkComplexity.Medium,
            taskType: NeuralNetworkTaskType.Regression,
            shouldReturnFullSequence: false,
            layers: layers,
            isDynamicSampleCount: true,
            isPlaceholder: false);
        
        // Create and return the network
        return new NeuralNetwork<T>(architecture);
    }
    
    /// <summary>
    /// Creates the risk assessment network.
    /// </summary>
    private NeuralNetwork<T> CreateRiskAssessmentNetwork()
    {
        // Create layers list
        var layers = new List<ILayer<T>>();
        
        // Input: state
        layers.Add(new InputLayer<T>(_stateDimension));
        layers.Add(new DenseLayer<T>(_stateDimension, 128, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU)));
        layers.Add(new DenseLayer<T>(128, 64, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU)));
        layers.Add(new DenseLayer<T>(64, 1, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.Sigmoid))); // Output in [0, 1]
        
        // Create neural network architecture
        var architecture = new NeuralNetworkArchitecture<T>(
            complexity: NetworkComplexity.Medium,
            taskType: NeuralNetworkTaskType.Regression,
            shouldReturnFullSequence: false,
            layers: layers,
            isDynamicSampleCount: true,
            isPlaceholder: false);
        
        // Create and return the network
        return new NeuralNetwork<T>(architecture);
    }
    
    /// <summary>
    /// Creates the distributional value network for risk assessment.
    /// </summary>
    private NeuralNetwork<T> CreateDistributionalValueNetwork()
    {
        // Create layers list
        var layers = new List<ILayer<T>>();
        
        // Distribution is represented by a vector of 51 values (common in distributional RL)
        int distributionSize = 51;
        
        // Input: state + action
        layers.Add(new InputLayer<T>(_stateDimension + _actionDimension));
        layers.Add(new DenseLayer<T>(_stateDimension + _actionDimension, 256, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU)));
        layers.Add(new DenseLayer<T>(256, 256, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU)));
        layers.Add(new DenseLayer<T>(256, 128, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU)));
        layers.Add(new DenseLayer<T>(128, distributionSize, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.Softmax))); // Output is a probability distribution
        
        // Create neural network architecture
        var architecture = new NeuralNetworkArchitecture<T>(
            complexity: NetworkComplexity.Medium,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            shouldReturnFullSequence: false,
            layers: layers,
            isDynamicSampleCount: true,
            isPlaceholder: false);
        
        // Create and return the network
        return new NeuralNetwork<T>(architecture);
    }
    
    /// <summary>
    /// Combines a state tensor and goal vector into a single input vector.
    /// </summary>
    private Vector<T> CombineStateAndGoal(TState state, Vector<T> goal)
    {
        var combined = new Vector<T>(_stateDimension + _goalDimension);
        
        // Copy state into the combined vector
        for (int i = 0; i < _stateDimension; i++)
        {
            combined[i] = state[0, i];
        }
        
        // Copy goal into the combined vector
        for (int i = 0; i < _goalDimension; i++)
        {
            combined[_stateDimension + i] = goal[i];
        }
        
        return combined;
    }
    
    /// <summary>
    /// Combines a state tensor and action vector into a single input vector.
    /// </summary>
    private Vector<T> CombineStateAndAction(TState state, Vector<T> action)
    {
        var combined = new Vector<T>(_stateDimension + _actionDimension);
        
        // Copy state into the combined vector
        for (int i = 0; i < _stateDimension; i++)
        {
            combined[i] = state[0, i];
        }
        
        // Copy action into the combined vector
        for (int i = 0; i < _actionDimension; i++)
        {
            combined[_stateDimension + i] = action[i];
        }
        
        return combined;
    }
    
    /// <summary>
    /// Creates an input vector from a state tensor.
    /// </summary>
    private Vector<T> CreateStateVector(TState state)
    {
        var stateVector = new Vector<T>(_stateDimension);
        
        for (int i = 0; i < _stateDimension; i++)
        {
            stateVector[i] = state[0, i];
        }
        
        return stateVector;
    }
    
    /// <summary>
    /// Creates input for the high-level policy network.
    /// </summary>
    private Vector<T> CreateHighLevelInput(TState state)
    {
        // For a simple feedforward policy, just convert the state tensor to a vector
        return CreateStateVector(state);
    }
    
    /// <summary>
    /// Samples an action from a Gaussian distribution with the given mean and log variance.
    /// </summary>
    private Vector<T> SampleAction(Vector<T> mean, Vector<T> logVar)
    {
        var action = new Vector<T>(_actionDimension);
        
        for (int i = 0; i < _actionDimension; i++)
        {
            // Convert log variance to standard deviation
            var stdDev = NumOps.Exp(NumOps.Multiply(logVar[i], NumOps.FromDouble(0.5)));
            
            // Sample from normal distribution: mean + std * random normal
            var noise = NumOps.FromDouble(SampleGaussian());
            action[i] = NumOps.Add(mean[i], NumOps.Multiply(stdDev, noise));
            
            // Clamp to ensure the action is in a valid range [-1, 1]
            action[i] = ClampToRange(action[i], NumOps.FromDouble(-1.0), NumOps.FromDouble(1.0));
        }
        
        return action;
    }
    
    /// <summary>
    /// Samples a value from a standard Gaussian distribution (mean 0, std 1).
    /// </summary>
    private double SampleGaussian()
    {
        // Box-Muller transform
        double u1 = 1.0 - Random.NextDouble(); // uniform(0,1] random doubles
        double u2 = 1.0 - Random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }
    
    /// <summary>
    /// Clamps a value to the specified range.
    /// </summary>
    private T ClampToRange(T value, T min, T max)
    {
        if (NumOps.LessThan(value, min))
            return min;
        if (NumOps.GreaterThan(value, max))
            return max;
        return value;
    }
    
    /// <summary>
    /// Accumulates rewards for the high-level policy over the time horizon.
    /// </summary>
    private T AccumulateHighLevelReward()
    {
        // Simple summation of recent low-level rewards
        // In a more sophisticated implementation, we might use discounting
        T accumulated = NumOps.Zero;
        
        int count = Math.Min(_lowLevelMemory.Count, _highLevelTimeHorizon);
        for (int i = 0; i < count; i++)
        {
            int idx = _lowLevelMemory.Count - 1 - i;
            if (idx >= 0)
            {
                accumulated = NumOps.Add(accumulated, _lowLevelMemory[idx].reward);
            }
        }
        
        return accumulated;
    }
    
    /// <summary>
    /// Updates the high-level networks (policy and value).
    /// </summary>
    private T UpdateHighLevelNetworks()
    {
        // Skip if we don't have enough high-level experiences
        if (_highLevelMemory.Count < 2)
            return NumOps.Zero;
        
        // Sample a batch of experiences
        var batchIndices = SampleIndices(_highLevelMemory.Count, Math.Min(8, _highLevelMemory.Count));
        T totalLoss = NumOps.Zero;
        
        foreach (int idx in batchIndices)
        {
            var (state, goal, reward) = _highLevelMemory[idx];
            
            // Update high-level policy
            var highLevelInput = CreateHighLevelInput(state);
            var highLevelInputTensor = new Tensor<T>(new[] { 1, highLevelInput.Length });
            for (int i = 0; i < highLevelInput.Length; i++)
            {
                highLevelInputTensor[0, i] = highLevelInput[i];
            }
            var highLevelPolicyOutput = _highLevelPolicy.Predict(highLevelInputTensor);
            
            // Extract goal mean and log variance
            var goalMean = new Vector<T>(_goalDimension);
            var goalLogVar = new Vector<T>(_goalDimension);
            for (int i = 0; i < _goalDimension; i++)
            {
                goalMean[i] = highLevelPolicyOutput[i];
                goalLogVar[i] = highLevelPolicyOutput[i + _goalDimension];
            }
            
            // Calculate log probability of the goal
            var logProb = CalculateLogProbability(goal, goalMean, goalLogVar);
            
            // Calculate entropy
            var entropy = CalculateEntropy(goalLogVar);
            
            // Get value estimate
            var stateGoalInput = CombineStateAndGoal(state, goal);
            var stateGoalTensor = new Tensor<T>(new[] { stateGoalInput.Length });
            for (int i = 0; i < stateGoalInput.Length; i++)
            {
                stateGoalTensor[i] = stateGoalInput[i];
            }
            var valueOutput = _highLevelValue.Predict(stateGoalTensor);
            
            // Policy loss = -log_prob * advantage - entropy_coef * entropy
            var advantage = NumOps.Subtract(reward, valueOutput[0]);
            var policyLoss = NumOps.Subtract(
                NumOps.Multiply(NumOps.Negate(logProb), advantage),
                NumOps.Multiply(NumOps.FromDouble(0.01), entropy) // Entropy coefficient
            );
            
            // Value loss = 0.5 * (value - reward)^2
            var valueDiff = NumOps.Subtract(valueOutput[0], reward);
            var valueLoss = NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Multiply(valueDiff, valueDiff));
            
            // Update networks
            _highLevelPolicy.Backward(policyLoss);
            // TODO: Implement Step() or use appropriate optimizer method
            // _highLevelPolicyOptimizer.Step();
            
            _highLevelValue.Backward(valueLoss);
            // TODO: Implement Step() or use appropriate optimizer method
            // _highLevelValueOptimizer.Step();
            
            totalLoss = NumOps.Add(totalLoss, NumOps.Add(policyLoss, valueLoss));
        }
        
        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchIndices.Count));
    }
    
    /// <summary>
    /// Updates the low-level networks (policy and value).
    /// </summary>
    private T UpdateLowLevelNetworks()
    {
        // Skip if we don't have enough low-level experiences
        if (_lowLevelMemory.Count < 2)
            return NumOps.Zero;
        
        // Sample a batch of experiences
        var batchIndices = SampleIndices(_lowLevelMemory.Count, Math.Min(32, _lowLevelMemory.Count));
        T totalLoss = NumOps.Zero;
        
        foreach (int idx in batchIndices)
        {
            var (state, goal, action, reward, nextState, done) = _lowLevelMemory[idx];
            
            // Update low-level policy
            var lowLevelInput = CombineStateAndGoal(state, goal);
            var lowLevelInputTensor = new Tensor<T>(new[] { lowLevelInput.Length });
            for (int i = 0; i < lowLevelInput.Length; i++)
            {
                lowLevelInputTensor[i] = lowLevelInput[i];
            }
            var lowLevelPolicyOutput = _lowLevelPolicy.Predict(lowLevelInputTensor);
            
            // Extract action mean and log variance
            var actionMean = new Vector<T>(_actionDimension);
            var actionLogVar = new Vector<T>(_actionDimension);
            for (int i = 0; i < _actionDimension; i++)
            {
                actionMean[i] = lowLevelPolicyOutput[i];
                actionLogVar[i] = lowLevelPolicyOutput[i + _actionDimension];
            }
            
            // Calculate log probability of the action
            var logProb = CalculateLogProbability(action, actionMean, actionLogVar);
            
            // Calculate entropy
            var entropy = CalculateEntropy(actionLogVar);
            
            // Get value estimate
            var stateActionInput = CombineStateAndAction(state, action);
            var stateActionTensor = new Tensor<T>(new[] { stateActionInput.Length });
            for (int i = 0; i < stateActionInput.Length; i++)
            {
                stateActionTensor[i] = stateActionInput[i];
            }
            var valueOutput = _lowLevelValue.Predict(stateActionTensor);
            
            // Get next state value for TD learning
            T nextValue = NumOps.Zero;
            if (!done)
            {
                // Use current policy to get next action
                var nextLowLevelInput = CombineStateAndGoal(nextState, goal);
                var nextLowLevelInputTensor = new Tensor<T>(new[] { nextLowLevelInput.Length });
                for (int i = 0; i < nextLowLevelInput.Length; i++)
                {
                    nextLowLevelInputTensor[i] = nextLowLevelInput[i];
                }
                var nextLowLevelPolicyOutput = _lowLevelPolicy.Predict(nextLowLevelInputTensor);
                
                // Extract next action mean
                var nextActionMean = new Vector<T>(_actionDimension);
                for (int i = 0; i < _actionDimension; i++)
                {
                    nextActionMean[i] = nextLowLevelPolicyOutput[i];
                }
                
                // Get value of next state-action pair
                var nextStateActionInput = CombineStateAndAction(nextState, nextActionMean);
                
                if (_useTargetNetwork && _lowLevelValueTarget != null)
                {
                    var nextStateActionTensor = new Tensor<T>(new[] { 1, nextStateActionInput.Length });
                    for (int i = 0; i < nextStateActionInput.Length; i++)
                    {
                        nextStateActionTensor[0, i] = nextStateActionInput[i];
                    }
                    var nextValueOutput = _lowLevelValueTarget.Predict(nextStateActionTensor);
                    nextValue = nextValueOutput.Shape.Length == 1 ? nextValueOutput[0] : nextValueOutput[0, 0];
                }
                else
                {
                    var nextStateActionTensor = new Tensor<T>(new[] { nextStateActionInput.Length });
                for (int i = 0; i < nextStateActionInput.Length; i++)
                {
                    nextStateActionTensor[i] = nextStateActionInput[i];
                }
                var nextValueOutput = _lowLevelValue.Predict(nextStateActionTensor);
                    nextValue = nextValueOutput[0];
                }
            }
            
            // Calculate the target value using TD learning
            var target = NumOps.Add(reward, NumOps.Multiply(_lowLevelGamma, nextValue));
            
            // Calculate advantage
            var advantage = NumOps.Subtract(target, valueOutput[0]);
            
            // Add intrinsic reward for exploration if enabled
            if (_useIntrinsicRewards)
            {
                var intrinsicReward = CalculateIntrinsicReward(state, action);
                advantage = NumOps.Add(advantage, NumOps.Multiply(_intrinsicRewardScale, intrinsicReward));
            }
            
            // Incorporate risk aversion
            T riskMeasure = GetRiskMeasure(state, action);
            advantage = NumOps.Subtract(advantage, NumOps.Multiply(_riskAversionParameter, riskMeasure));
            
            // Policy loss = -log_prob * advantage - entropy_coef * entropy
            var policyLoss = NumOps.Subtract(
                NumOps.Multiply(NumOps.Negate(logProb), advantage),
                NumOps.Multiply(NumOps.FromDouble(0.02), entropy) // Entropy coefficient
            );
            
            // Value loss = 0.5 * (value - target)^2
            var valueDiff = NumOps.Subtract(valueOutput[0], target);
            var valueLoss = NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Multiply(valueDiff, valueDiff));
            
            // Update networks
            // TODO: Fix parameter update process - need to implement proper gradient handling
            // _lowLevelPolicy.Backward(policyLoss);
            // _lowLevelPolicyOptimizer.UpdateParameters(_lowLevelPolicy.GetParameters());
            
            // _lowLevelValue.Backward(valueLoss);
            // _lowLevelValueOptimizer.UpdateParameters(_lowLevelValue.GetParameters());
            
            totalLoss = NumOps.Add(totalLoss, NumOps.Add(policyLoss, valueLoss));
        }
        
        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchIndices.Count));
    }
    
    /// <summary>
    /// Updates the distributional value network.
    /// </summary>
    private T UpdateDistributionalValueNetwork()
    {
        // Skip if we don't have enough experiences
        if (_lowLevelMemory.Count < 2)
            return NumOps.Zero;
        
        // Sample a batch of experiences
        var batchIndices = SampleIndices(_lowLevelMemory.Count, Math.Min(32, _lowLevelMemory.Count));
        T totalLoss = NumOps.Zero;
        
        foreach (int idx in batchIndices)
        {
            var (state, goal, action, reward, nextState, done) = _lowLevelMemory[idx];
            
            // TODO: Fix Forward/Backward methods - need to cast or use proper interface
            // Get current distributional value
            // var stateActionInput = CombineStateAndAction(state, action);
            // var distribution = _distributionalValueNetwork.Predict(stateActionInput);
            
            // Calculate target distribution (simplified version)
            // var targetDistribution = CalculateTargetDistribution(distribution, reward, nextState, action, done);
            
            // Calculate cross-entropy loss
            // var loss = CalculateCrossEntropyLoss(distribution, targetDistribution);
            
            // Update network
            // _distributionalValueNetwork.Backward(loss);
            // _distributionalValueOptimizer.UpdateParameters(_distributionalValueNetwork.GetParameters());
            
            // totalLoss = NumOps.Add(totalLoss, loss);
            totalLoss = NumOps.Add(totalLoss, NumOps.Zero); // Placeholder until loss calculation is fixed
        }
        
        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchIndices.Count));
    }
    
    /// <summary>
    /// Updates the target networks by copying weights from the current networks.
    /// </summary>
    private void UpdateTargetNetworks()
    {
        if (_useTargetNetwork)
        {
            _highLevelValueTarget = (NeuralNetwork<T>)_highLevelValue.Clone()!;
            _lowLevelValueTarget = (NeuralNetwork<T>)_lowLevelValue.Clone()!;
        }
    }
    
    /// <summary>
    /// Calculates the log probability of a value under a Gaussian distribution.
    /// </summary>
    private T CalculateLogProbability(Vector<T> value, Vector<T> mean, Vector<T> logVar)
    {
        T logProb = NumOps.Zero;
        T logTwoPI = NumOps.FromDouble(Math.Log(2 * Math.PI));
        
        for (int i = 0; i < value.Length; i++)
        {
            T diff = NumOps.Subtract(value[i], mean[i]);
            T variance = NumOps.Exp(logVar[i]);
            
            // Log probability of Gaussian: -0.5 * (log(2π) + log(σ²) + (x-μ)²/σ²)
            T term1 = NumOps.Add(logTwoPI, logVar[i]);
            T term2 = NumOps.Divide(NumOps.Multiply(diff, diff), variance);
            T logProbComp = NumOps.Multiply(
                NumOps.FromDouble(-0.5),
                NumOps.Add(term1, term2)
            );
            
            logProb = NumOps.Add(logProb, logProbComp);
        }
        
        return logProb;
    }
    
    /// <summary>
    /// Calculates the entropy of a Gaussian distribution with the given log variance.
    /// </summary>
    private T CalculateEntropy(Vector<T> logVar)
    {
        T entropy = NumOps.Zero;
        
        for (int i = 0; i < logVar.Length; i++)
        {
            // Entropy of Gaussian: 0.5 * (1 + log(2πσ²))
            entropy = NumOps.Add(
                entropy,
                NumOps.Multiply(
                    NumOps.FromDouble(0.5),
                    NumOps.Add(
                        NumOps.FromDouble(1 + Math.Log(2 * Math.PI)),
                        logVar[i]
                    )
                )
            );
        }
        
        return entropy;
    }
    
    /// <summary>
    /// Calculates an intrinsic reward to encourage exploration.
    /// </summary>
    private T CalculateIntrinsicReward(TState state, Vector<T> action)
    {
        // Simple random noise as intrinsic reward
        // In a more sophisticated implementation, this could be based on state novelty
        return NumOps.FromDouble(0.1 * (Random.NextDouble() - 0.5));
    }
    
    /// <summary>
    /// Calculates the target distribution for distributional RL.
    /// </summary>
    private Vector<T> CalculateTargetDistribution(Vector<T> currentDistribution, T reward, TState nextState, Vector<T> action, bool done)
    {
        // Simplified implementation - shift the distribution by the reward
        var targetDistribution = new Vector<T>(currentDistribution.Length);
        
        if (done)
        {
            // If episode is done, just set the distribution to be concentrated at the reward
            int rewardBin = GetRewardBin(reward, currentDistribution.Length);
            for (int i = 0; i < targetDistribution.Length; i++)
            {
                targetDistribution[i] = i == rewardBin ? NumOps.One : NumOps.Zero;
            }
        }
        else
        {
            // Shift the distribution by the reward and apply discount
            for (int i = 0; i < targetDistribution.Length; i++)
            {
                int shiftedBin = Math.Min(
                    Math.Max(0, i + GetRewardBin(reward, currentDistribution.Length) - targetDistribution.Length / 2),
                    targetDistribution.Length - 1);
                
                targetDistribution[i] = NumOps.Multiply(_lowLevelGamma, currentDistribution[shiftedBin]);
            }
        }
        
        // Normalize the distribution
        T sum = NumOps.Zero;
        for (int i = 0; i < targetDistribution.Length; i++)
        {
            sum = NumOps.Add(sum, targetDistribution[i]);
        }
        
        if (!NumOps.Equals(sum, NumOps.Zero))
        {
            for (int i = 0; i < targetDistribution.Length; i++)
            {
                targetDistribution[i] = NumOps.Divide(targetDistribution[i], sum);
            }
        }
        else
        {
            // If sum is zero, set to uniform distribution
            T uniformProb = NumOps.Divide(NumOps.One, NumOps.FromDouble(targetDistribution.Length));
            for (int i = 0; i < targetDistribution.Length; i++)
            {
                targetDistribution[i] = uniformProb;
            }
        }
        
        return targetDistribution;
    }
    
    /// <summary>
    /// Maps a reward value to a bin in the value distribution.
    /// </summary>
    private int GetRewardBin(T reward, int numBins)
    {
        // Map reward from [-1, 1] to [0, numBins-1]
        double normalizedReward = (Convert.ToDouble(reward) + 1.0) / 2.0;
        int bin = (int)(normalizedReward * (numBins - 1));
        return Math.Min(Math.Max(0, bin), numBins - 1);
    }
    
    /// <summary>
    /// Calculates the cross-entropy loss between two distributions.
    /// </summary>
    private T CalculateCrossEntropyLoss(Vector<T> predicted, Vector<T> target)
    {
        T loss = NumOps.Zero;
        T epsilon = NumOps.FromDouble(1e-10); // Small value to avoid log(0)
        
        for (int i = 0; i < predicted.Length; i++)
        {
            // Clip predicted values to avoid log(0)
            T clippedPred = NumOps.Add(predicted[i], epsilon);
            
            // Cross-entropy: -sum(target * log(predicted))
            T logPred = NumOps.Log(clippedPred);
            loss = NumOps.Subtract(loss, NumOps.Multiply(target[i], logPred));
        }
        
        return loss;
    }
    
    /// <summary>
    /// Samples a set of indices without replacement.
    /// </summary>
    private List<int> SampleIndices(int populationSize, int sampleSize)
    {
        var indices = new List<int>();
        var available = new List<int>();
        
        for (int i = 0; i < populationSize; i++)
        {
            available.Add(i);
        }
        
        for (int i = 0; i < sampleSize && available.Count > 0; i++)
        {
            int idx = Random.Next(available.Count);
            indices.Add(available[idx]);
            available.RemoveAt(idx);
        }
        
        return indices;
    }
    
    /// <summary>
    /// Simulates the next state based on the current state, action, and reward.
    /// </summary>
    private TState SimulateNextState(TState state, Vector<T> action, T reward)
    {
        // This is a simplified simulation for testing purposes
        // In a real implementation, this would use a learned dynamics model
        
        // Use explicit non-null assertion for Clone() since we know state cannot be null
        // and the result should not be null either
        var nextState = (TState)(state.Clone()!);
        
        // Apply some simple dynamics based on the action
        for (int i = 0; i < Math.Min(_actionDimension, _stateDimension / 2); i++)
        {
            // Update state based on action (simplified linear model)
            nextState[0, i] = NumOps.Add(
                nextState[0, i],
                NumOps.Multiply(action[i], NumOps.FromDouble(0.1))
            );
        }
        
        // Add some noise to simulate stochasticity
        for (int i = 0; i < _stateDimension; i++)
        {
            nextState[0, i] = NumOps.Add(
                nextState[0, i],
                NumOps.FromDouble(0.05 * SampleGaussian())
            );
        }
        
        return nextState;
    }
    
    /// <summary>
    /// Calculates the mean of a distribution.
    /// </summary>
    private T CalculateDistributionMean(Vector<T> distribution)
    {
        T mean = NumOps.Zero;
        
        for (int i = 0; i < distribution.Length; i++)
        {
            // The distribution values are assumed to be probabilities of discrete return values
            // For simplicity, we use the index as the return value (scaled appropriately)
            T value = NumOps.FromDouble((double)i / distribution.Length * 2 - 1); // Scale to [-1, 1]
            mean = NumOps.Add(mean, NumOps.Multiply(distribution[i], value));
        }
        
        return mean;
    }
    
    #endregion
}