namespace AiDotNet.ReinforcementLearning.Agents;

/// <summary>
/// Agent implementing the Quantile Regression Deep Q-Network (QR-DQN) algorithm for distributional reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// QR-DQN is a distributional reinforcement learning algorithm that models the entire distribution of
/// returns instead of just the expected value. By estimating quantiles of the return distribution,
/// it provides rich information about the uncertainty and risk associated with different actions.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Think of QR-DQN as an advanced trading algorithm that doesn't just predict how much money
/// you might make on average from a trade (like traditional methods). Instead, it gives you
/// a complete picture of all possible outcomes - from worst-case to best-case scenarios.
/// 
/// This is especially valuable for financial markets because:
/// - It helps you understand the risk of different trading strategies
/// - It can identify when two strategies have similar average returns but very different risks
/// - It allows for more cautious trading by focusing on downside protection
/// - It can better handle the complex, non-normal distributions common in financial returns
/// </para>
/// </remarks>
public class QRDQNAgent<T> : DQNAgent<Tensor<T>, T>
{
    private new readonly QRDQNOptions _options; // Hide base class _options
    
    // Neural network components (quantile-specific)
    private readonly QuantileNetwork<T> _quantileNetwork = default!;
    private readonly QuantileNetwork<T> _targetQuantileNetwork = default!;
    
    // Tau values for quantile regression (midpoints of quantile intervals)
    private readonly T[] _tauValues;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="QRDQNAgent{T}"/> class.
    /// </summary>
    /// <param name="options">Options for configuring the QR-DQN agent.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes the QR-DQN agent with the specified options, creating the neural
    /// networks, replay buffer, and other components needed for distributional reinforcement learning.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This sets up a specialized trading agent that can:
    /// - Learn from past market experiences
    /// - Predict entire distributions of possible returns for different actions
    /// - Make risk-aware decisions that consider both potential gains and losses
    /// - Adapt its exploration strategy to discover profitable trading patterns
    /// 
    /// The initialization process creates the neural networks and memory systems needed for
    /// the agent to learn and make decisions in financial markets.
    /// </para>
    /// </remarks>
    public QRDQNAgent(QRDQNOptions options)
        : base(new DQNOptions
        {
            StateSize = options.StateSize,
            ActionSize = options.ActionSize,
            LearningRate = options.LearningRate,
            Gamma = options.Gamma,
            BatchSize = options.BatchSize,
            ReplayBufferCapacity = options.ReplayBufferCapacity,
            TargetNetworkUpdateFrequency = options.TargetUpdateFrequency,
            UseSoftTargetUpdate = options.Tau < 1.0,
            InitialExplorationRate = options.InitialExplorationRate,
            FinalExplorationRate = options.FinalExplorationRate,
            ExplorationFraction = 0.1, // Default value since QRDQNOptions doesn't have this property
            NetworkArchitecture = options.HiddenLayerSizes,
            UsePrioritizedReplay = options.UsePrioritizedReplay,
            PrioritizedReplayAlpha = options.PriorityAlpha,
            PrioritizedReplayBetaInitial = options.PriorityBetaStart,
            Seed = options.Seed
        })
    {
        _options = options;
        
        // Initialize tau values (midpoints of quantile intervals)
        _tauValues = new T[options.NumQuantiles];
        for (int i = 0; i < options.NumQuantiles; i++)
        {
            double tau = (i + 0.5) / options.NumQuantiles;  // Midpoint of interval [i/N, (i+1)/N]
            _tauValues[i] = NumOps.FromDouble(tau);
        }
        
        // Create main and target networks
        _quantileNetwork = new QuantileNetwork<T>(
            options.StateSize,
            options.ActionSize,
            options.NumQuantiles,
            options.HiddenLayerSizes,
            options.UseNoisyNetworks,
            options.InitialNoiseStd);
            
        _targetQuantileNetwork = new QuantileNetwork<T>(
            options.StateSize,
            options.ActionSize,
            options.NumQuantiles,
            options.HiddenLayerSizes,
            options.UseNoisyNetworks,
            options.InitialNoiseStd);
            
        // Initialize target network with same weights as main network
        _targetQuantileNetwork.SetParameters(_quantileNetwork.GetParameters());
    }
    
    /// <summary>
    /// Selects an action for the given state.
    /// </summary>
    /// <param name="state">The current environment state.</param>
    /// <param name="isTraining">Whether the agent is in training mode (exploration) or evaluation mode (exploitation).</param>
    /// <returns>The selected action vector.</returns>
    /// <remarks>
    /// <para>
    /// This method selects an action based on the current state. In evaluation mode, it uses
    /// risk-aware action selection if configured (e.g., CVaR), otherwise it selects the action
    /// with the highest expected value. In training mode, it follows the configured exploration strategy.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is where the agent decides what trading action to take given the current market state.
    /// 
    /// What makes QR-DQN special is how it can incorporate risk awareness into decision making:
    /// - It can identify actions with similar average returns but different risk profiles
    /// - It can avoid actions with high downside risk
    /// - It can make decisions based on worst-case scenarios rather than just average outcomes
    /// 
    /// This leads to more robust trading strategies that can better navigate volatile markets.
    /// </para>
    /// </remarks>
    public override int SelectAction(Tensor<T> state, bool isTraining = true)
    {
        // In training mode, follow exploration policy
        if (isTraining && _options.UseNoisyNetworks)
        {
            // With noisy networks, we don't need epsilon-greedy exploration
            // The network itself provides exploration through parameter noise
            _quantileNetwork.ResetNoise();  // Generate new noise for exploration
        }
        
        // Get quantile values from network
        var quantileValues = _quantileNetwork.Forward(state);
        
        // Select action
        int action;
        
        if (isTraining && !_options.UseNoisyNetworks)
        {
            // Epsilon-greedy exploration for non-noisy networks
            // Calculate epsilon based on exploration schedule
            // Calculate progress using default values since QRDQNOptions doesn't have MaxSteps or ExplorationFraction
            double maxSteps = 1000000; // Default max steps
            double explorationFraction = 0.1; // Default exploration fraction
            double progress = Math.Min(1.0, _updateCounter / (maxSteps * explorationFraction));
            double epsilon = _options.InitialExplorationRate + 
                            (_options.FinalExplorationRate - _options.InitialExplorationRate) * progress;
            
            if (Random.NextDouble() < epsilon)
            {
                // Random action
                action = Random.Next(_options.ActionSize);
            }
            else
            {
                // Greedy action based on expected value or risk measure
                action = SelectGreedyActionIndex(quantileValues);
            }
        }
        else
        {
            // Evaluation mode or noisy networks
            if (_options.UseCVaR)
            {
                // Risk-sensitive action selection using CVaR
                action = SelectCVaRActionIndex(quantileValues, _options.CVaRAlpha);
            }
            else if (_options.RiskDistortion > 0.0)
            {
                // Risk-sensitive action selection using risk distortion
                action = SelectRiskDistortedActionIndex(quantileValues, _options.RiskDistortion);
            }
            else
            {
                // Standard action selection based on expected value
                action = SelectGreedyActionIndex(quantileValues);
            }
        }
        
        return action;
    }
    
    
    /// <summary>
    /// Selects the action index with the highest expected value.
    /// </summary>
    /// <param name="quantileValues">The quantile values for each action.</param>
    /// <returns>The selected action index.</returns>
    private int SelectGreedyActionIndex(Tensor<T> quantileValues)
    {
        // Calculate expected value for each action by averaging quantiles
        var expectedValues = new Vector<T>(_options.ActionSize);
        for (int a = 0; a < _options.ActionSize; a++)
        {
            T sum = NumOps.Zero;
            for (int q = 0; q < _options.NumQuantiles; q++)
            {
                sum = NumOps.Add(sum, quantileValues[a, q]);
            }
            expectedValues[a] = NumOps.Divide(sum, NumOps.FromDouble(_options.NumQuantiles));
        }
        
        // Find action with highest expected value
        int bestAction = 0;
        T bestValue = expectedValues[0];
        
        for (int a = 1; a < _options.ActionSize; a++)
        {
            if (NumOps.GreaterThan(expectedValues[a], bestValue))
            {
                bestValue = expectedValues[a];
                bestAction = a;
            }
        }
        
        return bestAction;
    }
    
    /// <summary>
    /// Selects an action based on the Conditional Value at Risk (CVaR) measure.
    /// </summary>
    /// <param name="quantileValues">The quantile values for each action.</param>
    /// <param name="alpha">The risk level (between 0 and 1).</param>
    /// <returns>The selected action index.</returns>
    /// <remarks>
    /// <para>
    /// CVaR measures the expected value in the worst alpha% of cases, making it a risk-sensitive
    /// measure that focuses on downside protection. Lower alpha values lead to more conservative
    /// decision making.
    /// </para>
    /// </remarks>
    private int SelectCVaRActionIndex(Tensor<T> quantileValues, double alpha)
    {
        // Calculate CVaR for each action
        var cvarValues = new Vector<T>(_options.ActionSize);
        int numQuantilesToConsider = Math.Max(1, (int)(_options.NumQuantiles * alpha));
        
        for (int a = 0; a < _options.ActionSize; a++)
        {
            // Sort quantiles for this action (ascending)
            var sortedQuantiles = new List<T>();
            for (int q = 0; q < _options.NumQuantiles; q++)
            {
                sortedQuantiles.Add(quantileValues[a, q]);
            }
            sortedQuantiles.Sort((x, y) => NumOps.LessThan(x, y) ? -1 : (NumOps.Equals(x, y) ? 0 : 1));
            
            // Calculate CVaR as average of worst quantiles
            T sum = NumOps.Zero;
            for (int i = 0; i < numQuantilesToConsider; i++)
            {
                sum = NumOps.Add(sum, sortedQuantiles[i]);
            }
            cvarValues[a] = NumOps.Divide(sum, NumOps.FromDouble(numQuantilesToConsider));
        }
        
        // Find action with highest CVaR
        int bestAction = 0;
        T bestValue = cvarValues[0];
        
        for (int a = 1; a < _options.ActionSize; a++)
        {
            if (NumOps.GreaterThan(cvarValues[a], bestValue))
            {
                bestValue = cvarValues[a];
                bestAction = a;
            }
        }
        
        return bestAction;
    }
    
    /// <summary>
    /// Selects an action using risk distortion, which overweights the probability of negative outcomes.
    /// </summary>
    /// <param name="quantileValues">The quantile values for each action.</param>
    /// <param name="distortion">The risk distortion parameter (between 0 and 1).</param>
    /// <returns>The selected action index.</returns>
    /// <remarks>
    /// <para>
    /// Risk distortion introduces risk aversion by distorting the probabilities used to weight the
    /// quantiles. Higher distortion values lead to more conservative decision making.
    /// </para>
    /// </remarks>
    private int SelectRiskDistortedActionIndex(Tensor<T> quantileValues, double distortion)
    {
        // Calculate risk-distorted values for each action
        var distortedValues = new Vector<T>(_options.ActionSize);
        
        for (int a = 0; a < _options.ActionSize; a++)
        {
            T weightedSum = NumOps.Zero;
            
            for (int q = 0; q < _options.NumQuantiles; q++)
            {
                // Calculate distorted probability weight
                double tau = Convert.ToDouble(_tauValues[q]);
                double distortedTau = Math.Pow(tau, 1.0 - distortion);
                double weight = distortedTau / _options.NumQuantiles;
                
                // Apply weight to quantile value
                T weightedQuantile = NumOps.Multiply(
                    quantileValues[a, q],
                    NumOps.FromDouble(weight));
                    
                // Add to sum
                weightedSum = NumOps.Add(weightedSum, weightedQuantile);
            }
            
            distortedValues[a] = weightedSum;
        }
        
        // Find action with highest risk-distorted value
        int bestAction = 0;
        T bestValue = distortedValues[0];
        
        for (int a = 1; a < _options.ActionSize; a++)
        {
            if (NumOps.GreaterThan(distortedValues[a], bestValue))
            {
                bestValue = distortedValues[a];
                bestAction = a;
            }
        }
        
        return bestAction;
    }
    
    /// <summary>
    /// Processes a new experience and adds it to the replay buffer.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <param name="action">The action taken.</param>
    /// <param name="reward">The reward received.</param>
    /// <param name="nextState">The next state.</param>
    /// <param name="done">Whether the episode is done.</param>
    /// <remarks>
    /// <para>
    /// This method adds a new experience to the replay buffer and potentially performs
    /// a training update if enough experiences have been collected.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is how the agent learns from its trading experiences:
    /// 1. It stores each new market experience in memory
    /// 2. Periodically, it samples a batch of experiences from memory
    /// 3. It uses these experiences to update its understanding of market patterns
    /// 4. Over time, it learns which trading actions lead to better outcomes in different scenarios
    /// </para>
    /// </remarks>
    public override void Learn(Tensor<T> state, int action, T reward, Tensor<T> nextState, bool done)
    {
        // Add experience to replay buffer
        _replayBuffer.Add(state, action, reward, nextState, done);
        
        // Update network if enough experiences have been collected
        _updateCounter++;
        
        if (_updateCounter % 4 == 0 && _replayBuffer.Size >= _options.BatchSize) // Using a default update frequency of 4 steps
        {
            // Sample batch of experiences
            var batch = _replayBuffer.SampleBatch(_options.BatchSize);
            
            // Convert batch data to appropriate types
            var states = Tensor<T>.Stack(batch.States);
            var actions = new Tensor<T>(new int[] { batch.Actions.Length, _options.ActionSize });
            for (int i = 0; i < batch.Actions.Length; i++)
            {
                actions[i, batch.Actions[i]] = NumOps.One;
            }
            var rewards = new Vector<T>(batch.Rewards);
            var nextStates = Tensor<T>.Stack(batch.NextStates);
            var dones = new Vector<T>(batch.Dones.Length);
            for (int i = 0; i < batch.Dones.Length; i++)
            {
                dones[i] = batch.Dones[i] ? NumOps.One : NumOps.Zero;
            }
            
            // Train on the batch
            var (loss, priorities) = TrainOnBatch(
                states, actions, rewards, nextStates, dones, batch.Indices);
            
            // Update priorities in replay buffer
            if (_options.UsePrioritizedReplay && batch.Indices != null && priorities != null)
            {
                var prioritizedBuffer = _replayBuffer as IPrioritizedReplayBuffer<Tensor<T>, int, T>;
                if (prioritizedBuffer != null)
                {
                    for (int i = 0; i < batch.Indices.Length; i++)
                    {
                        prioritizedBuffer.UpdatePriority(batch.Indices[i], priorities[i]);
                    }
                }
            }
            
            // Update target network periodically
            if (_updateCounter % _options.TargetUpdateFrequency == 0)
            {
                UpdateTargetNetworkQuantile(NumOps.FromDouble(_options.Tau));
            }
            
            _lastLoss = loss;
        }
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
    public T Train(Tensor<T>[] states, int[] actions, T[] rewards, Tensor<T>[] nextStates, bool[] dones)
    {
        // Convert actions to Vector<T> format for processing
        var actionVectors = new Vector<T>[actions.Length];
        for (int i = 0; i < actions.Length; i++)
        {
            actionVectors[i] = new Vector<T>(_options.ActionSize);
            actionVectors[i][actions[i]] = NumOps.One;
        }
        
        // Convert arrays to appropriate format
        var batch = PrepareBatch(states, actionVectors, rewards, nextStates, dones);
        
        // Train without updating priorities (used for external training)
        var (loss, _) = TrainOnBatch(
            batch.Item1, batch.Item2, batch.Item3, batch.Item4, batch.Item5, null);
            
        // Update target network if needed
        _updateCounter++;
        if (_updateCounter % _options.TargetUpdateFrequency == 0)
        {
            UpdateTargetNetworkQuantile(NumOps.FromDouble(_options.Tau));
        }
        
        _lastLoss = loss;
        return loss;
    }
    
    /// <summary>
    /// Prepares a batch of experiences for training.
    /// </summary>
    /// <param name="states">The batch of states.</param>
    /// <param name="actions">The batch of actions.</param>
    /// <param name="rewards">The batch of rewards.</param>
    /// <param name="nextStates">The batch of next states.</param>
    /// <param name="dones">The batch of done flags.</param>
    /// <returns>A tuple of tensors ready for training.</returns>
    private (Tensor<T>, Tensor<T>, Vector<T>, Tensor<T>, Vector<T>) PrepareBatch(
        Tensor<T>[] states, Vector<T>[] actions, T[] rewards, Tensor<T>[] nextStates, bool[] dones)
    {
        int batchSize = states.Length;
        
        // Create tensors for batch data
        var statesTensor = Tensor<T>.Stack(states);
        
        // Process actions
        var actionsTensor = new Tensor<T>(new int[] { batchSize, _options.ActionSize });
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < _options.ActionSize; j++)
            {
                actionsTensor[i, j] = actions[i][j];
            }
        }
        
        // Process rewards and dones
        var rewardsVector = new Vector<T>(batchSize);
        var donesVector = new Vector<T>(batchSize);
        
        for (int i = 0; i < batchSize; i++)
        {
            rewardsVector[i] = rewards[i];
            donesVector[i] = dones[i] ? NumOps.One : NumOps.Zero;
        }
        
        // Process next states
        var nextStatesTensor = Tensor<T>.Stack(nextStates);
        
        return (statesTensor, actionsTensor, rewardsVector, nextStatesTensor, donesVector);
    }
    
    /// <summary>
    /// Trains the network on a batch of experiences.
    /// </summary>
    /// <param name="states">The batch of states.</param>
    /// <param name="actions">The batch of actions.</param>
    /// <param name="rewards">The batch of rewards.</param>
    /// <param name="nextStates">The batch of next states.</param>
    /// <param name="dones">The batch of done flags.</param>
    /// <param name="indices">The indices of the experiences in the replay buffer (for priority updates).</param>
    /// <returns>A tuple containing the loss value and updated priorities.</returns>
    private (T, T[]?) TrainOnBatch(
        Tensor<T> states, Tensor<T> actions, Vector<T> rewards, 
        Tensor<T> nextStates, Vector<T> dones, int[]? indices)
    {
        int batchSize = states.Shape[0];
        
        // Get current quantile values for all state-action pairs
        var currentQuantiles = _quantileNetwork.Forward(states);
        
        // Get target quantile values
        var targetQuantiles = ComputeTargetQuantiles(nextStates, rewards, dones);
        
        // Calculate quantile regression loss
        var (loss, tdErrors) = ComputeQuantileRegressionLoss(
            currentQuantiles, targetQuantiles, actions);
            
        // Backpropagate and optimize
        // The quantile network's Backward method handles optimization internally
        _quantileNetwork.Backward(loss);
        
        // Calculate priorities for replay buffer update
        T[]? priorities = null;
        if (indices != null)
        {
            priorities = new T[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                // Priority is based on the absolute TD error, with a small constant to ensure non-zero priority
                priorities[i] = NumOps.Add(NumOps.Abs(tdErrors[i]), NumOps.FromDouble(1e-6));
            }
        }
        
        return (loss, priorities);
    }
    
    /// <summary>
    /// Computes the target quantile values for the QR-DQN update.
    /// </summary>
    /// <param name="nextStates">The batch of next states.</param>
    /// <param name="rewards">The batch of rewards.</param>
    /// <param name="dones">The batch of done flags.</param>
    /// <returns>The target quantile values.</returns>
    private Tensor<T> ComputeTargetQuantiles(Tensor<T> nextStates, Vector<T> rewards, Vector<T> dones)
    {
        int batchSize = nextStates.Shape[0];
        
        // Get next state quantile values from target network
        var nextQuantiles = _targetQuantileNetwork.Forward(nextStates);
        
        // Initialize target quantiles tensor
        var targetQuantiles = new Tensor<T>(
            new int[] { batchSize, _options.ActionSize, _options.NumQuantiles });
            
        // For each sample in the batch
        for (int i = 0; i < batchSize; i++)
        {
            // Check if the episode is done
            if (NumOps.GreaterThan(dones[i], NumOps.FromDouble(0.5)))
            {
                // If done, the target is just the reward
                for (int a = 0; a < _options.ActionSize; a++)
                {
                    for (int q = 0; q < _options.NumQuantiles; q++)
                    {
                        targetQuantiles[i, a, q] = rewards[i];
                    }
                }
            }
            else
            {
                // Not done, use the Bellman equation: Q(s,a) = r + gamma * Q(s',a')
                
                // Select next action using main or target network (double DQN)
                int nextAction;
                
                if (_options.UseDoubleDQN)
                {
                    // Double DQN: use main network for action selection
                    var nextStateMainQuantiles = _quantileNetwork.Forward(nextStates.GetSlice(i));
                    
                    // Calculate expected value for each action
                    var expectedValues = new Vector<T>(_options.ActionSize);
                    for (int a = 0; a < _options.ActionSize; a++)
                    {
                        T sum = NumOps.Zero;
                        for (int q = 0; q < _options.NumQuantiles; q++)
                        {
                            sum = NumOps.Add(sum, nextStateMainQuantiles[a, q]);
                        }
                        expectedValues[a] = NumOps.Divide(sum, NumOps.FromDouble(_options.NumQuantiles));
                    }
                    
                    // Find action with highest expected value
                    nextAction = 0;
                    T bestValue = expectedValues[0];
                    
                    for (int a = 1; a < _options.ActionSize; a++)
                    {
                        if (NumOps.GreaterThan(expectedValues[a], bestValue))
                        {
                            bestValue = expectedValues[a];
                            nextAction = a;
                        }
                    }
                }
                else
                {
                    // Standard DQN: use target network for action selection
                    var nextStateTargetQuantiles = nextQuantiles.GetSlice(i);
                    
                    // Calculate expected value for each action
                    var expectedValues = new Vector<T>(_options.ActionSize);
                    for (int a = 0; a < _options.ActionSize; a++)
                    {
                        T sum = NumOps.Zero;
                        for (int q = 0; q < _options.NumQuantiles; q++)
                        {
                            sum = NumOps.Add(sum, nextStateTargetQuantiles[a, q]);
                        }
                        expectedValues[a] = NumOps.Divide(sum, NumOps.FromDouble(_options.NumQuantiles));
                    }
                    
                    // Find action with highest expected value
                    nextAction = 0;
                    T bestValue = expectedValues[0];
                    
                    for (int a = 1; a < _options.ActionSize; a++)
                    {
                        if (NumOps.GreaterThan(expectedValues[a], bestValue))
                        {
                            bestValue = expectedValues[a];
                            nextAction = a;
                        }
                    }
                }
                
                // Calculate target quantiles using Bellman equation
                for (int a = 0; a < _options.ActionSize; a++)
                {
                    for (int q = 0; q < _options.NumQuantiles; q++)
                    {
                        if (a == nextAction)
                        {
                            // Q(s,a) = r + gamma * Q(s',a')
                            T discountedNextQuantile = NumOps.Multiply(
                                NumOps.FromDouble(_options.Gamma),
                                nextQuantiles[i, nextAction, q]);
                                
                            targetQuantiles[i, a, q] = NumOps.Add(rewards[i], discountedNextQuantile);
                        }
                        else
                        {
                            // For actions not taken, just copy the current quantile value
                            targetQuantiles[i, a, q] = nextQuantiles[i, a, q];
                        }
                    }
                }
            }
        }
        
        return targetQuantiles;
    }
    
    /// <summary>
    /// Computes the quantile regression loss for the QR-DQN update.
    /// </summary>
    /// <param name="currentQuantiles">The current quantile values.</param>
    /// <param name="targetQuantiles">The target quantile values.</param>
    /// <param name="actions">The batch of actions.</param>
    /// <returns>A tuple containing the loss value and TD errors.</returns>
    private (T, T[]) ComputeQuantileRegressionLoss(
        Tensor<T> currentQuantiles, Tensor<T> targetQuantiles, Tensor<T> actions)
    {
        int batchSize = currentQuantiles.Shape[0];
        T totalLoss = NumOps.Zero;
        var tdErrors = new T[batchSize];
        
        // Compute loss for each sample in the batch
        for (int i = 0; i < batchSize; i++)
        {
            T sampleLoss = NumOps.Zero;
            T maxTdError = NumOps.Zero;
            
            // Find the action that was taken
            int actionTaken = 0;
            T maxActionValue = actions[i, 0];
            
            for (int a = 1; a < _options.ActionSize; a++)
            {
                if (NumOps.GreaterThan(actions[i, a], maxActionValue))
                {
                    maxActionValue = actions[i, a];
                    actionTaken = a;
                }
            }
            
            // Compute quantile regression loss for the taken action
            for (int j = 0; j < _options.NumQuantiles; j++)
            {
                T predictionJ = currentQuantiles[i, actionTaken, j];
                
                for (int k = 0; k < _options.NumQuantiles; k++)
                {
                    T targetK = targetQuantiles[i, actionTaken, k];
                    
                    // Calculate TD error
                    T tdError = NumOps.Subtract(targetK, predictionJ);
                    
                    // Track maximum TD error for prioritized replay
                    if (NumOps.GreaterThan(NumOps.Abs(tdError), NumOps.Abs(maxTdError)))
                    {
                        maxTdError = tdError;
                    }
                    
                    // Calculate indicator function: I(delta < 0)
                    T indicator = NumOps.LessThan(tdError, NumOps.Zero) ? 
                        NumOps.One : NumOps.Zero;
                        
                    // Calculate absolute error for Huber loss
                    T absError = NumOps.Abs(tdError);
                    
                    // Apply Huber loss
                    T huberLoss;
                    T kappa = NumOps.FromDouble(_options.HuberKappa);
                    
                    if (NumOps.LessThanOrEquals(absError, kappa))
                    {
                        // Quadratic region
                        huberLoss = NumOps.Multiply(
                            NumOps.FromDouble(0.5),
                            NumOps.Multiply(tdError, tdError));
                    }
                    else
                    {
                        // Linear region
                        huberLoss = NumOps.Multiply(
                            kappa,
                            NumOps.Subtract(absError, NumOps.Multiply(kappa, NumOps.FromDouble(0.5))));
                    }
                    
                    // Calculate quantile weight
                    T tau = _tauValues[j];
                    T quantileWeight = NumOps.Subtract(tau, indicator);
                    
                    // Calculate weighted loss
                    T weightedLoss = NumOps.Multiply(
                        NumOps.Abs(quantileWeight),
                        huberLoss);
                        
                    // Add to sample loss
                    sampleLoss = NumOps.Add(sampleLoss, weightedLoss);
                }
            }
            
            // Normalize sample loss
            sampleLoss = NumOps.Divide(
                sampleLoss,
                NumOps.FromDouble(_options.NumQuantiles));
                
            // Store TD error for prioritized replay
            tdErrors[i] = maxTdError;
            
            // Add to total loss
            totalLoss = NumOps.Add(totalLoss, sampleLoss);
        }
        
        // Calculate mean loss
        totalLoss = NumOps.Divide(
            totalLoss,
            NumOps.FromDouble(batchSize));
            
        return (totalLoss, tdErrors);
    }
    
    
    /// <summary>
    /// Gets the predicted return distribution for a given state and action.
    /// </summary>
    /// <param name="state">The state to evaluate.</param>
    /// <param name="action">The action to evaluate. If null, returns distributions for all actions.</param>
    /// <returns>The quantile values representing the return distribution.</returns>
    public Tensor<T> GetReturnDistribution(Tensor<T> state, Vector<T>? action = null)
    {
        // Get quantile values from network
        var quantileValues = _quantileNetwork.Forward(state);
        
        if (action == null)
        {
            // Return distributions for all actions
            return quantileValues;
        }
        
        // Find the specified action
        int actionIndex = 0;
        T maxActionValue = action[0];
        
        for (int a = 1; a < _options.ActionSize; a++)
        {
            if (NumOps.GreaterThan(action[a], maxActionValue))
            {
                maxActionValue = action[a];
                actionIndex = a;
            }
        }
        
        // Extract distribution for the specified action
        var actionDistribution = new Tensor<T>(new int[] { 1, _options.NumQuantiles });
        for (int q = 0; q < _options.NumQuantiles; q++)
        {
            actionDistribution[0, q] = quantileValues[actionIndex, q];
        }
        
        return actionDistribution;
    }
    
    /// <summary>
    /// Calculates the Conditional Value at Risk (CVaR) for a given state and action.
    /// </summary>
    /// <param name="state">The state to evaluate.</param>
    /// <param name="action">The action to evaluate. If null, returns CVaR for all actions.</param>
    /// <param name="alpha">The risk level (between 0 and 1).</param>
    /// <returns>The CVaR value(s) representing the average of the worst alpha% of returns.</returns>
    public Tensor<T> GetCVaR(Tensor<T> state, Vector<T>? action = null, double alpha = 0.05)
    {
        // Get quantile values from network
        var quantileValues = _quantileNetwork.Forward(state);
        int numQuantilesToConsider = Math.Max(1, (int)(_options.NumQuantiles * alpha));
        
        if (action == null)
        {
            // Calculate CVaR for all actions
            var cvarValues = new Tensor<T>(new int[] { _options.ActionSize, 1 });
            
            for (int a = 0; a < _options.ActionSize; a++)
            {
                // Extract quantiles for this action
                var actionQuantiles = new List<T>();
                for (int q = 0; q < _options.NumQuantiles; q++)
                {
                    actionQuantiles.Add(quantileValues[a, q]);
                }
                
                // Sort quantiles (ascending)
                actionQuantiles.Sort((x, y) => 
                    NumOps.LessThan(x, y) ? -1 : (NumOps.Equals(x, y) ? 0 : 1));
                
                // Calculate CVaR as average of worst quantiles
                T actionSum = NumOps.Zero;
                for (int i = 0; i < numQuantilesToConsider; i++)
                {
                    actionSum = NumOps.Add(actionSum, actionQuantiles[i]);
                }
                cvarValues[a, 0] = NumOps.Divide(actionSum, NumOps.FromDouble(numQuantilesToConsider));
            }
            
            return cvarValues;
        }
        
        // Find the specified action
        int actionIndex = 0;
        T maxActionValue = action[0];
        
        for (int a = 1; a < _options.ActionSize; a++)
        {
            if (NumOps.GreaterThan(action[a], maxActionValue))
            {
                maxActionValue = action[a];
                actionIndex = a;
            }
        }
        
        // Calculate CVaR for the specified action
        var selectedActionQuantiles = new List<T>();
        for (int q = 0; q < _options.NumQuantiles; q++)
        {
            selectedActionQuantiles.Add(quantileValues[actionIndex, q]);
        }
        
        // Sort quantiles (ascending)
        selectedActionQuantiles.Sort((x, y) => 
            NumOps.LessThan(x, y) ? -1 : (NumOps.Equals(x, y) ? 0 : 1));
        
        // Calculate CVaR as average of worst quantiles
        T cvarSum = NumOps.Zero;
        for (int i = 0; i < numQuantilesToConsider; i++)
        {
            cvarSum = NumOps.Add(cvarSum, selectedActionQuantiles[i]);
        }
        T cvar = NumOps.Divide(cvarSum, NumOps.FromDouble(numQuantilesToConsider));
        
        // Return as tensor
        var cvarTensor = new Tensor<T>(new int[] { 1, 1 });
        cvarTensor[0, 0] = cvar;
        
        return cvarTensor;
    }
    
    /// <summary>
    /// Gets the latest loss value from training.
    /// </summary>
    /// <returns>The latest loss value.</returns>
    public override T GetLatestLoss()
    {
        return _lastLoss;
    }
    
    /// <summary>
    /// Gets the agent's parameters as a single vector.
    /// </summary>
    /// <returns>A vector containing all parameters of the agent.</returns>
    public new Vector<T> GetParameters()
    {
        return _quantileNetwork.GetParameters();
    }
    
    /// <summary>
    /// Sets the agent's parameters from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    public new void SetParameters(Vector<T> parameters)
    {
        _quantileNetwork.UpdateParameters(parameters);
        _targetQuantileNetwork.SetParameters(parameters);  // Update target network to match
    }
    
    /// <summary>
    /// Updates the target network parameters using soft update.
    /// </summary>
    /// <param name="tau">The interpolation parameter for soft update (0 < tau <= 1).</param>
    private void UpdateTargetNetworkQuantile(T tau)
    {
        var onlineParams = _quantileNetwork.GetParameters();
        var targetParams = _targetQuantileNetwork.GetParameters();
        
        // Perform soft update: target = tau * online + (1 - tau) * target
        var oneTau = NumOps.Subtract(NumOps.One, tau);
        for (int i = 0; i < onlineParams.Length; i++)
        {
            targetParams[i] = NumOps.Add(
                NumOps.Multiply(tau, onlineParams[i]),
                NumOps.Multiply(oneTau, targetParams[i])
            );
        }
        
        _targetQuantileNetwork.SetParameters(targetParams);
    }
    
    /// <summary>
    /// Saves the agent's state to a file.
    /// </summary>
    /// <param name="filePath">The path where the agent's state should be saved.</param>
    public override void Save(string filePath)
    {
        // Get parameters from the network
        var parameters = GetParameters();
        
        // Use serialization helper to save the parameters to a file
        SerializationHelper<T>.SaveVectorToFile(parameters, filePath);
    }
    
}

/// <summary>
/// Neural network for the QR-DQN algorithm that outputs quantile values.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class QuantileNetwork<T>
{
    private readonly int _stateSize;
    private readonly int _actionSize;
    private readonly int _numQuantiles;
    private readonly int[] _hiddenSizes;
    private readonly bool _useNoisyNetworks;
    private readonly double _noiseStd;
    private readonly INumericOperations<T> NumOps;
    private readonly Random _random = default!;
    
    // Network layers
    private readonly NeuralNetwork<T> _network = default!;
    private readonly T[] _tauValues;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="QuantileNetwork{T}"/> class.
    /// </summary>
    /// <param name="stateSize">The size of the state input.</param>
    /// <param name="actionSize">The number of possible actions.</param>
    /// <param name="numQuantiles">The number of quantiles to estimate.</param>
    /// <param name="hiddenSizes">The sizes of the hidden layers.</param>
    /// <param name="useNoisyNetworks">Whether to use noisy networks for exploration.</param>
    /// <param name="noiseStd">The standard deviation of the noise for noisy networks.</param>
    public QuantileNetwork(
        int stateSize, 
        int actionSize, 
        int numQuantiles, 
        int[] hiddenSizes,
        bool useNoisyNetworks,
        double noiseStd)
    {
        _stateSize = stateSize;
        _actionSize = actionSize;
        _numQuantiles = numQuantiles;
        _hiddenSizes = hiddenSizes;
        _useNoisyNetworks = useNoisyNetworks;
        _noiseStd = noiseStd;
        NumOps = MathHelper.GetNumericOperations<T>();
        _random = new Random();
        
        // Initialize tau values (midpoints of quantile intervals)
        _tauValues = new T[numQuantiles];
        for (int i = 0; i < numQuantiles; i++)
        {
            double tau = (i + 0.5) / numQuantiles;  // Midpoint of interval [i/N, (i+1)/N]
            _tauValues[i] = NumOps.FromDouble(tau);
        }
        
        // Create layers list
        var layers = new List<ILayer<T>>();
        
        // Add input layer
        layers.Add(new InputLayer<T>(_stateSize));
        
        // Add hidden layers
        for (int i = 0; i < _hiddenSizes.Length; i++)
        {
            int inputSize = i == 0 ? _stateSize : _hiddenSizes[i - 1];
            int outputSize = _hiddenSizes[i];
            
            layers.Add(new DenseLayer<T>(inputSize, outputSize, ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU) as IActivationFunction<T>));
        }
        
        // Add output layer (no activation for quantile values)
        layers.Add(new DenseLayer<T>(_hiddenSizes[_hiddenSizes.Length - 1], _actionSize * _numQuantiles, activationFunction: null));
        
        // Create neural network architecture
        var architecture = new NeuralNetworkArchitecture<T>(
            complexity: NetworkComplexity.Medium,
            taskType: NeuralNetworkTaskType.Regression,
            shouldReturnFullSequence: false,
            layers: layers,
            isDynamicSampleCount: true,
            isPlaceholder: false);
        
        // Create the neural network
        _network = new NeuralNetwork<T>(architecture);
        
        // Initialize noisy parameters if using noisy networks
        if (_useNoisyNetworks)
        {
            ResetNoise();
        }
    }
    
    /// <summary>
    /// Performs a forward pass through the network.
    /// </summary>
    /// <param name="state">The input state.</param>
    /// <returns>A tensor of quantile values for each action.</returns>
    public Tensor<T> Forward(Tensor<T> state)
    {
        // Forward pass through the network
        var output = _network.Predict(state);
        
        // Get the output as vector for single sample or as tensor for batch
        Vector<T> outputVector;
        bool isBatch = state.Rank > 1 && state.Shape[0] > 1;
        
        if (isBatch)
        {
            // Process batch output
            int batchSize = state.Shape[0];
            var reshapedOutput = new Tensor<T>(new[] { batchSize, _actionSize, _numQuantiles });
            
            for (int b = 0; b < batchSize; b++)
            {
                for (int a = 0; a < _actionSize; a++)
                {
                    for (int q = 0; q < _numQuantiles; q++)
                    {
                        reshapedOutput[b, a, q] = output[b, a * _numQuantiles + q];
                    }
                }
            }
            
            return reshapedOutput;
        }
        else
        {
            // Process single sample output
            outputVector = output.ToVector();
            var reshapedOutput = new Tensor<T>(new[] { 1, _actionSize, _numQuantiles });
            
            for (int a = 0; a < _actionSize; a++)
            {
                for (int q = 0; q < _numQuantiles; q++)
                {
                    reshapedOutput[0, a, q] = outputVector[a * _numQuantiles + q];
                }
            }
            
            return reshapedOutput;
        }
    }
    
    /// <summary>
    /// Performs a backward pass and updates the network parameters.
    /// </summary>
    /// <param name="loss">The loss value.</param>
    public void Backward(T loss)
    {
        // Use the extension method for backward propagation
        NeuralNetworkExtensions.Backward(_network, loss);
    }
    
    /// <summary>
    /// Resets the noise for noisy network layers.
    /// </summary>
    public void ResetNoise()
    {
        if (!_useNoisyNetworks)
        {
            return;
        }
        
        // Get all current parameters
        var parameters = _network.GetParameters();
        
        // Add noise to parameters
        for (int i = 0; i < parameters.Length; i++)
        {
            // Generate Gaussian noise with mean 0 and std _noiseStd
            double noise = SampleGaussian(0, _noiseStd);
            parameters[i] = NumOps.Add(
                parameters[i],
                NumOps.FromDouble(noise)
            );
        }
        
        // Set the noisy parameters back to the network
        _network.UpdateParameters(parameters);
    }
    
    /// <summary>
    /// Samples a value from a Gaussian distribution.
    /// </summary>
    /// <param name="mean">The mean of the distribution.</param>
    /// <param name="stdDev">The standard deviation of the distribution.</param>
    /// <returns>A random sample from the distribution.</returns>
    private double SampleGaussian(double mean, double stdDev)
    {
        // Box-Muller transform
        double u1 = 1.0 - _random.NextDouble(); // Uniform(0,1] random doubles
        double u2 = 1.0 - _random.NextDouble();
        double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        
        return mean + stdDev * z;
    }
    
    /// <summary>
    /// Gets all parameters of the network as a single vector.
    /// </summary>
    /// <returns>A vector containing all network parameters.</returns>
    public Vector<T> GetParameters()
    {
        return _network.GetParameters();
    }
    
    /// <summary>
    /// Sets all parameters of the network from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    public void SetParameters(Vector<T> parameters)
    {
        UpdateParameters(parameters);
    }
    
    /// <summary>
    /// Updates the parameters of the network.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    public void UpdateParameters(Vector<T> parameters)
    {
        _network.UpdateParameters(parameters);
    }
}