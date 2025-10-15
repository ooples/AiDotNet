namespace AiDotNet.ReinforcementLearning.Models;

/// <summary>
/// Implementation of Quantile Regression Deep Q-Network (QR-DQN) for distributional reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// QR-DQN is a distributional reinforcement learning algorithm that models the entire distribution
/// of returns rather than just the expected value. By estimating quantiles of the return distribution,
/// it provides a much richer representation of uncertainty and risk, which is particularly valuable
/// for financial market applications.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Traditional reinforcement learning models try to predict the average future reward for each action.
/// QR-DQN goes beyond this by predicting the entire range of possible outcomes. This is like the
/// difference between:
/// 
/// - A weather forecast that just says "70% chance of rain tomorrow"
/// - A forecast that tells you "10% chance of no rain, 20% chance of light rain, 40% chance of
///   moderate rain, 20% chance of heavy rain, 10% chance of thunderstorms"
/// 
/// For financial markets, this detailed distribution information is crucial because:
/// - It captures the risk and uncertainty of different trading actions
/// - It can identify strategies with similar average returns but very different risk profiles
/// - It allows for risk-aware decision making that considers potential downside
/// - It can better handle the complex, non-normal distributions common in financial returns
/// </para>
/// </remarks>
public class QRDQNModel<T> : ReinforcementLearningModelBase<T>
{
    private readonly QRDQNOptions _options = default!;
    private QRDQNAgent<T> _agent = null!;
    private DiscreteActionAdapter<T> _adapter = null!;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="QRDQNModel{T}"/> class.
    /// </summary>
    /// <param name="options">The options for configuring the QR-DQN algorithm.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes a new QR-DQN model with the specified options, setting up
    /// the neural network architecture, replay buffer, and other components needed for training
    /// and inference.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This sets up the QR-DQN model with your specified configuration. The model uses a specialized
    /// neural network architecture that can predict entire distributions of potential outcomes for
    /// each possible trading action, rather than just a single expected value.
    /// 
    /// Key aspects of the model include:
    /// - Distributional predictions that capture market uncertainty
    /// - Risk-aware decision making capabilities
    /// - Sophisticated exploration strategies to find profitable trading patterns
    /// </para>
    /// </remarks>
    public QRDQNModel(QRDQNOptions options)
        : base(options)
    {
        _options = options;
        
        // QR-DQN typically works with discrete action spaces
        IsContinuous = false;
        
        // Agent is initialized in InitializeAgent method
    }
    
    /// <summary>
    /// Initializes the QR-DQN agent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method creates and initializes the QR-DQN agent with the configured options.
    /// The agent includes the neural network for predicting quantile values, replay buffer
    /// for storing experiences, and exploration strategy.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This creates the "brain" of the QR-DQN agent, including:
    /// - Neural networks that predict distributions of potential returns
    /// - Memory systems for storing and learning from trading experiences
    /// - Exploration mechanisms to discover profitable strategies
    /// - Risk assessment capabilities for making better trading decisions
    /// 
    /// These components work together to help the agent learn to make trading decisions
    /// while properly accounting for market uncertainty and risk.
    /// </para>
    /// </remarks>
    protected override void InitializeAgent()
    {
        _agent = new QRDQNAgent<T>(_options);
        _adapter = new DiscreteActionAdapter<T>(_agent, _options.ActionSize);
    }
    
    /// <summary>
    /// Gets the QR-DQN agent instance.
    /// </summary>
    /// <returns>The QR-DQN agent as an IAgent interface.</returns>
    protected override IAgent<Tensor<T>, Vector<T>, T> GetAgent()
    {
        return _adapter ?? throw new InvalidOperationException("Agent has not been initialized");
    }
    
    /// <summary>
    /// Selects an action based on the current state.
    /// </summary>
    /// <param name="state">The current state observation.</param>
    /// <param name="isTraining">Whether the model is in training mode.</param>
    /// <returns>The selected action as a vector.</returns>
    /// <remarks>
    /// <para>
    /// This method selects an action based on the current state using the QR-DQN algorithm.
    /// During training, it balances exploration and exploitation according to the configured
    /// exploration strategy. During evaluation, it selects actions based on the quantile values,
    /// potentially using risk-sensitive approaches like CVaR if configured.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is where the model decides what trading action to take given the current market state.
    /// 
    /// The key difference from traditional methods is that QR-DQN doesn't just pick the action
    /// with the highest average expected return. Instead, it can:
    /// - Consider the full distribution of possible returns for each action
    /// - Be more cautious about actions with high volatility or downside risk
    /// - Make trading decisions that align with specific risk preferences
    /// 
    /// This leads to more robust trading strategies, especially in volatile markets.
    /// </para>
    /// </remarks>
    public override Vector<T> SelectAction(Tensor<T> state, bool isTraining = false)
    {
        // Forward the state to the adapter to select an action
        return _adapter.SelectAction(state, isTraining || IsTraining);
    }
    
    /// <summary>
    /// Updates the QR-DQN agent based on experience with the environment.
    /// </summary>
    /// <param name="state">The current state observation.</param>
    /// <param name="action">The action taken.</param>
    /// <param name="reward">The reward received.</param>
    /// <param name="nextState">The next state observation.</param>
    /// <param name="done">Whether the episode is done.</param>
    /// <returns>The loss value from the update, or zero if no update was performed.</returns>
    /// <remarks>
    /// <para>
    /// This method updates the QR-DQN model by storing the experience in a replay buffer
    /// and potentially performing a training update if enough experiences have been collected.
    /// The update computes the quantile regression loss across all quantiles of the return distribution.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is how the model learns from its trading experiences. The process works like this:
    /// 
    /// 1. A new market experience (state, action, reward, next state) is stored in memory
    /// 2. Periodically, the model samples a batch of experiences from memory
    /// 3. For each experience, the model:
    ///    - Predicts the distribution of potential returns for each action
    ///    - Computes how accurate these distribution predictions were
    ///    - Updates its neural network to make better predictions next time
    /// 
    /// The unique aspect of QR-DQN is that it's learning to predict entire distributions
    /// of possible outcomes, not just average values. This makes it much better at handling
    /// the inherent uncertainty in financial markets.
    /// </para>
    /// </remarks>
    public override T Update(Tensor<T> state, Vector<T> action, T reward, Tensor<T> nextState, bool done)
    {
        if (!IsTraining)
        {
            return NumOps.Zero; // No updates during evaluation
        }
        
        // Add the experience to the adapter's replay buffer and potentially update
        _adapter.Learn(state, action, reward, nextState, done);
        
        // Get the latest loss value from the agent
        LastLoss = _agent.GetLatestLoss();
        
        return LastLoss;
    }
    
    /// <summary>
    /// Trains the QR-DQN agent on a batch of experiences.
    /// </summary>
    /// <param name="states">The batch of states.</param>
    /// <param name="actions">The batch of actions.</param>
    /// <param name="rewards">The batch of rewards.</param>
    /// <param name="nextStates">The batch of next states.</param>
    /// <param name="dones">The batch of done flags.</param>
    /// <returns>The loss value from the training.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a training update on the QR-DQN model using a batch of experiences.
    /// It computes the quantile regression loss for all quantiles across the batch and updates
    /// the neural network weights accordingly.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is the detailed learning process where the model analyzes batches of trading experiences
    /// to improve its predictions. The QR-DQN uses a specialized loss function called "quantile regression"
    /// that helps it learn accurate probability distributions rather than just point estimates.
    /// 
    /// This distributional approach helps the model capture important aspects of market behavior:
    /// - The likelihood of different return outcomes
    /// - The potential magnitude of gains and losses
    /// - The overall uncertainty in different market conditions
    /// </para>
    /// </remarks>
    protected override T TrainOnBatch(
        Tensor<T> states,
        Tensor<T> actions,
        Vector<T> rewards,
        Tensor<T> nextStates,
        Vector<T> dones)
    {
        // Convert tensors to arrays
        Tensor<T>[] stateArray = new Tensor<T>[states.Shape[0]];
        Vector<T>[] actionArray = new Vector<T>[actions.Shape[0]];
        T[] rewardArray = rewards.ToArray();
        Tensor<T>[] nextStateArray = new Tensor<T>[nextStates.Shape[0]];
        bool[] doneArray = new bool[dones.Length];
        
        // Extract states and actions from tensors
        for (int i = 0; i < states.Shape[0]; i++)
        {
            stateArray[i] = states.GetSlice(i);
            
            // Extract action vector from tensor
            Vector<T> actionVector;
            if (actions.Rank > 1 && actions.Shape[1] > 1)
            {
                // Multi-dimensional action
                actionVector = new Vector<T>(actions.Shape[1]);
                for (int j = 0; j < actions.Shape[1]; j++)
                {
                    actionVector[j] = actions[i, j];
                }
            }
            else
            {
                // Single-dimensional action
                actionVector = new Vector<T>(1) { [0] = actions[i, 0] };
            }
            actionArray[i] = actionVector;
            
            nextStateArray[i] = nextStates.GetSlice(i);
            doneArray[i] = NumOps.GreaterThan(dones[i], NumOps.FromDouble(0.5));
        }
        
        // Use the adapter to train on the batch
        return _adapter.Train(stateArray, actionArray, rewardArray, nextStateArray, doneArray);
    }
    
    /// <summary>
    /// Gets all parameters of the QR-DQN model as a single vector.
    /// </summary>
    /// <returns>A vector containing all model parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method extracts all parameters from the neural networks and combines them into a single vector.
    /// This is useful for serialization and parameter management.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This gathers all the "knowledge" contained in the model's neural networks into a single
    /// list of numbers. These parameters represent everything the model has learned about how
    /// to make profitable trading decisions while accounting for risk and uncertainty.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        return _agent.GetParameters();
    }
    
    /// <summary>
    /// Sets all parameters of the QR-DQN model from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all model parameters.</param>
    /// <remarks>
    /// <para>
    /// This method distributes the parameters from a single vector to the neural networks.
    /// This is useful when loading a serialized model or after parameter optimization.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This takes a list of numbers representing the model's trading knowledge and loads them
    /// into the neural networks. It's like importing a saved strategy into the trading model,
    /// allowing it to use previously learned patterns without retraining.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        _agent.SetParameters(parameters);
    }
    
    /// <summary>
    /// Creates a new instance of the model.
    /// </summary>
    /// <returns>A new instance of the model with the same configuration but no learned parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the QRDQN model with the same configuration
    /// as this instance but without copying learned parameters.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This creates a brand new QRDQN agent with the same settings but none of the
    /// learned experience. It's like creating a new agent with the same capabilities
    /// but that hasn't been trained yet.
    /// </para>
    /// </remarks>
    public override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new QRDQNModel<T>(_options);
    }
    
    /// <summary>
    /// Creates a deep copy of this model.
    /// </summary>
    /// <returns>A deep copy of this model with the same parameters and state.</returns>
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        // Use the base class implementation which uses serialization/deserialization
        return base.DeepCopy();
    }
    
    /// <summary>
    /// Creates a shallow copy of this model.
    /// </summary>
    /// <returns>A shallow copy of this model.</returns>
    public override IFullModel<T, Tensor<T>, Tensor<T>> Clone()
    {
        // Use the base class implementation which creates a new instance
        return base.Clone();
    }
    
    /// <summary>
    /// Creates a new instance with the specified parameters.
    /// </summary>
    /// <param name="parameters">The parameters to set in the new instance.</param>
    /// <returns>A new instance with the specified parameters.</returns>
    public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        // Use the base class implementation which uses DeepCopy and SetParameters
        return base.WithParameters(parameters);
    }
    
    /// <summary>
    /// Gets the metadata for this model.
    /// </summary>
    /// <returns>The model metadata.</returns>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        
        metadata.ModelType = ModelType.QRDQNModel; // QR-DQN uses neural networks
        metadata.Description = "Distributional reinforcement learning for risk-sensitive decision making";
        metadata.FeatureCount = _options.StateSize;
        metadata.Complexity = _options.HiddenLayerSizes.Sum() * _options.NumQuantiles; // Complexity based on network size and quantiles
        metadata.AdditionalInfo = new Dictionary<string, object>
        {
            { "Algorithm", "QRDQN" },
            { "StateSize", _options.StateSize },
            { "ActionSize", _options.ActionSize },
            { "NetworkArchitecture", _options.NetworkArchitecture },
            { "NumQuantiles", _options.NumQuantiles },
            { "RiskMetric", _options.RiskMetric },
            { "RiskLevel", _options.RiskLevel }
        };

        return metadata;
    }
    
    
    /// <summary>
    /// Gets the latest loss value from training.
    /// </summary>
    /// <returns>The latest loss value.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the most recent loss value from model training.
    /// For QR-DQN, this represents the quantile regression loss across all quantiles.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This tells you how accurate the model's predictions are currently. Lower values generally
    /// indicate better performance, though for QR-DQN the loss is more complex since it's learning
    /// entire distributions rather than single values.
    /// </para>
    /// </remarks>
    public override T GetLoss()
    {
        return LastLoss;
    }
    
    
    /// <summary>
    /// Gets the predicted return distribution for a given state and action.
    /// </summary>
    /// <param name="state">The state to evaluate.</param>
    /// <param name="action">The action to evaluate. If null, returns distributions for all actions.</param>
    /// <returns>The quantile values representing the return distribution.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the predicted return distribution for a specific state-action pair
    /// as a set of quantile values. This provides a complete picture of the possible returns
    /// and their probabilities, enabling risk-aware decision making.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This gives you the model's prediction of all possible returns for a specific trading
    /// action in the current market state. The result shows the complete range of possible outcomes,
    /// from worst-case to best-case scenarios, and everything in between.
    /// 
    /// You can use this to:
    /// - Compare the risk profiles of different actions
    /// - Assess the potential downside risk of a trading decision
    /// - Understand the uncertainty associated with different strategies
    /// - Make decisions that align with specific risk preferences
    /// </para>
    /// </remarks>
    public Tensor<T> GetReturnDistribution(Tensor<T> state, Vector<T>? action = null)
    {
        return _agent.GetReturnDistribution(state, action);
    }
    
    /// <summary>
    /// Calculates the Conditional Value at Risk (CVaR) for a given state and action.
    /// </summary>
    /// <param name="state">The state to evaluate.</param>
    /// <param name="action">The action to evaluate. If null, returns CVaR for all actions.</param>
    /// <param name="alpha">The risk level (between 0 and 1).</param>
    /// <returns>The CVaR value(s) representing the average of the worst alpha% of returns.</returns>
    /// <remarks>
    /// <para>
    /// Conditional Value at Risk (CVaR), also known as Expected Shortfall, is a risk measure
    /// that calculates the expected loss in the worst alpha% of cases. For example, a CVaR(0.05) 
    /// represents the average loss in the worst 5% of scenarios. This allows for risk-sensitive
    /// decision making.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This gives you a measure of the "tail risk" or downside risk of a trading action.
    /// It answers the question: "If things go badly, how bad could they get on average?"
    /// 
    /// For example:
    /// - A CVaR(0.05) of -2.3 means "In the worst 5% of cases, you can expect to lose 2.3 on average"
    /// - A CVaR(0.25) of -0.8 means "In the worst 25% of cases, you can expect to lose 0.8 on average"
    /// 
    /// This is particularly valuable for risk management in trading, as it helps avoid strategies
    /// that might have good average performance but catastrophic worst-case scenarios.
    /// </para>
    /// </remarks>
    public Tensor<T> GetCVaR(Tensor<T> state, Vector<T>? action = null, double alpha = 0.05)
    {
        return _agent.GetCVaR(state, action, alpha);
    }
    
    /// <summary>
    /// Saves the QR-DQN model to a stream.
    /// </summary>
    /// <param name="stream">The stream to save the model to.</param>
    /// <remarks>
    /// <para>
    /// This method serializes the QR-DQN model's parameters and configuration
    /// to the provided stream, including the neural network weights and hyperparameters.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This saves the trading model's knowledge and configuration to a file stream,
    /// allowing you to store a trained model and reload it later without having to
    /// retrain it from scratch.
    /// </para>
    /// </remarks>
    public override void Save(Stream stream)
    {
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, true))
        {
            // Write model type identifier
            writer.Write("QRDQNModel");
            
            // Save agent parameters
            var parameters = GetParameters();
            writer.Write(parameters.Length);
            for (int i = 0; i < parameters.Length; i++)
            {
                writer.Write(Convert.ToDouble(parameters[i]));
            }
            
            // Save core options
            writer.Write(_options.StateSize);
            writer.Write(_options.ActionSize);
            writer.Write(_options.IsContinuous);
            writer.Write(_options.Gamma);
            writer.Write(_options.BatchSize);
            
            // Save QR-DQN specific options
            writer.Write(_options.NumQuantiles);
            writer.Write(_options.HuberKappa);
            writer.Write(_options.UseCVaR);
            writer.Write(_options.CVaRAlpha);
            writer.Write(_options.RiskDistortion);
            writer.Write(_options.UseNoisyNetworks);
            writer.Write(_options.InitialNoiseStd);
            writer.Write(_options.UseDoubleDQN);
            writer.Write(_options.UsePrioritizedReplay);
            writer.Write(_options.PriorityAlpha);
            writer.Write(_options.PriorityBetaStart);
            
            // Save neural network architecture
            writer.Write(_options.HiddenLayerSizes.Length);
            foreach (var layerSize in _options.HiddenLayerSizes)
            {
                writer.Write(layerSize);
            }
        }
    }
    
    /// <summary>
    /// Loads the QR-DQN model from a stream.
    /// </summary>
    /// <param name="stream">The stream to load the model from.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes the QR-DQN model's parameters and configuration
    /// from the provided stream, restoring the neural network weights and hyperparameters.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This loads a previously saved trading model from a file stream, restoring all of its
    /// learned knowledge and configuration. This allows you to deploy a trained model
    /// without having to retrain it from scratch.
    /// </para>
    /// </remarks>
    public override void Load(Stream stream)
    {
        using (var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, true))
        {
            // Read and verify model type identifier
            string modelType = reader.ReadString();
            if (modelType != "QRDQNModel")
            {
                throw new InvalidOperationException($"Expected QRDQNModel, but got {modelType}");
            }
            
            // Load parameters
            int paramCount = reader.ReadInt32();
            var parameters = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                parameters[i] = NumOps.FromDouble(reader.ReadDouble());
            }
            
            // Set parameters to the agent
            SetParameters(parameters);
            
            // Load and verify basic options
            int stateSize = reader.ReadInt32();
            int actionSize = reader.ReadInt32();
            bool isContinuous = reader.ReadBoolean();
            double gamma = reader.ReadDouble();
            int batchSize = reader.ReadInt32();
            
            // Verify that basic options match
            if (stateSize != _options.StateSize || actionSize != _options.ActionSize)
            {
                throw new InvalidOperationException(
                    $"Model dimensions mismatch. Saved model: State={stateSize}, Action={actionSize}, IsContinuous={isContinuous}. " +
                    $"Current options: State={_options.StateSize}, Action={_options.ActionSize}, IsContinuous={_options.IsContinuous}");
            }
            
            // Read QR-DQN specific options
            int numQuantiles = reader.ReadInt32();
            double huberKappa = reader.ReadDouble();
            bool useCVaR = reader.ReadBoolean();
            double cvarAlpha = reader.ReadDouble();
            double riskDistortion = reader.ReadDouble();
            bool useNoisyNetworks = reader.ReadBoolean();
            double initialNoiseStd = reader.ReadDouble();
            bool useDoubleDQN = reader.ReadBoolean();
            bool usePrioritizedReplay = reader.ReadBoolean();
            double priorityAlpha = reader.ReadDouble();
            double priorityBetaStart = reader.ReadDouble();
            
            // Load neural network architecture
            int numLayers = reader.ReadInt32();
            int[] hiddenLayerSizes = new int[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                hiddenLayerSizes[i] = reader.ReadInt32();
            }
            
            // You can optionally update the options if needed
            // _options.NumQuantiles = numQuantiles;
            // _options.HuberKappa = huberKappa;
            // etc.
        }
    }
    
    /// <summary>
    /// Adapter class that wraps agents with discrete actions to work with Vector<double> actions.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    internal class DiscreteActionAdapter<TNumeric> : IAgent<Tensor<TNumeric>, Vector<TNumeric>, TNumeric>
    {
        private readonly IAgent<Tensor<TNumeric>, int, TNumeric> _discreteAgent = default!;
        private readonly int _actionSize;
        private readonly INumericOperations<TNumeric> _numOps = default!;
        
        public DiscreteActionAdapter(IAgent<Tensor<TNumeric>, int, TNumeric> discreteAgent, int actionSize)
        {
            _discreteAgent = discreteAgent;
            _actionSize = actionSize;
            _numOps = MathHelper.GetNumericOperations<TNumeric>();
        }
        
        public bool IsTraining => _discreteAgent.IsTraining;
        
        public Vector<TNumeric> SelectAction(Tensor<TNumeric> state, bool isTraining = true)
        {
            // Get discrete action from the agent
            int action = _discreteAgent.SelectAction(state, isTraining);
            
            // Convert to one-hot vector
            var actionVector = new Vector<TNumeric>(_actionSize);
            for (int i = 0; i < _actionSize; i++)
            {
                actionVector[i] = i == action ? _numOps.One : _numOps.Zero;
            }
            return actionVector;
        }
        
        public void Learn(Tensor<TNumeric> state, Vector<TNumeric> action, TNumeric reward, Tensor<TNumeric> nextState, bool done)
        {
            // Convert vector action to discrete action
            int discreteAction = GetDiscreteAction(action);
            _discreteAgent.Learn(state, discreteAction, reward, nextState, done);
        }
        
        public void Save(string filePath)
        {
            _discreteAgent.Save(filePath);
        }
        
        public void Load(string filePath)
        {
            _discreteAgent.Load(filePath);
        }
        
        public void SetTrainingMode(bool isTraining)
        {
            _discreteAgent.SetTrainingMode(isTraining);
        }
        
        public TNumeric GetLatestLoss()
        {
            if (_discreteAgent is AgentBase<Tensor<TNumeric>, int, TNumeric> agentBase)
            {
                return agentBase.GetLatestLoss();
            }
            return _numOps.Zero;
        }
        
        public TNumeric Train(Tensor<TNumeric>[] states, Vector<TNumeric>[] actions, TNumeric[] rewards, Tensor<TNumeric>[] nextStates, bool[] dones)
        {
            // Convert vector actions to discrete actions
            int[] discreteActions = new int[actions.Length];
            for (int i = 0; i < actions.Length; i++)
            {
                discreteActions[i] = GetDiscreteAction(actions[i]);
            }
            
            // DQNAgent doesn't have a batch Train method that takes arrays,
            // so we need to learn from each experience individually
            for (int i = 0; i < states.Length; i++)
            {
                _discreteAgent.Learn(states[i], discreteActions[i], rewards[i], nextStates[i], dones[i]);
            }
            return GetLatestLoss();
        }
        
        public Vector<TNumeric> GetParameters()
        {
            if (_discreteAgent is QRDQNAgent<TNumeric> qrdqnAgent)
            {
                return qrdqnAgent.GetParameters();
            }
            throw new NotSupportedException("GetParameters not supported for this agent type");
        }
        
        public void SetParameters(Vector<TNumeric> parameters)
        {
            if (_discreteAgent is QRDQNAgent<TNumeric> qrdqnAgent)
            {
                qrdqnAgent.SetParameters(parameters);
            }
            else
            {
                throw new NotSupportedException("SetParameters not supported for this agent type");
            }
        }
        
        public int GetDiscreteAction(Vector<TNumeric> action)
        {
            // Find the index with the highest value (argmax)
            int maxIndex = 0;
            TNumeric maxValue = action[0];
            
            for (int i = 1; i < action.Length; i++)
            {
                if (_numOps.GreaterThan(action[i], maxValue))
                {
                    maxValue = action[i];
                    maxIndex = i;
                }
            }
            
            return maxIndex;
        }
    }
}