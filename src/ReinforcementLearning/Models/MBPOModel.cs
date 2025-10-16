namespace AiDotNet.ReinforcementLearning.Models;

/// <summary>
/// Implementation of Model-Based Policy Optimization (MBPO) for reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Model-Based Policy Optimization is a hybrid model-based/model-free reinforcement learning
/// algorithm that learns a model of the environment dynamics and uses it to generate synthetic
/// experiences to augment real experiences. This improves sample efficiency and accelerates
/// learning compared to purely model-free approaches.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// MBPO combines two powerful approaches to reinforcement learning:
/// 
/// 1. It builds a "simulator" of how financial markets behave (model-based)
/// 2. It uses this simulator to generate thousands of synthetic trading scenarios
/// 3. It trains a trading strategy using both real market data and simulated data
/// 
/// This hybrid approach gives you the best of both worlds:
/// - Fast learning like model-based methods (needs less real market data)
/// - Reliable performance like model-free methods (doesn't rely too heavily on the simulator)
/// 
/// For trading applications, this means you can develop sophisticated strategies with
/// less real-world market interaction, reducing risk during the learning phase.
/// </para>
/// </remarks>
public class MBPOModel<T> : ReinforcementLearningModelBase<T>
{
    private readonly Models.Options.MBPOOptions _options = default!;
    private MBPOAgent<T>? _agent;
    
    // Statistics for monitoring
    private int _totalRealExperiences;
    private int _totalModelExperiences;
    private int _modelTrainingIterations;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="MBPOModel{T}"/> class.
    /// </summary>
    /// <param name="options">The options for configuring the MBPO algorithm.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes a new MBPO model with the specified options, setting up
    /// the environment model, policy, and other components needed for the algorithm.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This sets up a trading system that learns in two ways:
    /// 1. From real market experiences (actual trades or historical data)
    /// 2. From a simulator that it builds and refines over time
    /// 
    /// The system has three main components:
    /// - A market simulator that predicts how markets will respond to different conditions
    /// - A trading strategy that decides what actions to take in different market states
    /// - A value estimator that predicts the long-term profit from different market situations
    /// 
    /// By combining these components, MBPO can learn effective trading strategies with
    /// much less real market data than traditional approaches.
    /// </para>
    /// </remarks>
    public MBPOModel(Models.Options.MBPOOptions options)
        : base(options)
    {
        _options = options;
        
        // MBPO can handle both continuous and discrete action spaces
        IsContinuous = options.IsContinuous;
        
        _totalRealExperiences = 0;
        _totalModelExperiences = 0;
        _modelTrainingIterations = 0;
        
        // Agent is initialized in InitializeAgent method
    }
    
    /// <summary>
    /// Initializes the MBPO agent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method creates and initializes the MBPO agent with the configured options.
    /// The agent includes the environment model, policy network, and value network components.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This creates the "brain" of the trading system, which includes:
    /// - A neural network that simulates market behavior
    /// - A neural network that decides on trading actions
    /// - A neural network that evaluates potential future returns
    /// - Memory systems for storing and learning from real and simulated trading experiences
    /// 
    /// These components work together to help the system learn to make profitable trading
    /// decisions while minimizing real-world market exposure.
    /// </para>
    /// </remarks>
    protected override void InitializeAgent()
    {
        _agent = new MBPOAgent<T>(_options);
    }
    
    /// <summary>
    /// Gets the MBPO agent instance.
    /// </summary>
    /// <returns>The MBPO agent as an IAgent interface.</returns>
    protected override IAgent<Tensor<T>, Vector<T>, T> GetAgent()
    {
        if (_agent == null)
            throw new InvalidOperationException("Agent not initialized");
        return _agent;
    }
    
    /// <summary>
    /// Selects an action based on the current state.
    /// </summary>
    /// <param name="state">The current state observation.</param>
    /// <param name="isTraining">Whether the model is in training mode.</param>
    /// <returns>The selected action as a vector.</returns>
    /// <remarks>
    /// <para>
    /// This method selects an action based on the current state using the MBPO policy.
    /// During training, it balances exploration and exploitation according to the entropy
    /// regularization settings. During evaluation, it selects the most likely action.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is where the system decides what trading action to take given the current market state.
    /// 
    /// Key aspects of this decision process:
    /// - During training, it balances trying new strategies vs. using proven ones
    /// - During actual trading, it focuses on the action most likely to maximize returns
    /// - It can handle both discrete actions (buy, hold, sell) and continuous actions (like
    ///   how much of the portfolio to allocate to different assets)
    /// 
    /// The policy network making these decisions has been trained on both real market data
    /// and thousands of simulated market scenarios.
    /// </para>
    /// </remarks>
    public override Vector<T> SelectAction(Tensor<T> state, bool isTraining = false)
    {
        if (_agent == null)
            throw new InvalidOperationException("Agent not initialized");
        return _agent.SelectAction(state, isTraining || IsTraining);
    }
    
    /// <summary>
    /// Updates the MBPO agent based on experience with the environment.
    /// </summary>
    /// <param name="state">The current state observation.</param>
    /// <param name="action">The action taken.</param>
    /// <param name="reward">The reward received.</param>
    /// <param name="nextState">The next state observation.</param>
    /// <param name="done">Whether the episode is done.</param>
    /// <returns>The loss value from the update, or zero if no update was performed.</returns>
    /// <remarks>
    /// <para>
    /// This method updates the MBPO model by storing the real experience in a replay buffer,
    /// potentially training the dynamics model, generating synthetic experiences, and updating
    /// the policy and value functions.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is how the system learns from each new market interaction. The process involves:
    /// 
    /// 1. Storing the real market experience in memory
    /// 2. Periodically updating the market simulator based on accumulated experiences
    /// 3. Using the simulator to generate many synthetic market scenarios
    /// 4. Updating the trading strategy based on both real and simulated experiences
    /// 
    /// This approach allows the system to extract much more learning from each real market
    /// interaction by supplementing it with simulated scenarios.
    /// </para>
    /// </remarks>
    public override T Update(Tensor<T> state, Vector<T> action, T reward, Tensor<T> nextState, bool done)
    {
        if (!IsTraining)
        {
            return NumOps.Zero; // No updates during evaluation
        }
        
        // Add the real experience to the agent
        if (_agent == null)
            throw new InvalidOperationException("Agent not initialized");
        _agent.AddRealExperience(state, action, reward, nextState, done);
        _totalRealExperiences++;
        
        // Check if we have enough real experiences to start model-based training
        if (_totalRealExperiences < _options.RealExpBeforeModel)
        {
            // Not enough real data yet, just do model-free updates
            if (_agent == null)
                throw new InvalidOperationException("Agent not initialized");
            LastLoss = _agent.UpdatePolicyFromRealData();
            return LastLoss;
        }
        
        // We have enough data for model-based training
        
        // Periodically train the dynamics model
        if (_totalRealExperiences % _options.ModelTrainingFrequency == 0)
        {
            // Train the dynamics model on real experiences
            if (_agent == null)
                throw new InvalidOperationException("Agent not initialized");
            var modelLoss = _agent.TrainDynamicsModel();
            _modelTrainingIterations++;
            
            // Generate synthetic experiences using the dynamics model
            int numSynthetic = _options.ModelRatio * _options.BatchSize;
            if (_agent == null)
                throw new InvalidOperationException("Agent not initialized");
            _agent.GenerateSyntheticExperiences(numSynthetic);
            _totalModelExperiences += numSynthetic;
        }
        
        // Update policy and value functions using both real and synthetic experiences
        if (_agent == null)
            throw new InvalidOperationException("Agent not initialized");
        LastLoss = _agent.UpdatePolicyFromAllData();
        
        return LastLoss;
    }
    
    /// <summary>
    /// Trains the MBPO agent on a batch of experiences.
    /// </summary>
    /// <param name="states">The batch of states.</param>
    /// <param name="actions">The batch of actions.</param>
    /// <param name="rewards">The batch of rewards.</param>
    /// <param name="nextStates">The batch of next states.</param>
    /// <param name="dones">The batch of done flags.</param>
    /// <returns>The loss value from the training.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a training update on the MBPO model using a batch of experiences.
    /// It adds the experiences to the real experience buffer and potentially triggers model
    /// training and synthetic data generation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is used for batch training on historical market data. It follows the same
    /// general process as the individual update method:
    /// 
    /// 1. Add all provided market experiences to memory
    /// 2. Train the market simulator if enough data is available
    /// 3. Generate synthetic market scenarios using the simulator
    /// 4. Update the trading strategy using both real and simulated data
    /// 
    /// This is particularly useful for initial training on historical data before
    /// deploying the system for actual trading.
    /// </para>
    /// </remarks>
    protected override T TrainOnBatch(
        Tensor<T> states,
        Tensor<T> actions,
        Vector<T> rewards,
        Tensor<T> nextStates,
        Vector<T> dones)
    {
        int batchSize = states.Shape[0];
        
        // Add all experiences to the real buffer
        for (int i = 0; i < batchSize; i++)
        {
            var state = states.GetSlice(i);
            
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
            
            var nextState = nextStates.GetSlice(i);
            bool done = NumOps.GreaterThan(dones[i], NumOps.FromDouble(0.5));
            
            if (_agent == null)
                throw new InvalidOperationException("Agent not initialized");
            _agent.AddRealExperience(state, actionVector, rewards[i], nextState, done);
        }
        
        _totalRealExperiences += batchSize;
        
        // Check if we have enough real experiences to start model-based training
        if (_totalRealExperiences < _options.RealExpBeforeModel)
        {
            // Not enough real data yet, just do model-free updates
            if (_agent == null)
                throw new InvalidOperationException("Agent not initialized");
            LastLoss = _agent.UpdatePolicyFromRealData();
            return LastLoss;
        }
        
        // Train the dynamics model
        if (_agent == null)
            throw new InvalidOperationException("Agent not initialized");
        var modelLoss = _agent.TrainDynamicsModel();
        _modelTrainingIterations++;
        
        // Generate synthetic experiences
        int numSynthetic = _options.ModelRatio * batchSize;
        if (_agent == null)
            throw new InvalidOperationException("Agent not initialized");
        _agent.GenerateSyntheticExperiences(numSynthetic);
        _totalModelExperiences += numSynthetic;
        
        // Update policy using both real and synthetic data
        if (_agent == null)
            throw new InvalidOperationException("Agent not initialized");
        LastLoss = _agent.UpdatePolicyFromAllData();
        
        return LastLoss;
    }
    
    /// <summary>
    /// Gets all parameters of the MBPO model as a single vector.
    /// </summary>
    /// <returns>A vector containing all model parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method extracts all parameters from the dynamics model, policy network, and
    /// value network, and combines them into a single vector for serialization.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This gathers all the learned knowledge from the market simulator, trading strategy,
    /// and value estimator into a single package. This comprehensive representation of the
    /// system's intelligence can be saved, loaded, or transferred.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        if (_agent == null)
            throw new InvalidOperationException("Agent not initialized");
        return _agent.GetParameters();
    }
    
    /// <summary>
    /// Sets all parameters of the MBPO model from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all model parameters.</param>
    /// <remarks>
    /// <para>
    /// This method distributes the parameters from a single vector to the dynamics model,
    /// policy network, and value network. This is useful when loading a serialized model.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This takes a saved package of the system's intelligence and loads it back into
    /// the market simulator, trading strategy, and value estimator components. It allows
    /// you to continue using a previously trained system without having to train it again.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        if (_agent == null)
            throw new InvalidOperationException("Agent not initialized");
        _agent.SetParameters(parameters);
    }
    
    /// <summary>
    /// Gets the latest loss value from training.
    /// </summary>
    /// <returns>The latest loss value.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the most recent loss value from model training.
    /// For MBPO, this represents a combination of policy, value, and possibly model losses.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This tells you how accurate the system's predictions are currently. MBPO has multiple
    /// components learning simultaneously, so this value represents the combined progress
    /// of the trading strategy and value estimator.
    /// </para>
    /// </remarks>
    public override T GetLoss()
    {
        return LastLoss;
    }
    
    /// <summary>
    /// Gets statistics about the model training process.
    /// </summary>
    /// <returns>A dictionary containing training statistics.</returns>
    /// <remarks>
    /// <para>
    /// This method returns information about the MBPO training process, including the number of
    /// real and synthetic experiences used, model training iterations, and the current
    /// mixture ratio of real to synthetic data.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This provides insights into how the system is learning, showing metrics like:
    /// - How many real market experiences have been collected
    /// - How many simulated experiences have been generated
    /// - How many times the market simulator has been updated
    /// - The current balance between real and simulated data in training
    /// 
    /// These statistics help you monitor the learning process and ensure the system
    /// is properly balancing real and simulated experiences.
    /// </para>
    /// </remarks>
    public Dictionary<string, double> GetTrainingStats()
    {
        var stats = new Dictionary<string, double>
        {
            { "TotalRealExperiences", _totalRealExperiences },
            { "TotalModelExperiences", _totalModelExperiences },
            { "ModelTrainingIterations", _modelTrainingIterations },
            { "RealToSyntheticRatio", _totalModelExperiences > 0 ? 
                (double)_totalRealExperiences / _totalModelExperiences : 0 }
        };
        
        // Add agent-specific metrics
        if (_agent == null)
            throw new InvalidOperationException("Agent not initialized");
        var agentStats = _agent.GetStats();
        foreach (var kvp in agentStats)
        {
            stats[kvp.Key] = kvp.Value;
        }
        
        return stats;
    }
    
    /// <summary>
    /// Uses the dynamics model to predict future states from a given state and action.
    /// </summary>
    /// <param name="state">The starting state.</param>
    /// <param name="action">The action to take.</param>
    /// <param name="numSteps">The number of steps to predict into the future.</param>
    /// <returns>A list of predicted future states.</returns>
    /// <remarks>
    /// <para>
    /// This method uses the learned dynamics model to predict how the environment will
    /// evolve over multiple timesteps given a starting state and action. This can be
    /// useful for planning and visualization purposes.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This uses the market simulator to predict what might happen in the future given
    /// the current market state and a specific trading action. It can predict multiple
    /// steps ahead, showing you how the market might evolve over time.
    /// 
    /// This forecast can be valuable for:
    /// - Visualizing potential market trajectories
    /// - Understanding the model's expectations
    /// - Planning trading strategies over multiple time steps
    /// - Identifying potential risks and opportunities in future market states
    /// </para>
    /// </remarks>
    public List<Tensor<T>> PredictFutureStates(Tensor<T> state, Vector<T> action, int numSteps)
    {
        if (_agent == null)
            throw new InvalidOperationException("Agent not initialized");
        return _agent.PredictFutureStates(state, action, numSteps);
    }
    
    /// <summary>
    /// Predicts the uncertainty in future state predictions.
    /// </summary>
    /// <param name="state">The starting state.</param>
    /// <param name="action">The action to take.</param>
    /// <returns>A measure of uncertainty in the prediction.</returns>
    /// <remarks>
    /// <para>
    /// This method uses the ensemble of dynamics models to estimate the uncertainty in
    /// the prediction of the next state. Higher values indicate greater disagreement
    /// among the models and thus higher uncertainty.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This tells you how confident the market simulator is in its predictions about what
    /// will happen after a certain trading action. It works by comparing the predictions
    /// from multiple simulators in the ensemble.
    /// 
    /// High uncertainty might indicate:
    /// - Unusual market conditions the system hasn't seen before
    /// - Potential for unexpected market movements
    /// - Areas where more caution might be warranted
    /// 
    /// This uncertainty measure can be valuable for risk management and decision making.
    /// </para>
    /// </remarks>
    public T GetPredictionUncertainty(Tensor<T> state, Vector<T> action)
    {
        if (_agent == null)
            throw new InvalidOperationException("Agent not initialized");
        return _agent.GetPredictionUncertainty(state, action);
    }
    
    /// <summary>
    /// Saves the MBPO model to a stream.
    /// </summary>
    /// <param name="stream">The stream to save the model to.</param>
    /// <remarks>
    /// <para>
    /// This method serializes the MBPO model's parameters and configuration
    /// to the provided stream, including the dynamics model, policy network, and
    /// value network parameters.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This saves all of the system's intelligence to a file stream. It captures:
    /// - What the market simulator has learned about market behavior
    /// - What the trading strategy has learned about optimal actions
    /// - What the value estimator has learned about expected returns
    /// - All the configuration settings that define how the system operates
    /// 
    /// This allows you to save a trained system and use it later without retraining.
    /// </para>
    /// </remarks>
    public override void Save(Stream stream)
    {
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, true))
        {
            // Write model type identifier
            writer.Write("MBPOModel");
            
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
            
            // Save MBPO specific options
            writer.Write(_options.EnsembleSize);
            writer.Write(_options.ModelRatio);
            writer.Write(_options.RolloutHorizon);
            writer.Write(_options.ProbabilisticModel);
            writer.Write(_options.EnsemblePolicy);
            writer.Write(_options.PolicyEnsembleSize);
            writer.Write(_options.InitialTemperature);
            writer.Write(_options.AutoTuneEntropy);
            writer.Write(_options.ModelPredictRewards);
            writer.Write(_options.BranchingRollouts);
            writer.Write(_options.NumBranches);
            
            // Save model network architecture
            writer.Write(_options.ModelHiddenSizes.Length);
            foreach (var size in _options.ModelHiddenSizes)
            {
                writer.Write(size);
            }
            
            // Save policy network architecture
            writer.Write(_options.PolicyHiddenSizes.Length);
            foreach (var size in _options.PolicyHiddenSizes)
            {
                writer.Write(size);
            }
            
            // Save value network architecture
            writer.Write(_options.ValueHiddenSizes.Length);
            foreach (var size in _options.ValueHiddenSizes)
            {
                writer.Write(size);
            }
            
            // Save training statistics
            writer.Write(_totalRealExperiences);
            writer.Write(_totalModelExperiences);
            writer.Write(_modelTrainingIterations);
        }
    }
    
    /// <summary>
    /// Loads the MBPO model from a stream.
    /// </summary>
    /// <param name="stream">The stream to load the model from.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes the MBPO model's parameters and configuration
    /// from the provided stream, restoring the dynamics model, policy network, and
    /// value network parameters.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This loads a previously saved system's intelligence from a file stream. It restores:
    /// - The market simulator's understanding of market behavior
    /// - The trading strategy's knowledge of optimal actions
    /// - The value estimator's predictions of expected returns
    /// - All the configuration settings that define the system's operation
    /// 
    /// This allows you to use a previously trained system without having to retrain it.
    /// </para>
    /// </remarks>
    public override void Load(Stream stream)
    {
        using (var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, true))
        {
            // Read and verify model type identifier
            string modelType = reader.ReadString();
            if (modelType != "MBPOModel")
            {
                throw new InvalidOperationException($"Expected MBPOModel, but got {modelType}");
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
            
            // Load MBPO specific options
            int ensembleSize = reader.ReadInt32();
            int modelRatio = reader.ReadInt32();
            int rolloutHorizon = reader.ReadInt32();
            bool probabilisticModel = reader.ReadBoolean();
            bool ensemblePolicy = reader.ReadBoolean();
            int policyEnsembleSize = reader.ReadInt32();
            double initialTemperature = reader.ReadDouble();
            bool autoTuneEntropy = reader.ReadBoolean();
            bool modelPredictRewards = reader.ReadBoolean();
            bool branchingRollouts = reader.ReadBoolean();
            int numBranches = reader.ReadInt32();
            
            // Load model network architecture
            int modelLayerCount = reader.ReadInt32();
            int[] modelHiddenSizes = new int[modelLayerCount];
            for (int i = 0; i < modelLayerCount; i++)
            {
                modelHiddenSizes[i] = reader.ReadInt32();
            }
            
            // Load policy network architecture
            int policyLayerCount = reader.ReadInt32();
            int[] policyHiddenSizes = new int[policyLayerCount];
            for (int i = 0; i < policyLayerCount; i++)
            {
                policyHiddenSizes[i] = reader.ReadInt32();
            }
            
            // Load value network architecture
            int valueLayerCount = reader.ReadInt32();
            int[] valueHiddenSizes = new int[valueLayerCount];
            for (int i = 0; i < valueLayerCount; i++)
            {
                valueHiddenSizes[i] = reader.ReadInt32();
            }
            
            // Load training statistics
            _totalRealExperiences = reader.ReadInt32();
            _totalModelExperiences = reader.ReadInt32();
            _modelTrainingIterations = reader.ReadInt32();
            
            // You can optionally update the options if needed
            // _options.EnsembleSize = ensembleSize;
            // _options.ModelRatio = modelRatio;
            // etc.
        }
    }
    
    /// <summary>
    /// Creates a new instance of the MBPO model with the same configuration.
    /// </summary>
    /// <returns>A new instance of the MBPO model.</returns>
    public override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new MBPOModel<T>(_options);
    }
}