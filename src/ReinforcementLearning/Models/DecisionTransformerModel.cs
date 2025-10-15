global using AiDotNet.ReinforcementLearning.Memory;

namespace AiDotNet.ReinforcementLearning.Models;

/// <summary>
/// Decision Transformer model for reinforcement learning through sequence modeling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Decision Transformer is a generative approach to reinforcement learning, which 
/// reformulates the reinforcement learning problem as a sequence modeling problem. 
/// Instead of training a policy or value function, it directly generates high-return 
/// sequences using a transformer architecture.
/// </para>
/// <para>
/// Key features:
/// - Works with both continuous and discrete action spaces
/// - Can be trained offline using historical data
/// - Conditions predictions on desired returns (return-to-go)
/// - Uses the transformer architecture for processing sequences of states, actions, and returns
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// The Decision Transformer is a modern approach to reinforcement learning that's 
/// particularly well-suited for financial applications like stock market prediction:
/// 
/// - It "thinks" about trading like completing a sequence - similar to how language models predict words
/// - Instead of learning through trial and error, it learns patterns from historical market data
/// - You can tell it what return you want to achieve, and it will suggest actions to reach that target
/// - It can handle complex, long-term patterns in financial data
/// - It works well with limited data and doesn't require actual market interaction during training
/// 
/// This makes it safer and more practical for financial applications than traditional
/// reinforcement learning approaches.
/// </para>
/// </remarks>
public class DecisionTransformerModel<T> : ReinforcementLearningModelBase<T>
{
    private readonly DecisionTransformerOptions<T> _options = default!;
    private DecisionTransformerAgent<Tensor<T>, T>? _agent;

    /// <summary>
    /// Gets whether the model uses a continuous action space.
    /// </summary>
    /// <remarks>
    /// Decision Transformer supports both continuous and discrete action spaces.
    /// For stock trading, continuous actions can represent trade sizes or allocations,
    /// while discrete actions can represent distinct trading decisions (buy, sell, hold).
    /// </remarks>
    public override bool IsContinuous => _options.IsContinuous;

    /// <summary>
    /// Initializes a new instance of the <see cref="DecisionTransformerModel{T}"/> class.
    /// </summary>
    /// <param name="options">The options for the Decision Transformer algorithm.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes a new Decision Transformer model with the specified options.
    /// It sets up the model's internal state and creates a Decision Transformer agent with the provided configuration.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is where we set up our Decision Transformer model for predicting stock market actions.
    /// 
    /// The options parameter specifies important settings like:
    /// - How much historical data the model considers (context length)
    /// - How complex the transformer network is (layers, heads, dimensions)
    /// - Whether to target specific returns when making decisions
    /// - How to balance learning from different types of market scenarios
    /// 
    /// These settings greatly impact how well the model adapts to different market conditions.
    /// </para>
    /// </remarks>
    public DecisionTransformerModel(DecisionTransformerOptions<T> options)
        : base(options)
    {
        _options = options;
        
        // Decision Transformer can work with continuous or discrete actions
        IsContinuous = options.IsContinuous;
        
        // Agent is initialized in InitializeAgent method
    }

    /// <summary>
    /// Initializes the Decision Transformer agent that will process sequences and make predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method creates and initializes the Decision Transformer agent with the configured options.
    /// The agent includes the transformer encoder, state/action/return encoders, and other components
    /// specific to the sequence modeling approach.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This creates the "brain" of our Decision Transformer agent, which includes:
    /// - Transformer model that processes sequences of market states, actions, and returns
    /// - Neural networks that encode raw market data into formats the transformer can understand
    /// - Components that determine how to target specific returns
    /// - Memory systems for storing and learning from trading histories
    /// 
    /// All these components work together to help the agent learn patterns from historical market data
    /// and make profitable trading decisions.
    /// </para>
    /// </remarks>
    protected override void InitializeAgent()
    {
        _agent = new DecisionTransformerAgent<Tensor<T>, T>(_options);
    }

    /// <summary>
    /// Gets the Decision Transformer agent that processes sequences and makes predictions.
    /// </summary>
    /// <returns>The Decision Transformer agent as an IAgent interface.</returns>
    /// <remarks>
    /// <para>
    /// This method provides access to the Decision Transformer agent through the common IAgent interface,
    /// allowing for interaction with the agent's core functionality.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This gives you access to the agent object, which handles the actual decision-making
    /// and learning processes. You can use this to directly interact with the agent
    /// if you need more control than the model-level methods provide.
    /// </para>
    /// </remarks>
    protected override IAgent<Tensor<T>, Vector<T>, T> GetAgent()
    {
        if (_agent == null)
        {
            throw new InvalidOperationException("Agent has not been initialized. Call InitializeAgent first.");
        }
        return _agent;
    }

    /// <summary>
    /// Selects an action based on the current state and the target return.
    /// </summary>
    /// <param name="state">The current state observation.</param>
    /// <param name="isTraining">Whether the model is in training mode.</param>
    /// <returns>The selected action as a vector.</returns>
    /// <remarks>
    /// <para>
    /// This method uses the Decision Transformer to predict the next action in a sequence,
    /// conditioned on the desired return. It leverages the transformer's ability to model
    /// sequential data and generate actions that are likely to achieve the target return.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is where the model decides what trading action to take given:
    /// - The current market state
    /// - The history of previous states and actions
    /// - The target return you want to achieve
    /// 
    /// The model looks at this information and predicts the next action in the sequence
    /// that's most likely to lead to your desired return.
    /// </para>
    /// </remarks>
    public override Vector<T> SelectAction(Tensor<T> state, bool isTraining = false)
    {
        if (_agent == null)
        {
            throw new InvalidOperationException("Agent has not been initialized. Call InitializeAgent first.");
        }
        return _agent.SelectAction(state, isTraining || IsTraining);
    }

    /// <summary>
    /// Updates the Decision Transformer agent based on experience with the environment.
    /// </summary>
    /// <param name="state">The current state observation.</param>
    /// <param name="action">The action taken.</param>
    /// <param name="reward">The reward received.</param>
    /// <param name="nextState">The next state observation.</param>
    /// <param name="done">Whether the episode is done.</param>
    /// <returns>The loss value from the update, or zero if no update was performed.</returns>
    /// <remarks>
    /// <para>
    /// This method updates the Decision Transformer by storing the experience in a sequence buffer
    /// and potentially performing a training update if enough experiences have been collected.
    /// Unlike traditional RL methods, the training focuses on predicting actions in sequences
    /// that lead to the target returns, rather than estimating value or policy functions.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is how the model learns from its trading experiences.
    /// 
    /// The method:
    /// 1. Stores the new market state, action, and reward in the model's memory
    /// 2. Decides whether it's time to learn from past experiences
    /// 3. If it's time to learn, the model updates its understanding of which action sequences
    ///    lead to high returns in different market conditions
    /// 4. Returns information about how much the model's understanding improved
    /// 
    /// This learning approach focuses on generating profitable trading sequences rather than
    /// estimating how good each action is in each state.
    /// </para>
    /// </remarks>
    public override T Update(Tensor<T> state, Vector<T> action, T reward, Tensor<T> nextState, bool done)
    {
        if (!IsTraining)
        {
            return NumOps.Zero; // No updates during evaluation
        }
        
        // Add the experience to the agent and potentially update
        if (_agent == null)
        {
            throw new InvalidOperationException("Agent has not been initialized. Call InitializeAgent first.");
        }
        _agent.Learn(state, action, reward, nextState, done);
        
        // Get the latest loss value from the agent
        LastLoss = _agent.GetLatestLoss();
        
        return LastLoss;
    }

    /// <summary>
    /// Trains the Decision Transformer agent on a batch of experiences.
    /// </summary>
    /// <param name="states">The batch of states.</param>
    /// <param name="actions">The batch of actions.</param>
    /// <param name="rewards">The batch of rewards.</param>
    /// <param name="nextStates">The batch of next states.</param>
    /// <param name="dones">The batch of done flags.</param>
    /// <returns>The loss value from the training.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a training update on the Decision Transformer using a batch of experiences.
    /// It creates sequences of states, actions, and returns, and trains the transformer to predict
    /// actions that follow the observed patterns in high-return trajectories.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is the detailed learning process where the model analyzes batches of trading histories.
    /// 
    /// In Decision Transformer, the learning process involves:
    /// 1. Creating sequences of market states, actions taken, and returns received
    /// 2. Training the transformer to predict what actions followed in successful trading sequences
    /// 3. Learning to generate similar action patterns when given similar market conditions
    /// 
    /// This allows the model to learn complex trading strategies from historical data
    /// without having to actively trade during training.
    /// </para>
    /// </remarks>
    protected override T TrainOnBatch(
        Tensor<T> states,
        Tensor<T> actions,
        Vector<T> rewards,
        Tensor<T> nextStates,
        Vector<T> dones)
    {
        // For Decision Transformer, we need to convert tensors to arrays
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
        
        // Use the Decision Transformer agent to train on the batch
        if (_agent == null)
        {
            throw new InvalidOperationException("Agent has not been initialized. Call InitializeAgent first.");
        }
        return _agent.Train(stateArray, actionArray, rewardArray, nextStateArray, doneArray);
    }

    /// <summary>
    /// Trains the model on a dataset of historical trajectories.
    /// </summary>
    /// <param name="trajectories">List of trajectory batches containing historical data.</param>
    /// <returns>A model result containing information about the training process.</returns>
    /// <remarks>
    /// <para>
    /// This method enables offline training of the Decision Transformer model using
    /// historical trajectory data. This is particularly useful for financial applications
    /// where collecting experiences through direct interaction with the market is impractical or risky.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method allows the model to learn from historical market data without having to actively trade.
    /// 
    /// You can provide:
    /// - Historical market states (price patterns, indicators, etc.)
    /// - Actions that were taken (buy, sell, hold decisions)
    /// - The returns that resulted from those actions
    /// 
    /// The model learns which sequences of actions in which market conditions led to high returns,
    /// without having to risk real money during the learning process.
    /// </para>
    /// </remarks>
    public ModelResult<T, Tensor<T>, Tensor<T>> TrainOffline(List<TrajectoryBatch<Tensor<T>, Vector<T>, T>> trajectories)
    {
        if (!_options.OfflineTraining)
        {
            throw new InvalidOperationException("Offline training is disabled in the options. Set OfflineTraining to true to use this method.");
        }

        if (trajectories == null || trajectories.Count == 0)
        {
            return new ModelResult<T, Tensor<T>, Tensor<T>>
            {
                Solution = this,
                Fitness = NumOps.Zero,
                EvaluationData = new ModelEvaluationData<T, Tensor<T>, Tensor<T>>(),
                SelectedFeatures = []
            };
        }

        // Track start time
        var startTime = DateTime.Now;
        
        // Start training mode
        IsTraining = true;
        
        // Pass the trajectories to the agent for offline training
        if (_agent == null)
        {
            throw new InvalidOperationException("Agent has not been initialized. Call InitializeAgent first.");
        }
        LastLoss = _agent.TrainOffline(trajectories);
        
        // Calculate elapsed time
        var elapsed = DateTime.Now - startTime;
        
        // Return result - store loss as fitness value
        return new ModelResult<T, Tensor<T>, Tensor<T>>
        {
            Solution = this,
            Fitness = LastLoss,
            EvaluationData = new ModelEvaluationData<T, Tensor<T>, Tensor<T>>(),
            SelectedFeatures = []
        };
    }

    /// <summary>
    /// Gets all parameters of the Decision Transformer model as a single vector.
    /// </summary>
    /// <returns>A vector containing all model parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method extracts all parameters from the transformer networks and combines them into a single vector.
    /// This is useful for serialization and parameter management.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method gathers all the learned knowledge from the transformer and other neural networks
    /// into a single list of numbers. These numbers represent the model's understanding of
    /// market patterns and effective trading strategies.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        if (_agent == null)
        {
            throw new InvalidOperationException("Agent has not been initialized. Call InitializeAgent first.");
        }
        return _agent.GetParameters();
    }

    /// <summary>
    /// Sets all parameters of the Decision Transformer model from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all model parameters.</param>
    /// <remarks>
    /// <para>
    /// This method distributes the parameters from a single vector to the transformer networks.
    /// This is useful when loading a serialized model or after parameter optimization.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method takes a list of numbers representing the model's trading knowledge
    /// and loads them into the transformer and other neural networks. It's like
    /// importing a saved strategy into the trading model, allowing it to
    /// continue using what it had previously learned.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        if (_agent == null)
        {
            throw new InvalidOperationException("Agent has not been initialized. Call InitializeAgent first.");
        }
        _agent.SetParameters(parameters);
    }

    /// <summary>
    /// Saves the Decision Transformer model to a stream.
    /// </summary>
    /// <param name="stream">The stream to save the model to.</param>
    /// <remarks>
    /// <para>
    /// This method serializes the Decision Transformer model's parameters and configuration
    /// to the provided stream. It saves the transformer architecture, encoder/decoder networks,
    /// and relevant optimizer states and configuration options.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This saves the trading model's "brain" to a file stream. The saved data includes:
    /// - The weights of all neural networks that process market data
    /// - The transformer parameters that understand market patterns over time
    /// - Configuration settings like context length and target return strategy
    /// 
    /// This allows you to save a trained trading model and restore it later without
    /// having to train it again from scratch.
    /// </para>
    /// </remarks>
    public override void Save(Stream stream)
    {
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, true))
        {
            // Write model type identifier
            writer.Write("DecisionTransformerModel");
            
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
            
            // Decision Transformer specific options
            writer.Write(_options.ContextLength);
            writer.Write(_options.NumTransformerLayers);
            writer.Write(_options.NumHeads);
            writer.Write(_options.EmbeddingDim);
            writer.Write(_options.ReturnConditioned);
            writer.Write(_options.OfflineTraining);
            writer.Write(_options.TransformerLearningRate);
            writer.Write(_options.DropoutRate);
            writer.Write((int)_options.PositionalEncodingType);
            writer.Write(_options.NormalizeReturns);
            writer.Write(_options.MaxTrajectoryLength);
        }
    }

    /// <summary>
    /// Loads the Decision Transformer model from a stream.
    /// </summary>
    /// <param name="stream">The stream to load the model from.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes the Decision Transformer model's parameters and configuration
    /// from the provided stream. It restores the transformer architecture, encoder/decoder networks,
    /// and relevant optimizer states and configuration options.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This loads a previously saved trading model's "brain" from a file stream. It restores:
    /// - All the neural network weights that determine trading decisions
    /// - The transformer parameters that understand market patterns
    /// - Configuration settings that control how the model operates
    /// 
    /// After loading, the trading model can continue operating with all the knowledge
    /// it had when it was saved, without having to relearn market patterns.
    /// </para>
    /// </remarks>
    public override void Load(Stream stream)
    {
        using (var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, true))
        {
            // Read and verify model type identifier
            string modelType = reader.ReadString();
            if (modelType != "DecisionTransformerModel")
            {
                throw new InvalidOperationException($"Expected DecisionTransformerModel, but got {modelType}");
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
            
            // Read Decision Transformer specific options
            int contextLength = reader.ReadInt32();
            int numTransformerLayers = reader.ReadInt32();
            int numHeads = reader.ReadInt32();
            int embeddingDim = reader.ReadInt32();
            bool returnConditioned = reader.ReadBoolean();
            bool offlineTraining = reader.ReadBoolean();
            double transformerLearningRate = reader.ReadDouble();
            double dropoutRate = reader.ReadDouble();
            int positionalEncodingType = reader.ReadInt32();
            bool normalizeReturns = reader.ReadBoolean();
            int maxTrajectoryLength = reader.ReadInt32();
            
            // You can optionally update the options if needed
            // _options.ContextLength = contextLength;
            // _options.NumTransformerLayers = numTransformerLayers;
            // etc.
        }
    }
    
    /// <summary>
    /// Creates a new instance of the model.
    /// </summary>
    /// <returns>A new instance of the model.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the Decision Transformer model with the same options
    /// as the current instance. This is useful for creating copies of the model
    /// for parallel training or comparison.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This creates a fresh copy of the trading model with the same settings but without
    /// any learned knowledge. It's like creating a new trader who follows
    /// the same strategy framework but starts with no market experience.
    /// </para>
    /// </remarks>
    public override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new DecisionTransformerModel<T>(_options);
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
        
        metadata.ModelType = ModelType.DecisionTransformerModel;
        metadata.Description = "A transformer-based reinforcement learning model that generates optimal action sequences";
        metadata.FeatureCount = _options.StateSize;
        metadata.Complexity = _options.NumTransformerLayers * _options.NumHeads * _options.EmbeddingDim; // Complexity based on transformer architecture
        metadata.AdditionalInfo = new Dictionary<string, object>
        {
            { "Algorithm", "DecisionTransformer" },
            { "StateSize", _options.StateSize },
            { "ActionSize", _options.ActionSize },
            { "SequenceLength", _options.ContextLength },
            { "EmbeddingDimension", _options.EmbeddingDim },
            { "NumTransformerLayers", _options.NumTransformerLayers },
            { "NumHeads", _options.NumHeads },
            { "PositionalEncodingType", _options.PositionalEncodingType }
        };

        return metadata;
    }
    
}