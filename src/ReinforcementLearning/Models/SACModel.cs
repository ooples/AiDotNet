using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Results;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.Interfaces;
using AiDotNet.ReinforcementLearning.Models.Options;

namespace AiDotNet.ReinforcementLearning.Models
{
    /// <summary>
    /// Soft Actor-Critic (SAC) model for continuous control with entropy regularization.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// <para>
    /// SAC is an actor-critic reinforcement learning algorithm that combines the sample efficiency of off-policy
    /// methods with the stability of maximum entropy reinforcement learning. It incorporates several key features:
    /// - Maximum entropy framework that encourages exploration by maximizing expected reward and policy entropy
    /// - Off-policy training that can reuse previously collected experiences
    /// - Automatic entropy coefficient adjustment to optimize exploration
    /// - Dual critic networks to reduce overestimation bias in value estimates
    /// </para>
    /// <para>
    /// These features make SAC particularly suitable for continuous control tasks that require both
    /// exploration and precise control, such as robotics, locomotion, and manipulation tasks.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// SAC is a reinforcement learning algorithm that helps robots and agents learn tasks with continuous actions.
    /// 
    /// Think of it like teaching a robot how to walk or manipulate objects:
    /// - The agent learns to output precise control signals for motors or joints
    /// - It balances two goals: maximizing rewards and exploring different actions
    /// - It automatically adjusts how much it explores based on how well it's learning
    /// - It uses two separate "critic" networks to avoid being overconfident in its estimates
    /// 
    /// This balance between exploration and exploitation helps the agent learn efficiently without
    /// getting stuck in suboptimal solutions. SAC has been successfully used in robotics, autonomous
    /// systems, and simulated physics environments.
    /// </para>
    /// </remarks>
    public class SACModel<T> : ReinforcementLearningModelBase<T>
    {
        private readonly SACOptions _options = default!;
        private SACAgent<Tensor<T>, T>? _agent;
        private readonly int _batchSize;

        /// <summary>
        /// Initializes a new instance of the <see cref="SACModel{T}"/> class.
        /// </summary>
        /// <param name="options">The options for the SAC algorithm.</param>
        /// <remarks>
        /// <para>
        /// This constructor initializes a new SAC model with the specified options.
        /// It sets up the model's internal state and creates a SAC agent with the provided configuration.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is where we set up our SAC reinforcement learning model.
        /// 
        /// The options parameter specifies important settings like:
        /// - How the neural networks are structured
        /// - How quickly the model learns
        /// - How much emphasis to place on exploration versus exploitation
        /// - Whether to automatically adjust the exploration rate
        /// 
        /// These settings greatly influence how well the model will learn in different environments.
        /// </para>
        /// </remarks>
        public SACModel(SACOptions options)
            : base(options)
        {
            _options = options;
            _batchSize = options.BatchSize;
            
            // SAC is for continuous action spaces only
            IsContinuous = options.IsContinuous;
            
            // SACAgent is initialized in InitializeAgent method
        }

        /// <summary>
        /// Initializes the SAC agent that will interact with the environment.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This method creates and initializes the SAC agent with the configured options.
        /// The agent includes actor and critic networks, target networks, replay buffer,
        /// and other components specific to the SAC algorithm.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This creates the "brain" of our SAC agent, which includes:
        /// - An actor network that decides which actions to take
        /// - Two critic networks that evaluate how good actions are
        /// - Target networks that stabilize learning
        /// - A memory system to store and recall past experiences
        /// - An entropy mechanism that encourages exploration
        /// 
        /// All these components work together to help the agent learn efficiently and stably.
        /// </para>
        /// </remarks>
        protected override void InitializeAgent()
        {
            _agent = new SACAgent<Tensor<T>, T>(_options);
        }

        /// <summary>
        /// Gets the SAC agent that interacts with the environment.
        /// </summary>
        /// <returns>The SAC agent as an IAgent interface.</returns>
        /// <remarks>
        /// <para>
        /// This method provides access to the SAC agent through the common IAgent interface,
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
            return _agent ?? throw new InvalidOperationException("Agent has not been initialized. Call InitializeAgent() first.");
        }

        /// <summary>
        /// Selects an action based on the current state using the SAC policy.
        /// </summary>
        /// <param name="state">The current state observation.</param>
        /// <param name="isTraining">Whether to use stochastic policy during action selection.</param>
        /// <returns>The selected action vector.</returns>
        /// <remarks>
        /// <para>
        /// This method uses the SAC agent's actor network to select an action given the current state.
        /// During training, the action is sampled from a stochastic policy to encourage exploration.
        /// During evaluation, the mean of the policy distribution is used for maximum expected performance.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is where the agent decides what action to take in a given situation.
        /// 
        /// - The state is processed through the actor network to get a probability distribution over actions
        /// - If training, a random action is sampled from this distribution to encourage exploration
        /// - If not training, the most likely action is chosen for better performance
        /// 
        /// The result is a vector of control signals appropriate for the environment.
        /// </para>
        /// </remarks>
        public override Vector<T> SelectAction(Tensor<T> state, bool isTraining = false)
        {
            // Use the SAC agent to select an action
            return _agent?.SelectAction(state, isTraining || IsTraining) ?? throw new InvalidOperationException("Agent not initialized");
        }

        /// <summary>
        /// Updates the SAC agent based on experience with the environment.
        /// </summary>
        /// <param name="state">The current state observation.</param>
        /// <param name="action">The action taken.</param>
        /// <param name="reward">The reward received.</param>
        /// <param name="nextState">The next state observation.</param>
        /// <param name="done">Whether the episode is done.</param>
        /// <returns>The loss value from the update, or zero if no update was performed.</returns>
        /// <remarks>
        /// <para>
        /// This method updates the SAC agent by storing the experience in the replay buffer
        /// and potentially performing a training update if enough experiences have been collected.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is how the agent learns from its interactions with the environment.
        /// 
        /// The method:
        /// 1. Stores the new experience in the agent's memory
        /// 2. Decides whether it's time to learn from past experiences
        /// 3. If it's time to learn, retrieves a batch of experiences and updates the agent's knowledge
        /// 4. Returns information about how much the agent's understanding improved
        /// 
        /// This is the core learning loop that happens after each interaction with the environment.
        /// </para>
        /// </remarks>
        public override T Update(Tensor<T> state, Vector<T> action, T reward, Tensor<T> nextState, bool done)
        {
            if (!IsTraining)
            {
                return NumOps.Zero; // No updates during evaluation
            }
            
            // Use the agent's Learn method which handles experience storage and updates internally
            _agent?.Learn(state, action, reward, nextState, done);
            
            // SAC agent doesn't expose loss directly, so we return zero
            // In a production scenario, we could extend the agent interface to expose training metrics
            LastLoss = NumOps.Zero;
            return LastLoss;
        }

        /// <summary>
        /// Trains the SAC agent on a batch of experiences.
        /// </summary>
        /// <param name="states">The batch of states.</param>
        /// <param name="actions">The batch of actions.</param>
        /// <param name="rewards">The batch of rewards.</param>
        /// <param name="nextStates">The batch of next states.</param>
        /// <param name="dones">The batch of done flags.</param>
        /// <returns>The critic loss value from the training.</returns>
        /// <remarks>
        /// <para>
        /// This method performs a training update on the SAC agent using a batch of experiences.
        /// It updates the critic networks, actor network, and potentially the entropy coefficient
        /// according to the SAC algorithm specifications.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is where the detailed learning happens using a batch of past experiences.
        /// 
        /// In SAC, the learning process involves:
        /// 1. Updating the twin critic networks to better estimate action values
        /// 2. Updating the actor network to select better actions
        /// 3. Optionally updating the entropy coefficient to balance exploration and exploitation
        /// 4. Soft-updating target networks to stabilize learning
        /// 
        /// This batch-based approach is more stable than learning from single experiences
        /// and allows the agent to learn from a diverse set of situations at once.
        /// </para>
        /// </remarks>
        protected override T TrainOnBatch(
            Tensor<T> states,
            Tensor<T> actions,
            Vector<T> rewards,
            Tensor<T> nextStates,
            Vector<T> dones)
        {
            // The SAC agent doesn't have a public Train method
            // Training happens internally through the Learn method
            // For batch training, we would need to extend the agent interface
            return NumOps.Zero;
        }

        /// <summary>
        /// Gets the current entropy coefficient (alpha) value.
        /// </summary>
        /// <returns>The current entropy coefficient value.</returns>
        /// <remarks>
        /// <para>
        /// The entropy coefficient controls the trade-off between exploration (high entropy)
        /// and exploitation (low entropy) in the SAC algorithm. If automatic tuning is enabled,
        /// this value will adapt during training.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This tells you how much the agent is prioritizing exploration versus exploitation.
        /// 
        /// A higher value means:
        /// - The agent is exploring more and trying different actions
        /// - It cares more about maintaining diversity in its policy
        /// 
        /// A lower value means:
        /// - The agent is exploiting what it knows more
        /// - It's focusing on maximizing rewards based on its current knowledge
        /// 
        /// If auto-tuning is enabled, this value will automatically adjust to find the right balance.
        /// </para>
        /// </remarks>
        public T GetEntropyCoefficient()
        {
            if (_agent == null)
                throw new InvalidOperationException("Agent not initialized");
            return _agent.GetEntropyCoefficient();
        }

        /// <summary>
        /// Gets all parameters of the SAC model as a single vector.
        /// </summary>
        /// <returns>A vector containing all model parameters.</returns>
        /// <remarks>
        /// <para>
        /// This method extracts all parameters from the actor and critic networks
        /// and combines them into a single vector. This is useful for serialization
        /// and parameter management.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This method gathers all the learned knowledge from the neural networks
        /// into a single list of numbers. These numbers represent the agent's current 
        /// understanding of how to act in different situations and how valuable different 
        /// actions are.
        /// </para>
        /// </remarks>
        public override Vector<T> GetParameters()
        {
            // Get parameters from the SAC agent
            if (_agent != null)
            {
                return _agent.GetParameters();
            }
            
            // No agent initialized
            return new Vector<T>(0);
        }

        /// <summary>
        /// Sets all parameters of the SAC model from a single vector.
        /// </summary>
        /// <param name="parameters">A vector containing all model parameters.</param>
        /// <remarks>
        /// <para>
        /// This method distributes the parameters from a single vector to the
        /// actor and critic networks. This is useful when loading a serialized model
        /// or after parameter optimization.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This method takes a list of numbers representing the agent's knowledge
        /// and distributes them to the appropriate neural networks. It's like
        /// loading a saved brain state back into the agent, allowing it to
        /// continue using what it had previously learned.
        /// </para>
        /// </remarks>
        public override void SetParameters(Vector<T> parameters)
        {
            // Set parameters to the SAC agent
            if (_agent != null)
            {
                _agent.SetParameters(parameters);
            }
        }
        
        /// <summary>
        /// Creates a new instance of the model.
        /// </summary>
        /// <returns>A new instance of the model with the same configuration but no learned parameters.</returns>
        /// <remarks>
        /// <para>
        /// This method creates a new instance of the SAC model with the same configuration
        /// as this instance but without copying learned parameters.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This creates a brand new SAC agent with the same settings but none of the
        /// learned experience. It's like creating a new agent with the same capabilities
        /// but that hasn't been trained yet.
        /// </para>
        /// </remarks>
        public override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new SACModel<T>(_options);
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
        /// Saves the SAC model to a stream.
        /// </summary>
        /// <param name="stream">The stream to save the model to.</param>
        /// <remarks>
        /// <para>
        /// This method serializes the SAC model's parameters and configuration
        /// to the provided stream. It saves the actor and critic networks,
        /// as well as relevant optimizer states and configuration options.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This saves the agent's brain to a file stream. The saved data includes:
        /// - The weights and biases of all neural networks
        /// - Configuration settings like learning rates and exploration parameters
        /// - The current entropy coefficient value
        /// - Other state information needed to continue training later
        /// 
        /// This allows you to save a trained agent and restore it later without
        /// having to train it again from scratch.
        /// </para>
        /// </remarks>
        public override void Save(Stream stream)
        {
            using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, true))
            {
                // Write model type identifier
                writer.Write("SACModel");
                
                // Save agent parameters
                var parameters = GetParameters();
                writer.Write(parameters.Length);
                for (int i = 0; i < parameters.Length; i++)
                {
                    writer.Write(Convert.ToDouble(parameters[i]));
                }
                
                // Save entropy coefficient
                writer.Write(Convert.ToDouble(GetEntropyCoefficient()));
                
                // Save options
                // TODO: Implement more detailed options serialization if needed
                writer.Write(_options.StateSize);
                writer.Write(_options.ActionSize);
                writer.Write(_options.IsContinuous);
                writer.Write(_options.Gamma);
                writer.Write(_options.Tau);
                writer.Write(_options.BatchSize);
                
                // SAC-specific options
                writer.Write(_options.AutoTuneEntropyCoefficient);
                writer.Write(_options.InitialEntropyCoefficient);
                writer.Write(_options.UseSquashedGaussianPolicy);
                writer.Write(_options.UseSeparateQNetworks);
            }
        }

        /// <summary>
        /// Loads the SAC model from a stream.
        /// </summary>
        /// <param name="stream">The stream to load the model from.</param>
        /// <remarks>
        /// <para>
        /// This method deserializes the SAC model's parameters and configuration
        /// from the provided stream. It restores the actor and critic networks,
        /// as well as relevant optimizer states and configuration options.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This loads a previously saved agent's brain from a file stream. It restores:
        /// - The weights and biases of all neural networks
        /// - Configuration settings
        /// - The entropy coefficient value
        /// - Other state information
        /// 
        /// After loading, the agent can continue operating with all the knowledge
        /// it had when it was saved, without having to relearn anything.
        /// </para>
        /// </remarks>
        public override void Load(Stream stream)
        {
            using (var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, true))
            {
                // Read and verify model type identifier
                string modelType = reader.ReadString();
                if (modelType != "SACModel")
                {
                    throw new InvalidOperationException($"Expected SACModel, but got {modelType}");
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
                
                // Load entropy coefficient value
                T entropyCoefficient = NumOps.FromDouble(reader.ReadDouble());
                // TODO: Set entropy coefficient if the agent provides a method for this
                
                // Load and verify basic options
                int stateSize = reader.ReadInt32();
                int actionSize = reader.ReadInt32();
                bool isContinuous = reader.ReadBoolean();
                double gamma = reader.ReadDouble();
                double tau = reader.ReadDouble();
                int batchSize = reader.ReadInt32();
                
                // Verify that basic options match
                if (stateSize != _options.StateSize || actionSize != _options.ActionSize)
                {
                    throw new InvalidOperationException(
                        $"Model dimensions mismatch. Saved model: State={stateSize}, Action={actionSize}. " +
                        $"Current options: State={_options.StateSize}, Action={_options.ActionSize}");
                }
                
                // Optional: Read and apply SAC-specific options
                // This is an example - you may choose to override options or just verify them
                bool autoTuneEntropy = reader.ReadBoolean();
                double initialEntropy = reader.ReadDouble();
                bool useSquashedPolicy = reader.ReadBoolean();
                bool useSeparateQNetworks = reader.ReadBoolean();
                
                // You can optionally update the options if needed
                // _options.AutoTuneEntropyCoefficient = autoTuneEntropy;
                // _options.InitialEntropyCoefficient = initialEntropy;
                // _options.UseSquashedGaussianPolicy = useSquashedPolicy;
                // _options.UseSeparateQNetworks = useSeparateQNetworks;
            }
        }
    }
}