using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Results;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.Interfaces;
using AiDotNet.ReinforcementLearning.Models.Options;

namespace AiDotNet.ReinforcementLearning.Models
{
    /// <summary>
    /// Twin Delayed Deep Deterministic Policy Gradient (TD3) model for continuous control.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// <para>
    /// TD3 is an actor-critic reinforcement learning algorithm designed for continuous action spaces.
    /// It extends the Deep Deterministic Policy Gradient (DDPG) algorithm with several stabilizing techniques,
    /// including:
    /// - Twin critic networks to reduce overestimation bias
    /// - Delayed policy updates to reduce variance
    /// - Target policy smoothing to further reduce variance and improve stability
    /// </para>
    /// <para>
    /// These enhancements make TD3 significantly more stable and efficient than its predecessor, DDPG,
    /// leading to better performance in continuous control tasks such as robotics and simulated physics environments.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// TD3 is a reinforcement learning algorithm that helps agents learn to control systems with continuous actions.
    /// 
    /// Think of it like teaching a robot to perform a complex physical task:
    /// - The agent learns to output precise control signals (like joint torques or motor speeds)
    /// - It uses two "critic" networks to avoid being overly optimistic about its actions
    /// - It updates its policy less frequently than it evaluates actions, which improves stability
    /// - It adds small random noise to its target actions to avoid overfitting to specific states
    /// 
    /// These techniques help the agent learn more reliably and efficiently in challenging environments.
    /// TD3 has been successfully applied to robotics, autonomous driving, and other continuous control problems.
    /// </para>
    /// </remarks>
    public class TD3Model<T> : ReinforcementLearningModelBase<T>
    {
        private readonly TD3Options _options = default!;
        private TD3Agent<Tensor<T>, T> _agent = null!;
        private readonly int _batchSize;

        /// <summary>
        /// Initializes a new instance of the <see cref="TD3Model{T}"/> class.
        /// </summary>
        /// <param name="options">The options for the TD3 algorithm.</param>
        /// <remarks>
        /// <para>
        /// This constructor initializes a new TD3 model with the specified options.
        /// It sets up the model's internal state and creates a TD3 agent with the provided configuration.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is where we set up our TD3 reinforcement learning model.
        /// 
        /// The options parameter specifies important settings like:
        /// - How the neural networks are structured
        /// - How quickly the model learns
        /// - How much exploration vs. exploitation to use
        /// - How often to update different parts of the model
        /// 
        /// These settings greatly influence how well the model will learn in different environments.
        /// </para>
        /// </remarks>
        public TD3Model(TD3Options options) 
            : base(options)
        {
            _options = options;
            _batchSize = options.BatchSize;
            
            // TD3 is for continuous action spaces only
            IsContinuous = options.IsContinuous;
            
            // TD3Agent is initialized in InitializeAgent method
        }

        /// <summary>
        /// Initializes the TD3 agent that will interact with the environment.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This method creates and initializes the TD3 agent with the configured options.
        /// The agent includes actor and critic networks, target networks, replay buffer,
        /// and other components specific to the TD3 algorithm.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This creates the "brain" of our TD3 agent, which includes:
        /// - An actor network that decides which actions to take
        /// - Two critic networks that evaluate how good actions are
        /// - Target networks that stabilize learning
        /// - A memory system to store and recall past experiences
        /// 
        /// All these components work together to help the agent learn efficiently and stably.
        /// </para>
        /// </remarks>
        protected override void InitializeAgent()
        {
            _agent = new TD3Agent<Tensor<T>, T>(_options);
        }

        /// <summary>
        /// Gets the TD3 agent that interacts with the environment.
        /// </summary>
        /// <returns>The TD3 agent as an IAgent interface.</returns>
        /// <remarks>
        /// <para>
        /// This method provides access to the TD3 agent through the common IAgent interface,
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
            return _agent;
        }

        /// <summary>
        /// Selects an action based on the current state using the TD3 policy.
        /// </summary>
        /// <param name="state">The current state observation.</param>
        /// <param name="isTraining">Whether to use exploration during action selection.</param>
        /// <returns>The selected action vector.</returns>
        /// <remarks>
        /// <para>
        /// This method uses the TD3 agent's actor network to select an action given the current state.
        /// During training, noise is added to the action for exploration. During evaluation, 
        /// the action is selected deterministically for maximum expected performance.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is where the agent decides what action to take in a given situation.
        /// 
        /// - The state is processed through the actor network to get a base action
        /// - If training, some noise is added to encourage exploration
        /// - If not training, the agent uses its best guess with no randomness
        /// 
        /// The result is a vector of control signals appropriate for the environment.
        /// </para>
        /// </remarks>
        public override Vector<T> SelectAction(Tensor<T> state, bool isTraining = false)
        {
            // Use the TD3 agent to select an action
            return _agent.SelectAction(state, isTraining || IsTraining);
        }

        /// <summary>
        /// Updates the TD3 agent based on experience with the environment.
        /// </summary>
        /// <param name="state">The current state observation.</param>
        /// <param name="action">The action taken.</param>
        /// <param name="reward">The reward received.</param>
        /// <param name="nextState">The next state observation.</param>
        /// <param name="done">Whether the episode is done.</param>
        /// <returns>The loss value from the update, or zero if no update was performed.</returns>
        /// <remarks>
        /// <para>
        /// This method updates the TD3 agent by storing the experience in the replay buffer
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
            _agent.Learn(state, action, reward, nextState, done);
            
            // TD3Agent doesn't expose loss directly, so we return zero
            // In a real implementation, the agent could expose a GetLastLoss() method
            LastLoss = NumOps.Zero;
            return LastLoss;
        }

        /// <summary>
        /// Trains the TD3 agent on a batch of experiences.
        /// </summary>
        /// <param name="states">The batch of states.</param>
        /// <param name="actions">The batch of actions.</param>
        /// <param name="rewards">The batch of rewards.</param>
        /// <param name="nextStates">The batch of next states.</param>
        /// <param name="dones">The batch of done flags.</param>
        /// <returns>The critic loss value from the training.</returns>
        /// <remarks>
        /// <para>
        /// This method performs a training update on the TD3 agent using a batch of experiences.
        /// It updates the critic networks and, less frequently, the actor network according
        /// to the TD3 algorithm specifications.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is where the detailed learning happens using a batch of past experiences.
        /// 
        /// In TD3, the learning process involves:
        /// 1. Updating the twin critic networks to better estimate action values
        /// 2. Occasionally updating the actor network to select better actions
        /// 3. Slowly updating target networks to stabilize learning
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
            // TD3Agent doesn't have a public Train method
            // The training is handled internally through its Update method
            // For now, return zero loss as the agent doesn't expose training metrics directly
            return NumOps.Zero;
        }

        /// <summary>
        /// Gets all parameters of the TD3 model as a single vector.
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
            // Get parameters from the TD3 agent
            if (_agent != null)
            {
                return _agent.GetParameters();
            }
            
            // No agent initialized
            return new Vector<T>(0);
        }

        /// <summary>
        /// Sets all parameters of the TD3 model from a single vector.
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
            // Set parameters to the TD3 agent
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
        /// This method creates a new instance of the TD3 model with the same configuration
        /// as this instance but without copying learned parameters.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This creates a brand new TD3 agent with the same settings but none of the
        /// learned experience. It's like creating a new agent with the same capabilities
        /// but that hasn't been trained yet.
        /// </para>
        /// </remarks>
        public override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new TD3Model<T>(_options);
        }

        /// <summary>
        /// Saves the TD3 model to a stream.
        /// </summary>
        /// <param name="stream">The stream to save the model to.</param>
        /// <remarks>
        /// <para>
        /// This method serializes the TD3 model's parameters and configuration
        /// to the provided stream. It saves the actor and critic networks,
        /// as well as relevant optimizer states and configuration options.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This saves the agent's brain to a file stream. The saved data includes:
        /// - The weights and biases of all neural networks
        /// - Configuration settings like learning rates and exploration parameters
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
                writer.Write("TD3Model");
                
                // Save agent parameters
                var parameters = GetParameters();
                writer.Write(parameters.Length);
                for (int i = 0; i < parameters.Length; i++)
                {
                    writer.Write(Convert.ToDouble(parameters[i]));
                }
                
                // Save options
                // TODO: Implement more detailed options serialization if needed
                writer.Write(_options.StateSize);
                writer.Write(_options.ActionSize);
                writer.Write(_options.IsContinuous);
                writer.Write(_options.Gamma);
                writer.Write(_options.Tau);
                writer.Write(_options.BatchSize);
                
                // TD3-specific options
                writer.Write(_options.PolicyUpdateFrequency);
                writer.Write(_options.TargetPolicyNoiseScale);
                writer.Write(_options.TargetPolicyNoiseClip);
                writer.Write(_options.UseMinimumQValue);
            }
        }

        /// <summary>
        /// Loads the TD3 model from a stream.
        /// </summary>
        /// <param name="stream">The stream to load the model from.</param>
        /// <remarks>
        /// <para>
        /// This method deserializes the TD3 model's parameters and configuration
        /// from the provided stream. It restores the actor and critic networks,
        /// as well as relevant optimizer states and configuration options.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This loads a previously saved agent's brain from a file stream. It restores:
        /// - The weights and biases of all neural networks
        /// - Configuration settings
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
                if (modelType != "TD3Model")
                {
                    throw new InvalidOperationException($"Expected TD3Model, but got {modelType}");
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
                double tau = reader.ReadDouble();
                int batchSize = reader.ReadInt32();
                
                // Verify that basic options match
                if (stateSize != _options.StateSize || actionSize != _options.ActionSize)
                {
                    throw new InvalidOperationException(
                        $"Model dimensions mismatch. Saved model: State={stateSize}, Action={actionSize}. " +
                        $"Current options: State={_options.StateSize}, Action={_options.ActionSize}");
                }
                
                // Optional: Read and apply TD3-specific options
                // This is an example - you may choose to override options or just verify them
                int policyUpdateFreq = reader.ReadInt32();
                double targetNoiseScale = reader.ReadDouble();
                double targetNoiseClip = reader.ReadDouble();
                bool useMinQValue = reader.ReadBoolean();
                
                // You can optionally update the options if needed
                // _options.PolicyUpdateFrequency = policyUpdateFreq;
                // _options.TargetPolicyNoiseScale = targetNoiseScale;
                // _options.TargetPolicyNoiseClip = targetNoiseClip;
                // _options.UseMinimumQValue = useMinQValue;
            }
        }
    }
}