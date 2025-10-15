using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.Interfaces;
using AiDotNet.ReinforcementLearning.Models.Options;
using AiDotNet.Helpers;
using System;
using System.IO;

namespace AiDotNet.ReinforcementLearning.Models
{
    /// <summary>
    /// Deep Q-Network (DQN) model for discrete action spaces.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// <para>
    /// DQN is a reinforcement learning algorithm that learns to approximate Q-values 
    /// (action values) using a deep neural network. It incorporates several key features:
    /// - Experience replay to break correlations between consecutive samples
    /// - Target networks to stabilize learning
    /// - Optional extensions like Double DQN, Dueling networks, and Prioritized Experience Replay
    /// </para>
    /// <para>
    /// DQN is particularly suitable for discrete action spaces where the number of possible actions
    /// is finite, such as game playing, robotics with discrete controls, and other decision-making tasks.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// DQN is a reinforcement learning algorithm that helps agents learn to make decisions.
    /// 
    /// Think of it like teaching a computer to play a video game:
    /// - The agent observes the game state (screen pixels, game score, etc.)
    /// - It learns which actions (buttons to press) lead to the highest scores
    /// - It stores its experiences and learns from them repeatedly
    /// - It uses special techniques to make the learning process more stable
    /// 
    /// DQN works best when there are a limited number of possible actions to take,
    /// like the buttons on a game controller or discrete choices in a decision problem.
    /// </para>
    /// </remarks>
    public class DQNModel<T> : ReinforcementLearningModelBase<T>
    {
        private readonly DQNOptions _options = default!;
        private DQNAgent<Tensor<T>, T> _agent = null!;
        private readonly int _batchSize;

        /// <summary>
        /// Initializes a new instance of the <see cref="DQNModel{T}"/> class.
        /// </summary>
        /// <param name="options">The options for the DQN algorithm.</param>
        /// <remarks>
        /// <para>
        /// This constructor initializes a new DQN model with the specified options.
        /// It sets up the model's internal state and creates a DQN agent with the provided configuration.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is where we set up our DQN reinforcement learning model.
        /// 
        /// The options parameter specifies important settings like:
        /// - How the neural network is structured
        /// - How quickly the model learns
        /// - How the agent balances exploration versus exploitation
        /// - Which advanced techniques to use (Double DQN, Dueling networks, etc.)
        /// 
        /// These settings greatly influence how well the model will learn in different environments.
        /// </para>
        /// </remarks>
        public DQNModel(DQNOptions options)
            : base(options)
        {
            _options = options;
            _batchSize = options.BatchSize;
            
            // DQN is for discrete action spaces only
            IsContinuous = options.IsContinuous;
            
            // DQNAgent is initialized in InitializeAgent method
        }

        /// <summary>
        /// Initializes the DQN agent that will interact with the environment.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This method creates and initializes the DQN agent with the configured options.
        /// The agent includes Q-network, target network, replay buffer, and other components
        /// specific to the DQN algorithm.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This creates the "brain" of our DQN agent, which includes:
        /// - A Q-network that estimates the value of different actions
        /// - A target network that helps stabilize learning
        /// - A memory system to store and recall past experiences
        /// - Exploration strategies to balance trying new actions and exploiting known good ones
        /// 
        /// All these components work together to help the agent learn efficiently and stably.
        /// </para>
        /// </remarks>
        protected override void InitializeAgent()
        {
            _agent = new DQNAgent<Tensor<T>, T>(_options);
        }

        /// <summary>
        /// Gets the DQN agent that interacts with the environment.
        /// </summary>
        /// <returns>The DQN agent as an IAgent interface.</returns>
        /// <remarks>
        /// <para>
        /// This method provides access to the DQN agent through the common IAgent interface,
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
            throw new NotSupportedException("DQNAgent uses int actions, not Vector<T>. Use SelectAction and Update methods instead.");
        }

        /// <summary>
        /// Selects an action based on the current state using the DQN policy.
        /// </summary>
        /// <param name="state">The current state observation.</param>
        /// <param name="isTraining">Whether to use exploration during action selection.</param>
        /// <returns>The selected action as a vector (containing a single value for discrete actions).</returns>
        /// <remarks>
        /// <para>
        /// This method uses the DQN agent's Q-network to select an action given the current state.
        /// During training, an exploration strategy (like epsilon-greedy) is used to balance
        /// exploration and exploitation. During evaluation, the best action is selected greedily.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is where the agent decides what action to take in a given situation.
        /// 
        /// - The state is processed through the Q-network to get values for each possible action
        /// - If training, sometimes a random action is chosen to explore
        /// - If not training, the action with the highest estimated value is chosen
        /// 
        /// The result is an action appropriate for the environment.
        /// </para>
        /// </remarks>
        public override Vector<T> SelectAction(Tensor<T> state, bool isTraining = false)
        {
            // For DQN, we need to convert the int action to a Vector<T>
            int action = _agent.SelectAction(state, isTraining || IsTraining);
            var result = new Vector<T>(1);
            result[0] = NumOps.FromDouble(action);
            return result;
        }

        /// <summary>
        /// Updates the DQN agent based on experience with the environment.
        /// </summary>
        /// <param name="state">The current state observation.</param>
        /// <param name="action">The action taken (as a Vector<double> containing a single value).</param>
        /// <param name="reward">The reward received.</param>
        /// <param name="nextState">The next state observation.</param>
        /// <param name="done">Whether the episode is done.</param>
        /// <returns>The loss value from the update, or zero if no update was performed.</returns>
        /// <remarks>
        /// <para>
        /// This method updates the DQN agent by storing the experience in the replay buffer
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
            
            // For DQN, we need to convert the Vector<T> action to an int
            int discreteAction = Convert.ToInt32(Convert.ToDouble(action[0]));
            
            // Add the experience to the replay buffer and potentially update
            _agent.Learn(state, discreteAction, reward, nextState, done);
            
            // Get the latest loss value from the agent if available
            LastLoss = _agent.GetLatestLoss();
            
            return LastLoss;
        }

        /// <summary>
        /// Trains the DQN agent on a batch of experiences.
        /// </summary>
        /// <param name="states">The batch of states.</param>
        /// <param name="actions">The batch of actions.</param>
        /// <param name="rewards">The batch of rewards.</param>
        /// <param name="nextStates">The batch of next states.</param>
        /// <param name="dones">The batch of done flags.</param>
        /// <returns>The loss value from the training.</returns>
        /// <remarks>
        /// <para>
        /// This method performs a training update on the DQN agent using a batch of experiences.
        /// It updates the Q-network based on the Bellman equation and TD learning.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is where the detailed learning happens using a batch of past experiences.
        /// 
        /// In DQN, the learning process involves:
        /// 1. Calculating target Q-values using the current reward and future estimated values
        /// 2. Updating the Q-network to better predict these target values
        /// 3. Occasionally updating the target network to stabilize learning
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
            // For DQN, we need to convert the tensor of actions to an array of ints
            int[] discreteActions = new int[actions.Shape[0]];
            for (int i = 0; i < actions.Shape[0]; i++)
            {
                discreteActions[i] = Convert.ToInt32(Convert.ToDouble(actions[i, 0]));
            }
            
            // Use the DQN agent to train on the batch
            // Convert dones to bool array
            var donesArray = new bool[dones.Length];
            for (int i = 0; i < dones.Length; i++)
            {
                donesArray[i] = !NumOps.Equals(dones[i], NumOps.Zero);
            }
            return _agent.Train(states, discreteActions, rewards, nextStates, donesArray);
        }

        /// <summary>
        /// Gets all parameters of the DQN model as a single vector.
        /// </summary>
        /// <returns>A vector containing all model parameters.</returns>
        /// <remarks>
        /// <para>
        /// This method extracts all parameters from the Q-network and combines them into a single vector.
        /// This is useful for serialization and parameter management.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This method gathers all the learned knowledge from the neural networks
        /// into a single list of numbers. These numbers represent the agent's current 
        /// understanding of how valuable different actions are in different situations.
        /// </para>
        /// </remarks>
        public override Vector<T> GetParameters()
        {
            return _agent.GetParameters();
        }

        /// <summary>
        /// Sets all parameters of the DQN model from a single vector.
        /// </summary>
        /// <param name="parameters">A vector containing all model parameters.</param>
        /// <remarks>
        /// <para>
        /// This method distributes the parameters from a single vector to the Q-network.
        /// This is useful when loading a serialized model or after parameter optimization.
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
            _agent.SetParameters(parameters);
        }

        /// <summary>
        /// Saves the DQN model to a stream.
        /// </summary>
        /// <param name="stream">The stream to save the model to.</param>
        /// <remarks>
        /// <para>
        /// This method serializes the DQN model's parameters and configuration
        /// to the provided stream. It saves the Q-network and target network,
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
                writer.Write("DQNModel");
                
                // Save agent parameters
                var parameters = GetParameters();
                writer.Write(parameters.Length);
                for (int i = 0; i < parameters.Length; i++)
                {
                    writer.Write(Convert.ToDouble(parameters[i]));
                }
                
                // Save options
                writer.Write(_options.StateSize);
                writer.Write(_options.ActionSize);
                writer.Write(_options.Gamma);
                writer.Write(_options.BatchSize);
                
                // DQN-specific options
                writer.Write(_options.UseDoubleDQN);
                writer.Write(_options.UseDuelingDQN);
                writer.Write(_options.UsePrioritizedReplay);
                writer.Write(_options.EpsilonStart);
                writer.Write(_options.EpsilonEnd);
                writer.Write(_options.EpsilonDecay);
            }
        }

        /// <summary>
        /// Loads the DQN model from a stream.
        /// </summary>
        /// <param name="stream">The stream to load the model from.</param>
        /// <remarks>
        /// <para>
        /// This method deserializes the DQN model's parameters and configuration
        /// from the provided stream. It restores the Q-network and target network,
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
                if (modelType != "DQNModel")
                {
                    throw new InvalidOperationException($"Expected DQNModel, but got {modelType}");
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
                double gamma = reader.ReadDouble();
                int batchSize = reader.ReadInt32();
                
                // Verify that basic options match
                if (stateSize != _options.StateSize || actionSize != _options.ActionSize)
                {
                    throw new InvalidOperationException(
                        $"Model dimensions mismatch. Saved model: State={stateSize}, Action={actionSize}. " +
                        $"Current options: State={_options.StateSize}, Action={_options.ActionSize}");
                }
                
                // Read DQN-specific options
                bool useDoubleDQN = reader.ReadBoolean();
                bool useDuelingDQN = reader.ReadBoolean();
                bool usePrioritizedReplay = reader.ReadBoolean();
                double epsilonStart = reader.ReadDouble();
                double epsilonEnd = reader.ReadDouble();
                double epsilonDecay = reader.ReadDouble();
                
                // You can optionally update the options if needed
                // _options.UseDoubleDQN = useDoubleDQN;
                // _options.UseDuelingDQN = useDuelingDQN;
                // _options.UsePrioritizedReplay = usePrioritizedReplay;
            }
        }
        
        /// <summary>
        /// Creates a new instance of the model.
        /// </summary>
        /// <returns>A new instance of the model.</returns>
        /// <remarks>
        /// <para>
        /// This method creates a new instance of the DQN model with the same options
        /// as the current instance. This is useful for creating copies of the model
        /// for parallel training or comparison.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This creates a fresh copy of the model with the same settings but without
        /// any learned knowledge. It's like creating a new student who will follow
        /// the same curriculum but start from scratch.
        /// </para>
        /// </remarks>
        public override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new DQNModel<T>(_options);
        }
    }
}