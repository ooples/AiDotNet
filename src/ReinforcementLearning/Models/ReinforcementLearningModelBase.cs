using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.ReinforcementLearning.Interfaces;
using AiDotNet.ReinforcementLearning.Models.Options;
using AiDotNet.Extensions;

namespace AiDotNet.ReinforcementLearning.Models;

/// <summary>
/// Provides a base class for all reinforcement learning models in the library.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This abstract class defines the common interface and functionality that all reinforcement learning models share,
/// including agent interactions, training, prediction, evaluation, and serialization/deserialization capabilities.
/// </para>
/// <para>
/// Reinforcement learning models learn to make sequences of decisions by interacting with an environment
/// and learning from the feedback (rewards) received. The base class provides a framework for implementing
/// various reinforcement learning algorithms like DQN, DDPG, SAC, TD3, PPO, and others.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// A reinforcement learning model helps an agent learn how to make good decisions in an environment.
/// 
/// Think of reinforcement learning like training a pet:
/// - The agent (pet) observes its environment and takes actions
/// - The environment provides feedback through rewards or penalties
/// - The agent learns to take actions that maximize the total reward over time
/// 
/// This base class serves as a blueprint that all specific reinforcement learning models follow.
/// It ensures that every model can:
/// - Interact with environments to collect experiences
/// - Learn from these experiences to improve decision-making
/// - Make predictions about which actions to take in new situations
/// - Be saved to disk and loaded later without retraining
/// 
/// Reinforcement learning models are used in many real-world applications, including:
/// - Game playing (like chess, Go, video games)
/// - Robotics and autonomous vehicles
/// - Resource management and scheduling
/// - Trading and finance
/// - Recommendation systems
/// </para>
/// </remarks>
public abstract class ReinforcementLearningModelBase<T> : IReinforcementLearningModel<T>
{
/// <summary>
/// Configuration options for the reinforcement learning model.
/// </summary>
/// <remarks>
/// <para>
/// These options control the core behavior of the reinforcement learning model, including
/// network architectures, learning rates, exploration parameters, and other algorithm-specific settings.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Think of these options as settings that determine how the model works:
/// - Learning rate: How quickly the model adapts to new information
/// - Discount factor: How much future rewards are valued compared to immediate rewards
/// - Exploration rate: How often the agent tries new actions versus exploiting known good ones
/// - Network architecture: The structure of the neural networks that power the agent's decision-making
/// </para>
/// </remarks>
protected ReinforcementLearningOptions Options { get; private set; }

/// <summary>
/// Provides numeric operations for the specific type T.
/// </summary>
/// <remarks>
/// <para>
/// This property provides mathematical operations appropriate for the generic type T,
/// allowing the algorithm to work consistently with different numeric types like
/// float, double, or decimal.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// This is a helper that knows how to do math (addition, multiplication, etc.) with 
/// your specific number type, whether that's a regular double, a precise decimal value,
/// or something else. It allows the model to work with different types of numbers
/// without changing its core logic.
/// </para>
/// </remarks>
protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();

/// <summary>
/// Gets or sets the last computed loss value during training.
/// </summary>
/// <remarks>
/// <para>
/// This value represents how well the model is currently performing. Lower values typically
/// indicate better performance, though the exact meaning depends on the specific algorithm.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Think of loss as a score that shows how far off the model's predictions are from optimal.
/// A lower loss generally means the model is doing better at its job.
/// </para>
/// </remarks>
protected T LastLoss { get; set; }

/// <summary>
/// Indicates whether the model is currently in training mode.
/// </summary>
/// <remarks>
/// <para>
/// When in training mode, the model might explore more and use different behavior than
/// when it's being used to make actual decisions (evaluation mode).
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// This is like a switch between "practice mode" and "performance mode":
/// - In training mode (true), the model tries new things to learn what works best
/// - In evaluation mode (false), the model uses what it has learned to make the best decisions
/// </para>
/// </remarks>
protected bool IsTraining { get; set; } = true;

/// <summary>
/// A random number generator for exploration and stochastic processes.
/// </summary>
protected Random Random { get; private set; }

/// <summary>
/// Gets whether the model uses a continuous action space.
/// </summary>
/// <remarks>
/// <para>
/// This property indicates whether the agent operates in a continuous action space (true)
/// or a discrete action space (false). This affects how actions are represented and processed.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// This tells you whether the agent's actions are:
/// - Continuous: Actions represented as floating-point values that can vary smoothly 
///   (like steering wheel angles, motor speeds, or joint torques)
/// - Discrete: Actions represented as distinct choices from a fixed set 
///   (like "move left", "move right", or selecting from numbered options)
/// 
/// Continuous action spaces are common in robotics and control tasks, while 
/// discrete action spaces are common in games and decision-making tasks.
/// </para>
/// </remarks>
public virtual bool IsContinuous { get; protected set; }

/// <summary>
/// Initializes a new instance of the <see cref="ReinforcementLearningModelBase{T}"/> class.
/// </summary>
/// <param name="options">The options that configure the reinforcement learning algorithm.</param>
/// <remarks>
/// <para>
/// This constructor initializes the reinforcement learning model with the provided options,
/// sets up the random number generator, and prepares the model for training or inference.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// This is where the reinforcement learning model gets set up with all its initial settings.
/// The options determine things like how quickly it learns, how much it explores, and
/// how it balances immediate versus future rewards.
/// </para>
/// </remarks>
protected ReinforcementLearningModelBase(ReinforcementLearningOptions options)
{
    // Check for null options and create default if needed
    if (options == null)
    {
        options = new ReinforcementLearningOptions();
    }

    Options = options;
    LastLoss = NumOps.Zero;

    // Initialize random number generator with seed if provided
    Random = options.Seed.HasValue ? new Random(options.Seed.Value) : new Random();

    // By default, set to discrete action space
    // Derived classes should override this as needed
    IsContinuous = false;

    InitializeAgent();
}

/// <summary>
/// Initializes the agent that will interact with the environment.
/// </summary>
/// <remarks>
/// <para>
/// This method sets up the reinforcement learning agent with neural networks,
/// memory buffers, and other components specific to the algorithm being implemented.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// This method creates the "brain" of the agent - the parts that allow it to:
/// - Remember past experiences
/// - Learn from those experiences
/// - Make decisions based on what it has learned
/// 
/// Different reinforcement learning algorithms have different ways of organizing
/// these components, which is why this method is abstract and implemented separately
/// for each specific algorithm.
/// </para>
/// </remarks>
protected abstract void InitializeAgent();

/// <summary>
/// Gets the agent that interacts with the environment.
/// </summary>
/// <remarks>
/// <para>
/// This property provides access to the reinforcement learning agent, which handles
/// the core logic of selecting actions, learning from experiences, and improving over time.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// The agent is the "actor" in reinforcement learning - it's the part that:
/// - Observes the environment state
/// - Decides which action to take
/// - Processes the reward feedback
/// - Updates its knowledge based on what it learns
/// 
/// Accessing the agent directly allows for more advanced interactions like
/// updating with specific experiences or controlling the exploration behavior.
/// </para>
/// </remarks>
protected abstract IAgent<Tensor<T>, Vector<T>, T> GetAgent();

/// <summary>
/// Makes predictions for the given inputs (states).
/// </summary>
/// <param name="X">The input features (states).</param>
/// <returns>The predicted values (actions or action values).</returns>
/// <remarks>
/// <para>
/// This method allows the reinforcement learning model to be used within the
/// supervised learning API of the library. It takes state representations and
/// returns action predictions.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// This method lets you ask the agent "what would you do in this situation?"
/// 
/// Given a set of states (situations), it returns the actions the agent would take
/// or values it assigns to different actions. This is useful for evaluating how the
/// agent would behave in specific scenarios without actually executing those actions
/// in an environment.
/// </para>
/// </remarks>
public virtual Vector<T> Predict(Matrix<T> X)
{
    // Create a vector for predictions
    var predictions = new Vector<T>(X.Rows);

    // Make predictions for each input row
    for (int i = 0; i < X.Rows; i++)
    {
        var input = X.GetRow(i);
        var inputTensor = Tensor<T>.FromVector(input);
        
        // Get action from the agent
        var action = SelectAction(inputTensor);
        
        // For simplicity, we return the first dimension of the action
        // Derived classes might implement this differently
        predictions[i] = action[0];
    }

    return predictions;
}

/// <summary>
/// Performs a forward pass to make predictions for tensor inputs.
/// </summary>
/// <param name="input">The input tensor representing a state.</param>
/// <returns>The output tensor representing an action or action values.</returns>
/// <remarks>
/// <para>
/// This method processes a state observation through the agent's policy network
/// to predict the best action to take or the values of different actions.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// This is how the agent decides what to do in a given situation.
/// 
/// The input tensor describes the current state (what the agent can observe),
/// and the output tensor represents the action the agent chooses to take
/// or how valuable it thinks different actions are.
/// </para>
/// </remarks>
public virtual Tensor<T> Predict(Tensor<T> input)
{
    // Check input shape
    if (input.Rank < 1)
    {
        throw new InvalidOperationException("Input tensor must have at least rank 1");
    }

        var batchSize = input.Shape[0];
        
        // If input is a batch, handle each state separately
        if (batchSize > 1)
        {
            var output = new Tensor<T>(new int[] { batchSize, 1 });
            
            for (int i = 0; i < batchSize; i++)
            {
                var singleInput = input.GetSlice(i);
                var singleOutput = SelectAction(singleInput);
                
                // Store the first dimension of the action
                output[i, 0] = singleOutput[0];
            }
            
            return output;
        }
        
        // For a single input, return the action as a tensor
        var action = SelectAction(input);
        return Tensor<T>.FromVector(action);
    }
    
    /// <summary>
    /// Selects an action based on the current state.
    /// </summary>
    /// <param name="state">The current state observation.</param>
    /// <param name="isTraining">Whether to use exploration during action selection.</param>
    /// <returns>The selected action.</returns>
    /// <remarks>
    /// <para>
    /// This method determines the agent's action in a given state. During training,
    /// it may incorporate exploration to try new actions. During evaluation, it typically
    /// selects the best action according to the learned policy.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is where the agent decides what to do in a given situation.
    /// 
    /// During training (learning), the agent sometimes tries random or exploratory actions
    /// to discover new strategies. During evaluation (when actually using what it's learned),
    /// it typically chooses what it believes is the best action.
    /// </para>
    /// </remarks>
    public abstract Vector<T> SelectAction(Tensor<T> state, bool isTraining = false);
    
    /// <summary>
    /// Updates the model based on experience with the environment.
    /// </summary>
    /// <param name="state">The current state observation.</param>
    /// <param name="action">The action taken.</param>
    /// <param name="reward">The reward received.</param>
    /// <param name="nextState">The next state observation.</param>
    /// <param name="done">Whether the episode is done.</param>
    /// <returns>The loss value from the update.</returns>
    /// <remarks>
    /// <para>
    /// This method updates the agent's knowledge based on a single step of interaction
    /// with the environment (a transition). It stores the experience and may perform
    /// a learning update if enough experiences have been collected.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is how the agent learns from its experiences.
    /// 
    /// When the agent takes an action and sees what happens, it:
    /// 1. Remembers what the situation was (state)
    /// 2. Remembers what it did (action)
    /// 3. Notes the feedback it got (reward)
    /// 4. Observes the new situation (next state)
    /// 5. Updates its understanding of which actions are good in which situations
    /// 
    /// Over time, this helps the agent learn to make better decisions to earn more rewards.
    /// </para>
    /// </remarks>
    public abstract T Update(Tensor<T> state, Vector<T> action, T reward, Tensor<T> nextState, bool done);
    
    /// <summary>
    /// Trains the agent on a batch of experiences.
    /// </summary>
    /// <param name="states">The batch of states.</param>
    /// <param name="actions">The batch of actions.</param>
    /// <param name="rewards">The batch of rewards.</param>
    /// <param name="nextStates">The batch of next states.</param>
    /// <param name="dones">The batch of done flags.</param>
    /// <returns>The loss value from the training.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a training update using a batch of experiences. It's typically
    /// called internally when enough experiences have been collected in the replay buffer.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is like a study session where the agent reviews multiple past experiences at once.
    /// 
    /// Rather than learning from just one experience at a time, the agent:
    /// 1. Gathers a collection of experiences (a batch)
    /// 2. Looks for patterns across these experiences
    /// 3. Makes a more substantial update to its understanding
    /// 
    /// This batch learning is often more efficient and stable than learning from
    /// individual experiences one at a time.
    /// </para>
    /// </remarks>
    protected abstract T TrainOnBatch(
        Tensor<T> states,
        Tensor<T> actions,
        Vector<T> rewards,
        Tensor<T> nextStates,
        Vector<T> dones);
    
    /// <summary>
    /// Sets the model to training mode.
    /// </summary>
    /// <remarks>
    /// <para>
    /// In training mode, the agent may use exploration strategies and update its parameters
    /// based on experiences.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This puts the agent in "learning mode," where it will:
    /// - Sometimes try random or exploratory actions to discover new strategies
    /// - Update its neural networks based on the experiences it collects
    /// - Focus on gathering information rather than maximizing immediate performance
    /// </para>
    /// </remarks>
    public virtual void SetTrainingMode()
    {
        IsTraining = true;
    }
    
    /// <summary>
    /// Sets the model to evaluation mode.
    /// </summary>
    /// <remarks>
    /// <para>
    /// In evaluation mode, the agent typically uses a deterministic policy without exploration
    /// and does not update its parameters.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This puts the agent in "performance mode," where it will:
    /// - Always choose what it believes is the best action
    /// - Not update its neural networks based on new experiences
    /// - Focus on maximizing performance using what it has already learned
    /// </para>
    /// </remarks>
    public virtual void SetEvaluationMode()
    {
        IsTraining = false;
    }
    
    /// <summary>
    /// Gets the parameters of the model as a single vector.
    /// </summary>
    /// <returns>A vector containing all parameters of the model.</returns>
    /// <remarks>
    /// <para>
    /// This method gathers all the parameters of the agent's neural networks into a single vector.
    /// It's useful for serialization, optimization, and analysis.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This collects all the "knowledge" the agent has learned into one long list of numbers.
    /// 
    /// These numbers represent the weights and biases in the neural networks that determine:
    /// - How the agent evaluates different states
    /// - How it decides which actions to take
    /// - How it predicts future rewards
    /// 
    /// This complete list can be saved and loaded later to restore the agent's knowledge.
    /// </para>
    /// </remarks>
    public abstract Vector<T> GetParameters();
    
    /// <summary>
    /// Sets the parameters of the model from a vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <remarks>
    /// <para>
    /// This method sets all the parameters of the agent's neural networks from a single vector.
    /// It's typically used when loading a saved model or after optimization.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This takes a saved list of numbers and restores the agent's "knowledge" from it.
    /// 
    /// It's like loading all the information the agent has learned about:
    /// - How to evaluate different states
    /// - Which actions to take in different situations
    /// - How to predict future rewards
    /// 
    /// This allows you to continue using a previously trained agent without having to
    /// train it all over again.
    /// </para>
    /// </remarks>
    public abstract void SetParameters(Vector<T> parameters);
    
    /// <summary>
    /// Saves the model to a stream.
    /// </summary>
    /// <param name="stream">The stream to save the model to.</param>
    /// <remarks>
    /// <para>
    /// This method serializes the model's parameters and configuration to a stream,
    /// allowing it to be persisted to disk or transmitted.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This saves the agent's "brain" to a file or data stream.
    /// 
    /// It captures:
    /// - What the agent has learned (neural network weights)
    /// - How the agent is configured (settings and hyperparameters)
    /// 
    /// This makes it possible to save a trained agent and use it later without
    /// having to train it all over again.
    /// </para>
    /// </remarks>
    public abstract void Save(Stream stream);
    
    /// <summary>
    /// Loads the model from a stream.
    /// </summary>
    /// <param name="stream">The stream to load the model from.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes the model's parameters and configuration from a stream,
    /// allowing it to be restored from disk or received from a transmission.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This loads a previously saved agent's "brain" from a file or data stream.
    /// 
    /// It restores:
    /// - What the agent had learned (neural network weights)
    /// - How the agent was configured (settings and hyperparameters)
    /// 
    /// This allows you to use a previously trained agent without having to train it again.
    /// </para>
    /// </remarks>
    public abstract void Load(Stream stream);
    
    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes the model to a byte array, which can be used for
    /// saving the model to a file or database, or transmitting it over a network.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This converts the agent's "brain" into a sequence of bytes that can be:
    /// - Saved to a file
    /// - Stored in a database
    /// - Sent over a network
    /// 
    /// It's a way to package up everything the agent has learned so it can be
    /// stored or shared and later restored.
    /// </para>
    /// </remarks>
    public virtual byte[] Serialize()
    {
        using (var stream = new MemoryStream())
        {
            Save(stream);
            return stream.ToArray();
        }
    }
    
    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="data">A byte array containing the serialized model.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes the model from a byte array, which could have been
    /// loaded from a file or database, or received over a network.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This converts a sequence of bytes back into the agent's "brain" after it was:
    /// - Loaded from a file
    /// - Retrieved from a database
    /// - Received over a network
    /// 
    /// It restores everything the agent had learned from this packed-up format.
    /// </para>
    /// </remarks>
    public virtual void Deserialize(byte[] data)
    {
        using (var stream = new MemoryStream(data))
        {
            Load(stream);
        }
    }
    
    /// <summary>
    /// Gets the last computed loss value.
    /// </summary>
    /// <returns>The last computed loss value.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the loss value from the most recent training update.
    /// It can be used to monitor the agent's learning progress.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This tells you how well the agent is currently performing.
    /// 
    /// The loss value is like a score that shows how far off the agent's
    /// predictions are from what they should be. A lower loss generally
    /// means the agent is doing better at its job.
    /// </para>
    /// </remarks>
    public virtual T GetLoss()
    {
        return LastLoss;
    }
    
    /// <summary>
    /// Creates a new instance of the model.
    /// </summary>
    /// <returns>A new instance of the model with the same configuration but no learned parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method is used to create a new instance of the specific model type with the same
    /// configuration as this instance but without copying learned parameters.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This creates a brand new copy of the agent with the same settings but none of
    /// the learned experience. It's like creating a new agent with the same capabilities
    /// but that hasn't been trained yet.
    /// </para>
    /// </remarks>
    public abstract IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance();
    
    /// <summary>
    /// Creates a deep copy of this model.
    /// </summary>
    /// <returns>A deep copy of this model with the same parameters and state.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a complete copy of the model, including all learned parameters
    /// and current state. It uses serialization and deserialization to ensure a proper deep copy.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This creates a complete duplicate of the agent with everything it has learned.
    /// The copy will behave exactly like the original because it has the same "knowledge"
    /// and settings.
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        // The most reliable way to create a deep copy is through serialization/deserialization
        byte[] serialized = Serialize();
        var copy = CreateNewInstance();
        copy.Deserialize(serialized);
        return copy;
    }
    
    /// <summary>
    /// Creates a shallow copy of this model.
    /// </summary>
    /// <returns>A shallow copy of this model.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the model with the same configuration
    /// but without copying learned parameters.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This creates a new agent with the same settings as the original but
    /// without any of the learned experience.
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, Tensor<T>, Tensor<T>> Clone()
    {
        return CreateNewInstance();
    }
    
    /// <summary>
    /// Creates a new instance with the specified parameters.
    /// </summary>
    /// <param name="parameters">The parameters to set in the new instance.</param>
    /// <returns>A new instance with the specified parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a deep copy of the model and sets its parameters to the specified values.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This creates a duplicate of the agent but with a specified set of parameters (weights and biases)
    /// instead of the original ones. It's like creating a new agent with the same structure
    /// but different learned knowledge.
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        var clone = Clone();
        if (clone is ReinforcementLearningModelBase<T> rlModel)
        {
            rlModel.SetParameters(parameters);
        }
        return clone;
    }
    
    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    /// <returns>Metadata describing the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the model, including its type, parameters,
    /// and performance characteristics.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This provides information about the agent, such as:
    /// - What type of reinforcement learning algorithm it uses
    /// - How many parameters (weights and biases) it has
    /// - How it's configured
    /// - How well it's currently performing
    /// </para>
    /// </remarks>
    public virtual ModelMetaData<T> GetModelMetaData()
    {
        return new ModelMetaData<T>
        {
            ModelType = Enums.ModelType.DQNModel, // Default RL model type
            Description = GetType().Name,
            FeatureCount = GetParameters().Length
        };
    }
    
    /// <summary>
    /// Backward compatibility method for GetModelMetaData.
    /// </summary>
    /// <returns>Metadata describing the model.</returns>
    public virtual ModelMetaData<T> GetModelMetadata()
    {
        return GetModelMetaData();
    }
    
    /// <summary>
    /// Trains the model on the given input and expected output.
    /// </summary>
    /// <param name="input">The input data.</param>
    /// <param name="expectedOutput">The expected output data.</param>
    /// <remarks>
    /// <para>
    /// This method is part of the IModel interface, but reinforcement learning typically
    /// doesn't train in the same supervised way. This implementation is meant to provide
    /// compatibility with the general model interface.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Reinforcement learning agents usually learn through interaction with an environment
    /// rather than from labeled examples like supervised learning. This method provides a way
    /// to fit within the common model interface, but the actual learning typically happens
    /// through the Update and TrainOnBatch methods.
    /// </para>
    /// </remarks>
    public virtual void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Reinforcement learning typically doesn't train in this supervised way
        // This method is provided for compatibility with the IModel interface
        
        // By default, we do nothing here as RL models learn through environment interaction
    }
    
    /// <summary>
    /// Gets the active feature indices used by the model.
    /// </summary>
    /// <returns>An enumerable of feature indices that are used by the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the indices of features that are actively used by the model.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This tells you which inputs or features the agent pays attention to when making decisions.
    /// It can be useful for understanding what information the agent considers important.
    /// </para>
    /// </remarks>
    public virtual IEnumerable<int> GetActiveFeatureIndices()
    {
        // By default, we assume all features are used
        // Derived classes might override this for models that use feature selection
        var agent = GetAgent();
        int featureCount = 0;
        
        // Try to determine feature count from agent
        if (agent != null)
        {
            // This is a simple approximation - specific models should override this method
            featureCount = GetParameters().Length;
        }
        
        // If we can't determine feature count, return an empty list
        if (featureCount <= 0)
            return new List<int>();
            
        // Return all feature indices
        var indices = new List<int>();
        for (int i = 0; i < featureCount; i++)
        {
            indices.Add(i);
        }
        
        return indices;
    }
    
    /// <summary>
    /// Checks if a specific feature is used by the model.
    /// </summary>
    /// <param name="featureIndex">The index of the feature to check.</param>
    /// <returns>True if the feature is used, false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// This method checks if a specific feature is actively used by the model.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This tells you whether the agent pays attention to a specific input or feature
    /// when making decisions.
    /// </para>
    /// </remarks>
    public virtual bool IsFeatureUsed(int featureIndex)
    {
        // By default, we assume all features are used
        // Derived classes might override this for models that use feature selection
        return true;
    }
    
    /// <summary>
    /// Sets the active feature indices to be used by the model.
    /// </summary>
    /// <param name="featureIndices">The indices of features to use.</param>
    /// <remarks>
    /// <para>
    /// This method sets which features should be actively used by the model.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This tells the agent which inputs or features it should pay attention to
    /// when making decisions.
    /// </para>
    /// </remarks>
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        // By default, we do nothing as most RL models don't support explicit feature selection
        // Derived classes might override this for models that support feature selection
    }
    
    /// <summary>
    /// Saves the model to the specified file path.
    /// </summary>
    /// <param name="path">The file path to save the model to.</param>
    /// <remarks>
    /// <para>
    /// This method saves the model to a file at the specified path. It uses the
    /// Save(Stream) method to perform the actual serialization.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This saves the agent to a file on disk, which allows you to use the trained agent
    /// again later without having to retrain it.
    /// </para>
    /// </remarks>
    public virtual void SaveModel(string path)
    {
        using (var stream = new FileStream(path, FileMode.Create))
        {
            Save(stream);
        }
    }
    
    /// <summary>
    /// Loads the model from the specified file path.
    /// </summary>
    /// <param name="path">The file path to load the model from.</param>
    /// <remarks>
    /// <para>
    /// This method loads the model from a file at the specified path. It uses the
    /// Load(Stream) method to perform the actual deserialization.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This loads a previously saved agent from a file on disk, allowing you to
    /// continue using a trained agent without having to train it again.
    /// </para>
    /// </remarks>
    public virtual void LoadModel(string path)
    {
        using (var stream = new FileStream(path, FileMode.Open))
        {
            Load(stream);
        }
    }

    #region IFullModel Interface Members - Added by Team 23

    /// <summary>
    /// Gets the total number of parameters in the model
    /// </summary>
    public virtual int ParameterCount => 0; // Base RL model has no direct trainable parameters

    /// <summary>
    /// Gets feature importance scores
    /// </summary>
    public virtual Dictionary<string, T> GetFeatureImportance()
    {
        // Return empty dictionary as default for RL models
        return new Dictionary<string, T>();
    }

    #endregion

    /// <summary>
    /// Saves the model to the specified file path.
    /// </summary>
    /// <param name="path">The file path to save the model to.</param>
    /// <remarks>
    /// <para>
    /// This method saves the model to a file at the specified path. It uses the
    /// Save(Stream) method to perform the actual serialization.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This saves the agent to a file on disk, which allows you to use the trained agent
    /// again later without having to retrain it.
    /// </para>
    /// </remarks>
    public virtual void SaveModel(string path)
    {
        using (var stream = new FileStream(path, FileMode.Create))
        {
            Save(stream);
        }
    }

    /// <summary>
    /// Loads the model from the specified file path.
    /// </summary>
    /// <param name="path">The file path to load the model from.</param>
    /// <remarks>
    /// <para>
    /// This method loads the model from a file at the specified path. It uses the
    /// Load(Stream) method to perform the actual deserialization.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This loads a previously saved agent from a file on disk, allowing you to
    /// continue using a trained agent without having to train it again.
    /// </para>
    /// </remarks>
    public virtual void LoadModel(string path)
    {
        using (var stream = new FileStream(path, FileMode.Open))
        {
            Load(stream);
        }
    }

    #region IInterpretableModel Implementation

    // Fields for interpretability support
    private readonly HashSet<InterpretationMethod> _enabledMethods = new();
    private Vector<int> _sensitiveFeatures;
    private readonly List<FairnessMetric> _fairnessMetrics = new();
    private IModel<Tensor<T>, Tensor<T>, ModelMetadata<T>> _baseModel;

    /// <summary>
    /// Gets the global feature importance across all predictions.
    /// </summary>
    public virtual async Task<Dictionary<int, T>> GetGlobalFeatureImportanceAsync()
    {
        return await InterpretableModelHelper.GetGlobalFeatureImportanceAsync(this, _enabledMethods);
    }

    /// <summary>
    /// Gets the local feature importance for a specific input.
    /// </summary>
    public virtual async Task<Dictionary<int, T>> GetLocalFeatureImportanceAsync(Tensor<T> input)
    {
        return await InterpretableModelHelper.GetLocalFeatureImportanceAsync(this, _enabledMethods, input);
    }

    /// <summary>
    /// Gets SHAP values for the given inputs.
    /// </summary>
    public virtual async Task<Matrix<T>> GetShapValuesAsync(Tensor<T> inputs)
    {
        return await InterpretableModelHelper.GetShapValuesAsync(this, _enabledMethods);
    }

    /// <summary>
    /// Gets LIME explanation for a specific input.
    /// </summary>
    public virtual async Task<LimeExplanation<T>> GetLimeExplanationAsync(Tensor<T> input, int numFeatures = 10)
    {
        return await InterpretableModelHelper.GetLimeExplanationAsync<T>(_enabledMethods, numFeatures);
    }

    /// <summary>
    /// Gets partial dependence data for specified features.
    /// </summary>
    public virtual async Task<PartialDependenceData<T>> GetPartialDependenceAsync(Vector<int> featureIndices, int gridResolution = 20)
    {
        return await InterpretableModelHelper.GetPartialDependenceAsync<T>(_enabledMethods, featureIndices, gridResolution);
    }

    /// <summary>
    /// Gets counterfactual explanation for a given input and desired output.
    /// </summary>
    public virtual async Task<CounterfactualExplanation<T>> GetCounterfactualAsync(Tensor<T> input, Tensor<T> desiredOutput, int maxChanges = 5)
    {
        return await InterpretableModelHelper.GetCounterfactualAsync<T>(_enabledMethods, maxChanges);
    }

    /// <summary>
    /// Gets model-specific interpretability information.
    /// </summary>
    public virtual async Task<Dictionary<string, object>> GetModelSpecificInterpretabilityAsync()
    {
        return await InterpretableModelHelper.GetModelSpecificInterpretabilityAsync(this);
    }

    /// <summary>
    /// Generates a text explanation for a prediction.
    /// </summary>
    public virtual async Task<string> GenerateTextExplanationAsync(Tensor<T> input, Tensor<T> prediction)
    {
        return await InterpretableModelHelper.GenerateTextExplanationAsync(this, input, prediction);
    }

    /// <summary>
    /// Gets feature interaction effects between two features.
    /// </summary>
    public virtual async Task<T> GetFeatureInteractionAsync(int feature1Index, int feature2Index)
    {
        return await InterpretableModelHelper.GetFeatureInteractionAsync<T>(_enabledMethods, feature1Index, feature2Index);
    }

    /// <summary>
    /// Validates fairness metrics for the given inputs.
    /// </summary>
    public virtual async Task<FairnessMetrics<T>> ValidateFairnessAsync(Tensor<T> inputs, int sensitiveFeatureIndex)
    {
        return await InterpretableModelHelper.ValidateFairnessAsync<T>(_fairnessMetrics);
    }

    /// <summary>
    /// Gets anchor explanation for a given input.
    /// </summary>
    public virtual async Task<AnchorExplanation<T>> GetAnchorExplanationAsync(Tensor<T> input, T threshold)
    {
        return await InterpretableModelHelper.GetAnchorExplanationAsync(_enabledMethods, threshold);
    }

    /// <summary>
    /// Sets the base model for interpretability analysis.
    /// </summary>
    public virtual void SetBaseModel(IModel<Tensor<T>, Tensor<T>, ModelMetadata<T>> model)
    {
        _baseModel = model ?? throw new ArgumentNullException(nameof(model));
    }

    /// <summary>
    /// Enables specific interpretation methods.
    /// </summary>
    public virtual void EnableMethod(params InterpretationMethod[] methods)
    {
        foreach (var method in methods)
        {
            _enabledMethods.Add(method);
        }
    }

    /// <summary>
    /// Configures fairness evaluation settings.
    /// </summary>
    public virtual void ConfigureFairness(Vector<int> sensitiveFeatures, params FairnessMetric[] fairnessMetrics)
    {
        _sensitiveFeatures = sensitiveFeatures ?? throw new ArgumentNullException(nameof(sensitiveFeatures));
        _fairnessMetrics.Clear();
        _fairnessMetrics.AddRange(fairnessMetrics);
    }

    #endregion
}
