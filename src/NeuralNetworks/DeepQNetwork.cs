using AiDotNet.ReinforcementLearning.ReplayBuffers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Deep Q-Network (DQN), a reinforcement learning algorithm that combines Q-learning with deep neural networks.
/// </summary>
/// <remarks>
/// <para>
/// A Deep Q-Network (DQN) is a reinforcement learning algorithm that uses a neural network to approximate the Q-function,
/// which represents the expected future rewards for taking specific actions in specific states. DQNs overcome the limitations
/// of traditional Q-learning by using neural networks to generalize across states, allowing them to handle problems with large
/// or continuous state spaces. Key features of DQNs include experience replay (storing and randomly sampling past experiences)
/// and the use of a separate target network to stabilize learning.
/// </para>
/// <para><b>For Beginners:</b> A Deep Q-Network is like a smart decision-maker that learns through trial and error.
/// 
/// Imagine you're teaching a robot to play a video game:
/// - The robot needs to learn which actions (button presses) are best in each situation (game screen)
/// - At first, the robot makes many random moves to explore the game
/// - Over time, it remembers which moves led to high scores and which led to game over
/// - The "Deep" part means it uses a neural network to recognize patterns in complex situations
/// - The "Q" refers to "Quality" - how good an action is in a specific situation
/// 
/// For example, in a maze game, the network learns that moving toward the exit is usually better than moving away from it,
/// even if it hasn't seen that exact maze position before.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class DeepQNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets or sets the number of possible actions the agent can take in the environment.
    /// </summary>
    /// <value>An integer representing the size of the action space.</value>
    /// <remarks>
    /// <para>
    /// The action space represents the set of all possible actions that an agent can take in the environment.
    /// In the context of a Deep Q-Network, this value determines the output size of the neural network,
    /// as each output neuron corresponds to the estimated Q-value for one specific action.
    /// </para>
    /// <para><b>For Beginners:</b> This is the number of different moves the agent can make.
    /// 
    /// Think of it like the number of buttons on a controller:
    /// - If _actionSpace is 4, the agent might have actions like "move up", "move down", "move left", and "move right"
    /// - If _actionSpace is 2, the agent might only have actions like "jump" and "don't jump"
    /// 
    /// The neural network has one output for each possible action, and each output gives a score
    /// for how good that action is in the current situation.
    /// </para>
    /// </remarks>
    private int _actionSpace;

    /// <summary>
    /// Gets the buffer that stores past experiences for experience replay.
    /// </summary>
    /// <value>A list of experiences, each containing a state, action, reward, next state, and done flag.</value>
    /// <remarks>
    /// <para>
    /// The replay buffer is a key component of DQN that stores past experiences (state, action, reward, next state, done)
    /// to enable experience replay. Experience replay breaks the correlation between consecutive training samples
    /// by randomly sampling from this buffer, which helps stabilize and improve the training process.
    /// </para>
    /// <para><b>For Beginners:</b> This is the agent's memory of past experiences.
    /// 
    /// Think of it like a journal where the agent records what happened each time it took an action:
    /// - What the situation looked like (state)
    /// - Which action it took
    /// - What reward it received (positive or negative)
    /// - What the new situation looked like afterward (next state)
    /// - Whether the episode ended (done)
    /// 
    /// Instead of learning only from its most recent actions, the agent can review this journal
    /// and learn from random past experiences, which helps it learn more efficiently and stably.
    /// </para>
    /// </remarks>
    private readonly List<Experience<T, Tensor<T>, int>> _replayBuffer = new List<Experience<T, Tensor<T>, int>>();

    /// <summary>
    /// Gets the target network, a copy of the main network used to generate target Q-values during training.
    /// </summary>
    /// <value>A separate Deep Q-Network with the same architecture as the main network.</value>
    /// <remarks>
    /// <para>
    /// The target network is a copy of the main Q-network that is used to generate target Q-values during training.
    /// Its parameters are periodically updated to match the main network. This separation between the network being
    /// updated and the network generating targets helps stabilize the training process by reducing the moving target problem.
    /// </para>
    /// <para><b>For Beginners:</b> This is a stable copy of the main network that helps the agent learn more effectively.
    /// 
    /// Think of it like having two versions of the agent:
    /// - One version (the main network) is actively learning and changing
    /// - The other version (the target network) stays more stable and changes less frequently
    /// 
    /// When evaluating how good an action was, the agent compares what it expected (from the main network)
    /// to what the stable version (target network) says should have happened. This prevents the agent from
    /// chasing a constantly moving target, which would make learning very difficult.
    /// </para>
    /// </remarks>
    private DeepQNetwork<T>? _targetNetwork;

    /// <summary>
    /// Gets the exploration rate, which controls how often the agent takes random actions versus exploiting learned knowledge.
    /// </summary>
    /// <value>A value between 0 and 1 representing the probability of taking a random action.</value>
    /// <remarks>
    /// <para>
    /// The epsilon value controls the exploration-exploitation trade-off in reinforcement learning.
    /// When epsilon is high, the agent explores more by taking random actions. As epsilon decreases,
    /// the agent exploits more by taking actions it believes will maximize rewards based on its current knowledge.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how often the agent tries random actions versus what it thinks are the best actions.
    /// 
    /// Think of epsilon like a dial between:
    /// - Exploration: Trying new, random actions to discover what works (high epsilon)
    /// - Exploitation: Using what it has already learned to pick the best known action (low epsilon)
    /// 
    /// For example:
    /// - With epsilon = 0.9, the agent will try random actions 90% of the time
    /// - With epsilon = 0.1, the agent will only try random actions 10% of the time
    /// 
    /// Usually, epsilon starts high and decreases over time as the agent learns more about its environment.
    /// </para>
    /// </remarks>
    private readonly T _epsilon;

    /// <summary>
    /// Gets the number of experiences to sample from the replay buffer during each training step.
    /// </summary>
    /// <value>An integer representing the batch size for training.</value>
    /// <remarks>
    /// <para>
    /// The batch size determines how many experiences are sampled from the replay buffer and processed
    /// together in each training step. Larger batch sizes can lead to more stable gradient updates
    /// but require more computation per step.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many past experiences the agent reviews at once during training.
    /// 
    /// Think of it like studying for a test:
    /// - Instead of reviewing just one experience at a time, the agent reviews a batch of them
    /// - This helps the agent see patterns across multiple experiences
    /// - It also makes training more stable and efficient
    /// 
    /// For example, with _batchSize = 32, the agent randomly picks 32 different experiences
    /// from its memory and learns from all of them at once.
    /// </para>
    /// </remarks>
    private readonly int _batchSize = 32;

    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Initializes a new instance of the <see cref="DeepQNetwork{T}"/> class with the specified architecture and exploration rate.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="epsilon">The initial exploration rate (probability of taking random actions). Default is 1.0 for full exploration.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Deep Q-Network with the specified architecture and exploration rate.
    /// It also initializes a separate target network with the same architecture, which is used to generate
    /// target Q-values during training.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up a new Deep Q-Network ready to start learning.
    /// 
    /// When creating a new DQN:
    /// - You provide an "architecture" that defines the neural network's structure
    /// - You can set an "epsilon" value that controls how often it will try random actions
    /// - The constructor also creates a copy of the network (target network) that helps with stable learning
    /// 
    /// Think of it like setting up a new student with blank notebooks (the neural networks) and
    /// a curiosity level (epsilon) that determines how often they'll experiment versus stick with what they know.
    /// </para>
    /// </remarks>
    public DeepQNetwork(NeuralNetworkArchitecture<T> architecture, ILossFunction<T>? lossFunction = null, double epsilon = 1.0) :
        this(architecture, lossFunction, epsilon, isTargetNetwork: false)
    {
    }

    /// <summary>
    /// Private constructor used to create the target network without infinite recursion.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="lossFunction">The loss function to use for training.</param>
    /// <param name="epsilon">The initial exploration rate.</param>
    /// <param name="isTargetNetwork">If true, this is a target network and won't create its own target network.</param>
    private DeepQNetwork(NeuralNetworkArchitecture<T> architecture, ILossFunction<T>? lossFunction, double epsilon, bool isTargetNetwork) :
        base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _epsilon = NumOps.FromDouble(epsilon);
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);

        // Only create the target network if this is not already a target network (prevents infinite recursion)
        if (!isTargetNetwork)
        {
            _targetNetwork = new DeepQNetwork<T>(architecture, lossFunction, epsilon, isTargetNetwork: true);
        }

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the Deep Q-Network based on the architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the layers of the Deep Q-Network. If custom layers are provided in the architecture,
    /// those layers are used. Otherwise, default layers are created based on the architecture's specifications.
    /// After setting up the layers, the method sets the action space based on the output size of the network.
    /// </para>
    /// <para><b>For Beginners:</b> This method builds the actual structure of the neural network.
    /// 
    /// When initializing the layers:
    /// - If you've specified your own custom layers, the network will use those
    /// - If not, the network will create a standard set of layers suitable for a DQN
    /// - The method also determines how many different actions the agent can take
    /// 
    /// This is like assembling the brain of our agent, which will learn to make decisions
    /// by associating situations with the best actions to take.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            // Use the layers provided by the user
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            // Use default layer configuration if no layers are provided
            Layers.AddRange(LayerHelper<T>.CreateDefaultDeepQNetworkLayers(Architecture));
        }

        _actionSpace = Architecture.OutputSize;
    }

    /// <summary>
    /// Gets the Q-values for all possible actions in the given state.
    /// </summary>
    /// <param name="state">The input tensor representing the current state.</param>
    /// <returns>A tensor of Q-values, one for each possible action.</returns>
    /// <remarks>
    /// <para>
    /// This method is a wrapper around the Predict method that makes it more semantically clear that
    /// the output represents Q-values for each possible action in the given state.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you how good the agent thinks each action is in the current situation.
    /// 
    /// The Q-values represent:
    /// - The agent's estimate of how much future reward it will get
    /// - If it takes a specific action in the current state
    /// - And then continues to act optimally afterward
    /// 
    /// For example, in a game, a Q-value of 100 for "move right" means the agent expects
    /// that moving right now will eventually lead to a total reward of about 100 points.
    /// </para>
    /// </remarks>
    public Tensor<T> GetQValues(Tensor<T> state)
    {
        return Predict(state);
    }

    /// <summary>
    /// Gets the best action to take in the given state based on current Q-values.
    /// </summary>
    /// <param name="state">The input vector representing the current state.</param>
    /// <returns>The index of the action with the highest Q-value.</returns>
    /// <remarks>
    /// <para>
    /// This method determines the optimal action to take in the given state by selecting the action
    /// with the highest predicted Q-value. This represents the action that the network currently believes
    /// will lead to the highest expected future reward.
    /// </para>
    /// <para><b>For Beginners:</b> This method finds the action that the agent thinks is best in the current situation.
    /// 
    /// The process works like this:
    /// - The agent gets Q-values (scores) for each possible action
    /// - It then selects the action with the highest score
    /// - This is called "exploitation" - using what the agent has learned so far
    /// 
    /// For example, if moving left has a Q-value of 5 and moving right has a Q-value of 10,
    /// the agent will choose to move right because it has the higher expected reward.
    /// </para>
    /// </remarks>
    public int GetBestAction(Tensor<T> state)
    {
        var qValues = GetQValues(state);
        return qValues.Max().maxIndex;
    }

    /// <summary>
    /// Gets an action to take in the given state, balancing exploration and exploitation.
    /// </summary>
    /// <param name="state">The input vector representing the current state.</param>
    /// <returns>The index of the selected action.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the epsilon-greedy policy for action selection. With probability epsilon,
    /// it selects a random action (exploration), and with probability 1-epsilon, it selects the action
    /// with the highest Q-value (exploitation). This balance between exploration and exploitation is
    /// crucial for effective reinforcement learning.
    /// </para>
    /// <para><b>For Beginners:</b> This method decides whether to try a random action or use what the agent has learned.
    /// 
    /// The process works like this:
    /// - The agent generates a random number between 0 and 1
    /// - If the number is less than epsilon, it takes a completely random action (exploration)
    /// - Otherwise, it takes the action it currently thinks is best (exploitation)
    /// 
    /// This balance is important because:
    /// - If the agent only exploits, it might miss better strategies it hasn't discovered yet
    /// - If the agent only explores, it never uses what it has learned
    /// 
    /// For example, with epsilon = 0.1:
    /// - 10% of the time, the agent will try a random action
    /// - 90% of the time, it will choose the action with the highest Q-value
    /// </para>
    /// </remarks>
    public int GetAction(Tensor<T> state)
    {
        if (NumOps.LessThan(NumOps.FromDouble(Random.NextDouble()), _epsilon))
        {
            return Random.Next(_actionSpace);
        }

        return GetBestAction(state);
    }

    /// <summary>
    /// Adds a new experience to the replay buffer.
    /// </summary>
    /// <param name="state">The state before taking the action.</param>
    /// <param name="action">The action taken.</param>
    /// <param name="reward">The reward received after taking the action.</param>
    /// <param name="nextState">The state after taking the action.</param>
    /// <param name="done">A flag indicating whether the episode ended after this action.</param>
    /// <remarks>
    /// <para>
    /// This method adds a new experience tuple (state, action, reward, next state, done) to the replay buffer
    /// for use in experience replay during training. If the buffer exceeds its maximum size (10,000 experiences),
    /// the oldest experiences are removed to make room for new ones.
    /// </para>
    /// <para><b>For Beginners:</b> This method stores a new experience in the agent's memory.
    /// 
    /// Each experience includes:
    /// - State: What the environment looked like before the action
    /// - Action: What the agent decided to do
    /// - Reward: The immediate feedback received (positive or negative)
    /// - Next State: What the environment looked like after the action
    /// - Done: Whether this action ended the episode (like finishing a game level)
    /// 
    /// The agent keeps a limited memory of past experiences (10,000 in this case) and
    /// removes the oldest ones when necessary, like a scrolling journal that only keeps
    /// the most recent entries.
    /// </para>
    /// </remarks>
    public void AddExperience(Tensor<T> state, int action, T reward, Tensor<T> nextState, bool done)
    {
        _replayBuffer.Add(new Experience<T, Tensor<T>, int>(state, action, reward, nextState, done));
        if (_replayBuffer.Count > 10000) // Limit buffer size
        {
            _replayBuffer.RemoveAt(0);
        }
    }

    /// <summary>
    /// Updates the target network to match the current state of the main network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method copies the parameters (weights and biases) from the main network to the target network.
    /// This update is performed periodically rather than after every training step to provide stable target values
    /// during training, which helps reduce the moving target problem and stabilize the learning process.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the stable copy of the network with the latest version.
    /// 
    /// Think of it like taking a snapshot of the main network:
    /// - The main network is constantly changing as it learns
    /// - The target network stays fixed to provide stable learning targets
    /// - Periodically, we update the target network to match the main network
    /// - This happens relatively infrequently (e.g., every 100 training steps)
    /// 
    /// This periodic updating helps the agent learn more stably by preventing
    /// a situation where it's constantly chasing a moving target.
    /// </para>
    /// </remarks>
    private void UpdateTargetNetwork()
    {
        // Copy weights from the main network to the target network
        if (_targetNetwork is null)
        {
            return;
        }

        for (int i = 0; i < Layers.Count; i++)
        {
            _targetNetwork.Layers[i].SetParameters(Layers[i].GetParameters());
        }
    }

    /// <summary>
    /// Updates the parameters of all layers in the Deep Q-Network.
    /// </summary>
    /// <param name="parameters">A vector containing the parameters to update all layers with.</param>
    /// <remarks>
    /// <para>
    /// This method distributes the provided parameter vector among all the layers in the network.
    /// Each layer receives a portion of the parameter vector corresponding to its number of parameters.
    /// The method keeps track of the starting index for each layer's parameters in the input vector.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the internal values of the neural network at once.
    /// 
    /// When updating parameters:
    /// - The input is a long list of numbers representing all values in the entire network
    /// - The method divides this list into smaller chunks
    /// - Each layer gets its own chunk of values
    /// - The layers use these values to adjust their internal settings
    /// 
    /// This method is less commonly used in DQN than the standard training process,
    /// but it provides a way to directly set all parameters at once, which can be
    /// useful in certain scenarios like loading pretrained weights or implementing advanced optimization algorithms.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int startIndex = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                Vector<T> layerParameters = parameters.SubVector(startIndex, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                startIndex += layerParameterCount;
            }
        }
    }

    /// <summary>
    /// Performs a forward pass with a tensor input.
    /// </summary>
    /// <param name="input">The input tensor representing the current state.</param>
    /// <returns>A tensor of Q-values for each possible action.</returns>
    /// <remarks>
    /// <para>
    /// This overload of the Predict method handles tensor inputs directly. It processes the input
    /// through all layers of the network and returns a tensor of Q-values.
    /// </para>
    /// <para><b>For Beginners:</b> This method does the same thing as the vector-based Predict,
    /// but works with tensors (multi-dimensional arrays) directly, which can be more efficient
    /// for certain types of inputs like images.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // GPU-resident optimization: when GPU engine is available and all layers support GPU,
        // use the GPU-resident path to avoid per-layer CPU downloads.
        // This can provide 10-50x speedup for inference workloads.
        if (Engine is DirectGpuTensorEngine && CanUseGpuResidentPath())
        {
            try
            {
                // ForwardGpu chains layers on GPU without intermediate CPU downloads
                using var gpuResult = ForwardGpu(input);
                return gpuResult.ToTensor();
            }
            catch
            {
                // Fall back to CPU path on any GPU error
            }
        }

        // CPU path: forward pass through all layers
        var current = input;

        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Checks if all layers support GPU execution for the GPU-resident optimization path.
    /// </summary>
    private bool CanUseGpuResidentPath()
    {
        foreach (var layer in Layers)
        {
            if (!layer.CanExecuteOnGpu)
            {
                return false;
            }
        }
        return true;
    }

    /// <summary>
    /// Trains the network using experience replay.
    /// </summary>
    /// <param name="input">Not used in DQN; experiences are sampled from the replay buffer.</param>
    /// <param name="expectedOutput">Not used in DQN; target Q-values are computed internally.</param>
    /// <remarks>
    /// <para>
    /// This method implements the core DQN training algorithm using experience replay. It samples a batch
    /// of experiences from the replay buffer, computes target Q-values using the target network, and
    /// updates the main network to minimize the difference between predicted and target Q-values.
    /// </para>
    /// <para><b>For Beginners:</b> This method is how the agent learns from its past experiences.
    /// 
    /// The training process works like this:
    /// 1. Randomly select a batch of experiences from memory
    /// 2. For each experience, calculate what the Q-values should have been:
    ///    - For terminal states (game over), the target is just the immediate reward
    ///    - For non-terminal states, the target is the immediate reward plus the discounted maximum future Q-value
    /// 3. Update the network to better predict these target values
    /// 4. Periodically update the target network to match the main network
    /// 
    /// This helps the agent gradually improve its ability to predict which actions will lead to higher rewards.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Skip if not enough experiences in the buffer
        if (_replayBuffer.Count < _batchSize)
        {
            return;
        }

        // Sample random batch of experiences
        var batch = SampleBatch(_batchSize);

        // Prepare input batch (states) and output batch (Q-values)
        var states = new Tensor<T>[_batchSize];
        var targetQValues = new Tensor<T>[_batchSize];

        for (int i = 0; i < _batchSize; i++)
        {
            var experience = batch[i];
            states[i] = experience.State;

            // Get current Q-values for the state
            var currentQValues = GetQValues(experience.State);

            // Create copy for updating
            var updatedQValues = currentQValues.Clone();

            // Calculate target Q-value for the action taken
            T targetQ;

            if (experience.Done || _targetNetwork is null)
            {
                // For terminal states (or if no target network), target Q-value is just the reward
                targetQ = experience.Reward;
            }
            else
            {
                // For non-terminal states, target Q-value is reward + gamma * max(Q(s', a'))
                // where gamma is the discount factor (typically 0.99)
                var nextStateQValues = _targetNetwork.GetQValues(experience.NextState);
                int bestNextAction = nextStateQValues.Max().maxIndex;
                T maxNextQ = nextStateQValues[bestNextAction];
                T gamma = NumOps.FromDouble(0.99); // Discount factor
                targetQ = NumOps.Add(experience.Reward, NumOps.Multiply(gamma, maxNextQ));
            }

            // Update only the Q-value for the action that was taken
            updatedQValues[experience.Action] = targetQ;

            targetQValues[i] = updatedQValues;
        }

        // Combine all states and target Q-values into batches
        var statesBatch = Tensor<T>.Stack(states);
        var targetsBatch = Tensor<T>.Stack(targetQValues);

        // Set network to training mode
        SetTrainingMode(true);

        // Forward pass with memory
        var predictions = ForwardWithMemoryBatch(statesBatch);

        // Calculate loss using the configured loss function
        LastLoss = _lossFunction.CalculateLoss(predictions.ToVector(), targetsBatch.ToVector());

        // Compute gradients and update network
        var outputGradients = predictions.Subtract(targetsBatch);
        BackpropagateBatch(outputGradients);
        UpdateParameters(NumOps.FromDouble(0.001)); // Learning rate

        // Set network back to inference mode
        SetTrainingMode(false);

        // Periodically update target network (e.g., every 100 training steps)
        if (Random.Next(100) == 0)
        {
            UpdateTargetNetwork();
        }
    }

    /// <summary>
    /// Samples a batch of experiences from the replay buffer.
    /// </summary>
    /// <param name="batchSize">The number of experiences to sample.</param>
    /// <returns>A list of randomly sampled experiences.</returns>
    /// <remarks>
    /// <para>
    /// This method randomly samples experiences from the replay buffer for use in training.
    /// Random sampling helps break correlations between consecutive experiences, which improves
    /// the stability of the learning process.
    /// </para>
    /// <para><b>For Beginners:</b> This method picks random memories from the agent's experience.
    /// 
    /// Instead of learning from its most recent experiences (which might be related),
    /// the agent randomly selects different experiences from its memory. This is like
    /// studying different topics rather than focusing on just one thing, which helps
    /// the agent develop a more balanced understanding.
    /// </para>
    /// </remarks>
    private List<Experience<T, Tensor<T>, int>> SampleBatch(int batchSize)
    {
        var batch = new List<Experience<T, Tensor<T>, int>>(batchSize);

        for (int i = 0; i < batchSize; i++)
        {
            int randomIndex = Random.Next(_replayBuffer.Count);
            batch.Add(_replayBuffer[randomIndex]);
        }

        return batch;
    }

    /// <summary>
    /// Performs a forward pass for a batch of inputs while storing intermediate values for backpropagation.
    /// </summary>
    /// <param name="inputs">A tensor containing a batch of input states.</param>
    /// <returns>A tensor containing a batch of predicted Q-values.</returns>
    /// <remarks>
    /// <para>
    /// This method processes a batch of inputs through the network, storing intermediate values
    /// needed for backpropagation during training. It's similar to ForwardWithMemory but handles
    /// batched inputs for more efficient training.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes multiple inputs at once and remembers
    /// intermediate values to help with learning.
    /// 
    /// Processing inputs in batches:
    /// - Is more efficient than processing one at a time
    /// - Helps the network learn more stable patterns
    /// - Requires keeping track of intermediate values for learning
    /// </para>
    /// </remarks>
    private Tensor<T> ForwardWithMemoryBatch(Tensor<T> inputs)
    {
        var current = inputs;

        for (int i = 0; i < Layers.Count; i++)
        {
            // Store input to each layer for backpropagation
            _layerInputs[i] = current;

            // Forward pass through layer
            current = Layers[i].Forward(current);

            // Store output from each layer for backpropagation
            _layerOutputs[i] = current;
        }

        return current;
    }

    /// <summary>
    /// Performs backpropagation for a batch of outputs.
    /// </summary>
    /// <param name="outputGradients">The gradients of the loss with respect to the network outputs.</param>
    /// <remarks>
    /// <para>
    /// This method propagates error gradients backwards through the network for a batch of outputs.
    /// It updates the parameter gradients based on how much each parameter contributed to the error.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how to adjust the network's internal values
    /// to make better predictions.
    /// 
    /// During backpropagation:
    /// - The network calculates how wrong its predictions were
    /// - It figures out how each of its internal values contributed to these errors
    /// - It determines how to adjust each value to reduce future errors
    /// - These adjustments are stored as gradients, which are applied during the update step
    /// </para>
    /// </remarks>
    private void BackpropagateBatch(Tensor<T> outputGradients)
    {
        var gradientTensor = outputGradients;

        // Backpropagate through layers in reverse order
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradientTensor = Layers[i].Backward(gradientTensor);
        }
    }

    /// <summary>
    /// Updates the parameters of all layers in the network using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to control the size of parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method applies the calculated gradients to update the parameters of each layer in the network.
    /// The learning rate controls how large the parameter updates are, with smaller values leading to more
    /// stable but slower learning.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts the network's internal values to improve its predictions.
    /// 
    /// After calculating how to adjust each value (in backpropagation):
    /// - This method applies those adjustments
    /// - The learning rate controls how big the adjustments are
    /// - Small learning rates mean small, cautious adjustments
    /// - Large learning rates mean bigger, faster adjustments
    /// 
    /// Finding the right learning rate is important - too small and learning is too slow,
    /// too large and learning can become unstable.
    /// </para>
    /// </remarks>
    private void UpdateParameters(T learningRate)
    {
        foreach (var layer in Layers)
        {
            if (layer.SupportsTraining)
            {
                layer.UpdateParameters(learningRate);
            }
        }
    }

    /// <summary>
    /// Gets metadata about this Deep Q-Network model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the model, including its name, description, architecture,
    /// and other relevant information that might be useful for users or tools working with the model.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides information about this neural network model.
    /// 
    /// The metadata includes:
    /// - The type of model (Deep Q-Network)
    /// - The network architecture (how many layers, neurons, etc.)
    /// - The action space (how many different actions the agent can take)
    /// - Other settings like the exploration rate
    /// 
    /// This information is useful for documentation, debugging, and when saving/loading models.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.DeepQNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ActionSpace", _actionSpace },
                { "ExplorationRate", Convert.ToDouble(_epsilon) },
                { "ReplayBufferSize", _replayBuffer.Count }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Saves Deep Q-Network specific data to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to save to.</param>
    /// <remarks>
    /// <para>
    /// This method serializes DQN-specific data that isn't part of the base neural network.
    /// This includes the exploration rate (epsilon), action space size, and target network state.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves special DQN settings to a file.
    /// 
    /// When saving the model:
    /// - The base neural network parts are saved by other methods
    /// - This method saves the DQN-specific settings
    /// - This includes values like how often the agent explores
    /// - It also saves the target network that helps stabilize learning
    /// 
    /// This allows you to load the exact same DQN later, with all its settings intact.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Save exploration rate (epsilon)
        writer.Write(Convert.ToDouble(_epsilon));

        // Save action space size
        writer.Write(_actionSpace);

        // Save replay buffer size (but not the actual experiences)
        writer.Write(_replayBuffer.Count);

        // Serialize target network (if present)
        writer.Write(_targetNetwork is not null);
        if (_targetNetwork is not null)
        {
            for (int i = 0; i < _targetNetwork.Layers.Count; i++)
            {
                _targetNetwork.Layers[i].Serialize(writer);
            }
        }
    }

    /// <summary>
    /// Loads Deep Q-Network specific data from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to load from.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes DQN-specific data that was previously saved using SerializeNetworkSpecificData.
    /// It restores the exploration rate, action space size, and target network state.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads special DQN settings from a file.
    /// 
    /// When loading a saved model:
    /// - The base neural network parts are loaded by other methods
    /// - This method loads the DQN-specific settings
    /// - It restores values like the exploration rate
    /// - It also loads the target network that helps stabilize learning
    /// 
    /// This allows you to continue using a previously trained DQN with all its settings intact.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Load exploration rate (epsilon)
        T epsilon = NumOps.FromDouble(reader.ReadDouble());

        // Load action space size
        int actionSpace = reader.ReadInt32();
        _actionSpace = actionSpace;

        // Load replay buffer size (but can't restore actual experiences)
        int replayBufferSize = reader.ReadInt32();
        _ = replayBufferSize;

        // Deserialize target network (if it was serialized)
        bool hasTargetNetwork = reader.ReadBoolean();
        if (hasTargetNetwork)
        {
            var targetNetwork = _targetNetwork ??
                new DeepQNetwork<T>(Architecture, _lossFunction, Convert.ToDouble(epsilon), isTargetNetwork: true);
            _targetNetwork = targetNetwork;

            for (int i = 0; i < targetNetwork.Layers.Count; i++)
            {
                targetNetwork.Layers[i].Deserialize(reader);
            }
        }
    }

    /// <summary>
    /// Creates a new instance of the Deep Q-Network with the same architecture and configuration.
    /// </summary>
    /// <returns>A new Deep Q-Network instance with the same architecture and configuration.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the Deep Q-Network with the same architecture and
    /// exploration rate (epsilon) as the current instance. It's used in scenarios where a fresh
    /// copy of the model is needed while maintaining the same configuration.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a brand new copy of the agent with the same setup.
    /// 
    /// Think of it like creating a clone of the agent:
    /// - The new agent has the same neural network architecture
    /// - The new agent has the same exploration rate (epsilon)
    /// - But it's a completely separate instance with its own memory and learning state
    /// 
    /// This is useful when you need multiple instances of the same DQN model,
    /// such as for parallel training or comparing different learning strategies.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new DeepQNetwork<T>(this.Architecture, _lossFunction, Convert.ToDouble(this._epsilon));
    }
}







