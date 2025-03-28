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
    private readonly List<Experience<T>> _replayBuffer = [];

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
    private readonly DeepQNetwork<T> _targetNetwork;

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

    /// <summary>
    /// Initializes a new instance of the <see cref="DeepQNetwork{T}"/> class with the specified architecture and exploration rate.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="epsilon">The initial exploration rate (probability of taking random actions). Default is a high value for exploration.</param>
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
    public DeepQNetwork(NeuralNetworkArchitecture<T> architecture, double epsilon = 1e16) : base(architecture)
    {
        _epsilon = NumOps.FromDouble(epsilon);
        _targetNetwork = new DeepQNetwork<T>(architecture, epsilon);
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
    /// Makes a prediction using the current state of the Deep Q-Network.
    /// </summary>
    /// <param name="input">The input vector representing the current state.</param>
    /// <returns>The predicted Q-values for each possible action in the given state.</returns>
    /// <remarks>
    /// <para>
    /// This method generates a prediction by passing the input state vector through each layer of the Deep Q-Network
    /// in sequence. The result is a vector where each element represents the estimated Q-value (expected future reward)
    /// for taking a specific action in the given state.
    /// </para>
    /// <para><b>For Beginners:</b> This method estimates how good each possible action is in the current situation.
    /// 
    /// The prediction process works like this:
    /// - The input is a description of the current state (what the agent can observe)
    /// - This information flows through all the layers of the neural network
    /// - The output is a score for each possible action the agent could take
    /// - Higher scores mean the network thinks that action will lead to better rewards
    /// 
    /// For example, if the agent is playing a game, the prediction might say:
    /// "Moving left has a value of 5, moving right has a value of 10, so moving right is probably better."
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Vector<T> input)
    {
        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(Tensor<T>.FromVector(current)).ToVector();
        }

        return current;
    }

    /// <summary>
    /// Gets the Q-values for all possible actions in the given state.
    /// </summary>
    /// <param name="state">The input vector representing the current state.</param>
    /// <returns>A vector of Q-values, one for each possible action.</returns>
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
    public Vector<T> GetQValues(Vector<T> state)
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
    public int GetBestAction(Vector<T> state)
    {
        var qValues = GetQValues(state);
        return ArgMax(qValues);
    }

    /// <summary>
    /// Finds the index of the maximum value in a vector.
    /// </summary>
    /// <param name="vector">The input vector to search.</param>
    /// <returns>The index of the maximum value in the vector.</returns>
    /// <remarks>
    /// <para>
    /// This method iterates through the elements of the input vector and returns the index of the element
    /// with the highest value. It is used to determine which action has the highest Q-value.
    /// </para>
    /// <para><b>For Beginners:</b> This method finds the position of the highest number in a list.
    /// 
    /// For example, given a list of scores [3, 7, 2, 5]:
    /// - The highest value is 7
    /// - The position (index) of 7 in the list is 1 (counting from 0)
    /// - So the method returns 1
    /// 
    /// In the context of Q-learning, if each position represents an action:
    /// - Position 0 might be "move up" with a score of 3
    /// - Position 1 might be "move right" with a score of 7
    /// - Position 2 might be "move down" with a score of 2
    /// - Position 3 might be "move left" with a score of 5
    /// - The method would return 1, indicating "move right" is the best action
    /// </para>
    /// </remarks>
    private int ArgMax(Vector<T> vector)
    {
        T max = vector[0];
        int maxIndex = 0;
        for (int i = 1; i < vector.Length; i++)
        {
            if (NumOps.GreaterThan(vector[i], max))
            {
                max = vector[i];
                maxIndex = i;
            }
        }
        return maxIndex;
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
    public int GetAction(Vector<T> state)
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
    public void AddExperience(Vector<T> state, int action, T reward, Vector<T> nextState, bool done)
    {
        _replayBuffer.Add(new Experience<T>(state, action, reward, nextState, done));
        if (_replayBuffer.Count > 10000) // Limit buffer size
        {
            _replayBuffer.RemoveAt(0);
        }
    }

    /// <summary>
    /// Trains the Deep Q-Network using experience replay and a target network.
    /// </summary>
    /// <param name="gamma">The discount factor for future rewards.</param>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method implements the core learning algorithm for the Deep Q-Network. It samples a batch of experiences
    /// from the replay buffer, computes target Q-values using the Bellman equation and the target network,
    /// calculates the loss between current and target Q-values, and performs backpropagation to update the network
    /// parameters. The target network is periodically updated to match the main network.
    /// </para>
    /// <para><b>For Beginners:</b> This method is how the agent learns from its past experiences.
    /// 
    /// The training process works like this:
    /// - If the agent doesn't have enough experiences yet, it waits until it does
    /// - It randomly selects a batch of past experiences to learn from
    /// - For each experience, it computes how much reward it actually got
    /// - It compares this with how much reward it expected to get (its prediction)
    /// - It adjusts its neural network to make better predictions in the future
    /// - Occasionally, it updates its target network to stabilize learning
    /// 
    /// The gamma parameter determines how much the agent values future rewards:
    /// - A low gamma (like 0.5) means the agent mostly cares about immediate rewards
    /// - A high gamma (like 0.99) means the agent values long-term rewards almost as much as immediate ones
    /// 
    /// This process is similar to how humans learn from experience - by comparing
    /// what we expected to happen with what actually happened, and updating our
    /// understanding accordingly.
    /// </para>
    /// </remarks>
    public void Train(T gamma, T learningRate)
    {
        if (_replayBuffer.Count < _batchSize) return;

        var batch = _replayBuffer.OrderBy(x => Random.Next()).Take(_batchSize).ToList();

        var states = new Matrix<T>(batch.Select(e => e.State).ToArray());
        var actions = batch.Select(e => e.Action).ToArray();
        var rewards = new Vector<T>(batch.Select(e => e.Reward).ToArray());
        var nextStates = new Matrix<T>(batch.Select(e => e.NextState).ToArray());
        var dones = batch.Select(e => e.Done).ToArray();

        // Predict Q-values for current states
        var currentQValues = PredictBatch(states);

        // Predict Q-values for next states
        var nextQValues = _targetNetwork.PredictBatch(nextStates);

        // Compute target Q-values
        var targetQValues = new Matrix<T>(currentQValues.Rows, currentQValues.Columns);
        for (int i = 0; i < _batchSize; i++)
        {
            for (int j = 0; j < _actionSpace; j++)
            {
                if (j == actions[i])
                {
                    T maxNextQ = nextQValues.GetRow(i).Max();
                    T target = dones[i] ? rewards[i] : NumOps.Add(rewards[i], NumOps.Multiply(gamma, maxNextQ));
                    targetQValues[i, j] = target;
                }
                else
                {
                    targetQValues[i, j] = currentQValues[i, j];
                }
            }
        }

        // Compute loss
        Matrix<T> loss = ComputeLoss(currentQValues, targetQValues);

        // Perform backpropagation
        BackPropagate(loss, learningRate);

        // Update target network periodically
        if (Random.Next(100) == 0) // Update every 100 steps on average
        {
            UpdateTargetNetwork();
        }
    }

    /// <summary>
    /// Predicts Q-values for a batch of states.
    /// </summary>
    /// <param name="inputs">A matrix where each row represents a state.</param>
    /// <returns>A matrix where each row contains Q-values for all actions in the corresponding state.</returns>
    /// <remarks>
    /// <para>
    /// This method is similar to the Predict method but operates on a batch of states rather than a single state.
    /// It passes the batch through each layer of the network and returns the resulting Q-values for all states
    /// and actions.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes multiple states at once to get their Q-values.
    /// 
    /// Instead of evaluating one state at a time:
    /// - This method takes a whole batch of different states
    /// - It runs them all through the neural network together
    /// - It returns a set of Q-values for each state in the batch
    /// 
    /// This batch processing is much more efficient than processing each state individually,
    /// especially when training the neural network.
    /// </para>
    /// </remarks>
    private Matrix<T> PredictBatch(Matrix<T> inputs)
    {
        var current = inputs;
        foreach (var layer in Layers)
        {
            current = layer.Forward(Tensor<T>.FromMatrix(current)).ToMatrix();
        }

        return current;
    }

    /// <summary>
    /// Computes the loss between predicted and target Q-values.
    /// </summary>
    /// <param name="predicted">The predicted Q-values from the main network.</param>
    /// <param name="target">The target Q-values computed using the Bellman equation.</param>
    /// <returns>A matrix representing the loss for each Q-value.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the mean squared error (MSE) loss between the predicted Q-values from the main network
    /// and the target Q-values computed using the Bellman equation and the target network. This loss is used
    /// to update the network parameters through backpropagation.
    /// </para>
    /// <para><b>For Beginners:</b> This method measures how wrong the agent's predictions were.
    /// 
    /// The loss calculation works like this:
    /// - For each action the agent took, we have:
    ///   - What the agent predicted would happen (predicted Q-values)
    ///   - What actually happened (target Q-values)
    /// - The method calculates the squared difference between these values
    /// - Larger differences result in higher loss, meaning the predictions were very wrong
    /// - Smaller differences result in lower loss, meaning the predictions were more accurate
    /// 
    /// This loss value is what the neural network tries to minimize during training,
    /// gradually improving its ability to predict the outcomes of different actions.
    /// </para>
    /// </remarks>
    private Matrix<T> ComputeLoss(Matrix<T> predicted, Matrix<T> target)
    {
        // Mean Squared Error loss
        return predicted.Subtract(target).PointwiseMultiply(predicted.Subtract(target));
    }

    /// <summary>
    /// Performs backpropagation to update the network parameters based on the computed loss.
    /// </summary>
    /// <param name="loss">The loss matrix representing the error for each Q-value.</param>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method propagates the computed loss backward through the network, calculating gradients
    /// for each layer's parameters. These gradients are then used to update the parameters in each layer
    /// to minimize the loss and improve the network's predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts the neural network to make better predictions next time.
    /// 
    /// The backpropagation process works like this:
    /// - Starting from the output (the predictions), we calculate how wrong they were
    /// - We then work backwards through each layer of the neural network
    /// - At each layer, we adjust the network's parameters (weights and biases)
    /// - The adjustments are proportional to how much each parameter contributed to the error
    /// - The learning rate controls how big these adjustments are
    /// 
    /// This is similar to learning from mistakes:
    /// - If a prediction was very wrong, the network makes bigger adjustments
    /// - If a prediction was mostly correct, the network makes smaller adjustments
    /// - Over time, this process leads to increasingly accurate predictions
    /// </para>
    /// </remarks>
    private void BackPropagate(Matrix<T> loss, T learningRate)
    {
        var gradient = Tensor<T>.FromMatrix(loss);
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(gradient);
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
    /// Serializes the Deep Q-Network to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to serialize to.</param>
    /// <exception cref="ArgumentNullException">Thrown when writer is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when a null layer is encountered or a layer type name cannot be determined.</exception>
    /// <remarks>
    /// <para>
    /// This method saves the state of the Deep Q-Network to a binary stream. It writes the number of layers,
    /// followed by the type name and serialized state of each layer. This allows the Deep Q-Network
    /// to be saved to disk and later restored with its trained parameters intact.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the trained network to a file.
    /// 
    /// When saving the Deep Q-Network:
    /// - First, it saves how many layers the network has
    /// - Then, for each layer, it saves:
    ///   - What type of layer it is
    ///   - All the weights and settings for that layer
    /// 
    /// This is like taking a complete snapshot of the neural network so you can:
    /// - Save your progress after training
    /// - Reload the trained network later without having to train it again
    /// - Share the trained network with others
    /// 
    /// For example, if you've trained an agent to play a game really well,
    /// you could save it and then load it later to see it play or continue training.
    /// </para>
    /// </remarks>
    public override void Serialize(BinaryWriter writer)
    {
        if (writer == null)
            throw new ArgumentNullException(nameof(writer));

        writer.Write(Layers.Count);
        foreach (var layer in Layers)
        {
            if (layer == null)
                throw new InvalidOperationException("Encountered a null layer during serialization.");

            string? fullName = layer.GetType().FullName;
            if (string.IsNullOrEmpty(fullName))
                throw new InvalidOperationException($"Unable to get full name for layer type {layer.GetType()}");

            writer.Write(fullName);
            layer.Serialize(writer);
        }
    }

    /// <summary>
    /// Deserializes the Deep Q-Network from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to deserialize from.</param>
    /// <exception cref="ArgumentNullException">Thrown when reader is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when layer type information is invalid or instance creation fails.</exception>
    /// <remarks>
    /// <para>
    /// This method restores the state of the Deep Q-Network from a binary stream. It reads the number of layers,
    /// followed by the type name and serialized state of each layer. This allows a previously saved
    /// Deep Q-Network to be restored from disk with all its trained parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved network from a file.
    /// 
    /// When loading the Deep Q-Network:
    /// - First, it reads how many layers the network had
    /// - Then, for each layer, it:
    ///   - Reads what type of layer it was
    ///   - Creates a new layer of that type
    ///   - Loads all the weights and settings for that layer
    ///   - Adds the layer to the network
    /// 
    /// This is like restoring a complete snapshot of your neural network,
    /// bringing back all the knowledge and patterns it had learned before.
    /// 
    /// For example, you could train an agent on your powerful computer,
    /// save it, and then load it on a different device to use its abilities
    /// without having to train it again.
    /// </para>
    /// </remarks>
    public override void Deserialize(BinaryReader reader)
    {
        if (reader == null)
            throw new ArgumentNullException(nameof(reader));

        int layerCount = reader.ReadInt32();
        Layers.Clear();

        for (int i = 0; i < layerCount; i++)
        {
            string layerTypeName = reader.ReadString();
            if (string.IsNullOrEmpty(layerTypeName))
                throw new InvalidOperationException("Encountered an empty layer type name during deserialization.");

            Type? layerType = Type.GetType(layerTypeName);
            if (layerType == null)
                throw new InvalidOperationException($"Cannot find type {layerTypeName}");

            if (!typeof(ILayer<T>).IsAssignableFrom(layerType))
                throw new InvalidOperationException($"Type {layerTypeName} does not implement ILayer<T>");

            object? instance = Activator.CreateInstance(layerType);
            if (instance == null)
                throw new InvalidOperationException($"Failed to create an instance of {layerTypeName}");

            var layer = (ILayer<T>)instance;
            layer.Deserialize(reader);
            Layers.Add(layer);
        }
    }
}