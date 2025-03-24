namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a single reinforcement learning experience tuple containing a state, action, reward, next state, and done flag.
/// </summary>
/// <remarks>
/// <para>
/// An Experience is a fundamental data structure in reinforcement learning algorithms that captures a single interaction
/// between an agent and its environment. It consists of five components: the current state, the action taken, the reward
/// received, the resulting next state, and a flag indicating whether the episode has ended. This tuple, often called a
/// SARSA tuple (State-Action-Reward-State-Action) or a transition, is used to train reinforcement learning agents,
/// particularly in methods like Q-learning and Deep Q-Networks (DQN).
/// </para>
/// <para><b>For Beginners:</b> An Experience is like a memory of a single decision and its outcome.
/// 
/// Think of an Experience as a learning moment that contains:
/// - State: What the situation looked like before making a decision
/// - Action: What decision was made
/// - Reward: The immediate feedback received (positive or negative)
/// - NextState: What the new situation looks like after the action
/// - Done: Whether this action ended the task or game
/// 
/// For example, if a robot is learning to navigate a maze:
/// - State might be its current position and what it can see
/// - Action might be "move forward"
/// - Reward might be +1 for moving closer to the exit, -1 for hitting a wall
/// - NextState would be its new position and view
/// - Done would be true if it reached the exit or false if it's still navigating
/// 
/// Reinforcement learning agents learn by collecting and analyzing thousands of these experiences.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for state representations and rewards, typically float or double.</typeparam>
public class Experience<T>
{
    /// <summary>
    /// Gets the state of the environment before the action was taken.
    /// </summary>
    /// <value>A vector representing the state of the environment.</value>
    /// <remarks>
    /// <para>
    /// The state represents the environment's condition or configuration at the moment the agent made its decision.
    /// In reinforcement learning, a state should contain all relevant information needed for the agent to make an
    /// optimal decision. The format and content of the state vector depend on the specific problem being solved.
    /// </para>
    /// <para><b>For Beginners:</b> This is a snapshot of the situation before making a decision.
    /// 
    /// Think of State as:
    /// - A photograph of the environment at a specific moment
    /// - All the information the agent can observe
    /// - The inputs the agent uses to decide what action to take
    /// 
    /// For example, in a game of chess, the state would be the current arrangement of all pieces on the board.
    /// In a self-driving car, the state might include sensor readings, camera images, and the car's position.
    /// </para>
    /// </remarks>
    public Vector<T> State { get; }

    /// <summary>
    /// Gets the action that was taken by the agent.
    /// </summary>
    /// <value>An integer representing the action taken.</value>
    /// <remarks>
    /// <para>
    /// The action represents the specific decision or move made by the agent in response to the observed state.
    /// Actions are typically represented as integers, with each integer corresponding to a specific possible action
    /// in the environment's action space. The mapping between integers and actual actions depends on the problem domain.
    /// </para>
    /// <para><b>For Beginners:</b> This is the decision or move that was made.
    /// 
    /// Think of Action as:
    /// - The choice the agent made from its available options
    /// - Represented as a simple number (like 0, 1, 2, 3)
    /// - Each number corresponds to a specific move or decision
    /// 
    /// For example:
    /// - In a simple grid world, 0 might mean "move up", 1 "move right", etc.
    /// - In an Atari game, each number might represent pressing a different button
    /// - In a robot, actions might be "turn left", "move forward", "grab object", etc.
    /// </para>
    /// </remarks>
    public int Action { get; }

    /// <summary>
    /// Gets the reward received after taking the action.
    /// </summary>
    /// <value>A numeric value representing the reward signal.</value>
    /// <remarks>
    /// <para>
    /// The reward is a scalar feedback signal that indicates the immediate benefit or cost of taking a specific action
    /// in a specific state. In reinforcement learning, the agent's goal is to maximize the cumulative reward over time.
    /// Rewards can be positive (indicating good outcomes), negative (indicating bad outcomes), or zero (neutral).
    /// </para>
    /// <para><b>For Beginners:</b> This is the immediate feedback received for the action.
    /// 
    /// Think of Reward as:
    /// - A score that tells the agent how good or bad its action was
    /// - Positive values mean the action was good
    /// - Negative values mean the action was bad
    /// - Zero means the action was neutral
    /// 
    /// For example:
    /// - In a game, +1 might be given for winning, -1 for losing
    /// - A robot might get +0.1 for moving closer to a goal, -1 for colliding with an obstacle
    /// - A stock trading agent might get a reward proportional to profit or loss
    /// 
    /// The reward is the primary signal that guides the agent's learning process.
    /// </para>
    /// </remarks>
    public T Reward { get; }

    /// <summary>
    /// Gets the state of the environment after the action was taken.
    /// </summary>
    /// <value>A vector representing the new state of the environment.</value>
    /// <remarks>
    /// <para>
    /// The next state represents the environment's condition or configuration after the agent's action has been executed.
    /// This new state allows the agent to observe the consequences of its action. In reinforcement learning algorithms,
    /// the next state is used to estimate future rewards and update the agent's policy or value function.
    /// </para>
    /// <para><b>For Beginners:</b> This is a snapshot of the situation after the action was taken.
    /// 
    /// Think of NextState as:
    /// - A new photograph of the environment after the action has been executed
    /// - How the world changed in response to the agent's decision
    /// - The starting point for the next decision
    /// 
    /// For example:
    /// - In chess, after moving a piece, NextState would be the new arrangement of the board
    /// - In a navigation task, it would be the new position and surroundings
    /// - In a video game, it would be the new screen the player sees
    /// 
    /// By comparing State and NextState, the agent can learn how its actions affect the environment.
    /// </para>
    /// </remarks>
    public Vector<T> NextState { get; }

    /// <summary>
    /// Gets a value indicating whether this action ended the episode.
    /// </summary>
    /// <value>True if the episode ended after this action; otherwise, false.</value>
    /// <remarks>
    /// <para>
    /// The done flag indicates whether the current episode has terminated after taking this action. An episode
    /// might end when a terminal state is reached, such as winning or losing a game, completing a task, or
    /// reaching a maximum number of steps. This flag is important for algorithms to know when to stop calculating
    /// future rewards and when to reset the environment for a new episode.
    /// </para>
    /// <para><b>For Beginners:</b> This indicates whether the task or game ended after this action.
    /// 
    /// Think of Done as:
    /// - A yes/no flag that signals if this was the final action in an episode
    /// - "True" means the episode is over (goal reached, game over, task complete, etc.)
    /// - "False" means the episode continues and more actions can be taken
    /// 
    /// For example:
    /// - In a maze, Done would be true if the agent reached the exit or got stuck
    /// - In a game, it would be true if the player won, lost, or the game timed out
    /// - In a robotic task, it would be true if the task was completed or failed
    /// 
    /// This flag helps the learning algorithm know when one episode ends and another begins.
    /// </para>
    /// </remarks>
    public bool Done { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="Experience{T}"/> class.
    /// </summary>
    /// <param name="state">The state before the action was taken.</param>
    /// <param name="action">The action that was taken.</param>
    /// <param name="reward">The reward received for taking the action.</param>
    /// <param name="nextState">The state after the action was taken.</param>
    /// <param name="done">A value indicating whether the episode ended after this action.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Experience instance with the specified components. These experiences
    /// are typically collected during the agent's interaction with the environment and stored in a replay buffer
    /// for training. In algorithms like Deep Q-Networks (DQN), experiences are sampled from this buffer to update
    /// the agent's policy in a process called experience replay.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new memory of a single interaction with the environment.
    /// 
    /// When creating a new Experience:
    /// - All five pieces of information are provided and stored
    /// - This forms a complete record of one step in the learning process
    /// - Many such experiences are collected during training
    /// - The agent learns by analyzing patterns across thousands of these experiences
    /// 
    /// Think of it like a journal entry recording: "In this situation, I took this action, 
    /// got this feedback, and ended up in this new situation."
    /// </para>
    /// </remarks>
    public Experience(Vector<T> state, int action, T reward, Vector<T> nextState, bool done)
    {
        State = state;
        Action = action;
        Reward = reward;
        NextState = nextState;
        Done = done;
    }
}