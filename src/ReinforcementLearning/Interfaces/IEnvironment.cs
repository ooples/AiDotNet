using AiDotNet.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Interfaces;

/// <summary>
/// Represents an environment in which a reinforcement learning agent can interact.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// An environment defines the world in which a reinforcement learning agent operates. It provides the interface
/// for the agent to observe the current state, take actions, and receive feedback in the form of rewards.
/// The environment follows the standard OpenAI Gym-like interface, making it compatible with common RL benchmarks.
/// </para>
/// <para><b>For Beginners:</b> Think of an environment as the game or world that an AI agent plays in.
///
/// Key concepts:
/// - State: What the agent can see or sense about its surroundings
/// - Action: The moves or decisions the agent can make
/// - Reward: The feedback (positive or negative) the agent gets for its actions
/// - Episode: A complete session from start to finish (like one game playthrough)
///
/// For example, in a maze:
/// - State: The agent's current position and what it can see
/// - Actions: Move up, down, left, or right
/// - Reward: +1 for reaching the goal, -1 for hitting walls, 0 otherwise
/// - Episode ends when the agent reaches the goal or exceeds a time limit
/// </para>
/// </remarks>
public interface IEnvironment<T>
{
    /// <summary>
    /// Gets the dimension of the observation space (state space).
    /// </summary>
    /// <value>The size of the state vector.</value>
    /// <remarks>
    /// <para>
    /// The observation space dimension defines how many values are needed to represent a state.
    /// For example, a 2D grid position would have dimension 2, while an Atari game screen might
    /// have dimension 84x84x4 = 28,224 (for a processed grayscale image).
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many numbers are needed to describe what the agent sees.
    ///
    /// Examples:
    /// - A simple grid position: 2 numbers (x and y coordinates)
    /// - A robot with sensors: Maybe 10 numbers (sensor readings)
    /// - A game screen: Many thousands of numbers (one for each pixel)
    /// </para>
    /// </remarks>
    int ObservationSpaceDimension { get; }

    /// <summary>
    /// Gets the number of possible actions the agent can take.
    /// </summary>
    /// <value>The size of the action space.</value>
    /// <remarks>
    /// <para>
    /// The action space size defines how many different actions are available to the agent.
    /// For discrete action spaces, this is the total number of distinct actions.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many different moves or buttons the agent has.
    ///
    /// Examples:
    /// - Grid movement: 4 actions (up, down, left, right)
    /// - Simple game: 3 actions (jump, duck, run)
    /// - Complex game: Might have dozens of action combinations
    /// </para>
    /// </remarks>
    int ActionSpaceSize { get; }

    /// <summary>
    /// Resets the environment to an initial state and returns the starting observation.
    /// </summary>
    /// <returns>The initial state of the environment.</returns>
    /// <remarks>
    /// <para>
    /// This method should be called at the start of each episode. It resets the environment to
    /// its initial conditions and returns the starting state. The exact nature of the reset
    /// (deterministic vs. randomized) depends on the specific environment implementation.
    /// </para>
    /// <para><b>For Beginners:</b> This starts a new game or episode from the beginning.
    ///
    /// When you call Reset:
    /// - The environment goes back to its starting position
    /// - All counters and timers restart
    /// - You get the initial state to begin making decisions
    ///
    /// It's like pressing the "restart" button on a game.
    /// </para>
    /// </remarks>
    Tensor<T> Reset();

    /// <summary>
    /// Executes the specified action in the environment and returns the resulting state, reward, and done flag.
    /// </summary>
    /// <param name="action">The action to execute (index into the action space).</param>
    /// <returns>A tuple containing:
    /// - nextState: The new state after executing the action
    /// - reward: The reward received for taking this action
    /// - done: Whether the episode has ended
    /// - info: Additional diagnostic information (optional)
    /// </returns>
    /// <remarks>
    /// <para>
    /// This is the core method for environment interaction. The agent provides an action, and the
    /// environment returns the consequences: the new state, the reward signal, and whether the episode
    /// has terminated. The info dictionary can contain additional diagnostic information that is not
    /// part of the state but may be useful for debugging or analysis.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the action happens! You tell the environment what you want to do,
    /// and it tells you what happened.
    ///
    /// The return values mean:
    /// - nextState: What you see after taking the action
    /// - reward: Your score or feedback (positive = good, negative = bad)
    /// - done: Did the game/task end? (reached goal, ran out of time, etc.)
    /// - info: Extra details that might be helpful but aren't part of the main game state
    ///
    /// For example, if you tell a robot to "move forward":
    /// - nextState might show its new position
    /// - reward might be +1 if it got closer to the goal
    /// - done might be true if it reached the goal or fell off a cliff
    /// - info might include details like "collided with wall"
    /// </para>
    /// </remarks>
    (Tensor<T> nextState, T reward, bool done, Dictionary<string, object>? info) Step(int action);

    /// <summary>
    /// Renders the current state of the environment (optional).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method provides a visual or textual representation of the environment's current state.
    /// It's primarily used for debugging, visualization, and human observation. Not all environments
    /// need to implement rendering.
    /// </para>
    /// <para><b>For Beginners:</b> This shows you what's happening visually.
    ///
    /// Render is like:
    /// - Drawing the game on screen
    /// - Printing text to show the agent's position
    /// - Displaying a video of what's happening
    ///
    /// It's optional and mainly used when you want to watch the agent learn or debug issues.
    /// Many environments skip this to train faster.
    /// </para>
    /// </remarks>
    void Render();

    /// <summary>
    /// Closes the environment and cleans up resources.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method should be called when you're done using the environment to release any resources
    /// like graphics contexts, file handles, or network connections.
    /// </para>
    /// <para><b>For Beginners:</b> This cleans up when you're done.
    ///
    /// Call Close when:
    /// - You've finished all your training or testing
    /// - You want to free up computer resources
    /// - You're about to create a new environment
    ///
    /// It's like closing a program properly instead of force-quitting it.
    /// </para>
    /// </remarks>
    void Close();

    /// <summary>
    /// Seeds the environment's random number generator for reproducibility.
    /// </summary>
    /// <param name="seed">The random seed value.</param>
    /// <remarks>
    /// <para>
    /// Setting a seed ensures that the environment behaves deterministically, which is useful for
    /// debugging and comparing different algorithms on exactly the same sequence of environment states.
    /// </para>
    /// <para><b>For Beginners:</b> This makes randomness predictable.
    ///
    /// Why use a seed:
    /// - Makes experiments repeatable (same results every time)
    /// - Helps with debugging (you can replay exact scenarios)
    /// - Allows fair comparison of different learning methods
    ///
    /// Think of it like setting a "random" dice to always roll the same sequence.
    /// This way, you can test if your changes actually made things better,
    /// not just luckier.
    /// </para>
    /// </remarks>
    void Seed(int seed);
}
