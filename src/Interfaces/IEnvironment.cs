using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a reinforcement learning environment that an agent interacts with.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// This interface defines the standard RL environment contract following the OpenAI Gym pattern.
/// All state observations and actions use AiDotNet's Vector type for consistency with the rest
/// of the library's type system.
/// </para>
/// <para><b>For Beginners:</b>
/// An environment is the "world" that the RL agent interacts with. Think of it like a video game:
/// - The agent sees the current state (like where characters are on screen)
/// - The agent takes actions (like pressing buttons)
/// - The environment responds with a new state and a reward (like points scored)
/// - The episode ends when certain conditions are met (like game over)
///
/// This interface ensures all environments work consistently with AiDotNet's RL agents.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("Environment")]
public interface IEnvironment<T>
{
    /// <summary>
    /// Gets the dimension of the observation space.
    /// </summary>
    /// <remarks>
    /// This is the length of the Vector returned by Reset() and Step().
    /// For example, CartPole has 4 dimensions: cart position, cart velocity, pole angle, pole angular velocity.
    /// </remarks>
    int ObservationSpaceDimension { get; }

    /// <summary>
    /// Gets the size of the action space (number of possible discrete actions or continuous action dimensions).
    /// </summary>
    /// <remarks>
    /// For discrete action spaces (like CartPole): this is the number of possible actions (e.g., 2 for left/right).
    /// For continuous action spaces: this is the dimensionality of the action vector.
    /// </remarks>
    int ActionSpaceSize { get; }

    /// <summary>
    /// Gets whether the action space is continuous (true) or discrete (false).
    /// </summary>
    bool IsContinuousActionSpace { get; }

    /// <summary>
    /// Resets the environment to an initial state and returns the initial observation.
    /// </summary>
    /// <returns>Initial state observation as a Vector.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Call this at the start of each episode to get a fresh starting state.
    /// Like pressing "restart" on a game.
    /// </remarks>
    Vector<T> Reset();

    /// <summary>
    /// Takes an action in the environment and returns the result.
    /// </summary>
    /// <param name="action">
    /// For discrete action spaces: a one-hot encoded Vector (length = ActionSpaceSize) or a Vector with a single element containing the action index.
    /// For continuous action spaces: a Vector of continuous values (length = ActionSpaceSize).
    /// </param>
    /// <returns>
    /// A tuple containing:
    /// - NextState: The resulting state observation
    /// - Reward: The reward received for this action
    /// - Done: Whether the episode has terminated
    /// - Info: Optional diagnostic information as a dictionary
    /// </returns>
    /// <remarks>
    /// <b>For Beginners:</b> This is like taking one action in the game - you press a button (action),
    /// and the game tells you what happened (new state, reward, whether game is over).
    /// </remarks>
    (Vector<T> NextState, T Reward, bool Done, Dictionary<string, object> Info) Step(Vector<T> action);

    /// <summary>
    /// Seeds the random number generator for reproducibility.
    /// </summary>
    /// <param name="seed">The random seed.</param>
    void Seed(int seed);

    /// <summary>
    /// Closes the environment and cleans up resources.
    /// </summary>
    void Close();
}
