using AiDotNet.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Interfaces
{
    /// <summary>
    /// Defines the interface for a reinforcement learning environment.
    /// </summary>
    /// <typeparam name="TState">The type used to represent the environment state, typically Tensor<double>&lt;T&gt;.</typeparam>
    /// <typeparam name="TAction">The type used to represent actions, typically int for discrete actions or Vector<double>&lt;T&gt; for continuous.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations (float, double, etc.).</typeparam>
    /// <remarks>
    /// <para>
    /// An environment represents the world with which the agent interacts. The agent takes actions
    /// in the environment, and the environment responds with new states and rewards. This interface
    /// follows the standard RL loop: the agent observes the current state, selects an action, and
    /// the environment returns a new state and reward.
    /// </para>
    /// </remarks>
    public interface IEnvironment<TState, TAction, T>
    {
        /// <summary>
        /// Gets the current state of the environment.
        /// </summary>
        /// <returns>The current state observation.</returns>
        TState GetState();

        /// <summary>
        /// Takes an action in the environment and returns the result.
        /// </summary>
        /// <param name="action">The action to take.</param>
        /// <returns>A tuple containing the new state, the reward received, and a flag indicating if the episode is done.</returns>
        (TState nextState, T reward, bool done) Step(TAction action);

        /// <summary>
        /// Resets the environment to an initial state and returns that state.
        /// </summary>
        /// <returns>The initial state observation.</returns>
        TState Reset();

        /// <summary>
        /// Gets the size of the state space (number of dimensions in the state).
        /// </summary>
        int StateSize { get; }

        /// <summary>
        /// Gets the size of the action space (number of possible actions for discrete spaces,
        /// or number of dimensions for continuous action spaces).
        /// </summary>
        int ActionSize { get; }

        /// <summary>
        /// Gets a value indicating whether the action space is continuous.
        /// </summary>
        bool IsContinuous { get; }

        /// <summary>
        /// Gets the minimum values for each dimension of the action space (for continuous action spaces).
        /// </summary>
        Vector<T>? ActionLowerBound { get; }

        /// <summary>
        /// Gets the maximum values for each dimension of the action space (for continuous action spaces).
        /// </summary>
        Vector<T>? ActionUpperBound { get; }

        /// <summary>
        /// Renders the environment (optional, for visualization).
        /// </summary>
        /// <param name="mode">The rendering mode (e.g., "human", "rgb_array").</param>
        /// <returns>Rendering result, if applicable.</returns>
        object? Render(string mode = "human");

        /// <summary>
        /// Closes the environment and releases any resources.
        /// </summary>
        void Close();
    }
}