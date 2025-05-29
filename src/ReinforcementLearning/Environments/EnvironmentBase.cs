using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.ReinforcementLearning.Interfaces;

namespace AiDotNet.ReinforcementLearning.Environments
{
    /// <summary>
    /// Base class for reinforcement learning environments.
    /// </summary>
    /// <typeparam name="TState">The type used to represent the environment state.</typeparam>
    /// <typeparam name="TAction">The type used to represent actions.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public abstract class EnvironmentBase<TState, TAction, T> : IEnvironment<TState, TAction, T> 
       
    {
        /// <summary>
        /// Gets the size of the state space.
        /// </summary>
        public abstract int StateSize { get; }

        /// <summary>
        /// Gets the size of the action space.
        /// </summary>
        public abstract int ActionSize { get; }

        /// <summary>
        /// Gets a value indicating whether the action space is continuous.
        /// </summary>
        public abstract bool IsContinuous { get; }

        /// <summary>
        /// Gets the minimum values for each dimension of the action space (for continuous action spaces).
        /// </summary>
        public virtual Vector<T>? ActionLowerBound => null;

        /// <summary>
        /// Gets the maximum values for each dimension of the action space (for continuous action spaces).
        /// </summary>
        public virtual Vector<T>? ActionUpperBound => null;

        /// <summary>
        /// Gets the current state of the environment.
        /// </summary>
        /// <returns>The current state observation.</returns>
        public abstract TState GetState();

        /// <summary>
        /// Takes an action in the environment and returns the result.
        /// </summary>
        /// <param name="action">The action to take.</param>
        /// <returns>A tuple containing the new state, the reward received, and a flag indicating if the episode is done.</returns>
        public abstract (TState nextState, T reward, bool done) Step(TAction action);

        /// <summary>
        /// Resets the environment to an initial state and returns that state.
        /// </summary>
        /// <returns>The initial state observation.</returns>
        public abstract TState Reset();

        /// <summary>
        /// Renders the environment (optional, for visualization).
        /// </summary>
        /// <param name="mode">The rendering mode (e.g., "human", "rgb_array").</param>
        /// <returns>Rendering result, if applicable.</returns>
        public virtual object? Render(string mode = "human")
        {
            return null;
        }

        /// <summary>
        /// Closes the environment and releases any resources.
        /// </summary>
        public virtual void Close()
        {
            // Default implementation does nothing
        }

        /// <summary>
        /// Gets information about the environment.
        /// </summary>
        /// <returns>A dictionary containing information about the environment.</returns>
        public virtual Dictionary<string, object> GetInfo()
        {
            return new Dictionary<string, object>
            {
                { "state_size", StateSize },
                { "action_size", ActionSize },
                { "is_continuous", IsContinuous }
            };
        }

        /// <summary>
        /// Validates an action to ensure it is within the allowed range.
        /// </summary>
        /// <param name="action">The action to validate.</param>
        /// <returns>The validated action (clipped to valid range if necessary).</returns>
        protected virtual TAction ValidateAction(TAction action)
        {
            return action;
        }
    }
}