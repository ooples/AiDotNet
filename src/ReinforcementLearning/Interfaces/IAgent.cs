using AiDotNet.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Interfaces
{
    /// <summary>
    /// Defines the interface for a reinforcement learning agent.
    /// </summary>
    /// <typeparam name="TState">The type used to represent the environment state, typically Tensor<double>&lt;T&gt;.</typeparam>
    /// <typeparam name="TAction">The type used to represent actions, typically int for discrete actions or Vector<double>&lt;T&gt; for continuous.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations (float, double, etc.).</typeparam>
    /// <remarks>
    /// <para>
    /// An agent is the decision-maker in reinforcement learning. It observes states from the environment,
    /// selects actions based on a policy, and learns from the resulting rewards. This interface defines
    /// the core functionality that all reinforcement learning agents should implement.
    /// </para>
    /// </remarks>
    public interface IAgent<TState, TAction, T>
    {
        /// <summary>
        /// Selects an action based on the current state.
        /// </summary>
        /// <param name="state">The current state observation.</param>
        /// <param name="isTraining">A flag indicating whether the agent is in training mode.</param>
        /// <returns>The selected action.</returns>
        /// <remarks>
        /// During training (isTraining=true), the agent typically uses an exploration strategy
        /// to balance exploration and exploitation. During evaluation (isTraining=false),
        /// the agent typically selects the best action according to its policy.
        /// </remarks>
        TAction SelectAction(TState state, bool isTraining = true);

        /// <summary>
        /// Updates the agent's knowledge based on an experience tuple.
        /// </summary>
        /// <param name="state">The state before the action was taken.</param>
        /// <param name="action">The action that was taken.</param>
        /// <param name="reward">The reward received after taking the action.</param>
        /// <param name="nextState">The state after the action was taken.</param>
        /// <param name="done">A flag indicating whether the episode ended after this action.</param>
        void Learn(TState state, TAction action, T reward, TState nextState, bool done);

        /// <summary>
        /// Saves the agent's state to a file.
        /// </summary>
        /// <param name="filePath">The path where the agent's state should be saved.</param>
        void Save(string filePath);

        /// <summary>
        /// Loads the agent's state from a file.
        /// </summary>
        /// <param name="filePath">The path from which to load the agent's state.</param>
        void Load(string filePath);

        /// <summary>
        /// Gets a value indicating whether the agent is currently in training mode.
        /// </summary>
        bool IsTraining { get; }

        /// <summary>
        /// Sets the agent's training mode.
        /// </summary>
        /// <param name="isTraining">A flag indicating whether the agent should be in training mode.</param>
        void SetTrainingMode(bool isTraining);
    }
}