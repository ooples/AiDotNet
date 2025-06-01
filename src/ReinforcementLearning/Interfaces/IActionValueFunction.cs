using AiDotNet.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Interfaces
{
    /// <summary>
    /// Defines the interface for a reinforcement learning action-value function (Q-function).
    /// </summary>
    /// <typeparam name="TState">The type used to represent the environment state, typically Tensor<double>&lt;T&gt;.</typeparam>
    /// <typeparam name="TAction">The type used to represent actions, typically int for discrete actions or Vector<double>&lt;T&gt; for continuous.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations (float, double, etc.).</typeparam>
    /// <remarks>
    /// <para>
    /// An action-value function (Q-function) estimates the expected future reward from taking a specific action
    /// in a specific state and then following a policy. This interface extends IValueFunction to add methods
    /// for working with state-action pairs.
    /// </para>
    /// </remarks>
    public interface IActionValueFunction<TState, TAction, T> : IValueFunction<TState, T>
    {
        /// <summary>
        /// Predicts the Q-value for a specific state-action pair.
        /// </summary>
        /// <param name="state">The state.</param>
        /// <param name="action">The action.</param>
        /// <returns>The predicted Q-value.</returns>
        T PredictQValue(TState state, TAction action);

        /// <summary>
        /// Predicts Q-values for a batch of state-action pairs.
        /// </summary>
        /// <param name="states">The batch of states.</param>
        /// <param name="actions">The batch of actions.</param>
        /// <returns>The predicted Q-values for each state-action pair.</returns>
        Vector<T> PredictQValues(TState[] states, TAction[] actions);

        /// <summary>
        /// Predicts Q-values for all possible actions in a given state.
        /// </summary>
        /// <param name="state">The state.</param>
        /// <returns>A vector of Q-values, one for each possible action.</returns>
        Vector<T> PredictQValues(TState state);

        /// <summary>
        /// Predicts Q-values for all possible actions for a batch of states.
        /// </summary>
        /// <param name="states">The batch of states.</param>
        /// <returns>A matrix of Q-values, where each row corresponds to a state and each column to an action.</returns>
        Matrix<T> PredictQValuesBatch(TState[] states);

        /// <summary>
        /// Gets the best action for a given state (the action with the highest Q-value).
        /// </summary>
        /// <param name="state">The state.</param>
        /// <returns>The best action.</returns>
        TAction GetBestAction(TState state);

        /// <summary>
        /// Updates the Q-function based on target Q-values for specific state-action pairs.
        /// </summary>
        /// <param name="states">The states.</param>
        /// <param name="actions">The actions taken in each state.</param>
        /// <param name="targets">The target Q-values for each state-action pair.</param>
        /// <returns>The loss value after the update.</returns>
        T UpdateQ(TState[] states, TAction[] actions, Vector<T> targets);

        /// <summary>
        /// Updates the Q-function based on target Q-values for specific state-action pairs.
        /// </summary>
        /// <param name="states">The states.</param>
        /// <param name="actions">The actions taken in each state.</param>
        /// <param name="targets">The target Q-values for each state-action pair.</param>
        /// <param name="weights">Optional importance sampling weights for prioritized replay.</param>
        /// <returns>The loss value after the update.</returns>
        T Update(TState[] states, TAction[] actions, Vector<T> targets, T[]? weights = null);

        /// <summary>
        /// Computes the gradients of the Q-value with respect to the action.
        /// </summary>
        /// <param name="state">The state to evaluate.</param>
        /// <param name="action">The action to evaluate.</param>
        /// <returns>The gradients of the Q-value with respect to each action dimension.</returns>
        /// <remarks>
        /// <para>
        /// This method is used in actor-critic methods like DDPG and TD3, where the policy (actor)
        /// is updated using the gradients of the Q-value function (critic) with respect to the actions.
        /// </para>
        /// </remarks>
        Vector<T> ActionGradients(TState state, TAction action);

        /// <summary>
        /// Copies the parameters from another action-value function.
        /// </summary>
        /// <param name="source">The source action-value function from which to copy parameters.</param>
        void CopyParametersFrom(IActionValueFunction<TState, TAction, T> source);

        /// <summary>
        /// Performs a soft update of parameters from another action-value function.
        /// </summary>
        /// <param name="source">The source action-value function from which to update parameters.</param>
        /// <param name="tau">The soft update factor (between 0 and 1).</param>
        /// <remarks>
        /// <para>
        /// A soft update blends the parameters of the target network with the source network:
        /// target_params = (1 - tau) * target_params + tau * source_params
        /// This is commonly used in algorithms like DDPG and TD3 to gradually update target networks.
        /// </para>
        /// </remarks>
        void SoftUpdate(IActionValueFunction<TState, TAction, T> source, T tau);

        /// <summary>
        /// Gets the number of actions in the action space (for discrete action spaces)
        /// or the dimensionality of the action space (for continuous action spaces).
        /// </summary>
        int ActionSize { get; }

        /// <summary>
        /// Gets a value indicating whether the action space is continuous.
        /// </summary>
        bool IsContinuous { get; }
    }
}