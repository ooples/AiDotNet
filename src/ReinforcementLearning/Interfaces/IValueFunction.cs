using AiDotNet.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Interfaces
{
    /// <summary>
    /// Defines the interface for a reinforcement learning value function.
    /// </summary>
    /// <typeparam name="TState">The type used to represent the environment state, typically Tensor<double>&lt;T&gt;.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations (float, double, etc.).</typeparam>
    /// <remarks>
    /// <para>
    /// A value function estimates the expected future reward from being in a certain state (state-value function)
    /// or from taking a specific action in a certain state (action-value function). This interface defines the
    /// common functionality for both types of value functions.
    /// </para>
    /// </remarks>
    public interface IValueFunction<TState, T>
    {
        /// <summary>
        /// Predicts the value for a given state.
        /// </summary>
        /// <param name="state">The state for which to predict the value.</param>
        /// <returns>The predicted value.</returns>
        T PredictValue(TState state);

        /// <summary>
        /// Predicts values for a batch of states.
        /// </summary>
        /// <param name="states">The batch of states.</param>
        /// <returns>The predicted values for each state.</returns>
        Vector<T> PredictValues(TState[] states);

        /// <summary>
        /// Updates the value function based on target values.
        /// </summary>
        /// <param name="states">The states for which to update values.</param>
        /// <param name="targets">The target values for each state.</param>
        /// <returns>The loss value after the update.</returns>
        T Update(TState[] states, Vector<T> targets);

        /// <summary>
        /// Gets the parameters (weights and biases) of the value function.
        /// </summary>
        /// <returns>The parameters as a flat vector.</returns>
        Vector<T> GetParameters();

        /// <summary>
        /// Sets the parameters of the value function.
        /// </summary>
        /// <param name="parameters">The new parameter values.</param>
        void SetParameters(Vector<T> parameters);

        /// <summary>
        /// Copies the parameters from another value function.
        /// </summary>
        /// <param name="source">The source value function from which to copy parameters.</param>
        void CopyParametersFrom(IValueFunction<TState, T> source);

        /// <summary>
        /// Performs a soft update of parameters from another value function.
        /// </summary>
        /// <param name="source">The source value function from which to update parameters.</param>
        /// <param name="tau">The soft update factor (between 0 and 1).</param>
        /// <remarks>
        /// A soft update blends the parameters of the target network with the source network:
        /// target_params = (1 - tau) * target_params + tau * source_params
        /// This is commonly used in algorithms like DDPG and SAC to gradually update target networks.
        /// </remarks>
        void SoftUpdate(IValueFunction<TState, T> source, T tau);
    }
}