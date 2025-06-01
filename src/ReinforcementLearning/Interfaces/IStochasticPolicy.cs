using AiDotNet.LinearAlgebra;
using System.Collections.Generic;

namespace AiDotNet.ReinforcementLearning.Interfaces
{
    /// <summary>
    /// Defines the interface for a stochastic policy in reinforcement learning.
    /// </summary>
    /// <typeparam name="TState">The type used to represent the environment state, typically Tensor<double>&lt;T&gt;.</typeparam>
    /// <typeparam name="TAction">The type used to represent actions, typically Vector<double>&lt;T&gt; for continuous action spaces.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations (float, double, etc.).</typeparam>
    /// <remarks>
    /// <para>
    /// A stochastic policy maps states to a distribution over actions rather than directly to actions.
    /// This is useful in algorithms like SAC that benefit from exploration through policy stochasticity 
    /// and entropy regularization.
    /// </para>
    /// </remarks>
    public interface IStochasticPolicy<TState, TAction, T> : IPolicy<TState, TAction, T>
    {
        /// <summary>
        /// Calculates the policy gradients for actor update.
        /// </summary>
        /// <param name="state">The state.</param>
        /// <param name="action">The action.</param>
        /// <param name="qValue">The Q-value.</param>
        /// <param name="entropyCoefficient">The entropy coefficient (alpha).</param>
        /// <returns>A tuple containing the gradients information.</returns>
        /// <remarks>
        /// <para>
        /// For Gaussian policies, this would typically return gradients for the mean and log standard deviation.
        /// </para>
        /// </remarks>
        (Vector<T> meanGradient, Vector<T> logStdGradient) CalculatePolicyGradients(TState state, TAction action, T qValue, T entropyCoefficient);
        
        /// <summary>
        /// Updates the policy parameters using policy gradients with entropy regularization.
        /// </summary>
        /// <param name="policyGradients">A list of state-action-gradient tuples.</param>
        /// <param name="useGradientClipping">Whether to clip gradients to prevent large updates.</param>
        /// <param name="maxGradientNorm">The maximum gradient norm for clipping.</param>
        /// <remarks>
        /// <para>
        /// This method is commonly used in algorithms like SAC where the policy is updated
        /// with a combination of critic gradients and entropy regularization.
        /// </para>
        /// </remarks>
        void UpdateParameters(List<(TState state, Vector<T> meanGradient, Vector<T> logStdGradient)> policyGradients, bool useGradientClipping, T maxGradientNorm);
        
        /// <summary>
        /// Selects a deterministic action (the mean of the policy) for evaluation.
        /// </summary>
        /// <param name="state">The current state observation.</param>
        /// <returns>The deterministic action.</returns>
        /// <remarks>
        /// <para>
        /// This is typically used during evaluation when we want to assess the performance
        /// of the learned policy without exploration noise.
        /// </para>
        /// </remarks>
        TAction SelectDeterministicAction(TState state);
        
        /// <summary>
        /// Copies the parameters from another stochastic policy.
        /// </summary>
        /// <param name="source">The source policy from which to copy parameters.</param>
        void CopyParametersFrom(IStochasticPolicy<TState, TAction, T> source);
        
        /// <summary>
        /// Performs a soft update of parameters from another stochastic policy.
        /// </summary>
        /// <param name="source">The source policy from which to update parameters.</param>
        /// <param name="tau">The soft update factor (between 0 and 1).</param>
        /// <remarks>
        /// <para>
        /// A soft update blends the parameters of the target network with the source network:
        /// target_params = (1 - tau) * target_params + tau * source_params
        /// </para>
        /// </remarks>
        void SoftUpdate(IStochasticPolicy<TState, TAction, T> source, T tau);
        
        /// <summary>
        /// Gets all parameters of the stochastic policy as a flattened vector.
        /// </summary>
        /// <returns>A vector containing all parameters of the policy.</returns>
        Vector<T> GetParameters();
        
        /// <summary>
        /// Sets the parameters of the stochastic policy from a flattened vector.
        /// </summary>
        /// <param name="parameters">The parameters to set.</param>
        void SetParameters(Vector<T> parameters);
    }
}