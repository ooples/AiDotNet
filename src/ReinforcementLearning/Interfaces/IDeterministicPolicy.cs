using AiDotNet.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Interfaces
{
    /// <summary>
    /// Defines the interface for a deterministic policy in reinforcement learning.
    /// </summary>
    /// <typeparam name="TState">The type used to represent the environment state, typically Tensor<double>&lt;T&gt;.</typeparam>
    /// <typeparam name="TAction">The type used to represent actions, typically Vector<double>&lt;T&gt;.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations (float, double, etc.).</typeparam>
    /// <remarks>
    /// <para>
    /// A deterministic policy maps states directly to actions without any stochasticity.
    /// This is useful in certain algorithms like DDPG and TD3 that operate on continuous action spaces.
    /// </para>
    /// </remarks>
    public interface IDeterministicPolicy<TState, TAction, T> : IPolicy<TState, TAction, T>
       
    {
        /// <summary>
        /// Updates the policy parameters using policy gradients from a critic.
        /// </summary>
        /// <param name="policyGradients">A list of state-action gradient pairs.</param>
        /// <param name="useGradientClipping">Whether to clip gradients to prevent large updates.</param>
        /// <param name="maxGradientNorm">The maximum gradient norm for clipping.</param>
        /// <remarks>
        /// <para>
        /// This method is used in actor-critic methods where the policy (actor) is updated
        /// based on the gradients from the critic with respect to the actions.
        /// </para>
        /// </remarks>
        void UpdateFromPolicyGradients(List<(TState state, TAction actionGradient)> policyGradients, bool useGradientClipping, T maxGradientNorm);
        
        /// <summary>
        /// Copies the parameters from another policy.
        /// </summary>
        /// <param name="source">The source policy from which to copy parameters.</param>
        void CopyParametersFrom(IDeterministicPolicy<TState, TAction, T> source);
        
        /// <summary>
        /// Performs a soft update of parameters from another policy.
        /// </summary>
        /// <param name="source">The source policy from which to update parameters.</param>
        /// <param name="tau">The soft update factor (between 0 and 1).</param>
        /// <remarks>
        /// <para>
        /// A soft update blends the parameters of the target network with the source network:
        /// target_params = (1 - tau) * target_params + tau * source_params
        /// This is commonly used in algorithms like DDPG and TD3 to gradually update target networks.
        /// </para>
        /// </remarks>
        void SoftUpdate(IDeterministicPolicy<TState, TAction, T> source, T tau);
    }
}