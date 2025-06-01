namespace AiDotNet.ReinforcementLearning.Interfaces
{
    /// <summary>
    /// Defines the interface for a reinforcement learning policy.
    /// </summary>
    /// <typeparam name="TState">The type used to represent the environment state, typically Tensor<double>&lt;T&gt;.</typeparam>
    /// <typeparam name="TAction">The type used to represent actions, typically int for discrete actions or Vector<double>&lt;T&gt; for continuous.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations (float, double, etc.).</typeparam>
    /// <remarks>
    /// <para>
    /// A policy defines the agent's behavior by mapping states to actions or action probabilities.
    /// It encapsulates the agent's decision-making strategy, which may be deterministic or stochastic.
    /// </para>
    /// </remarks>
    public interface IPolicy<TState, TAction, T>
    {
        /// <summary>
        /// Selects an action based on the current state.
        /// </summary>
        /// <param name="state">The current state observation.</param>
        /// <returns>The selected action.</returns>
        TAction SelectAction(TState state);

        /// <summary>
        /// Evaluates the policy for a given state and returns action probabilities (for discrete action spaces)
        /// or action means and standard deviations (for continuous action spaces).
        /// </summary>
        /// <param name="state">The state to evaluate.</param>
        /// <returns>For discrete actions: probabilities for each action. For continuous actions: means and standard deviations.</returns>
        object EvaluatePolicy(TState state);

        /// <summary>
        /// Calculates the log probability of taking a specific action in a given state.
        /// </summary>
        /// <param name="state">The state in which the action was taken.</param>
        /// <param name="action">The action that was taken.</param>
        /// <returns>The log probability of the action.</returns>
        T LogProbability(TState state, TAction action);

        /// <summary>
        /// Updates the policy parameters using the provided gradients.
        /// </summary>
        /// <param name="gradients">The gradients for the policy parameters.</param>
        /// <param name="learningRate">The learning rate for the update.</param>
        void UpdateParameters(object gradients, T learningRate);

        /// <summary>
        /// Gets the entropy of the policy for a given state.
        /// </summary>
        /// <param name="state">The state for which to calculate the entropy.</param>
        /// <returns>The entropy value.</returns>
        T GetEntropy(TState state);

        /// <summary>
        /// Gets a value indicating whether the policy is stochastic.
        /// </summary>
        bool IsStochastic { get; }

        /// <summary>
        /// Gets a value indicating whether the policy is for continuous action spaces.
        /// </summary>
        bool IsContinuous { get; }
    }
}