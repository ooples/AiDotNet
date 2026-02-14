using System;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Policies.Exploration
{
    /// <summary>
    /// Interface for exploration strategies used by policies.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    [AiDotNet.Configuration.YamlConfigurable("ExplorationStrategy")]
    public interface IExplorationStrategy<T>
    {
        /// <summary>
        /// Modifies or replaces the policy's action for exploration.
        /// </summary>
        /// <param name="state">The current state.</param>
        /// <param name="policyAction">The action suggested by the policy.</param>
        /// <param name="actionSpaceSize">The number of possible actions.</param>
        /// <param name="random">Random number generator for stochastic exploration.</param>
        /// <returns>The action to take after applying exploration.</returns>
        Vector<T> GetExplorationAction(Vector<T> state, Vector<T> policyAction, int actionSpaceSize, Random random);

        /// <summary>
        /// Updates internal parameters (e.g., epsilon decay, noise reduction).
        /// Called after each training step.
        /// </summary>
        void Update();

        /// <summary>
        /// Resets internal state (e.g., for new episodes or training sessions).
        /// </summary>
        void Reset();
    }
}
