using System;
using System.Collections.Generic;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.ReinforcementLearning.Policies
{
    /// <summary>
    /// Core interface for RL policies - defines how to select actions.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public interface IPolicy<T> : IDisposable
    {
        /// <summary>
        /// Selects an action given the current state.
        /// </summary>
        /// <param name="state">The current state observation.</param>
        /// <param name="training">Whether the agent is training (enables exploration).</param>
        /// <returns>The selected action vector.</returns>
        Vector<T> SelectAction(Vector<T> state, bool training = true);

        /// <summary>
        /// Computes the log probability of a given action in a given state.
        /// Used by policy gradient methods (PPO, A2C, etc.).
        /// </summary>
        /// <param name="state">The state observation.</param>
        /// <param name="action">The action taken.</param>
        /// <returns>The log probability of the action.</returns>
        T ComputeLogProb(Vector<T> state, Vector<T> action);

        /// <summary>
        /// Gets the neural networks used by this policy.
        /// </summary>
        IReadOnlyList<INeuralNetwork<T>> GetNetworks();

        /// <summary>
        /// Resets any internal state (e.g., for recurrent policies, exploration noise).
        /// </summary>
        void Reset();
    }
}
