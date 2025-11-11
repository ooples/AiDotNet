using System;
using System.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Interface for Context Flow mechanism - maintains distinct information pathways
    /// and update rates for each nested optimization problem.
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    public interface IContextFlow<T>
        where T : struct, IFloatingPoint<T>, IPowerFunctions<T>, IExponentialFunctions<T>
    {
        /// <summary>
        /// Propagates context through the flow network.
        /// </summary>
        /// <param name="input">Input context</param>
        /// <param name="currentLevel">Current optimization level</param>
        /// <returns>Transformed context</returns>
        Tensor<T> PropagateContext(Tensor<T> input, int currentLevel);

        /// <summary>
        /// Computes gradients with respect to context flow.
        /// </summary>
        /// <param name="upstreamGradient">Gradient from upstream</param>
        /// <param name="level">Level to compute gradients for</param>
        /// <returns>Context flow gradients</returns>
        Tensor<T> ComputeContextGradients(Tensor<T> upstreamGradient, int level);

        /// <summary>
        /// Updates the context flow based on multi-level optimization.
        /// </summary>
        /// <param name="gradients">Gradients for each level</param>
        /// <param name="learningRates">Learning rate per level</param>
        void UpdateFlow(Tensor<T>[] gradients, T[] learningRates);

        /// <summary>
        /// Gets the current context state for a specific level.
        /// </summary>
        /// <param name="level">Optimization level</param>
        /// <returns>Context state</returns>
        Tensor<T> GetContextState(int level);

        /// <summary>
        /// Resets the context flow to initial state.
        /// </summary>
        void Reset();

        /// <summary>
        /// Gets the number of context flow levels.
        /// </summary>
        int NumberOfLevels { get; }
    }
}
