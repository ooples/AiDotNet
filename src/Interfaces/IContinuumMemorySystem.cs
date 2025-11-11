using System;
using System.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Interface for Continuum Memory System (CMS) - a spectrum of memory modules
    /// operating at different frequencies instead of binary short/long-term memory.
    /// Core component of nested learning paradigm.
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    public interface IContinuumMemorySystem<T>
        where T : struct, IFloatingPoint<T>, IPowerFunctions<T>, IExponentialFunctions<T>
    {
        /// <summary>
        /// Stores a representation in the continuum memory at the specified frequency level.
        /// </summary>
        /// <param name="representation">The data to store</param>
        /// <param name="frequencyLevel">Memory frequency level (0 = fastest, higher = slower)</param>
        void Store(Tensor<T> representation, int frequencyLevel);

        /// <summary>
        /// Retrieves a representation from the continuum memory.
        /// </summary>
        /// <param name="query">Query tensor</param>
        /// <param name="frequencyLevel">Memory frequency level to retrieve from</param>
        /// <returns>Retrieved representation</returns>
        Tensor<T> Retrieve(Tensor<T> query, int frequencyLevel);

        /// <summary>
        /// Updates memory based on current context and frequency.
        /// </summary>
        /// <param name="context">Current context</param>
        /// <param name="updateMask">Which frequency levels to update</param>
        void Update(Tensor<T> context, bool[] updateMask);

        /// <summary>
        /// Consolidates memories across frequency levels (mimicking memory consolidation).
        /// </summary>
        void Consolidate();

        /// <summary>
        /// Gets the number of frequency levels in the continuum.
        /// </summary>
        int NumberOfFrequencyLevels { get; }

        /// <summary>
        /// Gets or sets the decay rate for each frequency level.
        /// Lower frequency levels decay faster.
        /// </summary>
        T[] DecayRates { get; set; }

        /// <summary>
        /// Gets the current memory state at each frequency level.
        /// </summary>
        Tensor<T>[] MemoryStates { get; }
    }
}
