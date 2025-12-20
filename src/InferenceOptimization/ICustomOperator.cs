using System;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.InferenceOptimization
{
    /// <summary>
    /// Defines the contract for custom operators with hardware-specific optimizations
    /// </summary>
    public interface ICustomOperator
    {
        /// <summary>
        /// Gets the unique name of the operator
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Gets the version of the operator implementation
        /// </summary>
        string Version { get; }

        /// <summary>
        /// Gets the priority level (higher values are preferred)
        /// </summary>
        int Priority { get; }

        /// <summary>
        /// Determines if the operator can run on the current platform
        /// </summary>
        bool IsSupported();

        /// <summary>
        /// Estimates the relative performance gain over reference implementation
        /// </summary>
        /// <returns>Expected speedup multiplier (e.g., 2.0 for 2x speedup)</returns>
        double EstimatedSpeedup();
    }

    /// <summary>
    /// Base interface for custom operators that work with tensors
    /// </summary>
    public interface ICustomOperator<T> : ICustomOperator where T : struct
    {
        /// <summary>
        /// Executes the operator on input tensors
        /// </summary>
        Tensor<T> Execute(params Tensor<T>[] inputs);
    }
}
