using System;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Compression.Pruning
{
    /// <summary>
    /// Represents a single pruned parameter from a model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This class encapsulates a pruned parameter (weight matrix or bias vector)
    /// along with its sparsity mask.
    /// </para>
    /// <para><b>For Beginners:</b> This represents one pruned weight matrix or vector.
    /// 
    /// It contains:
    /// - The values (with zeros for pruned weights)
    /// - A mask indicating which weights are pruned
    /// - Metadata about the pruning applied
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
    public class PrunedParameter<T> where T : unmanaged
    {
        /// <summary>
        /// Gets or sets the original shape of the parameter.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This is the shape (dimensions) of the parameter.
        /// </para>
        /// <para><b>For Beginners:</b> This records the dimensions of the parameter.
        /// 
        /// For example, a weight matrix might have shape [1000, 500], meaning 1000 rows and 500 columns.
        /// </para>
        /// </remarks>
        public int[] OriginalShape { get; set; } = Array.Empty<int>();

        /// <summary>
        /// Gets or sets the values for this parameter as a Tensor.
        /// </summary>
        /// <remarks>
        /// <para>
        /// These are the parameter values, with zeros for pruned weights.
        /// </para>
        /// <para><b>For Beginners:</b> These are the actual values of the weights.
        /// 
        /// This includes all weights, but pruned weights are set to zero.
        /// </para>
        /// </remarks>
        public Tensor<T> Values { get; set; } = new Tensor<T>(new int[] { 0 });

        /// <summary>
        /// Gets or sets the pruning mask as a Tensor.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This is a binary mask where 1 indicates a weight is kept and 0 indicates it is pruned.
        /// </para>
        /// <para><b>For Beginners:</b> This shows which weights are kept and which are pruned.
        /// 
        /// The mask has:
        /// - 1 for weights that are kept
        /// - 0 for weights that are pruned (set to zero)
        /// 
        /// This mask allows for efficient sparse operations and storage.
        /// </para>
        /// </remarks>
        public Tensor<byte> Mask { get; set; } = new Tensor<byte>(new int[] { 0 });

        /// <summary>
        /// Gets or sets the sparsity level of this parameter.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This is the fraction of weights that are pruned (set to zero).
        /// </para>
        /// <para><b>For Beginners:</b> This is the percentage of zeros in this parameter.
        /// 
        /// For example, a value of 0.7 means 70% of the weights are zero.
        /// </para>
        /// </remarks>
        public double SparsityLevel { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether structured pruning was applied.
        /// </summary>
        /// <remarks>
        /// <para>
        /// When true, pruning was applied in a structured manner (e.g., entire rows or columns).
        /// </para>
        /// <para><b>For Beginners:</b> This indicates if entire groups of weights were pruned together.
        /// 
        /// Structured pruning means:
        /// - Instead of pruning individual weights
        /// - Entire structures (channels, filters, neurons) are pruned
        /// - This can be more hardware-friendly but less flexible
        /// </para>
        /// </remarks>
        public bool IsStructured { get; set; }
    }
}