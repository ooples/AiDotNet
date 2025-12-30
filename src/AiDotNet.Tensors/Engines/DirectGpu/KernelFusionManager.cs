// Copyright (c) AiDotNet. All rights reserved.
// Generalized kernel fusion framework for combining GPU operations.
// Tracks operation sequences and matches against registered fused kernels.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu
{
    /// <summary>
    /// Represents a type of GPU operation that can be fused.
    /// </summary>
    public enum GpuOperationType
    {
        /// <summary>General matrix multiplication.</summary>
        Gemm = 0,
        /// <summary>Bias addition.</summary>
        BiasAdd = 1,
        /// <summary>ReLU activation.</summary>
        ReLU = 2,
        /// <summary>GELU activation.</summary>
        GELU = 3,
        /// <summary>Sigmoid activation.</summary>
        Sigmoid = 4,
        /// <summary>Tanh activation.</summary>
        Tanh = 5,
        /// <summary>Softmax activation.</summary>
        Softmax = 6,
        /// <summary>Element-wise addition.</summary>
        Add = 7,
        /// <summary>Element-wise multiplication.</summary>
        Multiply = 8,
        /// <summary>Scalar multiplication.</summary>
        Scale = 9,
        /// <summary>Layer normalization.</summary>
        LayerNorm = 10,
        /// <summary>Batch normalization.</summary>
        BatchNorm = 11,
        /// <summary>Dropout.</summary>
        Dropout = 12,
        /// <summary>Residual connection (skip connection).</summary>
        Residual = 13
    }

    /// <summary>
    /// Represents an operation in a sequence that can potentially be fused.
    /// </summary>
    public readonly struct FusableOperation : IEquatable<FusableOperation>
    {
        /// <summary>
        /// Gets the type of operation.
        /// </summary>
        public GpuOperationType OperationType { get; }

        /// <summary>
        /// Gets optional metadata for the operation (e.g., kernel name suffix).
        /// </summary>
        public string Metadata { get; }

        /// <summary>
        /// Creates a new fusable operation.
        /// </summary>
        /// <param name="operationType">The operation type.</param>
        /// <param name="metadata">Optional metadata.</param>
        public FusableOperation(GpuOperationType operationType, string metadata = "")
        {
            OperationType = operationType;
            Metadata = metadata ?? string.Empty;
        }

        /// <inheritdoc/>
        public bool Equals(FusableOperation other)
        {
            return OperationType == other.OperationType && Metadata == other.Metadata;
        }

        /// <inheritdoc/>
        public override bool Equals(object? obj)
        {
            return obj is FusableOperation other && Equals(other);
        }

        /// <inheritdoc/>
        public override int GetHashCode()
        {
            unchecked
            {
                return ((int)OperationType * 397) ^ Metadata.GetHashCode();
            }
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return string.IsNullOrEmpty(Metadata) ? OperationType.ToString() : $"{OperationType}({Metadata})";
        }

        public static bool operator ==(FusableOperation left, FusableOperation right) => left.Equals(right);
        public static bool operator !=(FusableOperation left, FusableOperation right) => !left.Equals(right);
    }

    /// <summary>
    /// Represents a pattern of operations that can be fused into a single kernel.
    /// </summary>
    public sealed class FusionPattern
    {
        /// <summary>
        /// Gets the sequence of operations in this pattern.
        /// </summary>
        public IReadOnlyList<FusableOperation> Operations { get; }

        /// <summary>
        /// Gets the name of the fused kernel.
        /// </summary>
        public string FusedKernelName { get; }

        /// <summary>
        /// Gets a human-readable description of the fusion.
        /// </summary>
        public string Description { get; }

        /// <summary>
        /// Gets the expected performance benefit (e.g., "20-50%").
        /// </summary>
        public string ExpectedSpeedup { get; }

        /// <summary>
        /// Creates a new fusion pattern.
        /// </summary>
        /// <param name="operations">The operation sequence.</param>
        /// <param name="fusedKernelName">The kernel name.</param>
        /// <param name="description">Human-readable description.</param>
        /// <param name="expectedSpeedup">Expected performance benefit.</param>
        public FusionPattern(
            IReadOnlyList<FusableOperation> operations,
            string fusedKernelName,
            string description = "",
            string expectedSpeedup = "")
        {
            Operations = operations ?? throw new ArgumentNullException(nameof(operations));
            FusedKernelName = fusedKernelName ?? throw new ArgumentNullException(nameof(fusedKernelName));
            Description = description ?? string.Empty;
            ExpectedSpeedup = expectedSpeedup ?? string.Empty;
        }

        /// <summary>
        /// Creates a pattern key for hashing.
        /// </summary>
        internal string GetPatternKey()
        {
            var parts = new string[Operations.Count];
            for (int i = 0; i < Operations.Count; i++)
            {
                parts[i] = Operations[i].ToString();
            }
            return string.Join("->", parts);
        }
    }

    /// <summary>
    /// Result of a fusion attempt.
    /// </summary>
    public sealed class FusionResult
    {
        /// <summary>
        /// Gets whether fusion was successful.
        /// </summary>
        public bool IsFused { get; }

        /// <summary>
        /// Gets the fused kernel name, or null if not fused.
        /// </summary>
        public string? FusedKernelName { get; }

        /// <summary>
        /// Gets the matched fusion pattern, or null if not fused.
        /// </summary>
        public FusionPattern? Pattern { get; }

        /// <summary>
        /// Gets the operations that could not be fused (when partial fusion).
        /// </summary>
        public IReadOnlyList<FusableOperation> RemainingOperations { get; }

        /// <summary>
        /// Gets the number of operations fused.
        /// </summary>
        public int FusedOperationCount { get; }

        private FusionResult(
            bool isFused,
            string? fusedKernelName,
            FusionPattern? pattern,
            IReadOnlyList<FusableOperation> remainingOperations,
            int fusedOperationCount)
        {
            IsFused = isFused;
            FusedKernelName = fusedKernelName;
            Pattern = pattern;
            RemainingOperations = remainingOperations;
            FusedOperationCount = fusedOperationCount;
        }

        /// <summary>
        /// Creates a successful fusion result.
        /// </summary>
        public static FusionResult Success(FusionPattern pattern, IReadOnlyList<FusableOperation>? remainingOperations = null)
        {
            return new FusionResult(
                isFused: true,
                fusedKernelName: pattern.FusedKernelName,
                pattern: pattern,
                remainingOperations: remainingOperations ?? Array.Empty<FusableOperation>(),
                fusedOperationCount: pattern.Operations.Count
            );
        }

        /// <summary>
        /// Creates a failed fusion result (no matching pattern).
        /// </summary>
        public static FusionResult NoFusion(IReadOnlyList<FusableOperation> operations)
        {
            return new FusionResult(
                isFused: false,
                fusedKernelName: null,
                pattern: null,
                remainingOperations: operations,
                fusedOperationCount: 0
            );
        }
    }

    /// <summary>
    /// Manages kernel fusion by tracking operation sequences and matching against registered fused kernels.
    /// </summary>
    /// <remarks>
    /// <para><b>Design Philosophy:</b></para>
    /// <para>
    /// This framework provides a generalized approach to kernel fusion by:
    /// 1. Registering available fused kernels with their operation patterns
    /// 2. Tracking incoming operation sequences
    /// 3. Pattern matching to find optimal fusion opportunities
    /// 4. Falling back to individual kernels when no fusion is available
    /// </para>
    /// <para><b>Performance Benefit:</b></para>
    /// <para>
    /// Kernel fusion eliminates memory round-trips between operations, providing
    /// 20-50% performance improvement for memory-bound workloads. For example,
    /// fusing GEMM + Bias + ReLU saves two global memory reads/writes.
    /// </para>
    /// </remarks>
    public sealed class KernelFusionManager
    {
        private readonly Dictionary<string, FusionPattern> _patternsByKey;
        private readonly List<FusionPattern> _allPatterns;
        private readonly List<FusableOperation> _pendingOperations;
        private readonly object _lock = new object();

        /// <summary>
        /// Gets all registered fusion patterns.
        /// </summary>
        public IReadOnlyList<FusionPattern> RegisteredPatterns => _allPatterns;

        /// <summary>
        /// Gets the current pending operation sequence.
        /// </summary>
        public IReadOnlyList<FusableOperation> PendingOperations
        {
            get
            {
                lock (_lock)
                {
                    return _pendingOperations.ToArray();
                }
            }
        }

        /// <summary>
        /// Initializes a new kernel fusion manager with default patterns registered.
        /// </summary>
        public KernelFusionManager()
        {
            _patternsByKey = new Dictionary<string, FusionPattern>();
            _allPatterns = new List<FusionPattern>();
            _pendingOperations = new List<FusableOperation>();

            RegisterDefaultPatterns();
        }

        /// <summary>
        /// Registers the default fused kernel patterns.
        /// </summary>
        private void RegisterDefaultPatterns()
        {
            // GEMM + Bias + Activation patterns
            RegisterPattern(new FusionPattern(
                new[] {
                    new FusableOperation(GpuOperationType.Gemm),
                    new FusableOperation(GpuOperationType.BiasAdd),
                    new FusableOperation(GpuOperationType.ReLU)
                },
                "gemm_bias_relu",
                "Fused GEMM + Bias + ReLU for Dense layers",
                "20-50%"
            ));

            RegisterPattern(new FusionPattern(
                new[] {
                    new FusableOperation(GpuOperationType.Gemm),
                    new FusableOperation(GpuOperationType.BiasAdd),
                    new FusableOperation(GpuOperationType.GELU)
                },
                "gemm_bias_gelu",
                "Fused GEMM + Bias + GELU for Transformer FFN",
                "20-50%"
            ));

            RegisterPattern(new FusionPattern(
                new[] {
                    new FusableOperation(GpuOperationType.Gemm),
                    new FusableOperation(GpuOperationType.BiasAdd),
                    new FusableOperation(GpuOperationType.Sigmoid)
                },
                "gemm_bias_sigmoid",
                "Fused GEMM + Bias + Sigmoid for binary classification",
                "20-50%"
            ));

            RegisterPattern(new FusionPattern(
                new[] {
                    new FusableOperation(GpuOperationType.Gemm),
                    new FusableOperation(GpuOperationType.BiasAdd),
                    new FusableOperation(GpuOperationType.Tanh)
                },
                "gemm_bias_tanh",
                "Fused GEMM + Bias + Tanh for RNN layers",
                "20-50%"
            ));

            RegisterPattern(new FusionPattern(
                new[] {
                    new FusableOperation(GpuOperationType.Gemm),
                    new FusableOperation(GpuOperationType.BiasAdd)
                },
                "gemm_bias",
                "Fused GEMM + Bias (no activation)",
                "10-20%"
            ));

            // Sparse patterns
            RegisterPattern(new FusionPattern(
                new[] {
                    new FusableOperation(GpuOperationType.Gemm, "sparse_2_4"),
                    new FusableOperation(GpuOperationType.BiasAdd),
                    new FusableOperation(GpuOperationType.ReLU)
                },
                "sparse_gemm_bias_relu",
                "Fused sparse GEMM (2:4) + Bias + ReLU",
                "2x (from sparsity) + 20-50% (from fusion)"
            ));
        }

        /// <summary>
        /// Registers a new fusion pattern.
        /// </summary>
        /// <param name="pattern">The pattern to register.</param>
        /// <returns>True if registered, false if already exists.</returns>
        public bool RegisterPattern(FusionPattern pattern)
        {
            if (pattern == null)
                throw new ArgumentNullException(nameof(pattern));

            var key = pattern.GetPatternKey();
            lock (_lock)
            {
                if (_patternsByKey.ContainsKey(key))
                    return false;

                _patternsByKey[key] = pattern;
                _allPatterns.Add(pattern);
                return true;
            }
        }

        /// <summary>
        /// Unregisters a fusion pattern by kernel name.
        /// </summary>
        /// <param name="kernelName">The kernel name to remove.</param>
        /// <returns>True if removed, false if not found.</returns>
        public bool UnregisterPattern(string kernelName)
        {
            lock (_lock)
            {
                for (int i = _allPatterns.Count - 1; i >= 0; i--)
                {
                    if (_allPatterns[i].FusedKernelName == kernelName)
                    {
                        var pattern = _allPatterns[i];
                        _allPatterns.RemoveAt(i);
                        _patternsByKey.Remove(pattern.GetPatternKey());
                        return true;
                    }
                }
                return false;
            }
        }

        /// <summary>
        /// Adds an operation to the pending sequence.
        /// </summary>
        /// <param name="operation">The operation to add.</param>
        public void AddOperation(FusableOperation operation)
        {
            lock (_lock)
            {
                _pendingOperations.Add(operation);
            }
        }

        /// <summary>
        /// Adds multiple operations to the pending sequence.
        /// </summary>
        /// <param name="operations">The operations to add.</param>
        public void AddOperations(IEnumerable<FusableOperation> operations)
        {
            if (operations == null)
                throw new ArgumentNullException(nameof(operations));

            lock (_lock)
            {
                foreach (var op in operations)
                {
                    _pendingOperations.Add(op);
                }
            }
        }

        /// <summary>
        /// Clears the pending operation sequence.
        /// </summary>
        public void ClearPendingOperations()
        {
            lock (_lock)
            {
                _pendingOperations.Clear();
            }
        }

        /// <summary>
        /// Attempts to fuse the pending operation sequence.
        /// Returns the best matching fusion pattern, or indicates no fusion is available.
        /// </summary>
        /// <returns>The fusion result.</returns>
        public FusionResult TryFusePendingOperations()
        {
            lock (_lock)
            {
                if (_pendingOperations.Count == 0)
                    return FusionResult.NoFusion(Array.Empty<FusableOperation>());

                var ops = _pendingOperations.ToArray();
                _pendingOperations.Clear();

                return TryFuseOperations(ops);
            }
        }

        /// <summary>
        /// Attempts to fuse a specific operation sequence.
        /// </summary>
        /// <param name="operations">The operations to fuse.</param>
        /// <returns>The fusion result.</returns>
        public FusionResult TryFuseOperations(IReadOnlyList<FusableOperation> operations)
        {
            if (operations == null || operations.Count == 0)
                return FusionResult.NoFusion(Array.Empty<FusableOperation>());

            lock (_lock)
            {
                // Try exact match first
                var key = BuildPatternKey(operations);
                if (_patternsByKey.TryGetValue(key, out var exactPattern))
                {
                    return FusionResult.Success(exactPattern);
                }

                // Try longest prefix match (greedy fusion)
                return TryLongestPrefixMatch(operations);
            }
        }

        /// <summary>
        /// Checks if a specific operation sequence can be fused.
        /// </summary>
        /// <param name="operations">The operations to check.</param>
        /// <returns>True if a fused kernel exists for this sequence.</returns>
        public bool CanFuse(IReadOnlyList<FusableOperation> operations)
        {
            if (operations == null || operations.Count == 0)
                return false;

            var key = BuildPatternKey(operations);
            lock (_lock)
            {
                return _patternsByKey.ContainsKey(key);
            }
        }

        /// <summary>
        /// Gets the fused kernel name for a specific operation sequence, if available.
        /// </summary>
        /// <param name="operations">The operations to check.</param>
        /// <returns>The kernel name, or null if no fusion is available.</returns>
        public string? GetFusedKernelName(IReadOnlyList<FusableOperation> operations)
        {
            if (operations == null || operations.Count == 0)
                return null;

            var key = BuildPatternKey(operations);
            lock (_lock)
            {
                return _patternsByKey.TryGetValue(key, out var pattern) ? pattern.FusedKernelName : null;
            }
        }

        /// <summary>
        /// Gets the fused kernel name for an activation type in a GEMM+Bias+Activation pattern.
        /// </summary>
        /// <param name="activation">The activation type.</param>
        /// <returns>The kernel name for the fused pattern.</returns>
        public string? GetGemmBiasActivationKernel(ActivationType activation)
        {
            return activation switch
            {
                ActivationType.ReLU => "gemm_bias_relu",
                ActivationType.GELU => "gemm_bias_gelu",
                ActivationType.Sigmoid => "gemm_bias_sigmoid",
                ActivationType.Tanh => "gemm_bias_tanh",
                ActivationType.None => "gemm_bias",
                _ => null
            };
        }

        /// <summary>
        /// Converts an ActivationType to a FusableOperation.
        /// </summary>
        /// <param name="activation">The activation type.</param>
        /// <returns>The corresponding fusable operation.</returns>
        public static FusableOperation ActivationToOperation(ActivationType activation)
        {
            return activation switch
            {
                ActivationType.ReLU => new FusableOperation(GpuOperationType.ReLU),
                ActivationType.GELU => new FusableOperation(GpuOperationType.GELU),
                ActivationType.Sigmoid => new FusableOperation(GpuOperationType.Sigmoid),
                ActivationType.Tanh => new FusableOperation(GpuOperationType.Tanh),
                _ => throw new ArgumentException($"Cannot convert {activation} to FusableOperation", nameof(activation))
            };
        }

        /// <summary>
        /// Gets statistics about the registered fusion patterns.
        /// </summary>
        /// <returns>A formatted string with statistics.</returns>
        public string GetStatistics()
        {
            lock (_lock)
            {
                var stats = new System.Text.StringBuilder();
                stats.AppendLine($"Registered fusion patterns: {_allPatterns.Count}");
                stats.AppendLine();

                foreach (var pattern in _allPatterns)
                {
                    stats.AppendLine($"  {pattern.FusedKernelName}:");
                    stats.AppendLine($"    Pattern: {pattern.GetPatternKey()}");
                    if (!string.IsNullOrEmpty(pattern.Description))
                        stats.AppendLine($"    Description: {pattern.Description}");
                    if (!string.IsNullOrEmpty(pattern.ExpectedSpeedup))
                        stats.AppendLine($"    Expected speedup: {pattern.ExpectedSpeedup}");
                    stats.AppendLine();
                }

                return stats.ToString();
            }
        }

        /// <summary>
        /// Attempts to find the longest prefix of operations that matches a registered pattern.
        /// </summary>
        private FusionResult TryLongestPrefixMatch(IReadOnlyList<FusableOperation> operations)
        {
            // Try progressively shorter prefixes
            for (int length = operations.Count; length >= 2; length--)
            {
                var prefix = new FusableOperation[length];
                for (int i = 0; i < length; i++)
                {
                    prefix[i] = operations[i];
                }

                var key = BuildPatternKey(prefix);
                if (_patternsByKey.TryGetValue(key, out var pattern))
                {
                    // Found a match - return remaining operations
                    var remaining = new FusableOperation[operations.Count - length];
                    for (int i = 0; i < remaining.Length; i++)
                    {
                        remaining[i] = operations[length + i];
                    }
                    return FusionResult.Success(pattern, remaining);
                }
            }

            // No fusion found
            return FusionResult.NoFusion(operations);
        }

        /// <summary>
        /// Builds a pattern key from an operation sequence.
        /// </summary>
        private static string BuildPatternKey(IReadOnlyList<FusableOperation> operations)
        {
            var parts = new string[operations.Count];
            for (int i = 0; i < operations.Count; i++)
            {
                parts[i] = operations[i].ToString();
            }
            return string.Join("->", parts);
        }
    }
}
