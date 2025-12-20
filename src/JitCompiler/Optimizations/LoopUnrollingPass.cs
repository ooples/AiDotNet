using System.Linq;
using AiDotNet.JitCompiler.IR;
using Operations = AiDotNet.JitCompiler.IR.Operations;

namespace AiDotNet.JitCompiler.Optimizations
{

    /// <summary>
    /// Optimization pass that unrolls loops for better performance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Loop unrolling is a classic compiler optimization that replaces loops with
    /// repeated copies of the loop body. This can improve performance by:
    /// - Reducing loop overhead (counter increments, comparisons, branches)
    /// - Enabling better instruction pipelining
    /// - Allowing more aggressive optimization of the unrolled body
    /// - Improving cache utilization
    /// </para>
    /// <para><b>For Beginners:</b> Loop unrolling makes repeated operations faster.
    ///
    /// Instead of:
    /// <code>
    /// for (int i = 0; i &lt; 4; i++) {
    ///     result[i] = input[i] * 2;
    /// }
    /// </code>
    ///
    /// Unrolled version:
    /// <code>
    /// result[0] = input[0] * 2;
    /// result[1] = input[1] * 2;
    /// result[2] = input[2] * 2;
    /// result[3] = input[3] * 2;
    /// </code>
    ///
    /// Benefits:
    /// - No loop overhead (no counter, no comparisons)
    /// - CPU can execute operations in parallel (instruction-level parallelism)
    /// - Better for small, fixed-size loops
    ///
    /// In neural networks, this helps with:
    /// - Fixed-size tensor operations
    /// - Small batch processing
    /// - Vectorized operations
    /// </para>
    /// </remarks>
    public class LoopUnrollingPass : IOptimizationPass
    {
        /// <inheritdoc/>
        public string Name => "Loop Unrolling";

        private int _nextTensorId;

        /// <summary>
        /// Configuration for loop unrolling behavior.
        /// </summary>
        public class UnrollConfig
        {
            /// <summary>Maximum times to fully unroll a loop.</summary>
            public int MaxFullUnrollFactor { get; set; } = 8;

            /// <summary>Partial unroll factor for larger loops.</summary>
            public int PartialUnrollFactor { get; set; } = 4;

            /// <summary>Maximum operations to unroll (prevents code bloat).</summary>
            public int MaxOpsToUnroll { get; set; } = 100;

            /// <summary>Minimum tensor size to consider for unrolling.</summary>
            public int MinTensorSize { get; set; } = 4;

            /// <summary>Maximum tensor size for full unrolling.</summary>
            public int MaxTensorSizeForFullUnroll { get; set; } = 64;

            /// <summary>Whether to unroll sequential operations.</summary>
            public bool UnrollSequential { get; set; } = true;

            /// <summary>Whether to create unrolled fused operations.</summary>
            public bool CreateFusedUnrolled { get; set; } = true;
        }

        private readonly UnrollConfig _config;

        /// <summary>
        /// Initializes a new instance with default configuration.
        /// </summary>
        public LoopUnrollingPass() : this(new UnrollConfig()) { }

        /// <summary>
        /// Initializes a new instance with custom configuration.
        /// </summary>
        public LoopUnrollingPass(UnrollConfig config)
        {
            _config = config;
        }

        /// <inheritdoc/>
        public IRGraph Optimize(IRGraph graph)
        {
            // Initialize tensor ID counter
            _nextTensorId = graph.Operations.Any()
                ? graph.Operations.Max(op => op.OutputId) + 1
                : graph.InputIds.Any() ? graph.InputIds.Max() + 1 : 0;

            var optimizedOps = new List<IROp>();
            var processedOps = new HashSet<int>(); // Track processed operations by output ID
            var tensorMapping = new Dictionary<int, int>();

            for (int i = 0; i < graph.Operations.Count; i++)
            {
                var op = graph.Operations[i];

                if (processedOps.Contains(op.OutputId))
                    continue;

                // Check for unrollable patterns
                var unrolled = TryUnrollOperation(graph.Operations, i, processedOps, tensorMapping);

                if (unrolled != null && unrolled.Count > 0)
                {
                    optimizedOps.AddRange(unrolled);
                }
                else
                {
                    // Keep operation as-is but remap inputs
                    var remappedOp = RemapInputs(op, tensorMapping);
                    optimizedOps.Add(remappedOp);
                    processedOps.Add(op.OutputId);
                }
            }

            // Create optimized graph
            var newGraph = new IRGraph
            {
                InputIds = new List<int>(graph.InputIds),
                OutputIds = RemapOutputIds(graph.OutputIds, tensorMapping),
                Operations = optimizedOps,
                TensorShapes = new Dictionary<int, int[]>(graph.TensorShapes),
                Metadata = new Dictionary<string, object>(graph.Metadata)
            };

            // Add unrolling metadata
            newGraph.Metadata["LoopUnrolling_OriginalOps"] = graph.Operations.Count;
            newGraph.Metadata["LoopUnrolling_OptimizedOps"] = optimizedOps.Count;

            return newGraph;
        }

        /// <summary>
        /// Attempts to unroll an operation or sequence of operations.
        /// </summary>
        private List<IROp>? TryUnrollOperation(
            List<IROp> allOps,
            int startIndex,
            HashSet<int> processedOps,
            Dictionary<int, int> tensorMapping)
        {
            var op = allOps[startIndex];

            // Strategy 1: Unroll small repeated element-wise operations
            if (_config.UnrollSequential && IsUnrollableElementWise(op))
            {
                var sequence = FindUnrollableSequence(allOps, startIndex, processedOps);
                if (sequence.Count >= 2 && ShouldUnroll(sequence))
                {
                    return UnrollSequence(sequence, processedOps, tensorMapping);
                }
            }

            // Strategy 2: Create unrolled operations for small tensors
            if (_config.CreateFusedUnrolled && CanCreateUnrolledOp(op))
            {
                return CreateUnrolledOperation(op, processedOps, tensorMapping);
            }

            // Strategy 3: Unroll reduction operations
            if (IsSmallReduction(op))
            {
                return UnrollReduction(op, processedOps, tensorMapping);
            }

            return null;
        }

        /// <summary>
        /// Finds a sequence of operations that can be unrolled together.
        /// </summary>
        private List<IROp> FindUnrollableSequence(
            List<IROp> allOps,
            int startIndex,
            HashSet<int> processedOps)
        {
            var sequence = new List<IROp>();
            var startOp = allOps[startIndex];

            if (processedOps.Contains(startOp.OutputId))
                return sequence;

            sequence.Add(startOp);

            // Look for sequential operations that can be unrolled together
            var currentOutput = startOp.OutputId;

            for (int i = startIndex + 1; i < allOps.Count && sequence.Count < _config.MaxFullUnrollFactor; i++)
            {
                var nextOp = allOps[i];

                if (processedOps.Contains(nextOp.OutputId))
                    continue;

                // Check if this operation uses the current output
                if (!nextOp.InputIds.Contains(currentOutput))
                    break;

                // Check if it's an unrollable element-wise operation
                if (!IsUnrollableElementWise(nextOp))
                    break;

                // Check if the output is only used by the next operation (single consumer)
                if (CountUsages(allOps, currentOutput, processedOps) > 1)
                    break;

                sequence.Add(nextOp);
                currentOutput = nextOp.OutputId;
            }

            return sequence;
        }

        /// <summary>
        /// Checks if an operation is element-wise and unrollable.
        /// </summary>
        private bool IsUnrollableElementWise(IROp op)
        {
            return op is Operations.AddOp or
                   Operations.SubtractOp or
                   Operations.ElementwiseMultiplyOp or
                   Operations.DivideOp or
                   Operations.NegateOp or
                   Operations.ReLUOp or
                   Operations.SigmoidOp or
                   Operations.TanhOp or
                   Operations.ExpOp or
                   Operations.LogOp or
                   Operations.SqrtOp;
        }

        /// <summary>
        /// Determines if a sequence should be unrolled.
        /// </summary>
        private bool ShouldUnroll(List<IROp> sequence)
        {
            if (sequence.Count < 2)
                return false;

            // Check total output size
            var totalSize = sequence.Sum(op => op.OutputShape.Aggregate(1, (a, b) => a * b));

            // Don't unroll very large sequences
            if (totalSize > _config.MaxTensorSizeForFullUnroll * sequence.Count)
                return false;

            // Don't create too many operations
            if (sequence.Count * _config.MaxFullUnrollFactor > _config.MaxOpsToUnroll)
                return false;

            return true;
        }

        /// <summary>
        /// Unrolls a sequence of operations.
        /// </summary>
        private List<IROp> UnrollSequence(
            List<IROp> sequence,
            HashSet<int> processedOps,
            Dictionary<int, int> tensorMapping)
        {
            var result = new List<IROp>();

            // Create an unrolled fused operation
            var fusedOp = new Operations.UnrolledSequenceOp
            {
                OutputId = sequence[^1].OutputId,
                InputIds = sequence[0].InputIds,
                OutputType = sequence[^1].OutputType,
                OutputShape = sequence[^1].OutputShape,
                Operations = sequence.Select(op => op.OpType).ToList(),
                OriginalOperations = sequence.Select(op => CloneOperation(op)).ToList(),
                UnrollFactor = _config.MaxFullUnrollFactor
            };

            result.Add(fusedOp);

            // Mark all operations as processed
            foreach (var op in sequence)
            {
                processedOps.Add(op.OutputId);
                if (op != sequence[^1])
                {
                    tensorMapping[op.OutputId] = sequence[^1].OutputId;
                }
            }

            return result;
        }

        /// <summary>
        /// Checks if an operation can have an unrolled version created.
        /// </summary>
        private bool CanCreateUnrolledOp(IROp op)
        {
            // Only unroll small tensors
            var totalSize = op.OutputShape.Aggregate(1, (a, b) => a * b);

            if (totalSize < _config.MinTensorSize || totalSize > _config.MaxTensorSizeForFullUnroll)
                return false;

            // Must be element-wise
            return IsUnrollableElementWise(op);
        }

        /// <summary>
        /// Creates an unrolled version of an operation.
        /// </summary>
        private List<IROp>? CreateUnrolledOperation(
            IROp op,
            HashSet<int> processedOps,
            Dictionary<int, int> tensorMapping)
        {
            var totalSize = op.OutputShape.Aggregate(1, (a, b) => a * b);
            var unrollFactor = Math.Min(totalSize, _config.MaxFullUnrollFactor);

            var unrolledOp = new Operations.UnrolledElementwiseOp
            {
                OutputId = op.OutputId,
                InputIds = op.InputIds,
                OutputType = op.OutputType,
                OutputShape = op.OutputShape,
                BaseOperation = op.OpType,
                UnrollFactor = unrollFactor,
                TotalElements = totalSize
            };

            processedOps.Add(op.OutputId);

            return new List<IROp> { unrolledOp };
        }

        /// <summary>
        /// Checks if an operation is a small reduction that can be unrolled.
        /// </summary>
        private bool IsSmallReduction(IROp op)
        {
            if (op is not (Operations.SumOp or Operations.MeanOp or Operations.ReduceMaxOp or Operations.ReduceMeanOp))
                return false;

            var inputSize = op.InputIds.Length > 0 ? op.OutputShape.Aggregate(1, (a, b) => a * b) : 0;

            // Only unroll small reductions
            return inputSize > 0 && inputSize <= _config.MaxTensorSizeForFullUnroll;
        }

        /// <summary>
        /// Unrolls a reduction operation.
        /// </summary>
        private List<IROp>? UnrollReduction(
            IROp op,
            HashSet<int> processedOps,
            Dictionary<int, int> tensorMapping)
        {
            var unrolledOp = new Operations.UnrolledReductionOp
            {
                OutputId = op.OutputId,
                InputIds = op.InputIds,
                OutputType = op.OutputType,
                OutputShape = op.OutputShape,
                ReductionType = op.OpType,
                UnrollFactor = Math.Min(
                    op.OutputShape.Aggregate(1, (a, b) => a * b),
                    _config.MaxFullUnrollFactor)
            };

            processedOps.Add(op.OutputId);

            return new List<IROp> { unrolledOp };
        }

        /// <summary>
        /// Counts how many operations use a tensor as input.
        /// </summary>
        private int CountUsages(List<IROp> allOps, int tensorId, HashSet<int> processedOps)
        {
            return allOps.Count(op => !processedOps.Contains(op.OutputId) && op.InputIds.Contains(tensorId));
        }

        /// <summary>
        /// Remaps input tensor IDs according to the mapping.
        /// </summary>
        private IROp RemapInputs(IROp op, Dictionary<int, int> tensorMapping)
        {
            var newInputIds = op.InputIds
                .Select(id => tensorMapping.TryGetValue(id, out var newId) ? newId : id)
                .ToArray();

            op.InputIds = newInputIds;
            return op;
        }

        /// <summary>
        /// Remaps output IDs according to the mapping.
        /// </summary>
        private List<int> RemapOutputIds(List<int> outputIds, Dictionary<int, int> tensorMapping)
        {
            return outputIds
                .Select(id => tensorMapping.TryGetValue(id, out var newId) ? newId : id)
                .ToList();
        }

        /// <summary>
        /// Creates a shallow clone of an operation.
        /// </summary>
        private IROp CloneOperation(IROp op)
        {
            // Use MemberwiseClone via reflection or create new instance
            var clone = (IROp)Activator.CreateInstance(op.GetType())!;
            clone.OutputId = op.OutputId;
            clone.InputIds = op.InputIds.ToArray();
            clone.OutputType = op.OutputType;
            clone.OutputShape = op.OutputShape.ToArray();
            return clone;
        }
    }
} // namespace AiDotNet.JitCompiler.Optimizations

namespace AiDotNet.JitCompiler.IR.Operations
{
    /// <summary>
    /// Represents an unrolled sequence of operations.
    /// </summary>
    public class UnrolledSequenceOp : IROp
    {
        /// <summary>Gets or sets the list of operation types in the sequence.</summary>
        public List<string> Operations { get; set; } = new();

        /// <summary>Gets or sets the original operations.</summary>
        public List<IROp> OriginalOperations { get; set; } = new();

        /// <summary>Gets or sets the unroll factor.</summary>
        public int UnrollFactor { get; set; } = 4;

        /// <summary>Validates the operation.</summary>
        public override bool Validate()
        {
            if (!base.Validate()) return false;
            if (Operations.Count < 2) return false;
            return true;
        }

        /// <summary>Returns a string representation.</summary>
        public override string ToString()
        {
            return $"t{OutputId} = UnrolledSequence[{string.Join("->", Operations)}] x{UnrollFactor}";
        }
    }

    /// <summary>
    /// Represents an unrolled element-wise operation.
    /// </summary>
    public class UnrolledElementwiseOp : IROp
    {
        /// <summary>Gets or sets the base operation type.</summary>
        public string BaseOperation { get; set; } = "";

        /// <summary>Gets or sets the unroll factor.</summary>
        public int UnrollFactor { get; set; } = 4;

        /// <summary>Gets or sets the total number of elements.</summary>
        public int TotalElements { get; set; }

        /// <summary>Validates the operation.</summary>
        public override bool Validate()
        {
            if (!base.Validate()) return false;
            if (string.IsNullOrEmpty(BaseOperation)) return false;
            if (UnrollFactor < 2) return false;
            return true;
        }

        /// <summary>Returns a string representation.</summary>
        public override string ToString()
        {
            return $"t{OutputId} = Unrolled{BaseOperation}[factor={UnrollFactor}, elements={TotalElements}]";
        }
    }

    /// <summary>
    /// Represents an unrolled reduction operation.
    /// </summary>
    public class UnrolledReductionOp : IROp
    {
        /// <summary>Gets or sets the reduction type (Sum, Mean, Max, etc.).</summary>
        public string ReductionType { get; set; } = "Sum";

        /// <summary>Gets or sets the unroll factor.</summary>
        public int UnrollFactor { get; set; } = 4;

        /// <summary>Validates the operation.</summary>
        public override bool Validate()
        {
            if (!base.Validate()) return false;
            if (string.IsNullOrEmpty(ReductionType)) return false;
            return true;
        }

        /// <summary>Returns a string representation.</summary>
        public override string ToString()
        {
            return $"t{OutputId} = UnrolledReduce{ReductionType}[factor={UnrollFactor}]";
        }
    }
}
