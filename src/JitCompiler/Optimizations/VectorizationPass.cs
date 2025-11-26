using System.Numerics;
using AiDotNet.JitCompiler.IR;
using Operations = AiDotNet.JitCompiler.IR.Operations;

namespace AiDotNet.JitCompiler.Optimizations;

/// <summary>
/// Optimization pass that vectorizes operations using SIMD instructions.
/// </summary>
/// <remarks>
/// <para>
/// Vectorization transforms scalar operations into vector operations that process
/// multiple data elements in parallel using SIMD (Single Instruction Multiple Data)
/// instructions like AVX, AVX-512, or NEON.
/// </para>
/// <para><b>For Beginners:</b> This makes operations faster by processing multiple numbers at once.
///
/// Modern CPUs have special registers that can hold multiple numbers (vectors):
/// - SSE: 4 floats at once (128-bit)
/// - AVX: 8 floats at once (256-bit)
/// - AVX-512: 16 floats at once (512-bit)
///
/// Instead of:
///   a[0] + b[0]
///   a[1] + b[1]
///   a[2] + b[2]
///   a[3] + b[3]
///
/// We do:
///   vector_add([a[0], a[1], a[2], a[3]], [b[0], b[1], b[2], b[3]])
///
/// One instruction processes all 4 additions simultaneously!
/// This can provide 4-16x speedup for math operations.
/// </para>
/// </remarks>
public class VectorizationPass : IOptimizationPass
{
    /// <inheritdoc/>
    public string Name => "Vectorization";

    private readonly VectorizationConfig _config;

    /// <summary>
    /// Configuration for vectorization behavior.
    /// </summary>
    public class VectorizationConfig
    {
        /// <summary>Gets or sets whether to enable vectorization.</summary>
        public bool Enabled { get; set; } = true;

        /// <summary>Gets or sets the minimum tensor size for vectorization.</summary>
        public int MinTensorSize { get; set; } = 32;

        /// <summary>Gets or sets whether to use aggressive vectorization.</summary>
        public bool AggressiveMode { get; set; } = false;

        /// <summary>Gets or sets the target vector width (0 = auto-detect).</summary>
        public int TargetVectorWidth { get; set; } = 0;

        /// <summary>Gets or sets whether to vectorize reductions.</summary>
        public bool VectorizeReductions { get; set; } = true;

        /// <summary>Gets or sets whether to vectorize matrix operations.</summary>
        public bool VectorizeMatrixOps { get; set; } = true;
    }

    /// <summary>
    /// Initializes a new instance with default configuration.
    /// </summary>
    public VectorizationPass() : this(new VectorizationConfig()) { }

    /// <summary>
    /// Initializes a new instance with custom configuration.
    /// </summary>
    public VectorizationPass(VectorizationConfig config)
    {
        _config = config;
    }

    /// <summary>
    /// Gets the hardware vector width.
    /// </summary>
    public int HardwareVectorWidth =>
        _config.TargetVectorWidth > 0
            ? _config.TargetVectorWidth
            : (Vector.IsHardwareAccelerated ? Vector<float>.Count : 1);

    /// <inheritdoc/>
    public IRGraph Optimize(IRGraph graph)
    {
        if (!_config.Enabled || !Vector.IsHardwareAccelerated)
        {
            return graph;
        }

        var vectorizedOps = new List<IROp>();
        var vectorizationCount = 0;

        foreach (var op in graph.Operations)
        {
            if (CanVectorize(op))
            {
                var vectorizedOp = VectorizeOperation(op);
                vectorizedOps.Add(vectorizedOp);
                vectorizationCount++;
            }
            else
            {
                vectorizedOps.Add(op);
            }
        }

        // Create optimized graph
        var newGraph = new IRGraph
        {
            InputIds = new List<int>(graph.InputIds),
            OutputIds = new List<int>(graph.OutputIds),
            Operations = vectorizedOps,
            TensorShapes = new Dictionary<int, int[]>(graph.TensorShapes),
            Metadata = new Dictionary<string, object>(graph.Metadata)
        };

        // Add vectorization metadata
        newGraph.Metadata["Vectorization_Count"] = vectorizationCount;
        newGraph.Metadata["Vectorization_VectorWidth"] = HardwareVectorWidth;
        newGraph.Metadata["Vectorization_HardwareAccelerated"] = Vector.IsHardwareAccelerated;

        return newGraph;
    }

    /// <summary>
    /// Checks if an operation can be vectorized.
    /// </summary>
    private bool CanVectorize(IROp op)
    {
        // Check tensor size
        var totalElements = op.OutputShape.Aggregate(1, (a, b) => a * b);
        if (totalElements < _config.MinTensorSize)
            return false;

        // Check if the operation type supports vectorization
        return op switch
        {
            // Element-wise operations - excellent vectorization candidates
            Operations.AddOp => true,
            Operations.SubtractOp => true,
            Operations.ElementwiseMultiplyOp => true,
            Operations.DivideOp => true,
            Operations.NegateOp => true,

            // Math operations
            Operations.ExpOp => true,
            Operations.LogOp => true,
            Operations.SqrtOp => true,
            Operations.PowerOp => true,

            // Activations
            Operations.ReLUOp => true,
            Operations.SigmoidOp => true,
            Operations.TanhOp => true,

            // Reductions (if enabled)
            Operations.SumOp => _config.VectorizeReductions,
            Operations.MeanOp => _config.VectorizeReductions,
            Operations.ReduceMaxOp => _config.VectorizeReductions,
            Operations.ReduceMeanOp => _config.VectorizeReductions,

            // Matrix operations (if enabled)
            Operations.MatMulOp => _config.VectorizeMatrixOps && IsMatrixLargeEnough(op),

            // Fused operations
            Operations.FusedLinearOp => _config.VectorizeMatrixOps,
            Operations.FusedLinearActivationOp => _config.VectorizeMatrixOps,
            Operations.FusedElementwiseActivationOp => true,

            _ => false
        };
    }

    /// <summary>
    /// Checks if a matrix operation is large enough to benefit from vectorization.
    /// </summary>
    private bool IsMatrixLargeEnough(IROp op)
    {
        var totalElements = op.OutputShape.Aggregate(1, (a, b) => a * b);
        return totalElements >= HardwareVectorWidth * 4;
    }

    /// <summary>
    /// Creates a vectorized version of an operation.
    /// </summary>
    private IROp VectorizeOperation(IROp op)
    {
        var totalElements = op.OutputShape.Aggregate(1, (a, b) => a * b);
        var vectorWidth = HardwareVectorWidth;

        // Calculate vectorization parameters
        int numVectors = totalElements / vectorWidth;
        int remainder = totalElements % vectorWidth;

        return op switch
        {
            // Element-wise binary operations
            Operations.AddOp add => CreateVectorizedBinaryOp(add, "Add", vectorWidth, numVectors, remainder),
            Operations.SubtractOp sub => CreateVectorizedBinaryOp(sub, "Subtract", vectorWidth, numVectors, remainder),
            Operations.ElementwiseMultiplyOp mul => CreateVectorizedBinaryOp(mul, "Multiply", vectorWidth, numVectors, remainder),
            Operations.DivideOp div => CreateVectorizedBinaryOp(div, "Divide", vectorWidth, numVectors, remainder),

            // Element-wise unary operations
            Operations.NegateOp neg => CreateVectorizedUnaryOp(neg, "Negate", vectorWidth, numVectors, remainder),
            Operations.ExpOp exp => CreateVectorizedUnaryOp(exp, "Exp", vectorWidth, numVectors, remainder),
            Operations.LogOp log => CreateVectorizedUnaryOp(log, "Log", vectorWidth, numVectors, remainder),
            Operations.SqrtOp sqrt => CreateVectorizedUnaryOp(sqrt, "Sqrt", vectorWidth, numVectors, remainder),

            // Activations
            Operations.ReLUOp relu => CreateVectorizedUnaryOp(relu, "ReLU", vectorWidth, numVectors, remainder),
            Operations.SigmoidOp sig => CreateVectorizedUnaryOp(sig, "Sigmoid", vectorWidth, numVectors, remainder),
            Operations.TanhOp tanh => CreateVectorizedUnaryOp(tanh, "Tanh", vectorWidth, numVectors, remainder),

            // Reductions
            Operations.SumOp sum => CreateVectorizedReduction(sum, "Sum", vectorWidth),
            Operations.MeanOp mean => CreateVectorizedReduction(mean, "Mean", vectorWidth),
            Operations.ReduceMaxOp max => CreateVectorizedReduction(max, "Max", vectorWidth),
            Operations.ReduceMeanOp rmean => CreateVectorizedReduction(rmean, "Mean", vectorWidth),

            // Matrix operations
            Operations.MatMulOp matmul => CreateVectorizedMatMul(matmul, vectorWidth),

            // Return original if no specific vectorization
            _ => op
        };
    }

    /// <summary>
    /// Creates a vectorized binary operation.
    /// </summary>
    private Operations.VectorizedBinaryOp CreateVectorizedBinaryOp(
        IROp original,
        string operation,
        int vectorWidth,
        int numVectors,
        int remainder)
    {
        return new Operations.VectorizedBinaryOp
        {
            OutputId = original.OutputId,
            InputIds = original.InputIds,
            OutputType = original.OutputType,
            OutputShape = original.OutputShape,
            Operation = operation,
            VectorWidth = vectorWidth,
            NumVectors = numVectors,
            Remainder = remainder
        };
    }

    /// <summary>
    /// Creates a vectorized unary operation.
    /// </summary>
    private Operations.VectorizedUnaryOp CreateVectorizedUnaryOp(
        IROp original,
        string operation,
        int vectorWidth,
        int numVectors,
        int remainder)
    {
        return new Operations.VectorizedUnaryOp
        {
            OutputId = original.OutputId,
            InputIds = original.InputIds,
            OutputType = original.OutputType,
            OutputShape = original.OutputShape,
            Operation = operation,
            VectorWidth = vectorWidth,
            NumVectors = numVectors,
            Remainder = remainder
        };
    }

    /// <summary>
    /// Creates a vectorized reduction operation.
    /// </summary>
    private Operations.VectorizedReductionOp CreateVectorizedReduction(
        IROp original,
        string reductionType,
        int vectorWidth)
    {
        int[]? axes = null;
        bool keepDims = false;

        if (original is Operations.SumOp sum)
        {
            axes = sum.Axes;
            keepDims = sum.KeepDims;
        }
        else if (original is Operations.ReduceMaxOp max)
        {
            axes = max.Axes;
            keepDims = max.KeepDims;
        }
        else if (original is Operations.ReduceMeanOp mean)
        {
            axes = mean.Axes;
            keepDims = mean.KeepDims;
        }

        return new Operations.VectorizedReductionOp
        {
            OutputId = original.OutputId,
            InputIds = original.InputIds,
            OutputType = original.OutputType,
            OutputShape = original.OutputShape,
            ReductionType = reductionType,
            VectorWidth = vectorWidth,
            Axes = axes,
            KeepDims = keepDims
        };
    }

    /// <summary>
    /// Creates a vectorized matrix multiplication operation.
    /// </summary>
    private Operations.VectorizedMatMulOp CreateVectorizedMatMul(
        Operations.MatMulOp original,
        int vectorWidth)
    {
        return new Operations.VectorizedMatMulOp
        {
            OutputId = original.OutputId,
            InputIds = original.InputIds,
            OutputType = original.OutputType,
            OutputShape = original.OutputShape,
            VectorWidth = vectorWidth,
            UseTiling = true,
            TileSize = Math.Max(16, vectorWidth * 2)
        };
    }

    /// <summary>
    /// Gets statistics about vectorization opportunities in a graph.
    /// </summary>
    public VectorizationStats GetStats(IRGraph graph)
    {
        var stats = new VectorizationStats
        {
            TotalOperations = graph.Operations.Count,
            VectorizableOperations = graph.Operations.Count(CanVectorize),
            HardwareVectorWidth = HardwareVectorWidth,
            IsHardwareAccelerated = Vector.IsHardwareAccelerated
        };

        // Calculate potential speedup
        foreach (var op in graph.Operations)
        {
            if (CanVectorize(op))
            {
                var elements = op.OutputShape.Aggregate(1, (a, b) => a * b);
                stats.TotalVectorizableElements += elements;
            }
        }

        return stats;
    }
}

/// <summary>
/// Statistics about vectorization opportunities.
/// </summary>
public class VectorizationStats
{
    /// <summary>Total number of operations in the graph.</summary>
    public int TotalOperations { get; set; }

    /// <summary>Number of operations that can be vectorized.</summary>
    public int VectorizableOperations { get; set; }

    /// <summary>Total elements that can be processed with vectors.</summary>
    public long TotalVectorizableElements { get; set; }

    /// <summary>Hardware vector width.</summary>
    public int HardwareVectorWidth { get; set; }

    /// <summary>Whether hardware acceleration is available.</summary>
    public bool IsHardwareAccelerated { get; set; }

    /// <summary>Estimated speedup from vectorization.</summary>
    public double EstimatedSpeedup
    {
        get
        {
            if (!IsHardwareAccelerated || TotalOperations == 0)
                return 1.0;

            var vectorizableRatio = (double)VectorizableOperations / TotalOperations;
            // Amdahl's law: Speedup = 1 / ((1 - P) + P/S) where P = parallel fraction, S = speedup factor
            var speedupFactor = HardwareVectorWidth * 0.7; // Account for overhead
            return 1.0 / ((1.0 - vectorizableRatio) + (vectorizableRatio / speedupFactor));
        }
    }

    /// <summary>Returns a string representation of the statistics.</summary>
    public override string ToString()
    {
        return $"Vectorization Stats: {VectorizableOperations}/{TotalOperations} ops vectorizable, " +
               $"Vector width: {HardwareVectorWidth}, " +
               $"Estimated speedup: {EstimatedSpeedup:F2}x";
    }
}

namespace AiDotNet.JitCompiler.IR.Operations
{
    /// <summary>
    /// Vectorized binary operation (Add, Subtract, Multiply, Divide).
    /// </summary>
    public class VectorizedBinaryOp : IROp
    {
        /// <summary>Gets or sets the operation type.</summary>
        public string Operation { get; set; } = "Add";

        /// <summary>Gets or sets the vector width.</summary>
        public int VectorWidth { get; set; } = 4;

        /// <summary>Gets or sets the number of full vectors to process.</summary>
        public int NumVectors { get; set; }

        /// <summary>Gets or sets the number of remaining scalar elements.</summary>
        public int Remainder { get; set; }

        /// <summary>Validates the operation.</summary>
        public override bool Validate()
        {
            if (!base.Validate()) return false;
            if (InputIds.Length != 2) return false;
            if (VectorWidth < 1) return false;
            return true;
        }

        /// <summary>Returns a string representation.</summary>
        public override string ToString()
        {
            return $"t{OutputId} = Vectorized{Operation}[width={VectorWidth}, vecs={NumVectors}](t{InputIds[0]}, t{InputIds[1]})";
        }
    }

    /// <summary>
    /// Vectorized unary operation (Negate, Exp, Log, Sqrt, ReLU, etc.).
    /// </summary>
    public class VectorizedUnaryOp : IROp
    {
        /// <summary>Gets or sets the operation type.</summary>
        public string Operation { get; set; } = "Negate";

        /// <summary>Gets or sets the vector width.</summary>
        public int VectorWidth { get; set; } = 4;

        /// <summary>Gets or sets the number of full vectors to process.</summary>
        public int NumVectors { get; set; }

        /// <summary>Gets or sets the number of remaining scalar elements.</summary>
        public int Remainder { get; set; }

        /// <summary>Validates the operation.</summary>
        public override bool Validate()
        {
            if (!base.Validate()) return false;
            if (InputIds.Length != 1) return false;
            if (VectorWidth < 1) return false;
            return true;
        }

        /// <summary>Returns a string representation.</summary>
        public override string ToString()
        {
            return $"t{OutputId} = Vectorized{Operation}[width={VectorWidth}, vecs={NumVectors}](t{InputIds[0]})";
        }
    }

    /// <summary>
    /// Vectorized reduction operation (Sum, Mean, Max).
    /// </summary>
    public class VectorizedReductionOp : IROp
    {
        /// <summary>Gets or sets the reduction type.</summary>
        public string ReductionType { get; set; } = "Sum";

        /// <summary>Gets or sets the vector width.</summary>
        public int VectorWidth { get; set; } = 4;

        /// <summary>Gets or sets the axes to reduce over.</summary>
        public int[]? Axes { get; set; }

        /// <summary>Gets or sets whether to keep reduced dimensions.</summary>
        public bool KeepDims { get; set; } = false;

        /// <summary>Validates the operation.</summary>
        public override bool Validate()
        {
            if (!base.Validate()) return false;
            if (InputIds.Length != 1) return false;
            return true;
        }

        /// <summary>Returns a string representation.</summary>
        public override string ToString()
        {
            var axesStr = Axes != null ? $"[{string.Join(",", Axes)}]" : "all";
            return $"t{OutputId} = VectorizedReduce{ReductionType}[width={VectorWidth}, axes={axesStr}](t{InputIds[0]})";
        }
    }

    /// <summary>
    /// Vectorized matrix multiplication operation.
    /// </summary>
    public class VectorizedMatMulOp : IROp
    {
        /// <summary>Gets or sets the vector width.</summary>
        public int VectorWidth { get; set; } = 4;

        /// <summary>Gets or sets whether to use tiling.</summary>
        public bool UseTiling { get; set; } = true;

        /// <summary>Gets or sets the tile size.</summary>
        public int TileSize { get; set; } = 32;

        /// <summary>Validates the operation.</summary>
        public override bool Validate()
        {
            if (!base.Validate()) return false;
            if (InputIds.Length != 2) return false;
            return true;
        }

        /// <summary>Returns a string representation.</summary>
        public override string ToString()
        {
            return $"t{OutputId} = VectorizedMatMul[width={VectorWidth}, tile={TileSize}](t{InputIds[0]}, t{InputIds[1]})";
        }
    }
}
