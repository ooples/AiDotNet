using AiDotNet.InferenceOptimization.IR.Common;

namespace AiDotNet.InferenceOptimization.IR.LowLevel;

/// <summary>
/// Low-Level Intermediate Representation Operation.
/// Hardware-oriented operations for efficient execution, similar to TVM's TIR or MLIR's low-level dialects.
/// </summary>
/// <remarks>
/// <para><b>Design Philosophy:</b></para>
/// <para>
/// LLIR represents operations at a hardware level, enabling fine-grained optimizations:
/// - Loop nest transformations (tiling, unrolling, fusion)
/// - Memory layout optimization
/// - Vectorization and parallelization
/// - Device-specific code generation
/// </para>
///
/// <para><b>Industry Comparison:</b></para>
/// <list type="bullet">
/// <item>TVM TIR: Imperative with loop primitives - we add richer scheduling</item>
/// <item>MLIR Linalg: Generic named operations - we add explicit buffer management</item>
/// <item>Halide: Scheduling separated from algorithm - we integrate both</item>
/// <item>Triton: Block-level programming - we support multiple granularities</item>
/// </list>
///
/// <para><b>Exceeds Standards By:</b></para>
/// <list type="bullet">
/// <item>Unified representation for CPU/GPU/TPU</item>
/// <item>Automatic vectorization width selection</item>
/// <item>Memory hierarchy awareness (L1/L2/L3/DRAM)</item>
/// <item>Built-in profiling and auto-tuning support</item>
/// <item>Cross-platform buffer management</item>
/// </list>
/// </remarks>
public abstract class LLIROp
{
    #region Identity

    /// <summary>
    /// Unique identifier for the output buffer.
    /// Defaults to -1 (invalid) to prevent silent buffer collisions from missed assignments.
    /// </summary>
    public int OutputId { get; set; } = -1;

    /// <summary>
    /// Input buffer identifiers.
    /// </summary>
    public int[] InputIds { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Operation type name.
    /// </summary>
    public virtual string OpType => GetType().Name.Replace("Op", "");

    /// <summary>
    /// Debug name.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    #endregion

    #region Type Information

    /// <summary>
    /// Output data type.
    /// </summary>
    public IRDataType OutputDataType { get; set; } = IRDataType.Float32;

    /// <summary>
    /// Output shape.
    /// </summary>
    public int[] OutputShape { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Output strides for memory access.
    /// </summary>
    public long[]? OutputStrides { get; set; }

    /// <summary>
    /// Memory layout for output.
    /// </summary>
    public MemoryLayout OutputLayout { get; set; } = MemoryLayout.RowMajor;

    #endregion

    #region Execution

    /// <summary>
    /// Target device for execution.
    /// </summary>
    public DeviceType Device { get; set; } = DeviceType.CPU;

    /// <summary>
    /// Scheduling information for this operation.
    /// </summary>
    public ScheduleInfo Schedule { get; set; } = new();

    /// <summary>
    /// Buffer allocation information.
    /// </summary>
    public BufferInfo? BufferAllocation { get; set; }

    #endregion

    #region Provenance

    /// <summary>
    /// ID of the HLIR node this was lowered from.
    /// </summary>
    public int SourceHLIRNodeId { get; set; } = -1;

    #endregion

    #region Methods

    /// <summary>
    /// Validates the operation.
    /// </summary>
    public virtual bool Validate()
    {
        if (OutputId < 0) return false;
        if (OutputShape == null || OutputShape.Length == 0) return false;
        return true;
    }

    /// <summary>
    /// Estimates the cost of this operation.
    /// </summary>
    public abstract OperationMetrics EstimateCost();

    public override string ToString()
    {
        var inputs = string.Join(", ", InputIds.Select(id => $"b{id}"));
        var shape = $"[{string.Join(", ", OutputShape)}]";
        return $"b{OutputId} = {OpType}({inputs}) : {OutputDataType}{shape}@{Device}";
    }

    #endregion
}

/// <summary>
/// Scheduling information for loop nest optimization.
/// </summary>
public class ScheduleInfo
{
    /// <summary>
    /// Tile sizes for each loop dimension.
    /// </summary>
    public int[] TileSizes { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Loop order (dimension indices).
    /// </summary>
    public int[] LoopOrder { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Parallelization axes.
    /// </summary>
    public int[] ParallelAxes { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Vectorization axis (-1 for none).
    /// </summary>
    public int VectorAxis { get; set; } = -1;

    /// <summary>
    /// Vector width (e.g., 4 for SSE, 8 for AVX, 16 for AVX-512).
    /// </summary>
    public int VectorWidth { get; set; } = 1;

    /// <summary>
    /// Unroll factor for innermost loop.
    /// </summary>
    public int UnrollFactor { get; set; } = 1;

    /// <summary>
    /// Whether to use software pipelining.
    /// </summary>
    public bool UseSoftwarePipelining { get; set; }

    /// <summary>
    /// Prefetch distance (in iterations).
    /// </summary>
    public int PrefetchDistance { get; set; }

    /// <summary>
    /// Thread block dimensions (for GPU).
    /// </summary>
    public int[] ThreadBlockDims { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Grid dimensions (for GPU).
    /// </summary>
    public int[] GridDims { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Shared memory usage in bytes (for GPU).
    /// </summary>
    public int SharedMemoryBytes { get; set; }

    /// <summary>
    /// Register pressure estimate.
    /// </summary>
    public int RegisterPressure { get; set; }

    public ScheduleInfo Clone() => new()
    {
        TileSizes = (int[])TileSizes.Clone(),
        LoopOrder = (int[])LoopOrder.Clone(),
        ParallelAxes = (int[])ParallelAxes.Clone(),
        VectorAxis = VectorAxis,
        VectorWidth = VectorWidth,
        UnrollFactor = UnrollFactor,
        UseSoftwarePipelining = UseSoftwarePipelining,
        PrefetchDistance = PrefetchDistance,
        ThreadBlockDims = (int[])ThreadBlockDims.Clone(),
        GridDims = (int[])GridDims.Clone(),
        SharedMemoryBytes = SharedMemoryBytes,
        RegisterPressure = RegisterPressure
    };
}

/// <summary>
/// Buffer allocation and memory management information.
/// </summary>
public class BufferInfo
{
    /// <summary>
    /// Buffer size in bytes.
    /// </summary>
    public long SizeBytes { get; set; }

    /// <summary>
    /// Memory alignment requirement.
    /// </summary>
    public int Alignment { get; set; } = 64; // Cache line aligned by default

    /// <summary>
    /// Memory pool ID for buffer reuse.
    /// </summary>
    public int MemoryPoolId { get; set; } = -1;

    /// <summary>
    /// Offset within memory pool.
    /// </summary>
    public long PoolOffset { get; set; }

    /// <summary>
    /// Memory hierarchy level (L1, L2, L3, DRAM, HBM).
    /// </summary>
    public MemoryLevel MemoryLevel { get; set; } = MemoryLevel.DRAM;

    /// <summary>
    /// First use index in topological order.
    /// </summary>
    public int FirstUseIndex { get; set; }

    /// <summary>
    /// Last use index in topological order.
    /// </summary>
    public int LastUseIndex { get; set; }

    /// <summary>
    /// Whether this buffer can be allocated in-place with an input.
    /// </summary>
    public bool CanInPlace { get; set; }

    /// <summary>
    /// Input buffer ID for in-place operation.
    /// </summary>
    public int InPlaceInputId { get; set; } = -1;

    /// <summary>
    /// Whether this buffer is persistent (survives across invocations).
    /// </summary>
    public bool IsPersistent { get; set; }
}

/// <summary>
/// Memory hierarchy level.
/// </summary>
public enum MemoryLevel
{
    Register,
    L1Cache,
    L2Cache,
    L3Cache,
    DRAM,
    HBM,        // High Bandwidth Memory (GPU)
    SharedMemory, // GPU shared memory
    GlobalMemory, // GPU global memory
    ConstantMemory, // GPU constant memory
    TextureMemory   // GPU texture memory
}

/// <summary>
/// Operation performance metrics.
/// </summary>
public class OperationMetrics
{
    /// <summary>
    /// Floating-point operations.
    /// </summary>
    public long FLOPs { get; set; }

    /// <summary>
    /// Integer operations.
    /// </summary>
    public long IntOps { get; set; }

    /// <summary>
    /// Memory read in bytes.
    /// </summary>
    public long MemoryRead { get; set; }

    /// <summary>
    /// Memory write in bytes.
    /// </summary>
    public long MemoryWrite { get; set; }

    /// <summary>
    /// Estimated cycles on target device.
    /// </summary>
    public long EstimatedCycles { get; set; }

    /// <summary>
    /// Estimated latency in nanoseconds.
    /// </summary>
    public long LatencyNs { get; set; }

    /// <summary>
    /// Arithmetic intensity (ops per byte).
    /// </summary>
    public double ArithmeticIntensity =>
        (MemoryRead + MemoryWrite) > 0
            ? (double)(FLOPs + IntOps) / (MemoryRead + MemoryWrite)
            : 0;

    /// <summary>
    /// Whether operation is memory-bound.
    /// </summary>
    public bool IsMemoryBound => ArithmeticIntensity < 10;

    /// <summary>
    /// Roofline model bound (theoretical max GFLOPS).
    /// </summary>
    public double RooflineGFLOPS(double peakGFLOPS, double memBandwidthGBps)
    {
        return Math.Min(peakGFLOPS, ArithmeticIntensity * memBandwidthGBps);
    }
}

#region Concrete Operations

/// <summary>
/// Matrix multiplication operation.
/// </summary>
public class MatMulOp : LLIROp
{
    public int M { get; set; }
    public int N { get; set; }
    public int K { get; set; }
    public bool TransposeA { get; set; }
    public bool TransposeB { get; set; }
    public double Alpha { get; set; } = 1.0;
    public double Beta { get; set; } = 0.0;

    public override OperationMetrics EstimateCost()
    {
        var flops = 2L * M * N * K;
        var memRead = (long)(M * K + K * N) * GetElementSize();
        var memWrite = (long)(M * N) * GetElementSize();

        return new OperationMetrics
        {
            FLOPs = flops,
            MemoryRead = memRead,
            MemoryWrite = memWrite,
            LatencyNs = flops / 100 // Rough estimate
        };
    }

    private int GetElementSize() => OutputDataType switch
    {
        IRDataType.Float32 => 4,
        IRDataType.Float64 => 8,
        IRDataType.Float16 or IRDataType.BFloat16 => 2,
        _ => 4
    };
}

/// <summary>
/// Elementwise operation.
/// </summary>
public class ElementwiseOp : LLIROp
{
    public ElementwiseOpType ElementwiseType { get; set; }

    public override OperationMetrics EstimateCost()
    {
        var elements = OutputShape.Aggregate(1L, (a, b) => a * b);
        // Use proper element size based on data type (Float16=2, Float32=4, Float64=8, etc.)
        var elemSize = OutputDataType.ElementSizeInBytes();

        return new OperationMetrics
        {
            FLOPs = elements * (ElementwiseType == ElementwiseOpType.FusedMultiplyAdd ? 2 : 1),
            MemoryRead = elements * elemSize * InputIds.Length,
            MemoryWrite = elements * elemSize,
            // Use ceiling division to ensure non-zero latency for small arrays
            LatencyNs = Math.Max(1, (elements + 999) / 1000)
        };
    }
}

public enum ElementwiseOpType
{
    Add, Subtract, Multiply, Divide,
    Exp, Log, Sqrt, Rsqrt,
    ReLU, Sigmoid, Tanh, GELU, SiLU, Swish,
    Max, Min, Abs, Neg,
    FusedMultiplyAdd,
    Compare, Select,
    Softmax, LogSoftmax,
    Identity
}

/// <summary>
/// Convolution operation.
/// </summary>
public class Conv2DOp : LLIROp
{
    public int BatchSize { get; set; }
    public int InputChannels { get; set; }
    public int OutputChannels { get; set; }
    public int InputHeight { get; set; }
    public int InputWidth { get; set; }
    public int KernelHeight { get; set; }
    public int KernelWidth { get; set; }
    public int StrideH { get; set; } = 1;
    public int StrideW { get; set; } = 1;
    public int PadH { get; set; }
    public int PadW { get; set; }
    public int DilationH { get; set; } = 1;
    public int DilationW { get; set; } = 1;
    public int Groups { get; set; } = 1;
    public ConvAlgorithm Algorithm { get; set; } = ConvAlgorithm.Auto;

    public override OperationMetrics EstimateCost()
    {
        var outH = (InputHeight + 2 * PadH - DilationH * (KernelHeight - 1) - 1) / StrideH + 1;
        var outW = (InputWidth + 2 * PadW - DilationW * (KernelWidth - 1) - 1) / StrideW + 1;

        // 2 * (multiply + add) per output element per kernel element
        var flops = 2L * BatchSize * OutputChannels * outH * outW *
                    (InputChannels / Groups) * KernelHeight * KernelWidth;

        // Use proper element size based on data type (Float16=2, Float32=4, Float64=8, etc.)
        var elemSize = OutputDataType.ElementSizeInBytes();
        var inputSize = (long)BatchSize * InputChannels * InputHeight * InputWidth * elemSize;
        var kernelSize = (long)OutputChannels * (InputChannels / Groups) * KernelHeight * KernelWidth * elemSize;
        var outputSize = (long)BatchSize * OutputChannels * outH * outW * elemSize;

        return new OperationMetrics
        {
            FLOPs = flops,
            MemoryRead = inputSize + kernelSize,
            MemoryWrite = outputSize,
            LatencyNs = flops / 100
        };
    }
}

public enum ConvAlgorithm
{
    Auto,
    Direct,         // Direct convolution
    Im2Col,         // Image to column + GEMM
    Winograd,       // Winograd transform (for small kernels)
    FFT,            // FFT-based (for large kernels)
    Implicit,       // Implicit GEMM (for GPU)
    TensorCore      // Using tensor cores (for GPU)
}

/// <summary>
/// Reduction operation.
/// </summary>
public class ReduceOp : LLIROp
{
    public ReduceType ReduceType { get; set; }
    public int[] Axes { get; set; } = Array.Empty<int>();
    public bool KeepDims { get; set; }

    /// <summary>
    /// Shape of the input tensor before reduction. Required for accurate cost estimation.
    /// </summary>
    public int[] InputShape { get; set; } = Array.Empty<int>();

    public override OperationMetrics EstimateCost()
    {
        // Calculate input and output element counts
        var inputElements = InputShape.Length > 0
            ? InputShape.Aggregate(1L, (a, b) => a * b)
            : OutputShape.Aggregate(1L, (a, b) => a * b); // Fallback if InputShape not set
        var outputElements = OutputShape.Aggregate(1L, (a, b) => a * b);

        // Use proper element size based on data type
        var elemSize = OutputDataType.ElementSizeInBytes();

        return new OperationMetrics
        {
            // Each output element requires processing all elements along reduced axes
            FLOPs = inputElements,
            MemoryRead = inputElements * elemSize,
            MemoryWrite = outputElements * elemSize,
            LatencyNs = Math.Max(1, inputElements / 100)
        };
    }
}

public enum ReduceType
{
    Sum, Mean, Max, Min, Prod,
    L1Norm, L2Norm, LogSumExp,
    All, Any
}

/// <summary>
/// Memory operation (copy, reshape, transpose).
/// </summary>
public class MemoryOp : LLIROp
{
    public MemoryOpType MemoryOpType { get; set; }
    public int[] Permutation { get; set; } = Array.Empty<int>(); // For transpose
    public int[] NewShape { get; set; } = Array.Empty<int>(); // For reshape

    public override OperationMetrics EstimateCost()
    {
        var elements = OutputShape.Aggregate(1L, (a, b) => a * b);
        // Use proper element size based on data type (Float16=2, Float32=4, Float64=8, etc.)
        var elemSize = OutputDataType.ElementSizeInBytes();

        return new OperationMetrics
        {
            FLOPs = 0,
            MemoryRead = elements * elemSize,
            MemoryWrite = elements * elemSize,
            LatencyNs = Math.Max(1, elements / 1000)
        };
    }
}

public enum MemoryOpType
{
    Copy,
    Reshape,
    Transpose,
    Slice,
    Concat,
    Broadcast,
    Pad,
    Gather,
    Scatter
}

/// <summary>
/// Fused operation combining multiple operations.
/// </summary>
public class FusedOp : LLIROp
{
    /// <summary>
    /// Sequence of fused operations.
    /// </summary>
    public List<LLIROp> FusedOps { get; set; } = new();

    /// <summary>
    /// Pattern name (e.g., "ConvBNReLU", "MatMulBiasGELU").
    /// </summary>
    public string FusionPattern { get; set; } = string.Empty;

    /// <summary>
    /// Additional attributes for the fused operation (e.g., pooling parameters).
    /// </summary>
    /// <remarks>
    /// Used to store operation-specific parameters that don't fit in the standard properties,
    /// such as kernel size, stride, and padding for pooling operations.
    /// </remarks>
    public Dictionary<string, object> Attributes { get; set; } = new();

    public override OperationMetrics EstimateCost()
    {
        var combined = new OperationMetrics();
        foreach (var op in FusedOps)
        {
            var opCost = op.EstimateCost();
            combined.FLOPs += opCost.FLOPs;
            combined.IntOps += opCost.IntOps;
            combined.LatencyNs += opCost.LatencyNs;
        }

        // Fusion reduces memory traffic - only read inputs once and write final output
        if (FusedOps.Count > 0)
        {
            var firstOp = FusedOps[0];
            var lastOp = FusedOps[^1];
            combined.MemoryRead = firstOp.EstimateCost().MemoryRead;
            combined.MemoryWrite = lastOp.EstimateCost().MemoryWrite;
        }

        return combined;
    }

    public override string ToString()
    {
        return $"b{OutputId} = Fused[{FusionPattern}]({string.Join(", ", InputIds.Select(id => $"b{id}"))}) : {OutputDataType}";
    }
}

/// <summary>
/// Constant/parameter loading operation.
/// </summary>
public class ConstantOp : LLIROp
{
    public byte[]? Data { get; set; }
    public bool IsParameter { get; set; }
    public string ParameterName { get; set; } = string.Empty;

    public override OperationMetrics EstimateCost()
    {
        // Use actual data length if available, otherwise calculate from shape and proper element size
        var bytes = Data?.Length ?? OutputShape.Aggregate(1L, (a, b) => a * b) * OutputDataType.ElementSizeInBytes();

        return new OperationMetrics
        {
            FLOPs = 0,
            MemoryRead = bytes,
            MemoryWrite = bytes,
            LatencyNs = Math.Max(1, bytes / 1000)
        };
    }
}

#endregion
