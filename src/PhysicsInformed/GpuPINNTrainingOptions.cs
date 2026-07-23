using AiDotNet.Engines;

namespace AiDotNet.PhysicsInformed;

/// <summary>
/// Configuration options for GPU-accelerated PINN training.
/// </summary>
/// <remarks>
/// For Beginners:
/// These options control how GPU acceleration is used during physics-informed training.
/// GPU acceleration can significantly speed up training by parallelizing operations
/// across thousands of collocation points simultaneously.
///
/// Key settings:
/// - EnableGpu: Master switch for GPU acceleration
/// - BatchSizeGpu: Larger batches benefit more from GPU parallelism
/// - ParallelDerivativeComputation: Compute derivatives for multiple points at once
/// - AsyncTransfers: Overlap CPU/GPU data transfers with computation
/// </remarks>
public class GpuPINNTrainingOptions
{
    /// <summary>
    /// Gets or sets whether to enable GPU acceleration (default: true if GPU available).
    /// </summary>
    public bool EnableGpu { get; set; } = true;

    /// <summary>
    /// Gets or sets the GPU acceleration configuration.
    /// </summary>
    public GpuAccelerationConfig GpuConfig { get; set; } = new GpuAccelerationConfig();

    /// <summary>
    /// Gets or sets the batch size for GPU operations.
    /// Larger batches provide better GPU utilization but require more memory.
    /// Default is 1024, which works well for most GPUs.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// GPUs are most efficient when processing many operations at once.
    /// A larger batch size means more parallel work, but requires more GPU memory.
    /// Start with 1024 and increase if you have a high-end GPU with lots of memory.
    /// </remarks>
    public int BatchSizeGpu { get; set; } = 1024;

    /// <summary>
    /// Gets or sets whether to compute derivatives in parallel across the batch.
    /// This can significantly speed up PDE residual computation.
    /// </summary>
    public bool ParallelDerivativeComputation { get; set; } = true;

    /// <summary>
    /// Gets or sets the minimum number of collocation points to trigger GPU usage.
    /// Below this threshold, CPU may be faster due to transfer overhead.
    /// </summary>
    public int MinPointsForGpu { get; set; } = 1000;

    /// <summary>
    /// Gets or sets whether to use asynchronous GPU transfers.
    /// When enabled, data transfers overlap with computation for better performance.
    /// </summary>
    public bool AsyncTransfers { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use mixed precision (FP16) for forward/backward passes.
    /// Can provide 2x speedup on modern GPUs but with reduced precision.
    /// </summary>
    public bool UseMixedPrecision { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to pin memory for faster GPU transfers.
    /// Pinned memory allows faster CPU-GPU data transfer but uses more resources.
    /// </summary>
    public bool UsePinnedMemory { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to enable verbose GPU logging for debugging.
    /// </summary>
    public bool VerboseLogging { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of CUDA streams for parallel operations.
    /// More streams allow more parallel work but have diminishing returns.
    /// </summary>
    public int NumStreams { get; set; } = 2;

    /// <summary>
    /// Creates default options suitable for most GPUs.
    /// </summary>
    public static GpuPINNTrainingOptions Default => new GpuPINNTrainingOptions();

    /// <summary>
    /// Creates options optimized for high-end GPUs (RTX 4090, A100, H100).
    /// </summary>
    public static GpuPINNTrainingOptions HighEnd => new GpuPINNTrainingOptions
    {
        BatchSizeGpu = 4096,
        ParallelDerivativeComputation = true,
        AsyncTransfers = true,
        UseMixedPrecision = true,
        UsePinnedMemory = true,
        NumStreams = 4,
        GpuConfig = new GpuAccelerationConfig
        {
            UsageLevel = GpuUsageLevel.Aggressive
        }
    };

    /// <summary>
    /// Creates options optimized for memory-constrained GPUs.
    /// </summary>
    public static GpuPINNTrainingOptions LowMemory => new GpuPINNTrainingOptions
    {
        BatchSizeGpu = 256,
        ParallelDerivativeComputation = true,
        AsyncTransfers = false,
        UseMixedPrecision = false,
        UsePinnedMemory = false,
        NumStreams = 1,
        GpuConfig = new GpuAccelerationConfig
        {
            UsageLevel = GpuUsageLevel.Conservative
        }
    };

    /// <summary>
    /// Creates options that disable GPU entirely (CPU-only mode).
    /// </summary>
    public static GpuPINNTrainingOptions CpuOnly => new GpuPINNTrainingOptions
    {
        EnableGpu = false,
        GpuConfig = new GpuAccelerationConfig
        {
            DeviceType = GpuDeviceType.CPU
        }
    };
}
