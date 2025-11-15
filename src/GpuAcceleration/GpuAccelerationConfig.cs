using AiDotNet.Gpu;

namespace AiDotNet.GpuAcceleration;

/// <summary>
/// Configuration settings for GPU-accelerated training and inference.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This class contains all the settings you can adjust for GPU acceleration.
/// The default values work well for most use cases - you can just call ConfigureGpuAcceleration() without
/// parameters and it will automatically detect your GPU and use sensible defaults.
///
/// Key concepts:
/// - **Automatic Placement**: GPU decides where to run operations (GPU vs CPU) based on tensor size
/// - **GPU Threshold**: Minimum number of elements before using GPU (avoids transfer overhead)
/// - **Placement Strategy**: How to decide between CPU and GPU execution
/// - **Device Selection**: Which GPU to use if you have multiple
/// </para>
/// </remarks>
public class GpuAccelerationConfig
{
    /// <summary>
    /// Enable GPU acceleration (default: true if GPU is available).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Set to false to disable GPU and use CPU only.
    /// By default, GPU is enabled if available and disabled if not.
    /// </para>
    /// </remarks>
    public bool? EnableGpu { get; set; } = null; // null = auto-detect

    /// <summary>
    /// Minimum number of elements in a tensor before using GPU (default: 100,000).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Small operations are faster on CPU due to transfer overhead.
    /// This threshold determines when to switch to GPU. For example:
    /// - 100x100 matrix (10,000 elements) → CPU (faster due to no transfer)
    /// - 1000x1000 matrix (1,000,000 elements) → GPU (much faster computation)
    ///
    /// Adjust based on your GPU:
    /// - Fast GPU (RTX 4090, A100): Lower threshold like 50,000
    /// - Mid-range GPU (RTX 3060): Default 100,000
    /// - Older GPU: Higher threshold like 200,000
    /// </para>
    /// </remarks>
    public int GpuThreshold { get; set; } = 100_000;

    /// <summary>
    /// Strategy for deciding CPU vs GPU placement (default: AutomaticPlacement).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how operations are assigned to CPU or GPU:
    /// - **AutomaticPlacement** (recommended): Uses GPU for large tensors, CPU for small ones
    /// - **ForceGpu**: All operations on GPU (good if all your data is large)
    /// - **ForceCpu**: All operations on CPU (for debugging or no GPU)
    /// - **MinimizeTransfers**: Keep data where it is (for advanced users)
    /// - **CostBased**: Analyzes transfer vs compute cost (for advanced optimization)
    /// </para>
    /// </remarks>
    public ExecutionContext.PlacementStrategy Strategy { get; set; } = ExecutionContext.PlacementStrategy.AutomaticPlacement;

    /// <summary>
    /// GPU device type to prefer (default: Default = automatic selection).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Specifies which type of GPU to use:
    /// - **Default**: Automatically select best available (CUDA → OpenCL → CPU)
    /// - **CUDA**: Force NVIDIA CUDA (fails if not available)
    /// - **OpenCL**: Force OpenCL (AMD/Intel GPUs)
    /// - **CPU**: Force CPU execution (for debugging)
    ///
    /// Leave as Default unless you have specific requirements.
    /// </para>
    /// </remarks>
    public GpuDeviceType PreferredDeviceType { get; set; } = GpuDeviceType.Default;

    /// <summary>
    /// GPU compute speedup factor vs CPU (default: 10.0, used for CostBased strategy).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Estimate of how much faster GPU is vs CPU for computation.
    /// Only used when Strategy is CostBased. Default of 10x is conservative.
    /// You can benchmark your specific hardware to find the actual speedup.
    /// </para>
    /// </remarks>
    public double GpuComputeSpeedup { get; set; } = 10.0;

    /// <summary>
    /// PCIe transfer bandwidth in GB/s (default: 12.0, used for CostBased strategy).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Speed of data transfer between CPU and GPU.
    /// Only used when Strategy is CostBased.
    /// - PCIe 3.0 x16: ~12 GB/s
    /// - PCIe 4.0 x16: ~24 GB/s
    /// - PCIe 5.0 x16: ~48 GB/s
    /// </para>
    /// </remarks>
    public double TransferBandwidthGBps { get; set; } = 12.0;

    /// <summary>
    /// Enable verbose logging of GPU operations (default: false).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When true, prints information about which operations
    /// are running on GPU vs CPU. Useful for debugging and optimization, but can be verbose.
    /// </para>
    /// </remarks>
    public bool VerboseLogging { get; set; } = false;

    /// <summary>
    /// Enable GPU acceleration for inference (prediction) as well as training (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> GPU can accelerate both training AND inference.
    /// Set to false if you only want GPU during training but CPU during inference
    /// (e.g., for deployment to CPU-only servers).
    /// </para>
    /// </remarks>
    public bool EnableForInference { get; set; } = true;

    /// <summary>
    /// Creates a configuration with default recommended settings.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this (or just call ConfigureGpuAcceleration() with no parameters)
    /// for automatic GPU acceleration with sensible defaults. Works well for most use cases.
    /// </para>
    /// </remarks>
    public GpuAccelerationConfig()
    {
    }

    /// <summary>
    /// Creates a configuration for conservative GPU usage (higher threshold, safer for smaller GPUs).
    /// </summary>
    /// <returns>A conservative GPU acceleration configuration.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this for older or lower-end GPUs, or when GPU memory is limited.
    /// It uses GPU less aggressively, only for very large operations.
    ///
    /// Good for:
    /// - GTX 1060, GTX 1660, RTX 3050
    /// - Limited GPU memory (4GB or less)
    /// - When running other GPU applications simultaneously
    /// </para>
    /// </remarks>
    public static GpuAccelerationConfig Conservative()
    {
        return new GpuAccelerationConfig
        {
            GpuThreshold = 200_000,        // Higher threshold
            Strategy = ExecutionContext.PlacementStrategy.AutomaticPlacement,
            GpuComputeSpeedup = 8.0,       // More conservative speedup estimate
        };
    }

    /// <summary>
    /// Creates a configuration for aggressive GPU usage (lower threshold, maximum performance).
    /// </summary>
    /// <returns>An aggressive GPU acceleration configuration.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this for high-end GPUs to maximize performance.
    /// It uses GPU more aggressively, even for medium-sized operations.
    ///
    /// Good for:
    /// - RTX 4070/4080/4090, RTX 3080/3090
    /// - A100, V100, H100 datacenter GPUs
    /// - Dedicated GPU servers with plenty of GPU memory
    /// - Workstation GPUs (A6000, etc.)
    /// </para>
    /// </remarks>
    public static GpuAccelerationConfig Aggressive()
    {
        return new GpuAccelerationConfig
        {
            GpuThreshold = 50_000,          // Lower threshold
            Strategy = ExecutionContext.PlacementStrategy.AutomaticPlacement,
            GpuComputeSpeedup = 20.0,       // Higher speedup estimate for modern GPUs
            TransferBandwidthGBps = 24.0,   // Assume PCIe 4.0
        };
    }

    /// <summary>
    /// Creates a configuration that forces all operations to GPU (for maximum GPU utilization).
    /// </summary>
    /// <returns>A GPU-only configuration.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this when ALL your operations work with large tensors
    /// and you want to keep everything on GPU to minimize transfers.
    ///
    /// Good for:
    /// - Training large neural networks
    /// - Batch processing with large batches
    /// - When all operations are compute-intensive
    ///
    /// Not recommended for:
    /// - Mixed workloads with small and large tensors
    /// - Limited GPU memory
    /// - First time using GPU acceleration (start with default instead)
    /// </para>
    /// </remarks>
    public static GpuAccelerationConfig GpuOnly()
    {
        return new GpuAccelerationConfig
        {
            Strategy = ExecutionContext.PlacementStrategy.ForceGpu,
            GpuThreshold = 0,               // Ignore threshold
        };
    }

    /// <summary>
    /// Creates a configuration with GPU disabled (CPU-only execution).
    /// </summary>
    /// <returns>A CPU-only configuration.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this to disable GPU acceleration entirely.
    ///
    /// Good for:
    /// - Debugging (compare CPU vs GPU results)
    /// - Deployment to CPU-only servers
    /// - Testing code without requiring GPU
    /// - Very small models where GPU overhead isn't worth it
    /// </para>
    /// </remarks>
    public static GpuAccelerationConfig CpuOnly()
    {
        return new GpuAccelerationConfig
        {
            EnableGpu = false,
            Strategy = ExecutionContext.PlacementStrategy.ForceCpu,
        };
    }

    /// <summary>
    /// Creates a configuration for development/debugging with verbose logging.
    /// </summary>
    /// <returns>A configuration with verbose logging enabled.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this when you want to see which operations
    /// are running on GPU vs CPU. Helpful for understanding and optimizing your code.
    /// </para>
    /// </remarks>
    public static GpuAccelerationConfig Debug()
    {
        return new GpuAccelerationConfig
        {
            VerboseLogging = true,
            Strategy = ExecutionContext.PlacementStrategy.AutomaticPlacement,
        };
    }

    /// <summary>
    /// Gets a summary of the configuration.
    /// </summary>
    /// <returns>A string describing the configuration.</returns>
    public override string ToString()
    {
        return $"GpuAccelerationConfig: " +
               $"Enabled={EnableGpu?.ToString() ?? "Auto"}, " +
               $"Strategy={Strategy}, " +
               $"Threshold={GpuThreshold:N0} elements, " +
               $"Device={PreferredDeviceType}, " +
               $"Speedup={GpuComputeSpeedup:F1}x, " +
               $"Bandwidth={TransferBandwidthGBps:F1} GB/s, " +
               $"Inference={EnableForInference}, " +
               $"Verbose={VerboseLogging}";
    }
}
