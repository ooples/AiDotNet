namespace AiDotNet.Engines;

/// <summary>
/// GPU device type for acceleration.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Different GPU types work with different graphics cards:
/// - **Auto**: Automatically select best available (CUDA → OpenCL → CPU)
/// - **CUDA**: NVIDIA GPUs only (GeForce, RTX, Quadro, Tesla, A100, H100)
/// - **OpenCL**: Cross-platform (AMD, Intel, NVIDIA, Apple)
/// - **CPU**: Force CPU-only execution (no GPU)
/// </para>
/// </remarks>
public enum GpuDeviceType
{
    /// <summary>
    /// Automatically select best available GPU (CUDA → OpenCL → CPU).
    /// </summary>
    Auto,

    /// <summary>
    /// NVIDIA CUDA (GeForce, RTX, Quadro, Tesla, A100, H100).
    /// </summary>
    CUDA,

    /// <summary>
    /// OpenCL (AMD, Intel, NVIDIA, Apple - cross-platform).
    /// </summary>
    OpenCL,

    /// <summary>
    /// Force CPU-only execution (no GPU acceleration).
    /// </summary>
    CPU
}

/// <summary>
/// GPU usage level controlling when to use GPU vs CPU for operations.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This controls how aggressively the system uses GPU:
/// - **Default**: Balanced for typical GPUs (recommended)
/// - **Conservative**: Only use GPU for very large operations (older/slower GPUs)
/// - **Aggressive**: Use GPU more often (high-end GPUs like RTX 4090/A100)
/// - **AlwaysGpu**: Force all operations to GPU (maximize GPU utilization)
/// - **AlwaysCpu**: Force all operations to CPU (disable GPU entirely)
/// </para>
/// </remarks>
public enum GpuUsageLevel
{
    /// <summary>
    /// Conservative GPU usage - only for very large operations (older/slower GPUs).
    /// </summary>
    Conservative,

    /// <summary>
    /// Balanced GPU usage - good for most desktop GPUs (default).
    /// </summary>
    Default,

    /// <summary>
    /// Aggressive GPU usage - use GPU more often (high-end GPUs).
    /// </summary>
    Aggressive,

    /// <summary>
    /// Always use GPU for all operations (maximize GPU utilization).
    /// </summary>
    AlwaysGpu,

    /// <summary>
    /// Always use CPU for all operations (disable GPU entirely).
    /// </summary>
    AlwaysCpu
}

/// <summary>
/// Configuration for GPU-accelerated training and inference.
/// </summary>
/// <remarks>
/// <para><b>Phase B: GPU Acceleration Configuration</b></para>
/// <para>
/// This configuration controls when and how GPU acceleration is used during training and inference.
/// The default settings work well for most desktop GPUs - just call ConfigureGpuAcceleration()
/// without parameters for automatic GPU detection and sensible defaults.
/// </para>
/// <para><b>For Beginners:</b> GPU makes training 10-100x faster for large models by using your
/// graphics card for parallel computation. This config lets you:
/// - Enable/disable GPU acceleration
/// - Choose which GPU to use (if you have multiple)
/// - Control when to use GPU vs CPU based on operation size
/// - Enable debug logging to see what's running where
/// </para>
/// </remarks>
public class GpuAccelerationConfig
{
    /// <summary>
    /// Gets or sets the GPU device type to use (default: Auto).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Specifies which type of GPU to use:
    /// - **Auto**: Automatically select best available (CUDA → OpenCL → CPU)
    /// - **CUDA**: Force NVIDIA CUDA (fails if not available)
    /// - **OpenCL**: Force OpenCL (AMD/Intel/NVIDIA GPUs)
    /// - **CPU**: Force CPU execution (disable GPU)
    ///
    /// Leave as Auto unless you have specific requirements.
    /// </para>
    /// </remarks>
    public GpuDeviceType DeviceType { get; set; } = GpuDeviceType.Auto;

    /// <summary>
    /// Gets or sets the GPU usage level (default: Default).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how aggressively GPU is used:
    /// - **Default**: Balanced for typical GPUs (recommended)
    /// - **Conservative**: Only use GPU for very large operations (older GPUs)
    /// - **Aggressive**: Use GPU more often (high-end GPUs like RTX 4090/A100)
    /// - **AlwaysGpu**: Force all operations to GPU
    /// - **AlwaysCpu**: Force all operations to CPU
    /// </para>
    /// </remarks>
    public GpuUsageLevel UsageLevel { get; set; } = GpuUsageLevel.Default;

    /// <summary>
    /// Gets or sets the GPU device index to use if multiple GPUs are available (default: 0).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If you have multiple GPUs, specify which one to use:
    /// - 0: First GPU (default)
    /// - 1: Second GPU
    /// - etc.
    ///
    /// The system will enumerate available GPUs and select the one at this index.
    /// If the specified index doesn't exist, falls back to the first available GPU.
    /// </para>
    /// </remarks>
    public int DeviceIndex { get; set; } = 0;

    /// <summary>
    /// Gets or sets whether to enable verbose logging of GPU operations (default: false).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When true, logs information about:
    /// - GPU initialization and device selection
    /// - Which operations run on GPU vs CPU
    /// - Memory transfers and sizes
    /// - Performance metrics
    ///
    /// Useful for debugging and optimization, but can produce a lot of output.
    /// </para>
    /// </remarks>
    public bool VerboseLogging { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to enable GPU acceleration for inference (prediction) as well as training (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> GPU can accelerate both training AND inference.
    /// Set to false if you only want GPU during training but CPU during inference,
    /// for example when deploying to CPU-only servers.
    /// </para>
    /// </remarks>
    public bool EnableForInference { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to enable GPU persistence for neural network weights (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>Phase B: Persistent GPU Tensors (US-GPU-030)</b></para>
    /// <para>
    /// When enabled, neural network weights and biases stay on GPU memory between operations,
    /// eliminating per-operation CPU-GPU memory transfers. This provides massive speedups
    /// (up to 100x) for training and inference.
    /// </para>
    /// <para><b>For Beginners:</b> This keeps your model's weights on the GPU permanently
    /// instead of copying them back and forth for each operation. This is the single most
    /// important optimization for GPU performance.
    ///
    /// Only disable if:
    /// - You're running out of GPU memory
    /// - You need weights on CPU for other purposes between operations
    /// - You're debugging GPU-related issues
    /// </para>
    /// <para><b>Memory Impact:</b> Weights stay in GPU memory until the model is disposed.
    /// For large models (e.g., 100M parameters at 4 bytes each = 400MB), this GPU memory
    /// is allocated and held for the model's lifetime.
    /// </para>
    /// </remarks>
    public bool EnableGpuPersistence { get; set; } = true;

    /// <summary>
    /// Creates a configuration with default GPU settings.
    /// </summary>
    public GpuAccelerationConfig()
    {
    }

    /// <summary>
    /// Gets a string representation of this configuration.
    /// </summary>
    /// <returns>A string describing the configuration settings.</returns>
    public override string ToString()
    {
        return $"GpuConfig: DeviceType={DeviceType}, UsageLevel={UsageLevel}, DeviceIndex={DeviceIndex}, EnableForInference={EnableForInference}, EnableGpuPersistence={EnableGpuPersistence}, Verbose={VerboseLogging}";
    }
}
