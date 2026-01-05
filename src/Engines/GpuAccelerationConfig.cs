namespace AiDotNet.Engines;

/// <summary>
/// GPU execution mode controlling how operations are scheduled and executed.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This controls how GPU operations are executed:
/// - **Auto**: Automatically select best mode based on GPU capabilities (recommended)
/// - **Eager**: Execute each operation immediately (most compatible, simplest debugging)
/// - **Deferred**: Batch operations for optimization (highest performance, 10-50x faster)
/// - **ScopedDeferred**: Batch within explicit scopes (balanced performance and control)
/// </para>
/// </remarks>
public enum GpuExecutionModeConfig
{
    /// <summary>
    /// Automatically select best execution mode based on GPU capabilities.
    /// Uses deferred execution if supported, falls back to eager otherwise.
    /// </summary>
    Auto,

    /// <summary>
    /// Eager execution - each operation runs immediately and synchronously.
    /// Maximum compatibility, easiest debugging, but lowest performance.
    /// </summary>
    Eager,

    /// <summary>
    /// Deferred execution - operations are recorded and executed as optimized graphs.
    /// Enables kernel fusion, stream parallelism, and scheduling optimization.
    /// Highest performance (10-50x speedup) but requires GPU with async support.
    /// </summary>
    Deferred,

    /// <summary>
    /// Scoped deferred execution - operations within explicit scopes are batched.
    /// Provides multi-stream parallelism without full graph compilation.
    /// Good balance between performance and control.
    /// </summary>
    ScopedDeferred
}

/// <summary>
/// GPU device type for acceleration.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Different GPU types work with different graphics cards:
/// - **Auto**: Automatically select best available (CUDA → OpenCL → HIP → CPU)
/// - **CUDA**: NVIDIA GPUs only (GeForce, RTX, Quadro, Tesla, A100, H100)
/// - **OpenCL**: Cross-platform (AMD, Intel, NVIDIA, Apple)
/// - **CPU**: Force CPU-only execution (no GPU)
/// </para>
/// </remarks>
public enum GpuDeviceType
{
    /// <summary>
    /// Automatically select best available GPU (CUDA → OpenCL → HIP → CPU).
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
    /// - **Auto**: Automatically select best available (CUDA → OpenCL → HIP → CPU)
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

    // ==================== Advanced Execution Options (Phase 2-3) ====================

    /// <summary>
    /// Gets or sets the GPU execution mode (default: Auto).
    /// </summary>
    /// <remarks>
    /// <para><b>Phase 2-3: Async Pipelining and Graph Execution</b></para>
    /// <para><b>For Beginners:</b> Controls how GPU operations are scheduled:
    /// - **Auto**: Automatically select best mode (recommended for most users)
    /// - **Eager**: Immediate execution, easiest debugging
    /// - **Deferred**: Batch and optimize operations for maximum performance (10-50x faster)
    /// - **ScopedDeferred**: Balanced - batch within explicit scopes
    ///
    /// Start with Auto. If you need predictable step-by-step execution for debugging,
    /// use Eager. For maximum performance with large models, use Deferred.
    /// </para>
    /// </remarks>
    public GpuExecutionModeConfig ExecutionMode { get; set; } = GpuExecutionModeConfig.Auto;

    /// <summary>
    /// Gets or sets whether to enable execution graph compilation (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When enabled, the system records operations and compiles
    /// them into an optimized execution graph. This enables:
    /// - Operation fusion (combine GEMM + Bias + ReLU into one kernel)
    /// - Stream scheduling (overlap compute and data transfer)
    /// - Memory planning (reuse buffers efficiently)
    ///
    /// Provides 1.5-3x speedup. Only disable for debugging.
    /// </para>
    /// </remarks>
    public bool EnableGraphCompilation { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to enable automatic kernel fusion (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Kernel fusion combines multiple small operations into
    /// a single GPU kernel. For example, GEMM + Bias + ReLU becomes one fused operation.
    /// This reduces kernel launch overhead and memory bandwidth usage.
    ///
    /// Provides 1.5-2x speedup for common patterns. Only disable for debugging.
    /// </para>
    /// </remarks>
    public bool EnableAutoFusion { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to enable compute/transfer overlap (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When enabled, the system uses separate GPU streams
    /// for compute operations and data transfers. This allows GPU compute to continue
    /// while data is being uploaded or downloaded.
    ///
    /// Provides 1.5-2x speedup for workloads with mixed compute and data movement.
    /// Requires GPU with multi-stream support.
    /// </para>
    /// </remarks>
    public bool EnableComputeTransferOverlap { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum number of compute streams (default: 3).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More streams allow more operations to run in parallel.
    /// Modern GPUs can execute multiple independent operations simultaneously using streams.
    ///
    /// Values:
    /// - 1: Single stream, no parallelism
    /// - 2-3: Moderate parallelism (recommended for most GPUs)
    /// - 4+: High parallelism (for high-end GPUs like A100/H100)
    ///
    /// More streams use more GPU resources. Start with 3 and increase if profiling
    /// shows idle GPU time.
    /// </para>
    /// </remarks>
    public int MaxComputeStreams { get; set; } = 3;

    /// <summary>
    /// Gets or sets the minimum number of elements to use GPU (default: 4096).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Small operations have overhead that makes GPU slower
    /// than CPU. This threshold determines when to use GPU vs CPU.
    ///
    /// Values:
    /// - 1024-2048: Aggressive GPU usage (high-end GPUs with low latency)
    /// - 4096: Balanced (recommended default)
    /// - 10000+: Conservative (older GPUs, PCIe bandwidth limited)
    ///
    /// If you see many small operations running on GPU, increase this value.
    /// </para>
    /// </remarks>
    public int MinGpuElements { get; set; } = 4096;

    /// <summary>
    /// Gets or sets the maximum GPU memory usage fraction (default: 0.8).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how much GPU memory the system will use.
    /// Value is a fraction from 0.0 to 1.0 (e.g., 0.8 = 80% of GPU memory).
    ///
    /// When this limit is approached, the system will:
    /// - Evict least-recently-used tensors to CPU
    /// - Block new allocations until memory is freed
    ///
    /// Lower values leave headroom for other applications using the GPU.
    /// Higher values maximize memory for large models.
    /// </para>
    /// </remarks>
    public double MaxGpuMemoryUsage { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets whether to enable data prefetching (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Prefetching uploads data to GPU before it's needed,
    /// hiding transfer latency. For example, while the GPU computes layer N,
    /// prefetch uploads data for layer N+1.
    ///
    /// Provides 1.2-1.5x speedup for workloads with predictable data access.
    /// Uses slightly more GPU memory for prefetch buffers.
    /// </para>
    /// </remarks>
    public bool EnablePrefetch { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to cache compiled execution graphs (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When enabled, compiled execution graphs are cached
    /// and reused for repeated execution patterns (like training epochs).
    ///
    /// Eliminates graph compilation overhead after the first execution.
    /// Uses some memory for the cache. Only disable for debugging.
    /// </para>
    /// </remarks>
    public bool CacheCompiledGraphs { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to enable GPU profiling (default: false).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When enabled, detailed timing information is collected
    /// for all GPU operations. This helps identify performance bottlenecks.
    ///
    /// Adds overhead (5-10%), so only enable when investigating performance issues.
    /// Profile data can be accessed via the GpuExecutionContext.
    /// </para>
    /// </remarks>
    public bool EnableProfiling { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of transfer streams (default: 2).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Transfer streams are used for data movement between
    /// CPU and GPU. Having dedicated transfer streams allows data transfers to overlap
    /// with computation.
    ///
    /// Values:
    /// - 1: Single transfer stream (no transfer parallelism)
    /// - 2: Separate H2D and D2H streams (recommended)
    /// - 3+: Additional transfer parallelism (for high-bandwidth systems)
    ///
    /// This can also be configured via AIDOTNET_GPU_TRANSFER_STREAMS environment variable.
    /// </para>
    /// </remarks>
    public int TransferStreams { get; set; } = 2;

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
        return $"GpuConfig: DeviceType={DeviceType}, UsageLevel={UsageLevel}, DeviceIndex={DeviceIndex}, " +
               $"ExecutionMode={ExecutionMode}, GraphCompilation={EnableGraphCompilation}, " +
               $"AutoFusion={EnableAutoFusion}, ComputeTransferOverlap={EnableComputeTransferOverlap}, " +
               $"MaxStreams={MaxComputeStreams}, MinElements={MinGpuElements}, MaxMemory={MaxGpuMemoryUsage:P0}, " +
               $"Prefetch={EnablePrefetch}, CacheGraphs={CacheCompiledGraphs}, Profiling={EnableProfiling}";
    }

    /// <summary>
    /// Converts this user-facing configuration to internal GpuExecutionOptions.
    /// </summary>
    /// <returns>A GpuExecutionOptions instance with matching settings.</returns>
    internal AiDotNet.Tensors.Engines.Gpu.GpuExecutionOptions ToExecutionOptions()
    {
        var options = new AiDotNet.Tensors.Engines.Gpu.GpuExecutionOptions
        {
            MinGpuElements = MinGpuElements,
            MaxComputeStreams = MaxComputeStreams,
            TransferStreams = TransferStreams,
            EnableGraphCompilation = EnableGraphCompilation,
            EnableAutoFusion = EnableAutoFusion,
            MaxMemoryUsage = MaxGpuMemoryUsage,
            EnablePrefetch = EnablePrefetch,
            EnableComputeTransferOverlap = EnableComputeTransferOverlap,
            EnableGpuResidency = EnableGpuPersistence,
            EnableProfiling = EnableProfiling,
            CacheCompiledGraphs = CacheCompiledGraphs,
            ExecutionMode = ConvertExecutionMode(ExecutionMode)
        };

        // Map usage level to force flags
        if (UsageLevel == GpuUsageLevel.AlwaysGpu)
        {
            options.ForceGpu = true;
        }
        else if (UsageLevel == GpuUsageLevel.AlwaysCpu)
        {
            options.ForceCpu = true;
        }

        return options;
    }

    private static AiDotNet.Tensors.Engines.Gpu.GpuExecutionMode ConvertExecutionMode(GpuExecutionModeConfig mode)
    {
        return mode switch
        {
            GpuExecutionModeConfig.Auto => AiDotNet.Tensors.Engines.Gpu.GpuExecutionMode.Auto,
            GpuExecutionModeConfig.Eager => AiDotNet.Tensors.Engines.Gpu.GpuExecutionMode.Eager,
            GpuExecutionModeConfig.Deferred => AiDotNet.Tensors.Engines.Gpu.GpuExecutionMode.Deferred,
            GpuExecutionModeConfig.ScopedDeferred => AiDotNet.Tensors.Engines.Gpu.GpuExecutionMode.ScopedDeferred,
            _ => AiDotNet.Tensors.Engines.Gpu.GpuExecutionMode.Auto
        };
    }
}
