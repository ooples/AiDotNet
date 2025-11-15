using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Gpu;

/// <summary>
/// Manages execution context for CPU/GPU placement of tensor operations.
/// </summary>
/// <remarks>
/// <para>
/// ExecutionContext provides intelligent placement decisions for tensor operations,
/// automatically choosing between CPU and GPU execution based on configurable policies.
/// </para>
/// <para><b>For Beginners:</b> This class decides when to use CPU vs GPU for operations.
///
/// Think of it like a smart traffic router:
/// - Small operations → CPU (faster due to no transfer overhead)
/// - Large operations → GPU (much faster computation)
/// - Sequential operations → Keep data where it is (minimize transfers)
///
/// Example usage:
/// <code>
/// var context = new ExecutionContext(backend)
/// {
///     Strategy = PlacementStrategy.AutomaticPlacement,
///     GpuThreshold = 100_000  // Use GPU for tensors > 100K elements
/// };
///
/// // Automatically uses GPU for large tensors
/// if (context.ShouldUseGpu(largeTensor))
/// {
///     using var gpu = context.Execute(largeTensor, t => backend.ReLU(t));
/// }
/// </code>
/// </para>
/// </remarks>
public class ExecutionContext : IDisposable
{
    /// <summary>
    /// Defines strategies for deciding where to execute tensor operations.
    /// </summary>
    public enum PlacementStrategy
    {
        /// <summary>
        /// Automatically chooses CPU or GPU based on tensor size threshold.
        /// Best for general use - balances performance and transfer overhead.
        /// </summary>
        AutomaticPlacement,

        /// <summary>
        /// Forces all operations to execute on GPU regardless of size.
        /// Use when you know all operations benefit from GPU acceleration.
        /// </summary>
        ForceGpu,

        /// <summary>
        /// Forces all operations to execute on CPU.
        /// Use for debugging or when GPU is unavailable.
        /// </summary>
        ForceCpu,

        /// <summary>
        /// Minimizes data transfers by keeping data on current device.
        /// Best for sequential operations on same tensor.
        /// </summary>
        MinimizeTransfers,

        /// <summary>
        /// Uses cost-based analysis considering transfer time and compute time.
        /// Most sophisticated but slightly more overhead for decision-making.
        /// </summary>
        CostBased
    }

    private readonly object _lock = new object();
    private bool _disposed;

    /// <summary>
    /// Gets or sets the GPU backend to use for GPU operations.
    /// </summary>
    public IGpuBackend<float>? GpuBackend { get; set; }

    /// <summary>
    /// Gets or sets whether GPU acceleration is enabled.
    /// </summary>
    /// <remarks>
    /// Even if true, actual GPU usage depends on the Strategy and other factors.
    /// Set to false to completely disable GPU usage.
    /// </remarks>
    public bool UseGpu { get; set; }

    /// <summary>
    /// Gets or sets the minimum number of elements before using GPU.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> GPUs are fast at computation but slow at data transfer.
    ///
    /// Default threshold of 100,000 elements means:
    /// - 100x100 matrix (10,000 elements) → CPU faster
    /// - 1000x1000 matrix (1,000,000 elements) → GPU much faster
    ///
    /// Adjust based on your hardware:
    /// - Faster PCIe/GPU → Lower threshold (e.g., 50,000)
    /// - Slower GPU → Higher threshold (e.g., 200,000)
    /// </para>
    /// </remarks>
    public int GpuThreshold { get; set; } = 100_000;

    /// <summary>
    /// Gets or sets the placement strategy to use.
    /// </summary>
    public PlacementStrategy Strategy { get; set; } = PlacementStrategy.AutomaticPlacement;

    /// <summary>
    /// Gets or sets the estimated computation speedup on GPU vs CPU.
    /// </summary>
    /// <remarks>
    /// Used for cost-based placement decisions. Default is 10x speedup.
    /// Adjust based on your specific GPU and operation types.
    /// </remarks>
    public double GpuComputeSpeedup { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the estimated PCIe transfer bandwidth in GB/s.
    /// </summary>
    /// <remarks>
    /// Used for cost-based decisions. Default is 12 GB/s (PCIe 3.0 x16 conservative).
    /// PCIe 4.0 x16: ~24 GB/s
    /// PCIe 5.0 x16: ~48 GB/s
    /// </remarks>
    public double TransferBandwidthGBps { get; set; } = 12.0;

    /// <summary>
    /// Gets statistics about GPU vs CPU usage.
    /// </summary>
    public ExecutionStats Statistics { get; } = new ExecutionStats();

    /// <summary>
    /// Initializes a new instance of the <see cref="ExecutionContext"/> class.
    /// </summary>
    /// <param name="gpuBackend">Optional GPU backend. If null, GPU will be disabled.</param>
    public ExecutionContext(IGpuBackend<float>? gpuBackend = null)
    {
        GpuBackend = gpuBackend;
        UseGpu = gpuBackend?.IsAvailable ?? false;
    }

    /// <summary>
    /// Determines whether a tensor operation should execute on GPU.
    /// </summary>
    /// <param name="tensor">The tensor to evaluate.</param>
    /// <returns>True if the operation should use GPU, false for CPU.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the decision-making logic!
    ///
    /// It considers:
    /// 1. Is GPU available and enabled?
    /// 2. What's the current strategy?
    /// 3. How large is the tensor?
    /// 4. Where is the data currently located?
    ///
    /// This method is called automatically by GPU-aware operations.
    /// </para>
    /// </remarks>
    public bool ShouldUseGpu<T>(Tensor<T> tensor)
    {
        // GPU not available or disabled
        if (!UseGpu || GpuBackend == null || !GpuBackend.IsAvailable)
        {
            return false;
        }

        return Strategy switch
        {
            PlacementStrategy.AutomaticPlacement => tensor.Length >= GpuThreshold,
            PlacementStrategy.ForceGpu => true,
            PlacementStrategy.ForceCpu => false,
            PlacementStrategy.MinimizeTransfers => false, // Default to CPU unless data already on GPU
            PlacementStrategy.CostBased => ShouldUseGpuCostBased(tensor),
            _ => false
        };
    }

    /// <summary>
    /// Determines GPU usage based on cost-benefit analysis.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor.</typeparam>
    /// <param name="tensor">The tensor to evaluate.</param>
    /// <returns>True if GPU is estimated to be faster overall.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This does the math to decide if GPU is worth it.
    ///
    /// Formula:
    /// - GPU Time = Transfer Time + (Compute Time / Speedup)
    /// - CPU Time = Compute Time
    /// - Use GPU if: GPU Time &lt; CPU Time
    ///
    /// Example for 1M element tensor:
    /// - Transfer: ~0.3ms (4MB / 12GB/s)
    /// - Compute on CPU: ~10ms
    /// - Compute on GPU: ~1ms (10x speedup)
    /// - Total GPU: 0.3 + 1 = 1.3ms vs CPU: 10ms → Use GPU!
    /// </para>
    /// </remarks>
    private bool ShouldUseGpuCostBased<T>(Tensor<T> tensor)
    {
        // Estimate transfer cost (round-trip)
        var elementSize = System.Runtime.InteropServices.Marshal.SizeOf<T>();
        var totalBytes = tensor.Length * elementSize;
        var transferTimeMs = (totalBytes / (TransferBandwidthGBps * 1_000_000_000.0)) * 2.0 * 1000.0;

        // Estimate compute time (very rough heuristic)
        // Assume ~10 FLOPs per element, CPU at ~100 GFLOPS, GPU at speedup factor
        const double CPU_GFLOPS = 100.0;
        const double FLOPS_PER_ELEMENT = 10.0;
        var totalFlops = tensor.Length * FLOPS_PER_ELEMENT;
        var cpuComputeTimeMs = (totalFlops / (CPU_GFLOPS * 1_000_000_000.0)) * 1000.0;
        var gpuComputeTimeMs = cpuComputeTimeMs / GpuComputeSpeedup;

        // Total GPU time includes transfer overhead
        var totalGpuTimeMs = transferTimeMs + gpuComputeTimeMs;

        // Use GPU if total time is less than CPU time
        return totalGpuTimeMs < cpuComputeTimeMs;
    }

    /// <summary>
    /// Executes an operation with automatic CPU/GPU placement.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="operation">The operation to perform on GPU.</param>
    /// <returns>The result tensor on CPU.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a convenience method that handles everything!
    ///
    /// It automatically:
    /// 1. Decides if GPU should be used
    /// 2. Transfers data if needed
    /// 3. Executes the operation
    /// 4. Transfers result back
    /// 5. Cleans up GPU memory
    ///
    /// Example:
    /// <code>
    /// var result = context.Execute(inputTensor, gpu =>
    /// {
    ///     var activated = backend.ReLU(gpu);
    ///     return backend.Add(activated, activated);
    /// });
    /// </code>
    /// </para>
    /// </remarks>
    public Tensor<T> Execute<T>(
        Tensor<T> tensor,
        Func<GpuTensor<T>, GpuTensor<T>> operation)
        where T : unmanaged
    {
        if (!ShouldUseGpu(tensor))
        {
            lock (_lock)
            {
                Statistics.CpuOperations++;
            }
            // Execute on CPU - caller should handle CPU operations
            throw new InvalidOperationException(
                "Operation should execute on CPU. Check ShouldUseGpu before calling Execute.");
        }

        lock (_lock)
        {
            Statistics.GpuOperations++;
        }

        // Get the appropriate GPU backend
        var backend = GetBackendForType<T>();
        if (backend == null)
        {
            throw new InvalidOperationException("GPU backend not available for type " + typeof(T).Name);
        }

        using var gpuInput = backend.ToGpu(tensor);
        using var gpuResult = operation(gpuInput);
        return backend.ToCpu(gpuResult);
    }

    /// <summary>
    /// Executes a binary operation with automatic CPU/GPU placement.
    /// </summary>
    public Tensor<T> Execute<T>(
        Tensor<T> tensor1,
        Tensor<T> tensor2,
        Func<GpuTensor<T>, GpuTensor<T>, GpuTensor<T>> operation)
        where T : unmanaged
    {
        // For binary ops, use the larger tensor for placement decision
        var shouldUseGpu = ShouldUseGpu(tensor1) || ShouldUseGpu(tensor2);

        if (!shouldUseGpu)
        {
            lock (_lock)
            {
                Statistics.CpuOperations++;
            }
            throw new InvalidOperationException(
                "Operation should execute on CPU. Check ShouldUseGpu before calling Execute.");
        }

        lock (_lock)
        {
            Statistics.GpuOperations++;
        }

        var backend = GetBackendForType<T>();
        if (backend == null)
        {
            throw new InvalidOperationException("GPU backend not available for type " + typeof(T).Name);
        }

        using var gpu1 = backend.ToGpu(tensor1);
        using var gpu2 = backend.ToGpu(tensor2);
        using var gpuResult = operation(gpu1, gpu2);
        return backend.ToCpu(gpuResult);
    }

    /// <summary>
    /// Gets the appropriate GPU backend for the specified type.
    /// </summary>
    private IGpuBackend<T>? GetBackendForType<T>() where T : unmanaged
    {
        // Currently only float is supported
        // This can be extended for double, int, etc.
        if (typeof(T) == typeof(float))
        {
            return GpuBackend as IGpuBackend<T>;
        }

        return null;
    }

    /// <summary>
    /// Resets execution statistics.
    /// </summary>
    public void ResetStatistics()
    {
        lock (_lock)
        {
            Statistics.Reset();
        }
    }

    /// <summary>
    /// Disposes the execution context and associated GPU resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        GpuBackend?.Dispose();
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// Tracks execution statistics for CPU vs GPU operations.
/// </summary>
public class ExecutionStats
{
    private long _gpuOperations;
    private long _cpuOperations;

    /// <summary>
    /// Gets the number of operations executed on GPU.
    /// </summary>
    public long GpuOperations => _gpuOperations;

    /// <summary>
    /// Gets the number of operations executed on CPU.
    /// </summary>
    public long CpuOperations => _cpuOperations;

    /// <summary>
    /// Gets the total number of operations.
    /// </summary>
    public long TotalOperations => _gpuOperations + _cpuOperations;

    /// <summary>
    /// Gets the percentage of operations executed on GPU.
    /// </summary>
    public double GpuPercentage => TotalOperations > 0
        ? (_gpuOperations * 100.0) / TotalOperations
        : 0.0;

    internal long CpuOperations1 { get => _cpuOperations; set => _cpuOperations = value; }

    /// <summary>
    /// Increments GPU operation count (thread-safe).
    /// </summary>
    internal void IncrementGpu() => Interlocked.Increment(ref _gpuOperations);

    /// <summary>
    /// Increments CPU operation count (thread-safe).
    /// </summary>
    internal void IncrementCpu() => Interlocked.Increment(ref _cpuOperations);

    /// <summary>
    /// Resets all statistics.
    /// </summary>
    internal void Reset()
    {
        Interlocked.Exchange(ref _gpuOperations, 0);
        Interlocked.Exchange(ref _cpuOperations, 0);
    }

    /// <summary>
    /// Returns a string representation of the statistics.
    /// </summary>
    public override string ToString()
    {
        return $"GPU: {GpuOperations}, CPU: {CpuOperations}, Total: {TotalOperations}, GPU%: {GpuPercentage:F1}%";
    }
}
