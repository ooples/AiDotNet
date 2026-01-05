using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Thread-local execution context for GPU operations.
/// Provides a facade for GPU-resident tensor management with auto-detection.
/// </summary>
/// <remarks>
/// <para><b>Usage Pattern:</b></para>
/// <code>
/// // Create context for GPU-resident execution
/// using var ctx = GpuExecutionContext.Begin(engine, options);
///
/// // Operations within the context stay GPU-resident
/// var h1 = model.Layer1.Forward(input);
/// var h2 = model.Layer2.Forward(h1);
/// var output = model.Layer3.Forward(h2);
///
/// // Data only downloaded when exiting context or explicitly requested
/// var predictions = output.ToTensor();  // Triggers sync here
/// </code>
/// </remarks>
public sealed class GpuExecutionContext : IDisposable
{
    [ThreadStatic]
    private static GpuExecutionContext? _current;

    private readonly IAsyncGpuBackend? _asyncBackend;
    private readonly IDirectGpuBackend _backend;
    private readonly GpuExecutionContext? _parent;
    private bool _disposed;

    /// <summary>
    /// Gets the current execution context for this thread, if any.
    /// </summary>
    public static GpuExecutionContext? Current => _current;

    /// <summary>
    /// Gets the GPU backend associated with this context.
    /// </summary>
    public IDirectGpuBackend Backend => _backend;

    /// <summary>
    /// Gets the async GPU backend if available, otherwise null.
    /// </summary>
    public IAsyncGpuBackend? AsyncBackend => _asyncBackend;

    /// <summary>
    /// Gets the tensor registry for this context.
    /// </summary>
    public GpuTensorRegistry Registry { get; }

    /// <summary>
    /// Gets the stream pool for this context.
    /// </summary>
    public GpuStreamPool? StreamPool { get; }

    /// <summary>
    /// Gets the execution options for this context.
    /// </summary>
    public GpuExecutionOptions Options { get; }

    /// <summary>
    /// Gets whether this context supports async operations.
    /// </summary>
    public bool SupportsAsync => _asyncBackend != null;

    /// <summary>
    /// Gets whether GPU execution is available and should be used.
    /// </summary>
    public bool IsGpuAvailable => _backend.IsAvailable;

    /// <summary>
    /// Gets the effective execution mode based on backend capabilities.
    /// </summary>
    public GpuExecutionMode EffectiveExecutionMode { get; }

    /// <summary>
    /// Creates a new GPU execution context.
    /// </summary>
    /// <param name="backend">The GPU backend to use.</param>
    /// <param name="options">Execution options.</param>
    private GpuExecutionContext(IDirectGpuBackend backend, GpuExecutionOptions? options)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        Options = options ?? GpuExecutionOptions.FromEnvironment();
        _parent = _current;

        // Try to get async backend
        _asyncBackend = backend as IAsyncGpuBackend;

        // Create registry and stream pool
        Registry = new GpuTensorRegistry(backend, Options);

        if (_asyncBackend != null && _asyncBackend.SupportsMultiStream)
        {
            StreamPool = new GpuStreamPool(_asyncBackend, Options);
        }

        // Determine effective execution mode
        EffectiveExecutionMode = DetermineExecutionMode();

        // Set as current context
        _current = this;
    }

    /// <summary>
    /// Begins a new GPU execution context.
    /// </summary>
    /// <param name="backend">The GPU backend to use.</param>
    /// <param name="options">Execution options.</param>
    /// <returns>The new execution context.</returns>
    public static GpuExecutionContext Begin(IDirectGpuBackend backend, GpuExecutionOptions? options = null)
    {
        return new GpuExecutionContext(backend, options);
    }

    /// <summary>
    /// Determines whether GPU should be used for an operation of the given size.
    /// </summary>
    /// <param name="elementCount">The number of elements in the operation.</param>
    /// <returns>True if GPU should be used.</returns>
    public bool ShouldUseGpu(int elementCount)
    {
        if (Options.ForceCpu)
        {
            return false;
        }

        if (Options.ForceGpu)
        {
            return _backend.IsAvailable;
        }

        return _backend.IsAvailable && elementCount >= Options.MinGpuElements;
    }

    /// <summary>
    /// Uploads a CPU tensor to GPU.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="tensor">The CPU tensor to upload.</param>
    /// <param name="role">The role of this tensor.</param>
    /// <returns>A GPU-resident tensor.</returns>
    public GpuTensor<T> Upload<T>(Tensor<T> tensor, GpuTensorRole role = GpuTensorRole.General)
    {
        ThrowIfDisposed();

        var gpuTensor = new GpuTensor<T>(_backend, tensor, role);
        Registry.Register(gpuTensor);
        return gpuTensor;
    }

    /// <summary>
    /// Uploads CPU data to GPU.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="data">The CPU data to upload.</param>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="role">The role of this tensor.</param>
    /// <returns>A GPU-resident tensor.</returns>
    public GpuTensor<T> Upload<T>(T[] data, int[] shape, GpuTensorRole role = GpuTensorRole.General)
    {
        ThrowIfDisposed();

        var gpuTensor = new GpuTensor<T>(_backend, data, shape, role);
        Registry.Register(gpuTensor);
        return gpuTensor;
    }

    /// <summary>
    /// Creates an empty GPU tensor.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="role">The role of this tensor.</param>
    /// <returns>A GPU-resident tensor with uninitialized data.</returns>
    public GpuTensor<T> Empty<T>(int[] shape, GpuTensorRole role = GpuTensorRole.Intermediate)
    {
        ThrowIfDisposed();

        var gpuTensor = GpuTensorFactory.Empty<T>(_backend, shape, role);
        Registry.Register(gpuTensor);
        return gpuTensor;
    }

    /// <summary>
    /// Creates a GPU tensor filled with zeros.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="role">The role of this tensor.</param>
    /// <returns>A GPU-resident tensor filled with zeros.</returns>
    public GpuTensor<T> Zeros<T>(int[] shape, GpuTensorRole role = GpuTensorRole.Intermediate)
    {
        ThrowIfDisposed();

        var gpuTensor = GpuTensorFactory.Zeros<T>(_backend, shape, role);
        Registry.Register(gpuTensor);
        return gpuTensor;
    }

    /// <summary>
    /// Synchronizes all pending GPU operations.
    /// </summary>
    public void Synchronize()
    {
        ThrowIfDisposed();

        if (StreamPool != null)
        {
            StreamPool.SynchronizeAll();
        }
        else
        {
            _backend.Synchronize();
        }
    }

    /// <summary>
    /// Gets memory statistics for the current context.
    /// </summary>
    /// <returns>Memory statistics.</returns>
    public GpuMemoryStats GetMemoryStats()
    {
        return new GpuMemoryStats
        {
            TotalAllocatedBytes = Registry.TotalAllocatedBytes,
            MaxMemoryBytes = Registry.MaxMemoryBytes,
            TensorCount = Registry.TensorCount,
            MemoryUsage = Registry.MemoryUsage,
            IsUnderPressure = Registry.IsUnderMemoryPressure,
            TensorsByRole = Registry.GetStatistics()
        };
    }

    private GpuExecutionMode DetermineExecutionMode()
    {
        if (Options.ExecutionMode != GpuExecutionMode.Auto)
        {
            return Options.ExecutionMode;
        }

        // Auto-detect based on backend capabilities
        if (_asyncBackend == null)
        {
            return GpuExecutionMode.Eager;
        }

        if (_asyncBackend.SupportsGraphCapture && Options.EnableGraphCompilation)
        {
            return GpuExecutionMode.Deferred;
        }

        if (_asyncBackend.SupportsMultiStream && _asyncBackend.SupportsEvents)
        {
            return GpuExecutionMode.ScopedDeferred;
        }

        return GpuExecutionMode.Eager;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(GpuExecutionContext));
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        // Synchronize before disposing
        try
        {
            Synchronize();
        }
        catch
        {
            // Ignore synchronization errors during disposal
        }

        // Dispose managed resources
        StreamPool?.Dispose();
        Registry.Dispose();

        // Restore parent context
        _current = _parent;
    }
}

/// <summary>
/// GPU memory statistics.
/// </summary>
public sealed class GpuMemoryStats
{
    /// <summary>
    /// Total bytes currently allocated on GPU.
    /// </summary>
    public long TotalAllocatedBytes { get; init; }

    /// <summary>
    /// Maximum bytes allowed before eviction.
    /// </summary>
    public long MaxMemoryBytes { get; init; }

    /// <summary>
    /// Number of registered tensors.
    /// </summary>
    public int TensorCount { get; init; }

    /// <summary>
    /// Current memory usage as a fraction of max.
    /// </summary>
    public double MemoryUsage { get; init; }

    /// <summary>
    /// Whether memory pressure is high.
    /// </summary>
    public bool IsUnderPressure { get; init; }

    /// <summary>
    /// Breakdown of tensors by role.
    /// </summary>
    public Dictionary<GpuTensorRole, (int Count, long Bytes)> TensorsByRole { get; init; } = new();

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"GPU Memory: {TotalAllocatedBytes / (1024.0 * 1024.0):F1} MB / " +
               $"{MaxMemoryBytes / (1024.0 * 1024.0):F1} MB ({MemoryUsage:P1}), " +
               $"{TensorCount} tensors" +
               (IsUnderPressure ? " [PRESSURE]" : "");
    }
}

/// <summary>
/// Extension methods for using GPU execution context.
/// </summary>
public static class GpuExecutionContextExtensions
{
    /// <summary>
    /// Executes an action within a GPU execution context.
    /// </summary>
    /// <param name="backend">The GPU backend.</param>
    /// <param name="action">The action to execute.</param>
    /// <param name="options">Execution options.</param>
    public static void WithGpuContext(this IDirectGpuBackend backend, Action<GpuExecutionContext> action, GpuExecutionOptions? options = null)
    {
        using var ctx = GpuExecutionContext.Begin(backend, options);
        action(ctx);
    }

    /// <summary>
    /// Executes a function within a GPU execution context.
    /// </summary>
    /// <typeparam name="T">The result type.</typeparam>
    /// <param name="backend">The GPU backend.</param>
    /// <param name="func">The function to execute.</param>
    /// <param name="options">Execution options.</param>
    /// <returns>The function result.</returns>
    public static T WithGpuContext<T>(this IDirectGpuBackend backend, Func<GpuExecutionContext, T> func, GpuExecutionOptions? options = null)
    {
        using var ctx = GpuExecutionContext.Begin(backend, options);
        return func(ctx);
    }
}
