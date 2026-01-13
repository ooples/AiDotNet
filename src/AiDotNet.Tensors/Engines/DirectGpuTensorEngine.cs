using System;
using System.Collections.Concurrent;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Cached GPU buffer entry for persistent tensor management.
/// </summary>
internal sealed class GpuBufferCacheEntry : IDisposable
{
    public IGpuBuffer Buffer { get; }
    public PersistentTensorRole Role { get; }
    public int Version { get; set; }

    public GpuBufferCacheEntry(IGpuBuffer buffer, PersistentTensorRole role)
    {
        Buffer = buffer;
        Role = role;
        Version = 0;
    }

    public void Dispose()
    {
        Buffer.Dispose();
    }
}

/// <summary>
/// Cache entry for intermediate activation tensors to avoid re-uploading between layers.
/// When a layer's output is downloaded, we cache the GPU buffer so the next layer
/// can reuse it without re-uploading if it uses the same data.
/// </summary>
internal sealed class ActivationCacheEntry : IDisposable
{
    public IGpuBuffer Buffer { get; }
    public int[] Shape { get; }
    public long Timestamp { get; }
    public IDirectGpuBackend Backend { get; }

    public ActivationCacheEntry(IGpuBuffer buffer, int[] shape, long timestamp, IDirectGpuBackend backend)
    {
        Buffer = buffer;
        Shape = shape;
        Timestamp = timestamp;
        Backend = backend;
    }

    public void Dispose()
    {
        Buffer.Dispose();
    }
}

/// <summary>
/// IEngine implementation that routes supported ops to DirectGpuEngine and falls back to CPU.
/// </summary>
/// <remarks>
/// <para><b>Threading Model:</b> This engine uses buffer caching for GPU memory efficiency.
/// Cache operations are thread-safe for concurrent reads, but cache invalidation/clearing
/// (InvalidateWeightCache, InvalidateAllWeightCaches, ClearActivationCache) should NOT be
/// called while GPU operations are in-flight using cached buffers.</para>
/// <para><b>Safe usage patterns:</b></para>
/// <list type="bullet">
/// <item>Single-threaded inference: fully safe</item>
/// <item>Multi-threaded inference with stable weights: safe (no invalidation during inference)</item>
/// <item>Weight updates during inference: call invalidation only between inference batches</item>
/// </list>
/// <para>For concurrent weight updates during inference, consider using separate engine instances
/// or implementing external synchronization around weight update + invalidation sequences.</para>
/// </remarks>
public partial class DirectGpuTensorEngine : CpuEngine, IEngine, IDisposable
{
    private readonly DirectGpuEngine? _directGpu;
    private readonly bool _ownsDirectGpu;

    // GPU buffer cache for persistent tensors - keyed by tensor data array reference
    // Thread-safety: ConcurrentDictionary provides atomic operations. Invalidation uses
    // _persistentBufferLock for dispose safety. CAUTION: Callers must not invalidate/clear
    // while GPU operations are actively using cached buffers.
    private readonly ConcurrentDictionary<object, GpuBufferCacheEntry> _persistentBufferCache = new();
    private readonly object _persistentBufferLock = new();

    // Version tracking for invalidation
    private readonly ConcurrentDictionary<object, int> _tensorVersions = new();

    // Activation cache for intermediate tensors - enables GPU-resident layer chaining
    // Key: tensor data array reference, Value: (buffer, shape, timestamp)
    // This cache holds the last N activation buffers to avoid re-uploading layer outputs
    // Thread-safety: ConcurrentDictionary + _activationCacheLock for atomic compound operations.
    // CAUTION: ClearActivationCache should not be called during active GPU operations.
    private readonly ConcurrentDictionary<object, ActivationCacheEntry> _activationCache = new();
    private readonly object _activationCacheLock = new();
    private const int DefaultActivationCacheSize = 16;
    private int _maxActivationCacheSize = DefaultActivationCacheSize;
    private long _activationCacheTimestamp = 0;

    public DirectGpuTensorEngine()
    {
        _directGpu = new DirectGpuEngine();
        _ownsDirectGpu = true;
    }

    public DirectGpuTensorEngine(DirectGpuEngine directGpu)
    {
        _directGpu = directGpu;
        _ownsDirectGpu = false;
    }

    public bool IsGpuAvailable => _directGpu?.IsAvailable == true;

    public new string Name => IsGpuAvailable
        ? $"Direct GPU Engine ({_directGpu!.BackendName} {_directGpu.DeviceName})"
        : "CPU Engine (DirectGpu unavailable)";

    public new bool SupportsGpu => IsGpuAvailable;

    /// <summary>
    /// Gets or sets the maximum number of activation cache entries.
    /// Larger values use more GPU memory but reduce re-uploads for deep networks.
    /// Default is 16.
    /// </summary>
    public int MaxActivationCacheSize
    {
        get => _maxActivationCacheSize;
        set => _maxActivationCacheSize = value > 0 ? value : DefaultActivationCacheSize;
    }

    DirectGpuEngine? IEngine.DirectGpu => _directGpu;

    string IEngine.Name => Name;

    bool IEngine.SupportsGpu => SupportsGpu;

    private bool TryGetBackend(out IDirectGpuBackend backend)
    {
        // Check if there's an active DeferredScope - use its RecordingBackend for deferred execution
        var deferredScope = Gpu.DeferredScope.Current;
        if (deferredScope != null && deferredScope.IsRecording)
        {
            backend = deferredScope.RecordingBackend;
            return true;
        }

        // No deferred scope - use regular backend
        backend = _directGpu?.Backend!;
        return IsGpuAvailable && backend != null;
    }

    /// <summary>
    /// Gets the GPU backend if available.
    /// </summary>
    /// <returns>The GPU backend, or null if not available.</returns>
    public IDirectGpuBackend? GetBackend()
    {
        if (TryGetBackend(out var backend))
        {
            return backend;
        }
        return null;
    }

    /// <summary>
    /// Gets the async GPU backend if available (supports deferred execution).
    /// </summary>
    /// <returns>The async GPU backend, or null if not available or not supported.</returns>
    public IAsyncGpuBackend? GetAsyncBackend()
    {
        if (TryGetBackend(out var backend))
        {
            return backend as IAsyncGpuBackend;
        }
        return null;
    }

    /// <summary>
    /// Begins a GPU execution context for GPU-resident operations.
    /// Operations within the context stay GPU-resident until explicitly downloaded.
    /// </summary>
    /// <param name="options">Optional execution options.</param>
    /// <returns>A GPU execution context, or null if GPU is not available.</returns>
    public GpuExecutionContext? BeginGpuContext(GpuExecutionOptions? options = null)
    {
        if (!TryGetBackend(out var backend))
        {
            return null;
        }

        return GpuExecutionContext.Begin(backend, options);
    }

    /// <summary>
    /// Begins a deferred execution scope that records operations to an execution graph
    /// for optimized batch execution.
    /// </summary>
    /// <param name="options">Optional execution options.</param>
    /// <returns>A deferred scope for recording operations, or null if not supported.</returns>
    /// <remarks>
    /// <para><b>Example:</b></para>
    /// <code>
    /// using var scope = engine.BeginDeferredScope();
    /// if (scope != null)
    /// {
    ///     // Operations recorded to scope.GraphBuilder
    ///     scope.Execute(); // Compile and execute all at once
    /// }
    /// </code>
    /// </remarks>
    public IDeferredScope? BeginDeferredScope(GpuExecutionOptions? options = null)
    {
        var asyncBackend = GetAsyncBackend();
        if (asyncBackend == null)
        {
            return null;
        }

        var effectiveOptions = options ?? GpuExecutionOptions.FromEnvironment();
        var streamPool = asyncBackend.SupportsMultiStream
            ? new GpuStreamPool(asyncBackend, effectiveOptions)
            : null;

        return new DeferredScope(asyncBackend, effectiveOptions, streamPool);
    }

    /// <summary>
    /// Gets whether deferred execution is supported on this engine.
    /// </summary>
    public bool SupportsDeferredExecution => GetAsyncBackend() != null;

    /// <summary>
    /// Gets the current GPU execution context for this thread, if any.
    /// This allows operations to check if GPU-resident mode is active.
    /// </summary>
    public static GpuExecutionContext? CurrentContext => GpuExecutionContext.Current;

    /// <summary>
    /// Gets whether a GPU execution context is currently active on this thread.
    /// When active, GPU tensors can stay resident on the GPU without downloading.
    /// </summary>
    public static bool IsGpuContextActive => GpuExecutionContext.Current != null;

    /// <summary>
    /// Determines whether GPU should be used for an operation of the given element count.
    /// Uses the current execution context options if available, otherwise uses defaults.
    /// </summary>
    /// <param name="elementCount">The number of elements in the operation.</param>
    /// <returns>True if GPU should be used.</returns>
    public bool ShouldUseGpu(int elementCount)
    {
        if (!IsGpuAvailable)
        {
            return false;
        }

        // Use context options if available
        var context = GpuExecutionContext.Current;
        if (context != null)
        {
            return context.ShouldUseGpu(elementCount);
        }

        // Default threshold
        return elementCount >= 4096;
    }

    /// <summary>
    /// Uploads a tensor to GPU within the current execution context.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="tensor">The CPU tensor to upload.</param>
    /// <param name="role">The role of this tensor.</param>
    /// <returns>A GPU-resident tensor, or null if no context is active.</returns>
    public GpuTensor<T>? UploadToContext<T>(Tensor<T> tensor, GpuTensorRole role = GpuTensorRole.General)
    {
        var context = GpuExecutionContext.Current;
        return context?.Upload(tensor, role);
    }

    /// <summary>
    /// Uploads data to GPU within the current execution context.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="data">The CPU data to upload.</param>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="role">The role of this tensor.</param>
    /// <returns>A GPU-resident tensor, or null if no context is active.</returns>
    public GpuTensor<T>? UploadToContext<T>(T[] data, int[] shape, GpuTensorRole role = GpuTensorRole.General)
    {
        var context = GpuExecutionContext.Current;
        return context?.Upload(data, shape, role);
    }

    /// <summary>
    /// Creates an empty GPU tensor within the current execution context.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="role">The role of this tensor.</param>
    /// <returns>A GPU-resident tensor with uninitialized data, or null if no context is active.</returns>
    public GpuTensor<T>? EmptyInContext<T>(int[] shape, GpuTensorRole role = GpuTensorRole.Intermediate)
    {
        var context = GpuExecutionContext.Current;
        return context?.Empty<T>(shape, role);
    }

    /// <summary>
    /// Creates a GPU tensor filled with zeros within the current execution context.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="role">The role of this tensor.</param>
    /// <returns>A GPU-resident tensor filled with zeros, or null if no context is active.</returns>
    public GpuTensor<T>? ZerosInContext<T>(int[] shape, GpuTensorRole role = GpuTensorRole.Intermediate)
    {
        var context = GpuExecutionContext.Current;
        return context?.Zeros<T>(shape, role);
    }

    /// <summary>
    /// Executes an action within a GPU execution context.
    /// </summary>
    /// <param name="action">The action to execute.</param>
    /// <param name="options">Optional execution options.</param>
    /// <returns>True if executed on GPU, false if GPU not available.</returns>
    public bool WithGpuContext(Action<GpuExecutionContext> action, GpuExecutionOptions? options = null)
    {
        var context = BeginGpuContext(options);
        if (context == null)
        {
            return false;
        }

        using (context)
        {
            action(context);
        }

        return true;
    }

    /// <summary>
    /// Executes a function within a GPU execution context.
    /// </summary>
    /// <typeparam name="TResult">The result type.</typeparam>
    /// <param name="func">The function to execute.</param>
    /// <param name="fallback">Fallback function if GPU is not available.</param>
    /// <param name="options">Optional execution options.</param>
    /// <returns>The function result.</returns>
    public TResult WithGpuContext<TResult>(Func<GpuExecutionContext, TResult> func, Func<TResult> fallback, GpuExecutionOptions? options = null)
    {
        var context = BeginGpuContext(options);
        if (context == null)
        {
            return fallback();
        }

        using (context)
        {
            return func(context);
        }
    }

    private static float ToFloatScalar<T>(T value)
    {
        if (typeof(T) == typeof(float))
            return (float)(object)value!;
        if (typeof(T) == typeof(double))
            return (float)(double)(object)value!;

        // Use numeric operations directly instead of allocating a single-element array
        return (float)MathHelper.GetNumericOperations<T>().ToDouble(value);
    }

    private static T FromFloatScalar<T>(float value)
    {
        if (typeof(T) == typeof(float))
            return (T)(object)value;
        if (typeof(T) == typeof(double))
            return (T)(object)(double)value;

        // Use numeric operations directly instead of allocating a single-element array
        return MathHelper.GetNumericOperations<T>().FromFloat(value);
    }

    /// <summary>
    /// Helper struct for tracking GPU buffer ownership. Implements IDisposable
    /// to only dispose buffers we own (not cached ones).
    /// </summary>
    private readonly struct OwnedBuffer : IDisposable
    {
        private readonly IGpuBuffer _buffer;
        private readonly bool _ownsBuffer;

        /// <summary>
        /// Gets the underlying GPU buffer.
        /// </summary>
        public IGpuBuffer Buffer => _buffer;

        /// <summary>
        /// Gets whether this wrapper owns the buffer (and should dispose it).
        /// </summary>
        public bool OwnsBuffer => _ownsBuffer;

        public OwnedBuffer(IGpuBuffer buffer, bool ownsBuffer)
        {
            _buffer = buffer;
            _ownsBuffer = ownsBuffer;
        }

        public void Dispose()
        {
            if (_ownsBuffer)
                _buffer.Dispose();
        }
    }

    /// <summary>
    /// Gets a GPU buffer for the tensor data, using cache if available.
    /// Returns an OwnedBuffer that only disposes if we allocated it (not cached).
    /// Checks both persistent tensor cache (weights/biases) and activation cache (layer outputs).
    /// Thread-safe: uses lock to prevent use-after-dispose during cache eviction.
    /// </summary>
    private OwnedBuffer GetOrAllocateBuffer<T>(IDirectGpuBackend backend, T[] data)
    {
        // First check persistent tensor cache (for weights/biases)
        var cached = TryGetCachedBuffer(data);
        if (cached != null)
            return new OwnedBuffer(cached, ownsBuffer: false);

        // Check activation cache (for intermediate layer outputs)
        // Only reuse if the cached buffer was created by the same backend to avoid cross-backend issues
        // Use lock to prevent eviction/clear while we're using the cached buffer
        lock (_activationCacheLock)
        {
            if (_activationCache.TryGetValue(data, out var activationEntry) &&
                ReferenceEquals(activationEntry.Backend, backend))
            {
                // Found in activation cache with matching backend - reuse buffer without re-uploading
                // Return with ownsBuffer=false so we don't dispose the cached buffer
                return new OwnedBuffer(activationEntry.Buffer, ownsBuffer: false);
            }
        }

        // Not cached - need to upload
        float[] floatData = DirectGpuEngine.ToFloatArray(data);
        return new OwnedBuffer(backend.AllocateBuffer(floatData), ownsBuffer: true);
    }

    /// <summary>
    /// Caches the result buffer for potential reuse by the next layer.
    /// The result data array serves as the cache key.
    /// Thread-safe: uses lock to coordinate with cache lookups.
    /// </summary>
    private void CacheActivation<T>(T[] resultData, IGpuBuffer buffer, int[] shape, IDirectGpuBackend backend)
    {
        lock (_activationCacheLock)
        {
            // Evict old entries if cache is full
            if (_activationCache.Count >= _maxActivationCacheSize)
            {
                EvictOldestActivationsUnsafe();
            }

            var timestamp = System.Threading.Interlocked.Increment(ref _activationCacheTimestamp);
            var entry = new ActivationCacheEntry(buffer, shape, timestamp, backend);
            bool added = false;
            try
            {
                added = _activationCache.TryAdd(resultData, entry);
            }
            finally
            {
                if (!added)
                {
                    // Entry was not added (key already exists or exception occurred); dispose to avoid leaking the buffer.
                    entry.Dispose();
                }
            }
        }
    }

    /// <summary>
    /// Evicts the oldest half of the activation cache entries.
    /// Must be called while holding _activationCacheLock.
    /// </summary>
    private void EvictOldestActivationsUnsafe()
    {
        var entries = _activationCache.ToArray();
        if (entries.Length == 0) return;

        // Sort by timestamp (oldest first)
        var sorted = entries.OrderBy(e => e.Value.Timestamp).ToArray();

        // Remove oldest half
        int removeCount = sorted.Length / 2;
        for (int i = 0; i < removeCount; i++)
        {
            if (_activationCache.TryRemove(sorted[i].Key, out var removed))
            {
                removed.Dispose();
            }
        }
    }

    /// <summary>
    /// Clears the activation cache to free GPU memory.
    /// Call this between inference batches if memory is tight.
    /// Thread-safe: uses lock to prevent clearing while buffers are in use.
    /// </summary>
    public void ClearActivationCache()
    {
        lock (_activationCacheLock)
        {
            foreach (var entry in _activationCache.Values)
            {
                entry.Dispose();
            }
            _activationCache.Clear();
        }
    }

    /// <summary>
    /// Gets a GPU buffer for weight/bias tensor, auto-caching if not already persistent.
    /// Unlike GetOrAllocateBuffer, this caches the buffer in the persistent cache
    /// so subsequent calls reuse the same GPU buffer without re-uploading.
    /// Thread-safe: uses lock to coordinate with cache invalidation.
    /// </summary>
    private OwnedBuffer GetOrCacheWeightBuffer<T>(IDirectGpuBackend backend, T[] data, PersistentTensorRole role)
    {
        lock (_persistentBufferLock)
        {
            // First check persistent tensor cache
            var cached = TryGetCachedBuffer(data);
            if (cached != null)
                return new OwnedBuffer(cached, ownsBuffer: false);

            // Not cached - upload and cache for future use
            float[] floatData = DirectGpuEngine.ToFloatArray(data);
            IGpuBuffer gpuBuffer = backend.AllocateBuffer(floatData);

            // Add to persistent cache so future calls don't re-upload
            var entry = new GpuBufferCacheEntry(gpuBuffer, role);
            if (_persistentBufferCache.TryAdd(data, entry))
            {
                _tensorVersions.TryAdd(data, 0);
                // Return with ownsBuffer=false since cache now owns it
                return new OwnedBuffer(gpuBuffer, ownsBuffer: false);
            }
            else
            {
                // Another thread may have cached it; try to use that one
                var alreadyCached = TryGetCachedBuffer(data);
                if (alreadyCached != null)
                {
                    gpuBuffer.Dispose();
                    return new OwnedBuffer(alreadyCached, ownsBuffer: false);
                }

                // Entry was removed between TryAdd and lookup; fall back to our buffer
                return new OwnedBuffer(gpuBuffer, ownsBuffer: true);
            }
        }
    }

    /// <summary>
    /// Allocates a GPU buffer from span data (no caching, avoids ToArray() allocation).
    /// </summary>
    private OwnedBuffer AllocateBufferFromSpan<T>(IDirectGpuBackend backend, ReadOnlySpan<T> data)
    {
        // Convert via numeric operations directly to float array
        // ToFloatSpan has built-in fast path for T=float
        float[] result = new float[data.Length];
        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.ToFloatSpan(data, new Span<float>(result));
        return new OwnedBuffer(backend.AllocateBuffer(result), ownsBuffer: true);
    }

    /// <summary>
    /// Allocates a new output buffer (always owned, never cached).
    /// </summary>
    private static OwnedBuffer AllocateOutputBuffer(IDirectGpuBackend backend, int size)
    {
        return new OwnedBuffer(backend.AllocateBuffer(size), ownsBuffer: true);
    }

    private T[]? TryRunUnary<T>(T[] input, Action<IDirectGpuBackend, IGpuBuffer, IGpuBuffer, int> op)
    {
        if (!TryGetBackend(out var backend))
            return null;

        using var bufferA = GetOrAllocateBuffer(backend, input);
        using var bufferB = AllocateOutputBuffer(backend, input.Length);
        op(backend, bufferA.Buffer, bufferB.Buffer, input.Length);
        // Note: DownloadBuffer uses blocking read (clEnqueueReadBuffer blocking=true),
        // so Synchronize() is redundant and has been removed for performance
        float[] resultFloat = backend.DownloadBuffer(bufferB.Buffer);
        return DirectGpuEngine.FromFloatArray<T>(resultFloat);
    }

    private T[]? TryRunBinary<T>(T[] left, T[] right, Action<IDirectGpuBackend, IGpuBuffer, IGpuBuffer, IGpuBuffer, int> op)
    {
        if (!TryGetBackend(out var backend))
            return null;
        if (left.Length != right.Length)
            return null;

        using var bufferA = GetOrAllocateBuffer(backend, left);
        using var bufferB = GetOrAllocateBuffer(backend, right);
        using var bufferC = AllocateOutputBuffer(backend, left.Length);
        op(backend, bufferA.Buffer, bufferB.Buffer, bufferC.Buffer, left.Length);
        // Note: DownloadBuffer uses blocking read, Synchronize() removed for performance
        float[] resultFloat = backend.DownloadBuffer(bufferC.Buffer);
        return DirectGpuEngine.FromFloatArray<T>(resultFloat);
    }

    private T[]? TryRunScalar<T>(T[] input, T scalar, Action<IDirectGpuBackend, IGpuBuffer, IGpuBuffer, float, int> op)
    {
        if (!TryGetBackend(out var backend))
            return null;

        using var bufferA = GetOrAllocateBuffer(backend, input);
        using var bufferB = AllocateOutputBuffer(backend, input.Length);
        op(backend, bufferA.Buffer, bufferB.Buffer, ToFloatScalar(scalar), input.Length);
        // Note: DownloadBuffer uses blocking read, Synchronize() removed for performance
        float[] resultFloat = backend.DownloadBuffer(bufferB.Buffer);
        return DirectGpuEngine.FromFloatArray<T>(resultFloat);
    }

    /// <summary>
    /// Span-based binary operation that avoids ToArray() allocation for matrix operations.
    /// </summary>
    private T[]? TryRunBinarySpan<T>(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Action<IDirectGpuBackend, IGpuBuffer, IGpuBuffer, IGpuBuffer, int> op)
    {
        if (!TryGetBackend(out var backend))
            return null;
        if (left.Length != right.Length)
            return null;

        using var bufferA = AllocateBufferFromSpan(backend, left);
        using var bufferB = AllocateBufferFromSpan(backend, right);
        using var bufferC = AllocateOutputBuffer(backend, left.Length);
        op(backend, bufferA.Buffer, bufferB.Buffer, bufferC.Buffer, left.Length);
        // Note: DownloadBuffer uses blocking read, Synchronize() removed for performance
        float[] resultFloat = backend.DownloadBuffer(bufferC.Buffer);
        return DirectGpuEngine.FromFloatArray<T>(resultFloat);
    }

    /// <summary>
    /// Span-based scalar operation that avoids ToArray() allocation for matrix operations.
    /// </summary>
    private T[]? TryRunScalarSpan<T>(ReadOnlySpan<T> input, T scalar, Action<IDirectGpuBackend, IGpuBuffer, IGpuBuffer, float, int> op)
    {
        if (!TryGetBackend(out var backend))
            return null;

        using var bufferA = AllocateBufferFromSpan(backend, input);
        using var bufferB = AllocateOutputBuffer(backend, input.Length);
        op(backend, bufferA.Buffer, bufferB.Buffer, ToFloatScalar(scalar), input.Length);
        // Note: DownloadBuffer uses blocking read, Synchronize() removed for performance
        float[] resultFloat = backend.DownloadBuffer(bufferB.Buffer);
        return DirectGpuEngine.FromFloatArray<T>(resultFloat);
    }

    private static bool ShapesMatch(int[] left, int[] right)
    {
        return left.Length == right.Length && left.SequenceEqual(right);
    }

    Vector<T> IEngine.Add<T>(Vector<T> a, Vector<T> b)
    {
        var result = TryRunBinary(a.Data, b.Data, static (backend, left, right, output, size) => backend.Add(left, right, output, size));
        return result != null ? new Vector<T>(result) : base.Add(a, b);
    }

    Vector<T> IEngine.Subtract<T>(Vector<T> a, Vector<T> b)
    {
        var result = TryRunBinary(a.Data, b.Data, static (backend, left, right, output, size) => backend.Subtract(left, right, output, size));
        return result != null ? new Vector<T>(result) : base.Subtract(a, b);
    }

    Vector<T> IEngine.Multiply<T>(Vector<T> a, Vector<T> b)
    {
        var result = TryRunBinary(a.Data, b.Data, static (backend, left, right, output, size) => backend.Multiply(left, right, output, size));
        return result != null ? new Vector<T>(result) : base.Multiply(a, b);
    }

    Vector<T> IEngine.Multiply<T>(Vector<T> vector, T scalar)
    {
        var result = TryRunScalar(vector.Data, scalar, static (backend, input, output, value, size) => backend.Scale(input, output, value, size));
        return result != null ? new Vector<T>(result) : base.Multiply(vector, scalar);
    }

    Vector<T> IEngine.Divide<T>(Vector<T> a, Vector<T> b)
    {
        var result = TryRunBinary(a.Data, b.Data, static (backend, left, right, output, size) => backend.Divide(left, right, output, size));
        return result != null ? new Vector<T>(result) : base.Divide(a, b);
    }

    Vector<T> IEngine.Divide<T>(Vector<T> vector, T scalar)
    {
        var scalarValue = ToFloatScalar(scalar);
        if (scalarValue == 0)
            return base.Divide(vector, scalar);

        var result = TryRunScalar(vector.Data, scalar, static (backend, input, output, value, size) => backend.Scale(input, output, 1.0f / value, size));
        return result != null ? new Vector<T>(result) : base.Divide(vector, scalar);
    }

    Vector<T> IEngine.Max<T>(Vector<T> a, Vector<T> b)
    {
        var result = TryRunBinary(a.Data, b.Data, static (backend, left, right, output, size) => backend.Max(left, right, output, size));
        return result != null ? new Vector<T>(result) : base.Max(a, b);
    }

    Vector<T> IEngine.Min<T>(Vector<T> a, Vector<T> b)
    {
        var result = TryRunBinary(a.Data, b.Data, static (backend, left, right, output, size) => backend.Min(left, right, output, size));
        return result != null ? new Vector<T>(result) : base.Min(a, b);
    }

    Vector<T> IEngine.Abs<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.Data, static (backend, input, output, size) => backend.Abs(input, output, size));
        return result != null ? new Vector<T>(result) : base.Abs(vector);
    }

    Vector<T> IEngine.Exp<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.Data, static (backend, input, output, size) => backend.Exp(input, output, size));
        return result != null ? new Vector<T>(result) : base.Exp(vector);
    }

    Vector<T> IEngine.Exp2<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.Data, static (backend, input, output, size) => backend.Exp2(input, output, size));
        return result != null ? new Vector<T>(result) : base.Exp2(vector);
    }

    Vector<T> IEngine.Exp10<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.Data, static (backend, input, output, size) => backend.Exp10(input, output, size));
        return result != null ? new Vector<T>(result) : base.Exp10(vector);
    }

    Vector<T> IEngine.Log<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.Data, static (backend, input, output, size) => backend.Log(input, output, size));
        return result != null ? new Vector<T>(result) : base.Log(vector);
    }

    Vector<T> IEngine.Log2<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.Data, static (backend, input, output, size) => backend.Log2(input, output, size));
        return result != null ? new Vector<T>(result) : base.Log2(vector);
    }

    Vector<T> IEngine.Sqrt<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.Data, static (backend, input, output, size) => backend.Sqrt(input, output, size));
        return result != null ? new Vector<T>(result) : base.Sqrt(vector);
    }

    Vector<T> IEngine.Power<T>(Vector<T> vector, T exponent)
    {
        var result = TryRunScalar(vector.Data, exponent, static (backend, input, output, value, size) => backend.Power(input, output, value, size));
        return result != null ? new Vector<T>(result) : base.Power(vector, exponent);
    }

    Vector<T> IEngine.Tanh<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.Data, static (backend, input, output, size) => backend.Tanh(input, output, size));
        return result != null ? new Vector<T>(result) : base.Tanh(vector);
    }

    Vector<T> IEngine.Sigmoid<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.Data, static (backend, input, output, size) => backend.Sigmoid(input, output, size));
        return result != null ? new Vector<T>(result) : base.Sigmoid(vector);
    }

    Vector<T> IEngine.ReLU<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.Data, static (backend, input, output, size) => backend.Relu(input, output, size));
        return result != null ? new Vector<T>(result) : base.ReLU(vector);
    }

    Vector<T> IEngine.GELU<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.Data, static (backend, input, output, size) => backend.Gelu(input, output, size));
        return result != null ? new Vector<T>(result) : base.GELU(vector);
    }

    Matrix<T> IEngine.MatrixMultiply<T>(Matrix<T> a, Matrix<T> b)
    {
        if (!IsGpuAvailable || _directGpu == null)
            return base.MatrixMultiply(a, b);

        if (a.Columns != b.Rows)
            return base.MatrixMultiply(a, b);

        try
        {
            var resultData = _directGpu.MatMul(a.AsSpan().ToArray(), b.AsSpan().ToArray(), a.Rows, a.Columns, b.Columns);
            if (resultData == null)
                return base.MatrixMultiply(a, b);

            return new Matrix<T>(a.Rows, b.Columns, resultData);
        }
        catch
        {
            return base.MatrixMultiply(a, b);
        }
    }

    Matrix<T> IEngine.MatrixAdd<T>(Matrix<T> a, Matrix<T> b)
    {
        if (a.Rows != b.Rows || a.Columns != b.Columns)
            return base.MatrixAdd(a, b);

        // Use span-based method to avoid ToArray() allocation
        var result = TryRunBinarySpan(a.AsSpan(), b.AsSpan(), static (backend, left, right, output, size) => backend.Add(left, right, output, size));
        if (result == null)
            return base.MatrixAdd(a, b);

        var matrix = new Matrix<T>(a.Rows, a.Columns);
        result.AsSpan().CopyTo(matrix.AsWritableSpan());
        return matrix;
    }

    Matrix<T> IEngine.MatrixSubtract<T>(Matrix<T> a, Matrix<T> b)
    {
        if (a.Rows != b.Rows || a.Columns != b.Columns)
            return base.MatrixSubtract(a, b);

        // Use span-based method to avoid ToArray() allocation
        var result = TryRunBinarySpan(a.AsSpan(), b.AsSpan(), static (backend, left, right, output, size) => backend.Subtract(left, right, output, size));
        if (result == null)
            return base.MatrixSubtract(a, b);

        var matrix = new Matrix<T>(a.Rows, a.Columns);
        result.AsSpan().CopyTo(matrix.AsWritableSpan());
        return matrix;
    }

    Matrix<T> IEngine.MatrixMultiplyScalar<T>(Matrix<T> matrix, T scalar)
    {
        // Use span-based method to avoid ToArray() allocation
        var result = TryRunScalarSpan(matrix.AsSpan(), scalar, static (backend, input, output, value, size) => backend.Scale(input, output, value, size));
        if (result == null)
            return base.MatrixMultiplyScalar(matrix, scalar);

        var output = new Matrix<T>(matrix.Rows, matrix.Columns);
        result.AsSpan().CopyTo(output.AsWritableSpan());
        return output;
    }

    Tensor<T> IEngine.TensorAdd<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!ShapesMatch(a.Shape, b.Shape))
            return base.TensorAdd(a, b);

        var result = TryRunBinary(a.Data, b.Data, static (backend, left, right, output, size) => backend.Add(left, right, output, size));
        return result != null ? new Tensor<T>(result, a.Shape) : base.TensorAdd(a, b);
    }

    Tensor<T> IEngine.TensorSubtract<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!ShapesMatch(a.Shape, b.Shape))
            return base.TensorSubtract(a, b);

        var result = TryRunBinary(a.Data, b.Data, static (backend, left, right, output, size) => backend.Subtract(left, right, output, size));
        return result != null ? new Tensor<T>(result, a.Shape) : base.TensorSubtract(a, b);
    }

    Tensor<T> IEngine.TensorMultiply<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!ShapesMatch(a.Shape, b.Shape))
            return base.TensorMultiply(a, b);

        var result = TryRunBinary(a.Data, b.Data, static (backend, left, right, output, size) => backend.Multiply(left, right, output, size));
        return result != null ? new Tensor<T>(result, a.Shape) : base.TensorMultiply(a, b);
    }

    Tensor<T> IEngine.TensorDivide<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!ShapesMatch(a.Shape, b.Shape))
            return base.TensorDivide(a, b);

        var result = TryRunBinary(a.Data, b.Data, static (backend, left, right, output, size) => backend.Divide(left, right, output, size));
        return result != null ? new Tensor<T>(result, a.Shape) : base.TensorDivide(a, b);
    }

    Tensor<T> IEngine.TensorMultiplyScalar<T>(Tensor<T> tensor, T scalar)
    {
        var result = TryRunScalar(tensor.Data, scalar, static (backend, input, output, value, size) => backend.Scale(input, output, value, size));
        return result != null ? new Tensor<T>(result, tensor.Shape) : base.TensorMultiplyScalar(tensor, scalar);
    }

    Tensor<T> IEngine.TensorDivideScalar<T>(Tensor<T> tensor, T scalar)
    {
        var scalarValue = ToFloatScalar(scalar);
        if (scalarValue == 0)
            return base.TensorDivideScalar(tensor, scalar);

        var result = TryRunScalar(tensor.Data, scalar, static (backend, input, output, value, size) => backend.Scale(input, output, 1.0f / value, size));
        return result != null ? new Tensor<T>(result, tensor.Shape) : base.TensorDivideScalar(tensor, scalar);
    }

    Tensor<T> IEngine.TensorAbs<T>(Tensor<T> tensor)
    {
        var result = TryRunUnary(tensor.Data, static (backend, input, output, size) => backend.Abs(input, output, size));
        return result != null ? new Tensor<T>(result, tensor.Shape) : base.TensorAbs(tensor);
    }

    Tensor<T> IEngine.TensorExp<T>(Tensor<T> tensor)
    {
        var result = TryRunUnary(tensor.Data, static (backend, input, output, size) => backend.Exp(input, output, size));
        return result != null ? new Tensor<T>(result, tensor.Shape) : base.TensorExp(tensor);
    }

    Tensor<T> IEngine.TensorLog<T>(Tensor<T> tensor)
    {
        var result = TryRunUnary(tensor.Data, static (backend, input, output, size) => backend.Log(input, output, size));
        return result != null ? new Tensor<T>(result, tensor.Shape) : base.TensorLog(tensor);
    }

    Tensor<T> IEngine.TensorSqrt<T>(Tensor<T> tensor)
    {
        var result = TryRunUnary(tensor.Data, static (backend, input, output, size) => backend.Sqrt(input, output, size));
        return result != null ? new Tensor<T>(result, tensor.Shape) : base.TensorSqrt(tensor);
    }

    Tensor<T> IEngine.TensorNegate<T>(Tensor<T> tensor)
    {
        var result = TryRunScalar(tensor.Data, FromFloatScalar<T>(-1.0f), static (backend, input, output, value, size) => backend.Scale(input, output, value, size));
        return result != null ? new Tensor<T>(result, tensor.Shape) : base.TensorNegate(tensor);
    }

    Tensor<T> IEngine.TensorPower<T>(Tensor<T> tensor, T exponent)
    {
        var result = TryRunScalar(tensor.Data, exponent, static (backend, input, output, value, size) => backend.Power(input, output, value, size));
        return result != null ? new Tensor<T>(result, tensor.Shape) : base.TensorPower(tensor, exponent);
    }

    Tensor<T> IEngine.TensorMax<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!ShapesMatch(a.Shape, b.Shape))
            return base.TensorMax(a, b);

        var result = TryRunBinary(a.Data, b.Data, static (backend, left, right, output, size) => backend.Max(left, right, output, size));
        return result != null ? new Tensor<T>(result, a.Shape) : base.TensorMax(a, b);
    }

    Tensor<T> IEngine.TensorMin<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!ShapesMatch(a.Shape, b.Shape))
            return base.TensorMin(a, b);

        var result = TryRunBinary(a.Data, b.Data, static (backend, left, right, output, size) => backend.Min(left, right, output, size));
        return result != null ? new Tensor<T>(result, a.Shape) : base.TensorMin(a, b);
    }

    Tensor<T> IEngine.Tanh<T>(Tensor<T> tensor)
    {
        var result = TryRunUnary(tensor.Data, static (backend, input, output, size) => backend.Tanh(input, output, size));
        return result != null ? new Tensor<T>(result, tensor.Shape) : base.Tanh(tensor);
    }

    Tensor<T> IEngine.Sigmoid<T>(Tensor<T> tensor)
    {
        var result = TryRunUnary(tensor.Data, static (backend, input, output, size) => backend.Sigmoid(input, output, size));
        return result != null ? new Tensor<T>(result, tensor.Shape) : base.Sigmoid(tensor);
    }

    Tensor<T> IEngine.ReLU<T>(Tensor<T> tensor)
    {
        var result = TryRunUnary(tensor.Data, static (backend, input, output, size) => backend.Relu(input, output, size));
        return result != null ? new Tensor<T>(result, tensor.Shape) : base.ReLU(tensor);
    }

    Tensor<T> IEngine.GELU<T>(Tensor<T> tensor)
    {
        var result = TryRunUnary(tensor.Data, static (backend, input, output, size) => backend.Gelu(input, output, size));
        return result != null ? new Tensor<T>(result, tensor.Shape) : base.GELU(tensor);
    }

    T IEngine.TensorSum<T>(Tensor<T> tensor)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorSum(tensor);

        using var bufferA = GetOrAllocateBuffer(backend, tensor.Data);
        backend.Synchronize();
        float sum = backend.Sum(bufferA.Buffer, tensor.Length);
        return FromFloatScalar<T>(sum);
    }

    T IEngine.TensorMaxValue<T>(Tensor<T> tensor)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorMaxValue(tensor);

        using var bufferA = GetOrAllocateBuffer(backend, tensor.Data);
        backend.Synchronize();
        float max = backend.Max(bufferA.Buffer, tensor.Length);
        return FromFloatScalar<T>(max);
    }

    #region Fused Operations

    /// <summary>
    /// GPU-accelerated fused linear transformation: output = activation(input @ weights + bias).
    /// Uses cached GPU buffers for registered persistent tensors (weights/bias) to avoid
    /// redundant CPU→GPU transfers on every forward pass.
    /// </summary>
    public new Tensor<T> FusedLinear<T>(Tensor<T> input, Tensor<T> weights, Tensor<T>? bias, FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            return base.FusedLinear(input, weights, bias, activation);

        if (input.Rank < 1 || weights.Rank != 2)
            return base.FusedLinear(input, weights, bias, activation);

        int batchSize = input.Shape[0];
        int inputFeatures = weights.Shape[0];
        int outputFeatures = weights.Shape[1];

        // Use cache-aware buffer allocation (OwnedBuffer auto-disposes only if we allocated)
        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        // Auto-cache weights and biases so they stay on GPU for subsequent calls
        using var weightsBuffer = GetOrCacheWeightBuffer(backend, weights.Data, PersistentTensorRole.Weights);
        using var biasBuffer = bias != null ? GetOrCacheWeightBuffer(backend, bias.Data, PersistentTensorRole.Biases) : default;

        try
        {
            IGpuBuffer resultBuffer;

            // Use fused GPU kernels when available
            // Only use GPU path for natively supported fused ops (with bias)
            // For cases with bias and activation
            if (bias != null && activation != FusedActivationType.None)
            {
                // Use fused kernels for common activations (most efficient)
                switch (activation)
                {
                    case FusedActivationType.ReLU:
                        resultBuffer = backend.GemmBiasRelu(inputBuffer.Buffer, weightsBuffer.Buffer, biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures);
                        break;
                    case FusedActivationType.GELU:
                        resultBuffer = backend.GemmBiasGelu(inputBuffer.Buffer, weightsBuffer.Buffer, biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures);
                        break;
                    case FusedActivationType.Sigmoid:
                        resultBuffer = backend.GemmBiasSigmoid(inputBuffer.Buffer, weightsBuffer.Buffer, biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures);
                        break;
                    case FusedActivationType.Tanh:
                        resultBuffer = backend.GemmBiasTanh(inputBuffer.Buffer, weightsBuffer.Buffer, biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures);
                        break;
                    default:
                        // For other activations (LeakyReLU, Swish, etc.), use GemmBias + separate activation kernel
                        resultBuffer = backend.GemmBias(inputBuffer.Buffer, weightsBuffer.Buffer, biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures);
                        int size = batchSize * outputFeatures;
                        ApplyGpuActivation(backend, resultBuffer, size, activation);
                        break;
                }
            }
            else if (bias != null && activation == FusedActivationType.None)
            {
                // GEMM + Bias only (no activation) - use GPU GemmBias kernel
                resultBuffer = backend.GemmBias(inputBuffer.Buffer, weightsBuffer.Buffer, biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures);
            }
            else if (bias == null && activation == FusedActivationType.None)
            {
                // Simple MatMul only - use GPU
                resultBuffer = backend.MatMul(inputBuffer.Buffer, weightsBuffer.Buffer, batchSize, outputFeatures, inputFeatures);
            }
            else if (bias == null && activation != FusedActivationType.None)
            {
                // MatMul + activation (no bias) - use GPU MatMul followed by activation
                resultBuffer = backend.MatMul(inputBuffer.Buffer, weightsBuffer.Buffer, batchSize, outputFeatures, inputFeatures);
                int size = batchSize * outputFeatures;
                ApplyGpuActivation(backend, resultBuffer, size, activation);
            }
            else
            {
                // Fall back to CPU for other combinations (should not reach here now)
                return base.FusedLinear(input, weights, bias, activation);
            }

            // Download result - DownloadBuffer uses blocking read, Synchronize() removed for performance
            int resultSize = batchSize * outputFeatures;
            float[] resultFloat = new float[resultSize];
            backend.DownloadBuffer(resultBuffer, resultFloat);

            // Convert back to T
            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            int[] resultShape = new[] { batchSize, outputFeatures };

            // Cache the result buffer for potential reuse by the next layer
            // The next layer's input will be this layer's output (same data array)
            // So when GetOrAllocateBuffer is called with resultData, it can reuse this buffer
            CacheActivation(resultData, resultBuffer, resultShape, backend);

            return new Tensor<T>(resultData, resultShape);
        }
        catch
        {
            // Fall back to CPU on any GPU error
            return base.FusedLinear(input, weights, bias, activation);
        }
    }

    /// <summary>
    /// GPU-resident fused linear transformation that keeps result on GPU.
    /// Returns an IGpuTensor that can be passed to subsequent GPU operations
    /// without CPU round-trips. Only download the final result using ToTensor().
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">Input tensor (will be uploaded to GPU).</param>
    /// <param name="weights">Weight tensor (cached if registered).</param>
    /// <param name="bias">Optional bias tensor (cached if registered).</param>
    /// <param name="activation">Activation function to fuse.</param>
    /// <returns>GPU-resident tensor with the result. Caller must dispose this tensor to free GPU memory.</returns>
    /// <remarks>
    /// The returned tensor owns its GPU buffer. In GPU-resident workflows, these tensors should be
    /// disposed when no longer needed to prevent GPU memory leaks. Use 'using' statements or explicit
    /// Dispose() calls to ensure proper cleanup.
    /// </remarks>
    public IGpuTensor<T> FusedLinearGpu<T>(Tensor<T> input, Tensor<T> weights, Tensor<T>? bias, FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for FusedLinearGpu");

        if (input.Rank < 1 || weights.Rank != 2)
            throw new ArgumentException("Invalid tensor dimensions for FusedLinearGpu");

        int batchSize = input.Shape[0];
        int inputFeatures = weights.Shape[0];
        int outputFeatures = weights.Shape[1];

        // Upload input to GPU (activations are not cached persistently)
        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        // Auto-cache weights and biases so they stay on GPU for subsequent calls
        using var weightsBuffer = GetOrCacheWeightBuffer(backend, weights.Data, PersistentTensorRole.Weights);
        using var biasBuffer = bias != null ? GetOrCacheWeightBuffer(backend, bias.Data, PersistentTensorRole.Biases) : default;

        // Execute the fused kernel and get result buffer
        var resultBuffer = ExecuteFusedLinearKernel(backend, inputBuffer.Buffer, weightsBuffer.Buffer,
            biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures, activation);

        // Return GPU-resident tensor - NO DOWNLOAD
        // IMPORTANT: Caller is responsible for disposing the returned tensor to free GPU memory
        return new GpuTensor<T>(backend, resultBuffer, new[] { batchSize, outputFeatures },
            GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident fused linear transformation with GPU-resident input.
    /// Avoids re-uploading input that's already on GPU from a previous layer.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor.</param>
    /// <param name="weights">Weight tensor (cached if registered).</param>
    /// <param name="bias">Optional bias tensor (cached if registered).</param>
    /// <param name="activation">Activation function to fuse.</param>
    /// <returns>GPU-resident tensor with the result. Caller must dispose this tensor to free GPU memory.</returns>
    /// <remarks>
    /// The returned tensor owns its GPU buffer. In GPU-resident workflows, these tensors should be
    /// disposed when no longer needed to prevent GPU memory leaks. Use 'using' statements or explicit
    /// Dispose() calls to ensure proper cleanup.
    /// </remarks>
    public IGpuTensor<T> FusedLinearGpu<T>(IGpuTensor<T> input, Tensor<T> weights, Tensor<T>? bias, FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for FusedLinearGpu");

        if (input.Shape.Length < 1 || weights.Rank != 2)
            throw new ArgumentException("Invalid tensor dimensions for FusedLinearGpu");

        int batchSize = input.Shape[0];
        int inputFeatures = weights.Shape[0];
        int outputFeatures = weights.Shape[1];

        // Input is already on GPU - use its buffer directly
        // Auto-cache weights and biases so they stay on GPU for subsequent calls
        using var weightsBuffer = GetOrCacheWeightBuffer(backend, weights.Data, PersistentTensorRole.Weights);
        using var biasBuffer = bias != null ? GetOrCacheWeightBuffer(backend, bias.Data, PersistentTensorRole.Biases) : default;

        // Execute the fused kernel and get result buffer
        var resultBuffer = ExecuteFusedLinearKernel(backend, input.Buffer, weightsBuffer.Buffer,
            biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures, activation);

        // Return GPU-resident tensor - NO DOWNLOAD
        return new GpuTensor<T>(backend, resultBuffer, new[] { batchSize, outputFeatures },
            GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// Executes the fused linear kernel and returns the result buffer.
    /// Shared implementation for both CPU and GPU input variants.
    /// </summary>
    private static IGpuBuffer ExecuteFusedLinearKernel(
        IDirectGpuBackend backend,
        IGpuBuffer inputBuffer,
        IGpuBuffer weightsBuffer,
        IGpuBuffer? biasBuffer,
        int batchSize,
        int outputFeatures,
        int inputFeatures,
        FusedActivationType activation)
    {
        IGpuBuffer resultBuffer;

        // Use fused GPU kernels when available
        if (biasBuffer != null && activation != FusedActivationType.None)
        {
            // Use fused kernels for common activations (most efficient)
            switch (activation)
            {
                case FusedActivationType.ReLU:
                    resultBuffer = backend.GemmBiasRelu(inputBuffer, weightsBuffer, biasBuffer, batchSize, outputFeatures, inputFeatures);
                    break;
                case FusedActivationType.GELU:
                    resultBuffer = backend.GemmBiasGelu(inputBuffer, weightsBuffer, biasBuffer, batchSize, outputFeatures, inputFeatures);
                    break;
                case FusedActivationType.Sigmoid:
                    resultBuffer = backend.GemmBiasSigmoid(inputBuffer, weightsBuffer, biasBuffer, batchSize, outputFeatures, inputFeatures);
                    break;
                case FusedActivationType.Tanh:
                    resultBuffer = backend.GemmBiasTanh(inputBuffer, weightsBuffer, biasBuffer, batchSize, outputFeatures, inputFeatures);
                    break;
                default:
                    // For other activations, use GemmBias + separate activation kernel
                    resultBuffer = backend.GemmBias(inputBuffer, weightsBuffer, biasBuffer, batchSize, outputFeatures, inputFeatures);
                    int size = batchSize * outputFeatures;
                    ApplyGpuActivation(backend, resultBuffer, size, activation);
                    break;
            }
        }
        else if (biasBuffer != null && activation == FusedActivationType.None)
        {
            // GEMM + Bias only (no activation)
            resultBuffer = backend.GemmBias(inputBuffer, weightsBuffer, biasBuffer, batchSize, outputFeatures, inputFeatures);
        }
        else if (biasBuffer == null && activation == FusedActivationType.None)
        {
            // Simple MatMul only
            resultBuffer = backend.MatMul(inputBuffer, weightsBuffer, batchSize, outputFeatures, inputFeatures);
        }
        else
        {
            // MatMul + activation (no bias)
            resultBuffer = backend.MatMul(inputBuffer, weightsBuffer, batchSize, outputFeatures, inputFeatures);
            int size = batchSize * outputFeatures;
            ApplyGpuActivation(backend, resultBuffer, size, activation);
        }

        return resultBuffer;
    }

    private static IGpuBuffer GemmBiasNoActivation(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer bias, int M, int N, int K)
    {
        // Use GemmBiasRelu with a subsequent inverse to get just GEMM + Bias
        // This is a workaround since there's no direct GemmBias function
        // Fall back to return just MatMul result and let caller handle bias on CPU
        return backend.MatMul(input, weights, M, N, K);
    }

    private static IGpuBuffer GemmBiasWithActivation(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer bias, int M, int N, int K, FusedActivationType activation)
    {
        // For activations without native fused support, use MatMul + activation
        var result = backend.MatMul(input, weights, M, N, K);
        int size = M * N;
        ApplyGpuActivation(backend, result, size, activation);
        return result;
    }

    private static void ApplyGpuActivation(IDirectGpuBackend backend, IGpuBuffer buffer, int size, FusedActivationType activation)
    {
        switch (activation)
        {
            case FusedActivationType.ReLU:
                backend.Relu(buffer, buffer, size);
                break;
            case FusedActivationType.LeakyReLU:
                backend.LeakyRelu(buffer, buffer, 0.01f, size);
                break;
            case FusedActivationType.Sigmoid:
                backend.Sigmoid(buffer, buffer, size);
                break;
            case FusedActivationType.Tanh:
                backend.Tanh(buffer, buffer, size);
                break;
            case FusedActivationType.GELU:
                backend.Gelu(buffer, buffer, size);
                break;
            case FusedActivationType.Swish:
                backend.Swish(buffer, buffer, size);
                break;
            case FusedActivationType.ELU:
                backend.Elu(buffer, buffer, 1.0f, size); // alpha = 1.0 is standard
                break;
            case FusedActivationType.SELU:
                // SELU: scale * (x if x > 0, else alpha * (exp(x) - 1))
                // Standard parameters: scale ≈ 1.0507, alpha ≈ 1.6733
                backend.Selu(buffer, buffer, 1.6732632423543772f, 1.0507009873554805f, size);
                break;
            case FusedActivationType.Softplus:
                backend.Softplus(buffer, buffer, size);
                break;
            case FusedActivationType.Mish:
                backend.Mish(buffer, buffer, size);
                break;
            case FusedActivationType.HardSwish:
                backend.Hardswish(buffer, buffer, size);
                break;
            case FusedActivationType.HardSigmoid:
                backend.Hardsigmoid(buffer, buffer, size);
                break;
            case FusedActivationType.HardTanh:
                backend.Hardtanh(buffer, buffer, -1.0f, 1.0f, size);
                break;
            case FusedActivationType.None:
                break;
        }
    }

    /// <summary>
    /// Uploads a tensor to GPU memory, returning a GPU-resident tensor handle.
    /// </summary>
    /// <typeparam name="T">The element type of the tensor.</typeparam>
    /// <param name="tensor">The CPU tensor to upload.</param>
    /// <param name="role">The role of this tensor for memory management.</param>
    /// <returns>A GPU-resident tensor that can be used in subsequent GPU operations.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no GPU backend is available.</exception>
    /// <remarks>
    /// <para>
    /// Use this method to explicitly upload data to GPU for use in GPU-resident operations.
    /// The returned tensor can be passed to methods like <see cref="FusedLinearGpu{T}(IGpuTensor{T}, Tensor{T}, Tensor{T}?, FusedActivationType)"/>
    /// to avoid redundant uploads.
    /// </para>
    /// <para>
    /// The caller is responsible for disposing the returned GPU tensor when done.
    /// </para>
    /// </remarks>
    public IGpuTensor<T> UploadToGpu<T>(Tensor<T> tensor, GpuTensorRole role)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for UploadToGpu");

        // Convert tensor data to float and allocate GPU buffer
        float[] floatData = DirectGpuEngine.ToFloatArray(tensor.Data);
        var buffer = backend.AllocateBuffer(floatData);

        // Return GPU tensor that owns the buffer
        return new GpuTensor<T>(backend, buffer, tensor.Shape.ToArray(), role, ownsBuffer: true);
    }

    /// <summary>
    /// Uploads raw float data to GPU memory, returning a GPU-resident tensor handle.
    /// </summary>
    /// <typeparam name="T">The element type of the tensor.</typeparam>
    /// <param name="data">The float data to upload.</param>
    /// <param name="shape">The shape of the resulting tensor.</param>
    /// <param name="role">The role of this tensor for memory management.</param>
    /// <returns>A GPU-resident tensor that can be used in subsequent GPU operations.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no GPU backend is available.</exception>
    public IGpuTensor<T> UploadToGpu<T>(float[] data, int[] shape, GpuTensorRole role)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for UploadToGpu");

        var buffer = backend.AllocateBuffer(data);
        return new GpuTensor<T>(backend, buffer, shape, role, ownsBuffer: true);
    }

    /// <summary>
    /// Uploads weight/bias tensor to GPU with automatic caching. If the data is already cached,
    /// returns the cached GPU tensor without re-uploading. This is the recommended method for
    /// layer weights and biases that don't change between forward passes during inference.
    /// </summary>
    /// <typeparam name="T">The element type of the tensor.</typeparam>
    /// <param name="tensor">The CPU tensor containing weight/bias data.</param>
    /// <param name="role">The role indicating the type of persistent tensor (Weight, Bias, Statistics, AttentionCache, or Constant).</param>
    /// <returns>
    /// A GPU-resident tensor with ownership semantics determined by cache state:
    /// - If cached: ownsBuffer=false, disposing the tensor is safe (no-op, cache retains buffer).
    /// - If not cached (rare race condition): ownsBuffer=true, caller owns the buffer and disposal will free it.
    /// In both cases, disposing the returned tensor is safe and recommended.
    /// </returns>
    /// <exception cref="InvalidOperationException">Thrown when no GPU backend is available.</exception>
    /// <exception cref="ArgumentException">Thrown when an unsupported role is passed (e.g., General, Activation, Gradient).</exception>
    public IGpuTensor<T> GetOrCacheWeightsGpu<T>(Tensor<T> tensor, GpuTensorRole role = GpuTensorRole.Weight)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for GetOrCacheWeightsGpu");

        var persistentRole = role switch
        {
            GpuTensorRole.Weight => PersistentTensorRole.Weights,
            GpuTensorRole.Bias => PersistentTensorRole.Biases,
            GpuTensorRole.Statistics => PersistentTensorRole.NormalizationParams,
            GpuTensorRole.AttentionCache => PersistentTensorRole.AttentionCache,
            GpuTensorRole.Constant => PersistentTensorRole.Constant,
            _ => throw new ArgumentException(
                $"GetOrCacheWeightsGpu only supports Weight, Bias, Statistics, AttentionCache, or Constant roles. " +
                $"Got: {role}. Use UploadToGpu for other tensor types.", nameof(role))
        };
        var ownedBuffer = GetOrCacheWeightBuffer(backend, tensor.Data, persistentRole);

        // Propagate ownership: if cache owns buffer, GpuTensor shouldn't dispose;
        // if we own buffer (race condition fallback), GpuTensor should take ownership
        return new GpuTensor<T>(backend, ownedBuffer.Buffer, tensor.Shape.ToArray(), role, ownsBuffer: ownedBuffer.OwnsBuffer);
    }

    /// <summary>
    /// Invalidates a cached weight buffer, forcing a re-upload on the next GetOrCacheWeightsGpu call.
    /// Thread-safe: Uses _persistentBufferLock to synchronize with GetOrCacheWeightBuffer.
    /// </summary>
    public bool InvalidateWeightCache<T>(T[] data)
    {
        lock (_persistentBufferLock)
        {
            if (_persistentBufferCache.TryRemove(data, out var entry))
            {
                entry.Dispose();
                _tensorVersions.TryRemove(data, out _);
                return true;
            }
            return false;
        }
    }

    /// <summary>
    /// Invalidates all cached weight buffers.
    /// Thread-safe: Uses _persistentBufferLock to synchronize with GetOrCacheWeightBuffer.
    /// </summary>
    public void InvalidateAllWeightCaches()
    {
        lock (_persistentBufferLock)
        {
            foreach (var entry in _persistentBufferCache.Values)
            {
                entry.Dispose();
            }
            _persistentBufferCache.Clear();
            _tensorVersions.Clear();
        }
    }

    /// <summary>
    /// Applies an activation function to a GPU-resident tensor, returning a new GPU tensor.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">The GPU-resident input tensor.</param>
    /// <param name="activation">The activation type to apply.</param>
    /// <returns>A new GPU tensor with the activation applied.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no GPU backend is available.</exception>
    /// <remarks>
    /// <para>
    /// This method applies the specified activation function entirely on the GPU,
    /// without downloading data to CPU. Supported activations: ReLU, LeakyReLU, Sigmoid, Tanh, GELU, Swish.
    /// </para>
    /// </remarks>
    public IGpuTensor<T> ActivationGpu<T>(IGpuTensor<T> input, FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ActivationGpu");

        // Allocate output buffer
        int size = input.ElementCount;
        var outputBuffer = backend.AllocateBuffer(size);

        // Copy input to output first (activations work in-place)
        backend.Copy(input.Buffer, outputBuffer, size);

        // Apply activation in-place on output buffer
        ApplyGpuActivation(backend, outputBuffer, size, activation);

        // Return new GPU tensor
        return new GpuTensor<T>(backend, outputBuffer, input.Shape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// Performs GPU-resident dropout forward pass with random mask generation.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">The GPU-resident input tensor.</param>
    /// <param name="dropoutRate">Probability of dropping each element (0-1).</param>
    /// <param name="isTraining">If true, applies dropout; if false, passes through unchanged.</param>
    /// <param name="seed">Random seed for mask generation (use different seed per batch for variety).</param>
    /// <returns>A tuple of (output tensor, mask tensor) for use in backward pass.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no GPU backend is available.</exception>
    public (IGpuTensor<T> Output, IGpuTensor<T> Mask) DropoutGpu<T>(
        IGpuTensor<T> input,
        float dropoutRate,
        bool isTraining,
        ulong seed)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for DropoutGpu");

        int size = input.ElementCount;

        // Allocate output and mask buffers
        var outputBuffer = backend.AllocateBuffer(size);
        var maskBuffer = backend.AllocateBuffer(size);

        // Run dropout kernel (handles both training and inference modes)
        backend.Dropout(input.Buffer, outputBuffer, maskBuffer, size, dropoutRate, seed, isTraining);

        // Return GPU tensors
        var output = new GpuTensor<T>(backend, outputBuffer, input.Shape, GpuTensorRole.Activation, ownsBuffer: true);
        var mask = new GpuTensor<T>(backend, maskBuffer, input.Shape, GpuTensorRole.Intermediate, ownsBuffer: true);

        return (output, mask);
    }

    /// <summary>
    /// Performs GPU-resident dropout backward pass.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">The GPU-resident gradient from the next layer.</param>
    /// <param name="mask">The dropout mask from the forward pass.</param>
    /// <param name="dropoutRate">The dropout rate used in forward pass.</param>
    /// <returns>The gradient with respect to the input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no GPU backend is available.</exception>
    public IGpuTensor<T> DropoutBackwardGpu<T>(
        IGpuTensor<T> gradOutput,
        IGpuTensor<T> mask,
        float dropoutRate)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for DropoutBackwardGpu");

        int size = gradOutput.ElementCount;
        var gradInputBuffer = backend.AllocateBuffer(size);

        backend.DropoutBackward(gradOutput.Buffer, mask.Buffer, gradInputBuffer, size, dropoutRate);

        return new GpuTensor<T>(backend, gradInputBuffer, gradOutput.Shape, GpuTensorRole.Gradient, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident 2D max pooling that keeps output and indices on GPU.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="input">The input tensor on GPU.</param>
    /// <param name="poolSize">Pool size [height, width].</param>
    /// <param name="stride">Stride [height, width].</param>
    /// <param name="gpuIndices">Output GPU buffer containing pooling indices.</param>
    /// <returns>The pooled output as GPU-resident tensor.</returns>
    public IGpuTensor<T> MaxPool2DGpu<T>(
        IGpuTensor<T> input,
        int[] poolSize,
        int[] stride,
        out IGpuBuffer gpuIndices)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for MaxPool2DGpu");

        if (input.Shape.Length != 4 || poolSize.Length != 2 || stride.Length != 2)
            throw new ArgumentException("Input must be 4D [batch, channels, height, width] with 2D poolSize and stride");

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int outHeight = (inHeight - poolSize[0]) / stride[0] + 1;
        int outWidth = (inWidth - poolSize[1]) / stride[1] + 1;

        if (outHeight <= 0 || outWidth <= 0)
            throw new ArgumentException($"Invalid pooling parameters: output dimensions ({outHeight}, {outWidth}) are non-positive");

        int outputSize = batch * channels * outHeight * outWidth;
        var outputBuffer = backend.AllocateBuffer(outputSize);
        var indicesBuffer = backend.AllocateBuffer(outputSize);

        backend.MaxPool2D(input.Buffer, outputBuffer, indicesBuffer,
            batch, channels, inHeight, inWidth,
            outHeight, outWidth,
            poolSize[0], poolSize[1],
            stride[0], stride[1], 0, 0);

        var outputShape = new[] { batch, channels, outHeight, outWidth };
        gpuIndices = indicesBuffer;

        return new GpuTensor<T>(backend, outputBuffer, outputShape, GpuTensorRole.Intermediate, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident backward pass for 2D max pooling.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="gradOutput">The gradient of the output on GPU.</param>
    /// <param name="gpuIndices">The GPU buffer containing pooling indices from forward pass.</param>
    /// <param name="inputShape">The shape of the original input.</param>
    /// <param name="poolSize">Pool size [height, width].</param>
    /// <param name="stride">Stride [height, width].</param>
    /// <returns>The gradient with respect to input as GPU-resident tensor.</returns>
    public IGpuTensor<T> MaxPool2DBackwardGpu<T>(
        IGpuTensor<T> gradOutput,
        IGpuBuffer gpuIndices,
        int[] inputShape,
        int[] poolSize,
        int[] stride)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for MaxPool2DBackwardGpu");

        if (gradOutput.Shape.Length != 4 || inputShape.Length != 4)
            throw new ArgumentException("GradOutput and inputShape must be 4D");

        int batch = inputShape[0];
        int channels = inputShape[1];
        int inHeight = inputShape[2];
        int inWidth = inputShape[3];

        int gradInputSize = batch * channels * inHeight * inWidth;
        var gradInputBuffer = backend.AllocateBuffer(gradInputSize);

        backend.MaxPool2DBackward(gradOutput.Buffer, gpuIndices, gradInputBuffer,
            batch, channels, inHeight, inWidth,
            gradOutput.Shape[2], gradOutput.Shape[3],
            poolSize[0], poolSize[1],
            stride[0], stride[1], 0, 0);

        return new GpuTensor<T>(backend, gradInputBuffer, inputShape, GpuTensorRole.Gradient, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident 3D max pooling that keeps output and indices on GPU.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="input">The input tensor on GPU with shape [batch, channels, depth, height, width].</param>
    /// <param name="poolSize">Pool size [depth, height, width].</param>
    /// <param name="stride">Stride [depth, height, width].</param>
    /// <param name="gpuIndices">Output GPU buffer containing flat pooling indices.</param>
    /// <returns>The pooled output as GPU-resident tensor.</returns>
    public IGpuTensor<T> MaxPool3DGpu<T>(
        IGpuTensor<T> input,
        int[] poolSize,
        int[] stride,
        out IGpuBuffer gpuIndices)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for MaxPool3DGpu");

        if (input.Shape.Length != 5 || poolSize.Length != 3 || stride.Length != 3)
            throw new ArgumentException("Input must be 5D [batch, channels, depth, height, width] with 3D poolSize and stride");

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int inDepth = input.Shape[2];
        int inHeight = input.Shape[3];
        int inWidth = input.Shape[4];

        int outDepth = (inDepth - poolSize[0]) / stride[0] + 1;
        int outHeight = (inHeight - poolSize[1]) / stride[1] + 1;
        int outWidth = (inWidth - poolSize[2]) / stride[2] + 1;

        if (outDepth <= 0 || outHeight <= 0 || outWidth <= 0)
            throw new ArgumentException($"Invalid pooling parameters: output dimensions ({outDepth}, {outHeight}, {outWidth}) are non-positive");

        int outputSize = batch * channels * outDepth * outHeight * outWidth;
        var outputBuffer = backend.AllocateBuffer(outputSize);
        gpuIndices = backend.AllocateBuffer(outputSize * sizeof(int) / sizeof(float));

        backend.MaxPool3D(input.Buffer, outputBuffer, gpuIndices,
            batch, channels,
            inDepth, inHeight, inWidth,
            outDepth, outHeight, outWidth,
            poolSize[0], poolSize[1], poolSize[2],
            stride[0], stride[1], stride[2]);

        var outputShape = new[] { batch, channels, outDepth, outHeight, outWidth };
        return new GpuTensor<T>(backend, outputBuffer, outputShape, GpuTensorRole.Intermediate, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident backward pass for 3D max pooling.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="gradOutput">The gradient of the output on GPU.</param>
    /// <param name="gpuIndices">The GPU buffer containing flat pooling indices from forward pass.</param>
    /// <param name="inputShape">The shape of the original input [batch, channels, depth, height, width].</param>
    /// <param name="poolSize">Pool size [depth, height, width].</param>
    /// <param name="stride">Stride [depth, height, width].</param>
    /// <returns>The gradient with respect to input as GPU-resident tensor.</returns>
    public IGpuTensor<T> MaxPool3DBackwardGpu<T>(
        IGpuTensor<T> gradOutput,
        IGpuBuffer gpuIndices,
        int[] inputShape,
        int[] poolSize,
        int[] stride)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for MaxPool3DBackwardGpu");

        if (gradOutput.Shape.Length != 5 || inputShape.Length != 5)
            throw new ArgumentException("GradOutput and inputShape must be 5D");

        int batch = inputShape[0];
        int channels = inputShape[1];
        int inDepth = inputShape[2];
        int inHeight = inputShape[3];
        int inWidth = inputShape[4];

        int gradInputSize = batch * channels * inDepth * inHeight * inWidth;
        var gradInputBuffer = backend.AllocateBuffer(gradInputSize);

        backend.MaxPool3DBackward(gradOutput.Buffer, gpuIndices, gradInputBuffer,
            batch, channels,
            inDepth, inHeight, inWidth,
            gradOutput.Shape[2], gradOutput.Shape[3], gradOutput.Shape[4]);

        return new GpuTensor<T>(backend, gradInputBuffer, inputShape, GpuTensorRole.Gradient, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident 3D nearest neighbor upsampling.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="input">The input tensor on GPU with shape [batch, channels, depth, height, width].</param>
    /// <param name="scaleDepth">Scale factor for depth dimension.</param>
    /// <param name="scaleHeight">Scale factor for height dimension.</param>
    /// <param name="scaleWidth">Scale factor for width dimension.</param>
    /// <returns>The upsampled output as GPU-resident tensor.</returns>
    public IGpuTensor<T> NearestNeighborUpsample3DGpu<T>(
        IGpuTensor<T> input,
        int scaleDepth,
        int scaleHeight,
        int scaleWidth)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for NearestNeighborUpsample3DGpu");

        if (input.Shape.Length != 5)
            throw new ArgumentException("Input must be 5D [batch, channels, depth, height, width]");

        if (scaleDepth <= 0 || scaleHeight <= 0 || scaleWidth <= 0)
            throw new ArgumentException("Scale factors must be positive");

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int inDepth = input.Shape[2];
        int inHeight = input.Shape[3];
        int inWidth = input.Shape[4];

        int outDepth = inDepth * scaleDepth;
        int outHeight = inHeight * scaleHeight;
        int outWidth = inWidth * scaleWidth;

        int outputSize = batch * channels * outDepth * outHeight * outWidth;
        var outputBuffer = backend.AllocateBuffer(outputSize);

        backend.NearestNeighborUpsample3D(input.Buffer, outputBuffer,
            batch, channels,
            inDepth, inHeight, inWidth,
            scaleDepth, scaleHeight, scaleWidth);

        var outputShape = new[] { batch, channels, outDepth, outHeight, outWidth };
        return new GpuTensor<T>(backend, outputBuffer, outputShape, GpuTensorRole.Intermediate, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident backward pass for 3D nearest neighbor upsampling.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="gradOutput">The gradient of the output on GPU.</param>
    /// <param name="inputShape">The shape of the original input [batch, channels, depth, height, width].</param>
    /// <param name="scaleDepth">Scale factor for depth dimension.</param>
    /// <param name="scaleHeight">Scale factor for height dimension.</param>
    /// <param name="scaleWidth">Scale factor for width dimension.</param>
    /// <returns>The gradient with respect to input as GPU-resident tensor.</returns>
    public IGpuTensor<T> NearestNeighborUpsample3DBackwardGpu<T>(
        IGpuTensor<T> gradOutput,
        int[] inputShape,
        int scaleDepth,
        int scaleHeight,
        int scaleWidth)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for NearestNeighborUpsample3DBackwardGpu");

        if (gradOutput.Shape.Length != 5 || inputShape.Length != 5)
            throw new ArgumentException("GradOutput and inputShape must be 5D");

        int batch = inputShape[0];
        int channels = inputShape[1];
        int inDepth = inputShape[2];
        int inHeight = inputShape[3];
        int inWidth = inputShape[4];

        int gradInputSize = batch * channels * inDepth * inHeight * inWidth;
        var gradInputBuffer = backend.AllocateBuffer(gradInputSize);

        backend.NearestNeighborUpsample3DBackward(gradOutput.Buffer, gradInputBuffer,
            batch, channels,
            inDepth, inHeight, inWidth,
            scaleDepth, scaleHeight, scaleWidth);

        return new GpuTensor<T>(backend, gradInputBuffer, inputShape, GpuTensorRole.Gradient, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident 2D average pooling that keeps output on GPU.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="input">The input tensor on GPU.</param>
    /// <param name="poolSize">Pool size [height, width].</param>
    /// <param name="stride">Stride [height, width].</param>
    /// <returns>The pooled output as GPU-resident tensor.</returns>
    public IGpuTensor<T> AvgPool2DGpu<T>(
        IGpuTensor<T> input,
        int[] poolSize,
        int[] stride)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for AvgPool2DGpu");

        if (input.Shape.Length != 4 || poolSize.Length != 2 || stride.Length != 2)
            throw new ArgumentException("Input must be 4D [batch, channels, height, width] with 2D poolSize and stride");

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int outHeight = (inHeight - poolSize[0]) / stride[0] + 1;
        int outWidth = (inWidth - poolSize[1]) / stride[1] + 1;

        if (outHeight <= 0 || outWidth <= 0)
            throw new ArgumentException($"Invalid pooling parameters: output dimensions ({outHeight}, {outWidth}) are non-positive");

        int outputSize = batch * channels * outHeight * outWidth;
        var outputBuffer = backend.AllocateBuffer(outputSize);

        backend.AvgPool2D(input.Buffer, outputBuffer,
            batch, channels, inHeight, inWidth,
            outHeight, outWidth,
            poolSize[0], poolSize[1],
            stride[0], stride[1], 0, 0,
            countIncludePad: true);

        var outputShape = new[] { batch, channels, outHeight, outWidth };
        return new GpuTensor<T>(backend, outputBuffer, outputShape, GpuTensorRole.Intermediate, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident backward pass for 2D average pooling.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="gradOutput">The gradient of the output on GPU.</param>
    /// <param name="inputShape">The shape of the original input.</param>
    /// <param name="poolSize">Pool size [height, width].</param>
    /// <param name="stride">Stride [height, width].</param>
    /// <returns>The gradient with respect to input as GPU-resident tensor.</returns>
    public IGpuTensor<T> AvgPool2DBackwardGpu<T>(
        IGpuTensor<T> gradOutput,
        int[] inputShape,
        int[] poolSize,
        int[] stride)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for AvgPool2DBackwardGpu");

        if (gradOutput.Shape.Length != 4 || inputShape.Length != 4)
            throw new ArgumentException("GradOutput and inputShape must be 4D");

        int batch = inputShape[0];
        int channels = inputShape[1];
        int inHeight = inputShape[2];
        int inWidth = inputShape[3];

        int gradInputSize = batch * channels * inHeight * inWidth;
        var gradInputBuffer = backend.AllocateBuffer(gradInputSize);

        backend.AvgPool2DBackward(gradOutput.Buffer, gradInputBuffer,
            batch, channels, inHeight, inWidth,
            gradOutput.Shape[2], gradOutput.Shape[3],
            poolSize[0], poolSize[1],
            stride[0], stride[1], 0, 0,
            countIncludePad: true);

        return new GpuTensor<T>(backend, gradInputBuffer, inputShape, GpuTensorRole.Gradient, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-accelerated fused 2D convolution with activation.
    /// Uses cached GPU buffers for registered persistent tensors (kernel/bias) to avoid
    /// redundant CPU→GPU transfers on every forward pass.
    /// </summary>
    public new Tensor<T> FusedConv2D<T>(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T>? bias,
        int strideH, int strideW,
        int padH, int padW,
        int dilationH, int dilationW,
        FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            return base.FusedConv2D(input, kernel, bias, strideH, strideW, padH, padW, dilationH, dilationW, activation);

        // Expected input shape: [batch, inChannels, height, width]
        // Expected kernel shape: [outChannels, inChannels, kernelH, kernelW]
        if (input.Rank != 4 || kernel.Rank != 4)
            return base.FusedConv2D(input, kernel, bias, strideH, strideW, padH, padW, dilationH, dilationW, activation);

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int outChannels = kernel.Shape[0];
        int kernelH = kernel.Shape[2];
        int kernelW = kernel.Shape[3];

        // Calculate output dimensions with dilation
        int effectiveKernelH = kernelH + (kernelH - 1) * (dilationH - 1);
        int effectiveKernelW = kernelW + (kernelW - 1) * (dilationW - 1);
        int outHeight = (inHeight + 2 * padH - effectiveKernelH) / strideH + 1;
        int outWidth = (inWidth + 2 * padW - effectiveKernelW) / strideW + 1;

        if (outHeight <= 0 || outWidth <= 0)
            return base.FusedConv2D(input, kernel, bias, strideH, strideW, padH, padW, dilationH, dilationW, activation);

        // Use cache-aware buffer allocation (OwnedBuffer auto-disposes only if we allocated)
        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        using var kernelBuffer = GetOrCacheWeightBuffer(backend, kernel.Data, PersistentTensorRole.Weights);
        using var outputBuffer = AllocateOutputBuffer(backend, batch * outChannels * outHeight * outWidth);

        try
        {
            // Execute GPU convolution
            backend.Conv2D(inputBuffer.Buffer, kernelBuffer.Buffer, outputBuffer.Buffer,
                batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth,
                kernelH, kernelW,
                strideH, strideW, padH, padW,
                dilationH, dilationW);

            // Add bias if present
            if (bias != null)
            {
                // Bias is added per output channel, broadcast across batch and spatial dimensions
                int outputSize = batch * outChannels * outHeight * outWidth;
                int spatialSize = outHeight * outWidth;

                // Download, add bias, re-upload (GPU bias broadcast kernel would be more efficient)
                float[] outputFloat = new float[outputSize];
                backend.DownloadBuffer(outputBuffer.Buffer, outputFloat);

                // Get bias data (check cache first)
                using var biasBuffer = GetOrCacheWeightBuffer(backend, bias.Data, PersistentTensorRole.Biases);
                float[] biasFloat = new float[bias.Length];
                backend.DownloadBuffer(biasBuffer.Buffer, biasFloat);

                for (int b = 0; b < batch; b++)
                {
                    for (int c = 0; c < outChannels; c++)
                    {
                        float biasVal = biasFloat[c];
                        int baseIdx = (b * outChannels + c) * spatialSize;
                        for (int s = 0; s < spatialSize; s++)
                        {
                            outputFloat[baseIdx + s] += biasVal;
                        }
                    }
                }

                // Re-upload for activation
                using var biasedBuffer = backend.AllocateBuffer(outputFloat);

                // Apply activation on GPU
                if (activation != FusedActivationType.None)
                {
                    ApplyGpuActivation(backend, biasedBuffer, outputSize, activation);
                }

                // DownloadBuffer uses blocking read, Synchronize() removed for performance
                float[] resultFloat = new float[outputSize];
                backend.DownloadBuffer(biasedBuffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, new[] { batch, outChannels, outHeight, outWidth });
            }
            else
            {
                // No bias - apply activation directly
                int outputSize = batch * outChannels * outHeight * outWidth;

                if (activation != FusedActivationType.None)
                {
                    ApplyGpuActivation(backend, outputBuffer.Buffer, outputSize, activation);
                }

                // DownloadBuffer uses blocking read, Synchronize() removed for performance
                float[] resultFloat = new float[outputSize];
                backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, new[] { batch, outChannels, outHeight, outWidth });
            }
        }
        catch
        {
            // Fall back to CPU on any GPU error
            return base.FusedConv2D(input, kernel, bias, strideH, strideW, padH, padW, dilationH, dilationW, activation);
        }
    }

    /// <summary>
    /// GPU-resident fused 2D convolution with activation.
    /// Keeps input and output on GPU for chained layer execution.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor [batch, inChannels, height, width].</param>
    /// <param name="kernel">Kernel tensor (cached if registered).</param>
    /// <param name="bias">Optional bias tensor (cached if registered).</param>
    /// <param name="strideH">Vertical stride.</param>
    /// <param name="strideW">Horizontal stride.</param>
    /// <param name="padH">Vertical padding.</param>
    /// <param name="padW">Horizontal padding.</param>
    /// <param name="dilationH">Vertical dilation.</param>
    /// <param name="dilationW">Horizontal dilation.</param>
    /// <param name="activation">Activation function to fuse.</param>
    /// <returns>GPU-resident output tensor.</returns>
    public IGpuTensor<T> FusedConv2DGpu<T>(
        IGpuTensor<T> input,
        Tensor<T> kernel,
        Tensor<T>? bias,
        int strideH, int strideW,
        int padH, int padW,
        int dilationH, int dilationW,
        FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for FusedConv2DGpu");

        // Expected input shape: [batch, inChannels, height, width]
        // Expected kernel shape: [outChannels, inChannels, kernelH, kernelW]
        if (input.Shape.Length != 4 || kernel.Rank != 4)
            throw new ArgumentException("FusedConv2DGpu requires 4D input and kernel tensors");

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int outChannels = kernel.Shape[0];
        int kernelH = kernel.Shape[2];
        int kernelW = kernel.Shape[3];

        // Calculate output dimensions with dilation
        int effectiveKernelH = kernelH + (kernelH - 1) * (dilationH - 1);
        int effectiveKernelW = kernelW + (kernelW - 1) * (dilationW - 1);
        int outHeight = (inHeight + 2 * padH - effectiveKernelH) / strideH + 1;
        int outWidth = (inWidth + 2 * padW - effectiveKernelW) / strideW + 1;

        if (outHeight <= 0 || outWidth <= 0)
            throw new ArgumentException($"Invalid convolution parameters result in non-positive output dimensions: {outHeight}x{outWidth}");

        int outputSize = batch * outChannels * outHeight * outWidth;

        // Input is already on GPU - use its buffer directly
        using var kernelBuffer = GetOrCacheWeightBuffer(backend, kernel.Data, PersistentTensorRole.Weights);
        var outputBuffer = backend.AllocateBuffer(outputSize);

        try
        {
            // Execute GPU convolution
            backend.Conv2D(input.Buffer, kernelBuffer.Buffer, outputBuffer,
                batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth,
                kernelH, kernelW,
                strideH, strideW, padH, padW,
                dilationH, dilationW);

            // Add bias if present using GPU kernel for NCHW format
            if (bias != null)
            {
                int spatialSize = outHeight * outWidth;
                using var biasBuffer = GetOrCacheWeightBuffer(backend, bias.Data, PersistentTensorRole.Biases);
                backend.Conv2DBiasAdd(outputBuffer, biasBuffer.Buffer, batch, outChannels, spatialSize);
            }

            // Apply activation on GPU
            if (activation != FusedActivationType.None)
            {
                ApplyGpuActivation(backend, outputBuffer, outputSize, activation);
            }

            // Return GPU-resident tensor - NO DOWNLOAD
            return new GpuTensor<T>(backend, outputBuffer, new[] { batch, outChannels, outHeight, outWidth },
                GpuTensorRole.Activation, ownsBuffer: true);
        }
        catch
        {
            outputBuffer.Dispose();
            throw;
        }
    }

    /// <summary>
    /// GPU-accelerated fused 3D convolution with activation.
    /// Uses cached GPU buffers for registered persistent tensors (kernel/bias) to avoid
    /// redundant CPU→GPU transfers on every forward pass.
    /// </summary>
    public new Tensor<T> FusedConv3D<T>(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T>? bias,
        int strideD, int strideH, int strideW,
        int padD, int padH, int padW,
        int dilationD, int dilationH, int dilationW,
        FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            return base.FusedConv3D(input, kernel, bias, strideD, strideH, strideW, padD, padH, padW, dilationD, dilationH, dilationW, activation);

        // Expected input shape: [batch, inChannels, depth, height, width]
        // Expected kernel shape: [outChannels, inChannels, kernelD, kernelH, kernelW]
        if (input.Rank != 5 || kernel.Rank != 5)
            return base.FusedConv3D(input, kernel, bias, strideD, strideH, strideW, padD, padH, padW, dilationD, dilationH, dilationW, activation);

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inDepth = input.Shape[2];
        int inHeight = input.Shape[3];
        int inWidth = input.Shape[4];

        int outChannels = kernel.Shape[0];
        int kernelD = kernel.Shape[2];
        int kernelH = kernel.Shape[3];
        int kernelW = kernel.Shape[4];

        // Calculate output dimensions with dilation
        int effectiveKernelD = kernelD + (kernelD - 1) * (dilationD - 1);
        int effectiveKernelH = kernelH + (kernelH - 1) * (dilationH - 1);
        int effectiveKernelW = kernelW + (kernelW - 1) * (dilationW - 1);
        int outDepth = (inDepth + 2 * padD - effectiveKernelD) / strideD + 1;
        int outHeight = (inHeight + 2 * padH - effectiveKernelH) / strideH + 1;
        int outWidth = (inWidth + 2 * padW - effectiveKernelW) / strideW + 1;

        if (outDepth <= 0 || outHeight <= 0 || outWidth <= 0)
            return base.FusedConv3D(input, kernel, bias, strideD, strideH, strideW, padD, padH, padW, dilationD, dilationH, dilationW, activation);

        // Use cache-aware buffer allocation (OwnedBuffer auto-disposes only if we allocated)
        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        using var kernelBuffer = GetOrCacheWeightBuffer(backend, kernel.Data, PersistentTensorRole.Weights);
        using var outputBuffer = AllocateOutputBuffer(backend, batch * outChannels * outDepth * outHeight * outWidth);

        try
        {
            // Execute GPU 3D convolution
            backend.Conv3D(inputBuffer.Buffer, kernelBuffer.Buffer, outputBuffer.Buffer,
                batch, inChannels, inDepth, inHeight, inWidth,
                outChannels, outDepth, outHeight, outWidth,
                kernelD, kernelH, kernelW,
                strideD, strideH, strideW,
                padD, padH, padW,
                dilationD, dilationH, dilationW);

            // Add bias if present
            if (bias != null)
            {
                int outputSize = batch * outChannels * outDepth * outHeight * outWidth;
                int spatialSize = outDepth * outHeight * outWidth;

                // Download, add bias, re-upload
                float[] outputFloat = new float[outputSize];
                backend.DownloadBuffer(outputBuffer.Buffer, outputFloat);

                using var biasBuffer = GetOrCacheWeightBuffer(backend, bias.Data, PersistentTensorRole.Biases);
                float[] biasFloat = new float[bias.Length];
                backend.DownloadBuffer(biasBuffer.Buffer, biasFloat);

                for (int b = 0; b < batch; b++)
                {
                    for (int c = 0; c < outChannels; c++)
                    {
                        float biasVal = biasFloat[c];
                        int baseIdx = (b * outChannels + c) * spatialSize;
                        for (int s = 0; s < spatialSize; s++)
                        {
                            outputFloat[baseIdx + s] += biasVal;
                        }
                    }
                }

                using var biasedBuffer = backend.AllocateBuffer(outputFloat);

                if (activation != FusedActivationType.None)
                {
                    ApplyGpuActivation(backend, biasedBuffer, outputSize, activation);
                }

                // DownloadBuffer uses blocking read, Synchronize() removed for performance
                float[] resultFloat = new float[outputSize];
                backend.DownloadBuffer(biasedBuffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, new[] { batch, outChannels, outDepth, outHeight, outWidth });
            }
            else
            {
                int outputSize = batch * outChannels * outDepth * outHeight * outWidth;

                if (activation != FusedActivationType.None)
                {
                    ApplyGpuActivation(backend, outputBuffer.Buffer, outputSize, activation);
                }

                // DownloadBuffer uses blocking read, Synchronize() removed for performance
                float[] resultFloat = new float[outputSize];
                backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, new[] { batch, outChannels, outDepth, outHeight, outWidth });
            }
        }
        catch
        {
            return base.FusedConv3D(input, kernel, bias, strideD, strideH, strideW, padD, padH, padW, dilationD, dilationH, dilationW, activation);
        }
    }

    /// <summary>
    /// GPU-resident fused 3D convolution with activation.
    /// Keeps input and output on GPU for chained layer execution.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor [batch, inChannels, depth, height, width].</param>
    /// <param name="kernel">Kernel tensor (cached if registered).</param>
    /// <param name="bias">Optional bias tensor (cached if registered).</param>
    /// <param name="strideD">Depth stride.</param>
    /// <param name="strideH">Height stride.</param>
    /// <param name="strideW">Width stride.</param>
    /// <param name="padD">Depth padding.</param>
    /// <param name="padH">Height padding.</param>
    /// <param name="padW">Width padding.</param>
    /// <param name="dilationD">Depth dilation.</param>
    /// <param name="dilationH">Height dilation.</param>
    /// <param name="dilationW">Width dilation.</param>
    /// <param name="activation">Fused activation type.</param>
    /// <returns>GPU-resident output tensor [batch, outChannels, outDepth, outHeight, outWidth].</returns>
    public IGpuTensor<T> FusedConv3DGpu<T>(
        IGpuTensor<T> input,
        Tensor<T> kernel,
        Tensor<T>? bias,
        int strideD, int strideH, int strideW,
        int padD, int padH, int padW,
        int dilationD, int dilationH, int dilationW,
        FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for FusedConv3DGpu");

        // Expected input shape: [batch, inChannels, depth, height, width]
        // Expected kernel shape: [outChannels, inChannels, kernelD, kernelH, kernelW]
        if (input.Shape.Length != 5 || kernel.Rank != 5)
            throw new ArgumentException("FusedConv3DGpu requires 5D input and kernel tensors");

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inDepth = input.Shape[2];
        int inHeight = input.Shape[3];
        int inWidth = input.Shape[4];

        int outChannels = kernel.Shape[0];
        int kernelD = kernel.Shape[2];
        int kernelH = kernel.Shape[3];
        int kernelW = kernel.Shape[4];

        // Calculate output dimensions with dilation
        int effectiveKernelD = kernelD + (kernelD - 1) * (dilationD - 1);
        int effectiveKernelH = kernelH + (kernelH - 1) * (dilationH - 1);
        int effectiveKernelW = kernelW + (kernelW - 1) * (dilationW - 1);
        int outDepth = (inDepth + 2 * padD - effectiveKernelD) / strideD + 1;
        int outHeight = (inHeight + 2 * padH - effectiveKernelH) / strideH + 1;
        int outWidth = (inWidth + 2 * padW - effectiveKernelW) / strideW + 1;

        if (outDepth <= 0 || outHeight <= 0 || outWidth <= 0)
            throw new ArgumentException($"Invalid 3D convolution parameters result in non-positive output dimensions: {outDepth}x{outHeight}x{outWidth}");

        int outputSize = batch * outChannels * outDepth * outHeight * outWidth;
        int spatialSize = outDepth * outHeight * outWidth;

        // Use cache-aware buffer allocation
        using var kernelBuffer = GetOrCacheWeightBuffer(backend, kernel.Data, PersistentTensorRole.Weights);
        var outputBuffer = backend.AllocateBuffer(outputSize);

        try
        {
            // Execute GPU 3D convolution
            backend.Conv3D(input.Buffer, kernelBuffer.Buffer, outputBuffer,
                batch, inChannels, inDepth, inHeight, inWidth,
                outChannels, outDepth, outHeight, outWidth,
                kernelD, kernelH, kernelW,
                strideD, strideH, strideW,
                padD, padH, padW,
                dilationD, dilationH, dilationW);

            // Add bias if present (Conv2DBiasAdd works for any spatial size)
            if (bias != null)
            {
                using var biasBuffer = GetOrCacheWeightBuffer(backend, bias.Data, PersistentTensorRole.Biases);
                backend.Conv2DBiasAdd(outputBuffer, biasBuffer.Buffer, batch, outChannels, spatialSize);
            }

            // Apply activation on GPU
            if (activation != FusedActivationType.None)
            {
                ApplyGpuActivation(backend, outputBuffer, outputSize, activation);
            }

            // Return GPU-resident tensor - NO DOWNLOAD
            return new GpuTensor<T>(backend, outputBuffer, new[] { batch, outChannels, outDepth, outHeight, outWidth },
                GpuTensorRole.Activation, ownsBuffer: true);
        }
        catch
        {
            outputBuffer.Dispose();
            throw;
        }
    }

    /// <summary>
    /// GPU-accelerated fused transposed 2D convolution with activation.
    /// Uses cached GPU buffers for registered persistent tensors (kernel/bias) to avoid
    /// redundant CPU→GPU transfers on every forward pass.
    /// </summary>
    public new Tensor<T> FusedConvTranspose2D<T>(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T>? bias,
        int strideH, int strideW,
        int padH, int padW,
        int outputPadH, int outputPadW,
        FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            return base.FusedConvTranspose2D(input, kernel, bias, strideH, strideW, padH, padW, outputPadH, outputPadW, activation);

        // Expected input shape: [batch, inChannels, height, width]
        // Expected kernel shape: [inChannels, outChannels, kernelH, kernelW]
        if (input.Rank != 4 || kernel.Rank != 4)
            return base.FusedConvTranspose2D(input, kernel, bias, strideH, strideW, padH, padW, outputPadH, outputPadW, activation);

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int outChannels = kernel.Shape[1];
        int kernelH = kernel.Shape[2];
        int kernelW = kernel.Shape[3];

        // Calculate output dimensions for transposed convolution
        int outHeight = (inHeight - 1) * strideH - 2 * padH + kernelH + outputPadH;
        int outWidth = (inWidth - 1) * strideW - 2 * padW + kernelW + outputPadW;

        if (outHeight <= 0 || outWidth <= 0)
            return base.FusedConvTranspose2D(input, kernel, bias, strideH, strideW, padH, padW, outputPadH, outputPadW, activation);

        // Use cache-aware buffer allocation (OwnedBuffer auto-disposes only if we allocated)
        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        using var kernelBuffer = GetOrCacheWeightBuffer(backend, kernel.Data, PersistentTensorRole.Weights);
        using var outputBuffer = AllocateOutputBuffer(backend, batch * outChannels * outHeight * outWidth);

        try
        {
            // Execute GPU transposed convolution
            backend.ConvTranspose2D(inputBuffer.Buffer, kernelBuffer.Buffer, outputBuffer.Buffer,
                batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth,
                kernelH, kernelW,
                strideH, strideW, padH, padW,
                outputPadH, outputPadW);

            // Add bias if present
            if (bias != null)
            {
                int outputSize = batch * outChannels * outHeight * outWidth;
                int spatialSize = outHeight * outWidth;

                // Download, add bias, re-upload
                float[] outputFloat = new float[outputSize];
                backend.DownloadBuffer(outputBuffer.Buffer, outputFloat);

                using var biasBuffer = GetOrCacheWeightBuffer(backend, bias.Data, PersistentTensorRole.Biases);
                float[] biasFloat = new float[bias.Length];
                backend.DownloadBuffer(biasBuffer.Buffer, biasFloat);

                for (int b = 0; b < batch; b++)
                {
                    for (int c = 0; c < outChannels; c++)
                    {
                        float biasVal = biasFloat[c];
                        int baseIdx = (b * outChannels + c) * spatialSize;
                        for (int s = 0; s < spatialSize; s++)
                        {
                            outputFloat[baseIdx + s] += biasVal;
                        }
                    }
                }

                using var biasedBuffer = backend.AllocateBuffer(outputFloat);

                if (activation != FusedActivationType.None)
                {
                    ApplyGpuActivation(backend, biasedBuffer, outputSize, activation);
                }

                // DownloadBuffer uses blocking read, Synchronize() removed for performance
                float[] resultFloat = new float[outputSize];
                backend.DownloadBuffer(biasedBuffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, new[] { batch, outChannels, outHeight, outWidth });
            }
            else
            {
                int outputSize = batch * outChannels * outHeight * outWidth;

                if (activation != FusedActivationType.None)
                {
                    ApplyGpuActivation(backend, outputBuffer.Buffer, outputSize, activation);
                }

                // DownloadBuffer uses blocking read, Synchronize() removed for performance
                float[] resultFloat = new float[outputSize];
                backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, new[] { batch, outChannels, outHeight, outWidth });
            }
        }
        catch
        {
            return base.FusedConvTranspose2D(input, kernel, bias, strideH, strideW, padH, padW, outputPadH, outputPadW, activation);
        }
    }

    /// <summary>
    /// GPU-resident fused transposed 2D convolution with activation.
    /// Keeps input and output on GPU for chained layer execution.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor [batch, inChannels, height, width].</param>
    /// <param name="kernel">Kernel tensor (cached if registered).</param>
    /// <param name="bias">Optional bias tensor (cached if registered).</param>
    /// <param name="strideH">Vertical stride.</param>
    /// <param name="strideW">Horizontal stride.</param>
    /// <param name="padH">Vertical padding.</param>
    /// <param name="padW">Horizontal padding.</param>
    /// <param name="outputPadH">Vertical output padding.</param>
    /// <param name="outputPadW">Horizontal output padding.</param>
    /// <param name="activation">Activation function to fuse.</param>
    /// <returns>GPU-resident output tensor.</returns>
    public IGpuTensor<T> FusedConvTranspose2DGpu<T>(
        IGpuTensor<T> input,
        Tensor<T> kernel,
        Tensor<T>? bias,
        int strideH, int strideW,
        int padH, int padW,
        int outputPadH, int outputPadW,
        FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for FusedConvTranspose2DGpu");

        // Expected input shape: [batch, inChannels, height, width]
        // Expected kernel shape: [inChannels, outChannels, kernelH, kernelW]
        if (input.Shape.Length != 4 || kernel.Rank != 4)
            throw new ArgumentException("FusedConvTranspose2DGpu requires 4D input and kernel tensors");

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int outChannels = kernel.Shape[1];
        int kernelH = kernel.Shape[2];
        int kernelW = kernel.Shape[3];

        // Calculate output dimensions for transposed convolution
        int outHeight = (inHeight - 1) * strideH - 2 * padH + kernelH + outputPadH;
        int outWidth = (inWidth - 1) * strideW - 2 * padW + kernelW + outputPadW;

        if (outHeight <= 0 || outWidth <= 0)
            throw new ArgumentException($"Invalid transposed convolution parameters result in non-positive output dimensions: {outHeight}x{outWidth}");

        int outputSize = batch * outChannels * outHeight * outWidth;

        // Input is already on GPU - use its buffer directly
        using var kernelBuffer = GetOrCacheWeightBuffer(backend, kernel.Data, PersistentTensorRole.Weights);
        var outputBuffer = backend.AllocateBuffer(outputSize);

        try
        {
            // Execute GPU transposed convolution
            backend.ConvTranspose2D(input.Buffer, kernelBuffer.Buffer, outputBuffer,
                batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth,
                kernelH, kernelW,
                strideH, strideW, padH, padW,
                outputPadH, outputPadW);

            // Add bias if present using GPU kernel for NCHW format
            if (bias != null)
            {
                int spatialSize = outHeight * outWidth;
                using var biasBuffer = GetOrCacheWeightBuffer(backend, bias.Data, PersistentTensorRole.Biases);
                backend.Conv2DBiasAdd(outputBuffer, biasBuffer.Buffer, batch, outChannels, spatialSize);
            }

            // Apply activation on GPU
            if (activation != FusedActivationType.None)
            {
                ApplyGpuActivation(backend, outputBuffer, outputSize, activation);
            }

            // Return GPU-resident tensor - NO DOWNLOAD
            return new GpuTensor<T>(backend, outputBuffer, new[] { batch, outChannels, outHeight, outWidth },
                GpuTensorRole.Activation, ownsBuffer: true);
        }
        catch
        {
            outputBuffer.Dispose();
            throw;
        }
    }

    /// <summary>
    /// GPU-accelerated 2D max pooling operation.
    /// Uses GPU kernels for efficient parallel computation of maximum values within pooling windows.
    /// </summary>
    public new Tensor<T> MaxPool2D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0)
    {
        if (stride == 0) stride = poolSize;

        if (!TryGetBackend(out var backend))
            return base.MaxPool2D(input, poolSize, stride, padding);

        // Expected input shape: [batch, channels, height, width]
        if (input.Rank != 4)
            return base.MaxPool2D(input, poolSize, stride, padding);

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        // Calculate output dimensions
        int outHeight = (inHeight + 2 * padding - poolSize) / stride + 1;
        int outWidth = (inWidth + 2 * padding - poolSize) / stride + 1;

        if (outHeight <= 0 || outWidth <= 0)
            return base.MaxPool2D(input, poolSize, stride, padding);

        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        using var outputBuffer = AllocateOutputBuffer(backend, batch * channels * outHeight * outWidth);

        try
        {
            // Execute GPU max pooling (indices buffer is null for forward-only)
            backend.MaxPool2D(inputBuffer.Buffer, outputBuffer.Buffer, null,
                batch, channels, inHeight, inWidth,
                outHeight, outWidth,
                poolSize, poolSize,
                stride, stride, padding, padding);

            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            int outputSize = batch * channels * outHeight * outWidth;
            float[] resultFloat = new float[outputSize];
            backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, new[] { batch, channels, outHeight, outWidth });
        }
        catch
        {
            return base.MaxPool2D(input, poolSize, stride, padding);
        }
    }

    /// <summary>
    /// GPU-accelerated 2D max pooling with indices for backward pass.
    /// Returns both pooled output and indices of maximum values for gradient computation.
    /// </summary>
    public new Tensor<T> MaxPool2DWithIndices<T>(Tensor<T> input, int[] poolSize, int[] stride, out int[,,,,] maxIndices)
    {
        if (!TryGetBackend(out var backend))
            return base.MaxPool2DWithIndices(input, poolSize, stride, out maxIndices);

        if (input.Rank != 4 || poolSize.Length != 2 || stride.Length != 2)
            return base.MaxPool2DWithIndices(input, poolSize, stride, out maxIndices);

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int outHeight = (inHeight - poolSize[0]) / stride[0] + 1;
        int outWidth = (inWidth - poolSize[1]) / stride[1] + 1;

        if (outHeight <= 0 || outWidth <= 0)
            return base.MaxPool2DWithIndices(input, poolSize, stride, out maxIndices);

        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        using var outputBuffer = AllocateOutputBuffer(backend, batch * channels * outHeight * outWidth);
        using var indicesBuffer = AllocateOutputBuffer(backend, batch * channels * outHeight * outWidth);

        try
        {
            backend.MaxPool2D(inputBuffer.Buffer, outputBuffer.Buffer, indicesBuffer.Buffer,
                batch, channels, inHeight, inWidth,
                outHeight, outWidth,
                poolSize[0], poolSize[1],
                stride[0], stride[1], 0, 0);

            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            int outputSize = batch * channels * outHeight * outWidth;
            float[] resultFloat = new float[outputSize];
            float[] indicesFloat = new float[outputSize];
            backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);
            backend.DownloadBuffer(indicesBuffer.Buffer, indicesFloat);

            // Convert indices to int array
            maxIndices = new int[batch, channels, outHeight, outWidth, 2];
            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int oh = 0; oh < outHeight; oh++)
                    {
                        for (int ow = 0; ow < outWidth; ow++)
                        {
                            int flatIdx = ((b * channels + c) * outHeight + oh) * outWidth + ow;
                            int idx = (int)indicesFloat[flatIdx];
                            maxIndices[b, c, oh, ow, 0] = idx / inWidth;
                            maxIndices[b, c, oh, ow, 1] = idx % inWidth;
                        }
                    }
                }
            }

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, new[] { batch, channels, outHeight, outWidth });
        }
        catch
        {
            return base.MaxPool2DWithIndices(input, poolSize, stride, out maxIndices);
        }
    }

    /// <summary>
    /// GPU-accelerated backward pass for 2D max pooling.
    /// Propagates gradients back through the max pooling operation using stored indices.
    /// </summary>
    public new Tensor<T> MaxPool2DBackward<T>(Tensor<T> gradOutput, int[,,,,] maxIndices, int[] inputShape, int[] poolSize, int[] stride)
    {
        if (!TryGetBackend(out var backend))
            return base.MaxPool2DBackward(gradOutput, maxIndices, inputShape, poolSize, stride);

        if (gradOutput.Rank != 4 || inputShape.Length != 4)
            return base.MaxPool2DBackward(gradOutput, maxIndices, inputShape, poolSize, stride);

        int batch = inputShape[0];
        int channels = inputShape[1];
        int inHeight = inputShape[2];
        int inWidth = inputShape[3];
        int outHeight = gradOutput.Shape[2];
        int outWidth = gradOutput.Shape[3];

        // Convert indices to flat GPU buffer
        int indexCount = batch * channels * outHeight * outWidth;
        float[] indicesFlat = new float[indexCount];
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outHeight; oh++)
                {
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        int flatIdx = ((b * channels + c) * outHeight + oh) * outWidth + ow;
                        int h = maxIndices[b, c, oh, ow, 0];
                        int w = maxIndices[b, c, oh, ow, 1];
                        indicesFlat[flatIdx] = h * inWidth + w;
                    }
                }
            }
        }

        using var gradOutputBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
        using var indicesBuffer = backend.AllocateBuffer(indicesFlat);
        using var gradInputBuffer = AllocateOutputBuffer(backend, batch * channels * inHeight * inWidth);

        try
        {
            backend.MaxPool2DBackward(gradOutputBuffer.Buffer, indicesBuffer, gradInputBuffer.Buffer,
                batch, channels, inHeight, inWidth,
                outHeight, outWidth,
                poolSize[0], poolSize[1],
                stride[0], stride[1], 0, 0);

            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            int inputSize = batch * channels * inHeight * inWidth;
            float[] resultFloat = new float[inputSize];
            backend.DownloadBuffer(gradInputBuffer.Buffer, resultFloat);

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, inputShape);
        }
        catch
        {
            return base.MaxPool2DBackward(gradOutput, maxIndices, inputShape, poolSize, stride);
        }
    }

    /// <summary>
    /// GPU-accelerated 2D average pooling operation.
    /// Uses GPU kernels for efficient parallel computation of average values within pooling windows.
    /// </summary>
    public new Tensor<T> AvgPool2D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0)
    {
        if (stride == 0) stride = poolSize;

        if (!TryGetBackend(out var backend))
            return base.AvgPool2D(input, poolSize, stride, padding);

        // Expected input shape: [batch, channels, height, width]
        if (input.Rank != 4)
            return base.AvgPool2D(input, poolSize, stride, padding);

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        // Calculate output dimensions
        int outHeight = (inHeight + 2 * padding - poolSize) / stride + 1;
        int outWidth = (inWidth + 2 * padding - poolSize) / stride + 1;

        if (outHeight <= 0 || outWidth <= 0)
            return base.AvgPool2D(input, poolSize, stride, padding);

        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        using var outputBuffer = AllocateOutputBuffer(backend, batch * channels * outHeight * outWidth);

        try
        {
            // Execute GPU average pooling
            backend.AvgPool2D(inputBuffer.Buffer, outputBuffer.Buffer,
                batch, channels, inHeight, inWidth,
                outHeight, outWidth,
                poolSize, poolSize,
                stride, stride, padding, padding,
                countIncludePad: true);

            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            int outputSize = batch * channels * outHeight * outWidth;
            float[] resultFloat = new float[outputSize];
            backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, new[] { batch, channels, outHeight, outWidth });
        }
        catch
        {
            return base.AvgPool2D(input, poolSize, stride, padding);
        }
    }

    /// <summary>
    /// GPU-accelerated 2D average pooling with array parameters.
    /// </summary>
    public new Tensor<T> AvgPool2D<T>(Tensor<T> input, int[] poolSize, int[] stride)
    {
        if (!TryGetBackend(out var backend))
            return base.AvgPool2D(input, poolSize, stride);

        if (input.Rank != 4 || poolSize.Length != 2 || stride.Length != 2)
            return base.AvgPool2D(input, poolSize, stride);

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int outHeight = (inHeight - poolSize[0]) / stride[0] + 1;
        int outWidth = (inWidth - poolSize[1]) / stride[1] + 1;

        if (outHeight <= 0 || outWidth <= 0)
            return base.AvgPool2D(input, poolSize, stride);

        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        using var outputBuffer = AllocateOutputBuffer(backend, batch * channels * outHeight * outWidth);

        try
        {
            backend.AvgPool2D(inputBuffer.Buffer, outputBuffer.Buffer,
                batch, channels, inHeight, inWidth,
                outHeight, outWidth,
                poolSize[0], poolSize[1],
                stride[0], stride[1], 0, 0,
                countIncludePad: true);

            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            int outputSize = batch * channels * outHeight * outWidth;
            float[] resultFloat = new float[outputSize];
            backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, new[] { batch, channels, outHeight, outWidth });
        }
        catch
        {
            return base.AvgPool2D(input, poolSize, stride);
        }
    }

    /// <summary>
    /// GPU-accelerated backward pass for 2D average pooling.
    /// Distributes gradients evenly across the input elements that contributed to each output.
    /// </summary>
    public new Tensor<T> AvgPool2DBackward<T>(Tensor<T> gradOutput, int[] inputShape, int[] poolSize, int[] stride)
    {
        if (!TryGetBackend(out var backend))
            return base.AvgPool2DBackward(gradOutput, inputShape, poolSize, stride);

        if (gradOutput.Rank != 4 || inputShape.Length != 4)
            return base.AvgPool2DBackward(gradOutput, inputShape, poolSize, stride);

        int batch = inputShape[0];
        int channels = inputShape[1];
        int inHeight = inputShape[2];
        int inWidth = inputShape[3];
        int outHeight = gradOutput.Shape[2];
        int outWidth = gradOutput.Shape[3];

        using var gradOutputBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
        using var gradInputBuffer = AllocateOutputBuffer(backend, batch * channels * inHeight * inWidth);

        try
        {
            backend.AvgPool2DBackward(gradOutputBuffer.Buffer, gradInputBuffer.Buffer,
                batch, channels, inHeight, inWidth,
                outHeight, outWidth,
                poolSize[0], poolSize[1],
                stride[0], stride[1], 0, 0,
                countIncludePad: true);

            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            int inputSize = batch * channels * inHeight * inWidth;
            float[] resultFloat = new float[inputSize];
            backend.DownloadBuffer(gradInputBuffer.Buffer, resultFloat);

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, inputShape);
        }
        catch
        {
            return base.AvgPool2DBackward(gradOutput, inputShape, poolSize, stride);
        }
    }

    /// <summary>
    /// GPU-accelerated depthwise 2D convolution.
    /// Each input channel is convolved with its own filter, commonly used in MobileNets.
    /// </summary>
    public new Tensor<T> DepthwiseConv2D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding)
    {
        if (!TryGetBackend(out var backend))
            return base.DepthwiseConv2D(input, kernel, stride, padding);

        // Expected input shape: [batch, channels, height, width]
        // Expected kernel shape: [channels, 1, kernelH, kernelW] or [channels, kernelH, kernelW]
        if (input.Rank != 4)
            return base.DepthwiseConv2D(input, kernel, stride, padding);

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int kernelH = kernel.Rank == 4 ? kernel.Shape[2] : kernel.Shape[1];
        int kernelW = kernel.Rank == 4 ? kernel.Shape[3] : kernel.Shape[2];

        int strideH = stride.Length >= 1 ? stride[0] : 1;
        int strideW = stride.Length >= 2 ? stride[1] : strideH;
        int padH = padding.Length >= 1 ? padding[0] : 0;
        int padW = padding.Length >= 2 ? padding[1] : padH;

        int outHeight = (inHeight + 2 * padH - kernelH) / strideH + 1;
        int outWidth = (inWidth + 2 * padW - kernelW) / strideW + 1;

        if (outHeight <= 0 || outWidth <= 0)
            return base.DepthwiseConv2D(input, kernel, stride, padding);

        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        using var kernelBuffer = GetOrCacheWeightBuffer(backend, kernel.Data, PersistentTensorRole.Weights);
        using var outputBuffer = AllocateOutputBuffer(backend, batch * channels * outHeight * outWidth);

        try
        {
            backend.DepthwiseConv2D(inputBuffer.Buffer, kernelBuffer.Buffer, outputBuffer.Buffer,
                batch, channels, inHeight, inWidth,
                outHeight, outWidth,
                kernelH, kernelW,
                strideH, strideW, padH, padW);

            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            int outputSize = batch * channels * outHeight * outWidth;
            float[] resultFloat = new float[outputSize];
            backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, new[] { batch, channels, outHeight, outWidth });
        }
        catch
        {
            return base.DepthwiseConv2D(input, kernel, stride, padding);
        }
    }

    /// <summary>
    /// GPU-resident depthwise 2D convolution with optional bias and activation.
    /// Keeps input and output on GPU for chained layer execution.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor [batch, channels, height, width].</param>
    /// <param name="kernel">Kernel tensor (cached if registered). Shape: [channels, 1, kH, kW].</param>
    /// <param name="bias">Optional bias tensor (cached if registered).</param>
    /// <param name="strideH">Vertical stride.</param>
    /// <param name="strideW">Horizontal stride.</param>
    /// <param name="padH">Vertical padding.</param>
    /// <param name="padW">Horizontal padding.</param>
    /// <param name="activation">Activation function to fuse.</param>
    /// <returns>GPU-resident output tensor.</returns>
    public IGpuTensor<T> DepthwiseConv2DGpu<T>(
        IGpuTensor<T> input,
        Tensor<T> kernel,
        Tensor<T>? bias,
        int strideH, int strideW,
        int padH, int padW,
        FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for DepthwiseConv2DGpu");

        // Expected input shape: [batch, channels, height, width]
        // Expected kernel shape: [channels, 1, kernelH, kernelW] or [channels, kernelH, kernelW]
        if (input.Shape.Length != 4)
            throw new ArgumentException("DepthwiseConv2DGpu requires 4D input tensor");

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int kernelH = kernel.Rank == 4 ? kernel.Shape[2] : kernel.Shape[1];
        int kernelW = kernel.Rank == 4 ? kernel.Shape[3] : kernel.Shape[2];

        int outHeight = (inHeight + 2 * padH - kernelH) / strideH + 1;
        int outWidth = (inWidth + 2 * padW - kernelW) / strideW + 1;

        if (outHeight <= 0 || outWidth <= 0)
            throw new ArgumentException($"Invalid depthwise convolution parameters result in non-positive output dimensions: {outHeight}x{outWidth}");

        int outputSize = batch * channels * outHeight * outWidth;

        // Input is already on GPU - use its buffer directly
        using var kernelBuffer = GetOrCacheWeightBuffer(backend, kernel.Data, PersistentTensorRole.Weights);
        var outputBuffer = backend.AllocateBuffer(outputSize);

        try
        {
            // Execute GPU depthwise convolution
            backend.DepthwiseConv2D(input.Buffer, kernelBuffer.Buffer, outputBuffer,
                batch, channels, inHeight, inWidth,
                outHeight, outWidth,
                kernelH, kernelW,
                strideH, strideW, padH, padW);

            // Add bias if present using GPU kernel for NCHW format
            if (bias != null)
            {
                int spatialSize = outHeight * outWidth;
                using var biasBuffer = GetOrCacheWeightBuffer(backend, bias.Data, PersistentTensorRole.Biases);
                // Depthwise output has same channels as input - use Conv2DBiasAdd pattern
                backend.Conv2DBiasAdd(outputBuffer, biasBuffer.Buffer, batch, channels, spatialSize);
            }

            // Apply activation on GPU
            if (activation != FusedActivationType.None)
            {
                ApplyGpuActivation(backend, outputBuffer, outputSize, activation);
            }

            // Return GPU-resident tensor - NO DOWNLOAD
            return new GpuTensor<T>(backend, outputBuffer, new[] { batch, channels, outHeight, outWidth },
                GpuTensorRole.Activation, ownsBuffer: true);
        }
        catch
        {
            outputBuffer.Dispose();
            throw;
        }
    }

    /// <summary>
    /// GPU-accelerated locally connected 2D convolution.
    /// Uses cached GPU buffers for registered persistent tensors (weights/bias) to avoid
    /// redundant CPU→GPU transfers on every forward pass.
    /// Falls back to CPU implementation if GPU is unavailable.
    /// </summary>
    public new Tensor<T> LocallyConnectedConv2D<T>(Tensor<T> input, Tensor<T> weights, Tensor<T>? bias, int[] stride)
    {
        if (!TryGetBackend(out var backend))
            return base.LocallyConnectedConv2D(input, weights, bias, stride);

        // Expected input shape: [batch, inChannels, height, width]
        // Expected weights shape: [outH, outW, outC, inC, kH, kW]
        if (input.Rank != 4 || weights.Rank != 6)
            return base.LocallyConnectedConv2D(input, weights, bias, stride);

        if (stride == null || stride.Length != 2)
            throw new ArgumentException("Stride must be an array of 2 elements", nameof(stride));
        if (stride[0] <= 0 || stride[1] <= 0)
            throw new ArgumentException("Stride elements must be positive", nameof(stride));

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int outHeight = weights.Shape[0];
        int outWidth = weights.Shape[1];
        int outChannels = weights.Shape[2];
        int kernelH = weights.Shape[4];
        int kernelW = weights.Shape[5];

        if (outHeight <= 0 || outWidth <= 0)
            return base.LocallyConnectedConv2D(input, weights, bias, stride);

        int outputSize = batch * outChannels * outHeight * outWidth;

        // Use cache-aware buffer allocation for weights/bias (persistent tensors)
        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        using var weightsBuffer = GetOrCacheWeightBuffer(backend, weights.Data, PersistentTensorRole.Weights);
        using var outputBuffer = AllocateOutputBuffer(backend, outputSize);

        try
        {
            // Handle bias - check cache for persistent bias tensors
            IGpuBuffer? biasGpuBuffer = null;
            OwnedBuffer? biasOwned = null;
            if (bias != null)
            {
                biasOwned = GetOrCacheWeightBuffer(backend, bias.Data, PersistentTensorRole.Biases);
                biasGpuBuffer = biasOwned.Value.Buffer;
            }

            try
            {
                // Execute GPU locally connected convolution
                backend.LocallyConnectedConv2D(
                    inputBuffer.Buffer, weightsBuffer.Buffer, biasGpuBuffer, outputBuffer.Buffer,
                    batch, inChannels, inHeight, inWidth,
                    outChannels, outHeight, outWidth,
                    kernelH, kernelW, stride[0], stride[1]);

                // Download result
                float[] resultFloat = new float[outputSize];
                backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, new[] { batch, outChannels, outHeight, outWidth });
            }
            finally
            {
                biasOwned?.Dispose();
            }
        }
        catch
        {
            // Fall back to CPU on any GPU error
            return base.LocallyConnectedConv2D(input, weights, bias, stride);
        }
    }

    /// <summary>
    /// GPU-accelerated locally connected 2D backward pass for input gradients.
    /// Falls back to CPU implementation if GPU is unavailable.
    /// </summary>
    public new Tensor<T> LocallyConnectedConv2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> weights, int[] inputShape, int[] stride)
    {
        if (!TryGetBackend(out var backend))
            return base.LocallyConnectedConv2DBackwardInput(gradOutput, weights, inputShape, stride);

        // Validate inputs
        if (gradOutput.Rank != 4 || weights.Rank != 6)
            return base.LocallyConnectedConv2DBackwardInput(gradOutput, weights, inputShape, stride);

        if (inputShape == null || inputShape.Length != 4)
            throw new ArgumentException("Input shape must be an array of 4 elements", nameof(inputShape));
        if (stride == null || stride.Length != 2)
            throw new ArgumentException("Stride must be an array of 2 elements", nameof(stride));

        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int inHeight = inputShape[2];
        int inWidth = inputShape[3];

        int outHeight = weights.Shape[0];
        int outWidth = weights.Shape[1];
        int outChannels = weights.Shape[2];
        int kernelH = weights.Shape[4];
        int kernelW = weights.Shape[5];

        int inputSize = batch * inChannels * inHeight * inWidth;

        using var gradOutputBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
        using var weightsBuffer = GetOrCacheWeightBuffer(backend, weights.Data, PersistentTensorRole.Weights);
        using var gradInputBuffer = AllocateOutputBuffer(backend, inputSize);

        try
        {
            backend.LocallyConnectedConv2DBackwardInput(
                gradOutputBuffer.Buffer, weightsBuffer.Buffer, gradInputBuffer.Buffer,
                batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth,
                kernelH, kernelW, stride[0], stride[1]);

            float[] resultFloat = new float[inputSize];
            backend.DownloadBuffer(gradInputBuffer.Buffer, resultFloat);

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, inputShape);
        }
        catch
        {
            return base.LocallyConnectedConv2DBackwardInput(gradOutput, weights, inputShape, stride);
        }
    }

    /// <summary>
    /// GPU-accelerated locally connected 2D backward pass for weight gradients.
    /// Falls back to CPU implementation if GPU is unavailable.
    /// </summary>
    public new Tensor<T> LocallyConnectedConv2DBackwardWeights<T>(Tensor<T> gradOutput, Tensor<T> input, int[] weightsShape, int[] stride)
    {
        if (!TryGetBackend(out var backend))
            return base.LocallyConnectedConv2DBackwardWeights(gradOutput, input, weightsShape, stride);

        // Validate inputs
        if (gradOutput.Rank != 4 || input.Rank != 4)
            return base.LocallyConnectedConv2DBackwardWeights(gradOutput, input, weightsShape, stride);

        if (weightsShape == null || weightsShape.Length != 6)
            throw new ArgumentException("Weights shape must be an array of 6 elements", nameof(weightsShape));
        if (stride == null || stride.Length != 2)
            throw new ArgumentException("Stride must be an array of 2 elements", nameof(stride));

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int outHeight = weightsShape[0];
        int outWidth = weightsShape[1];
        int outChannels = weightsShape[2];
        int kernelH = weightsShape[4];
        int kernelW = weightsShape[5];

        int weightsSize = outHeight * outWidth * outChannels * inChannels * kernelH * kernelW;

        using var gradOutputBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        using var gradWeightsBuffer = AllocateOutputBuffer(backend, weightsSize);

        try
        {
            backend.LocallyConnectedConv2DBackwardWeights(
                inputBuffer.Buffer, gradOutputBuffer.Buffer, gradWeightsBuffer.Buffer,
                batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth,
                kernelH, kernelW, stride[0], stride[1]);

            float[] resultFloat = new float[weightsSize];
            backend.DownloadBuffer(gradWeightsBuffer.Buffer, resultFloat);

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, weightsShape);
        }
        catch
        {
            return base.LocallyConnectedConv2DBackwardWeights(gradOutput, input, weightsShape, stride);
        }
    }

    /// <summary>
    /// GPU-accelerated locally connected 2D backward pass for bias gradients.
    /// Falls back to CPU implementation if GPU is unavailable.
    /// </summary>
    public new Tensor<T> LocallyConnectedConv2DBackwardBias<T>(Tensor<T> gradOutput)
    {
        if (!TryGetBackend(out var backend))
            return base.LocallyConnectedConv2DBackwardBias<T>(gradOutput);

        // Validate input
        if (gradOutput.Rank != 4)
            return base.LocallyConnectedConv2DBackwardBias<T>(gradOutput);

        int batch = gradOutput.Shape[0];
        int outChannels = gradOutput.Shape[1];
        int outHeight = gradOutput.Shape[2];
        int outWidth = gradOutput.Shape[3];

        using var gradOutputBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
        using var gradBiasBuffer = AllocateOutputBuffer(backend, outChannels);

        try
        {
            backend.LocallyConnectedConv2DBackwardBias(
                gradOutputBuffer.Buffer, gradBiasBuffer.Buffer,
                batch, outChannels, outHeight, outWidth);

            float[] resultFloat = new float[outChannels];
            backend.DownloadBuffer(gradBiasBuffer.Buffer, resultFloat);

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, new[] { outChannels });
        }
        catch
        {
            return base.LocallyConnectedConv2DBackwardBias<T>(gradOutput);
        }
    }

    /// <summary>
    /// GPU-accelerated locally connected 2D convolution.
    /// Unlike standard convolution, each spatial position uses unique weights.
    /// </summary>
    /// <param name="input">Input tensor [batch, inChannels, inHeight, inWidth]</param>
    /// <param name="weights">Weight tensor [outH, outW, outC, inC, kH, kW]</param>
    /// <param name="bias">Optional bias tensor [outChannels]</param>
    /// <param name="strideH">Vertical stride</param>
    /// <param name="strideW">Horizontal stride</param>
    /// <param name="activation">Fused activation type</param>
    /// <returns>Output GPU tensor [batch, outChannels, outHeight, outWidth]</returns>
    public IGpuTensor<T> LocallyConnectedConv2DGpu<T>(
        IGpuTensor<T> input,
        Tensor<T> weights,
        Tensor<T>? bias,
        int strideH, int strideW,
        FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for LocallyConnectedConv2DGpu");

        // Expected input shape: [batch, inChannels, height, width]
        // Expected weights shape: [outH, outW, outC, inC, kH, kW]
        if (input.Shape.Length != 4)
            throw new ArgumentException("LocallyConnectedConv2DGpu requires 4D input tensor");
        if (weights.Rank != 6)
            throw new ArgumentException("LocallyConnectedConv2DGpu requires 6D weights tensor [outH, outW, outC, inC, kH, kW]");

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int outHeight = weights.Shape[0];
        int outWidth = weights.Shape[1];
        int outChannels = weights.Shape[2];
        int kernelH = weights.Shape[4];
        int kernelW = weights.Shape[5];

        if (outHeight <= 0 || outWidth <= 0)
            throw new ArgumentException($"Invalid locally connected convolution parameters result in non-positive output dimensions: {outHeight}x{outWidth}");

        int outputSize = batch * outChannels * outHeight * outWidth;

        using var weightsBuffer = GetOrCacheWeightBuffer(backend, weights.Data, PersistentTensorRole.Weights);
        using var biasBuffer = bias != null ? GetOrCacheWeightBuffer(backend, bias.Data, PersistentTensorRole.Biases) : default(OwnedBuffer?);
        var outputBuffer = backend.AllocateBuffer(outputSize);

        try
        {
            backend.LocallyConnectedConv2D(input.Buffer, weightsBuffer.Buffer, biasBuffer?.Buffer, outputBuffer,
                batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth,
                kernelH, kernelW, strideH, strideW);

            // Apply activation on GPU
            if (activation != FusedActivationType.None)
            {
                ApplyGpuActivation(backend, outputBuffer, outputSize, activation);
            }

            return new GpuTensor<T>(backend, outputBuffer, new[] { batch, outChannels, outHeight, outWidth },
                GpuTensorRole.Activation, ownsBuffer: true);
        }
        catch
        {
            outputBuffer.Dispose();
            throw;
        }
    }

    /// <summary>
    /// GPU-accelerated deformable 2D convolution (DCNv2).
    /// Uses cached GPU buffers for registered persistent tensors to avoid
    /// redundant CPU→GPU transfers. Falls back to CPU if GPU unavailable.
    /// </summary>
    public new Tensor<T> DeformableConv2D<T>(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T> offsets,
        Tensor<T>? mask,
        int[] stride,
        int[] padding,
        int[] dilation)
    {
        if (!TryGetBackend(out var backend))
            return base.DeformableConv2D(input, kernel, offsets, mask, stride, padding, dilation);

        // Validate inputs
        if (input.Rank != 4 || kernel.Rank != 4 || offsets.Rank != 4)
            return base.DeformableConv2D(input, kernel, offsets, mask, stride, padding, dilation);

        if (stride == null || stride.Length != 2)
            throw new ArgumentException("Stride must be an array of 2 elements", nameof(stride));
        if (padding == null || padding.Length != 2)
            throw new ArgumentException("Padding must be an array of 2 elements", nameof(padding));
        if (dilation == null || dilation.Length != 2)
            throw new ArgumentException("Dilation must be an array of 2 elements", nameof(dilation));

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int outChannels = kernel.Shape[0];
        int kernelH = kernel.Shape[2];
        int kernelW = kernel.Shape[3];

        int outHeight = (inHeight + 2 * padding[0] - dilation[0] * (kernelH - 1) - 1) / stride[0] + 1;
        int outWidth = (inWidth + 2 * padding[1] - dilation[1] * (kernelW - 1) - 1) / stride[1] + 1;

        if (outHeight <= 0 || outWidth <= 0)
            return base.DeformableConv2D(input, kernel, offsets, mask, stride, padding, dilation);

        // Calculate deformGroups from offsets shape: [batch, deformGroups*2*kH*kW, outH, outW]
        int offsetChannels = offsets.Shape[1];
        int deformGroups = offsetChannels / (2 * kernelH * kernelW);
        int groups = 1; // Standard deformable conv uses groups=1

        int outputSize = batch * outChannels * outHeight * outWidth;

        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        using var weightsBuffer = GetOrCacheWeightBuffer(backend, kernel.Data, PersistentTensorRole.Weights);
        using var offsetsBuffer = GetOrAllocateBuffer(backend, offsets.Data);
        using var outputBuffer = AllocateOutputBuffer(backend, outputSize);

        try
        {
            OwnedBuffer? maskBuffer = mask != null ? GetOrAllocateBuffer(backend, mask.Data) : null;
            try
            {
                backend.DeformableConv2D(
                    inputBuffer.Buffer, weightsBuffer.Buffer, offsetsBuffer.Buffer, maskBuffer?.Buffer, outputBuffer.Buffer,
                    batch, inChannels, inHeight, inWidth,
                    outChannels, outHeight, outWidth,
                    kernelH, kernelW, stride[0], stride[1], padding[0], padding[1],
                    dilation[0], dilation[1], groups, deformGroups);

                float[] resultFloat = new float[outputSize];
                backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, new[] { batch, outChannels, outHeight, outWidth });
            }
            finally
            {
                maskBuffer?.Dispose();
            }
        }
        catch
        {
            return base.DeformableConv2D(input, kernel, offsets, mask, stride, padding, dilation);
        }
    }

    /// <summary>
    /// GPU-accelerated deformable conv2D backward pass for input gradients.
    /// Falls back to CPU implementation if GPU is unavailable.
    /// </summary>
    public new Tensor<T> DeformableConv2DBackwardInput<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T> offsets,
        Tensor<T>? mask,
        int[] inputShape,
        int[] stride,
        int[] padding,
        int[] dilation)
    {
        if (!TryGetBackend(out var backend))
            return base.DeformableConv2DBackwardInput(gradOutput, input, kernel, offsets, mask, inputShape, stride, padding, dilation);

        if (gradOutput.Rank != 4 || input.Rank != 4 || kernel.Rank != 4)
            return base.DeformableConv2DBackwardInput(gradOutput, input, kernel, offsets, mask, inputShape, stride, padding, dilation);

        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int inHeight = inputShape[2];
        int inWidth = inputShape[3];

        int outChannels = kernel.Shape[0];
        int kernelH = kernel.Shape[2];
        int kernelW = kernel.Shape[3];

        int outHeight = gradOutput.Shape[2];
        int outWidth = gradOutput.Shape[3];

        int offsetChannels = offsets.Shape[1];
        int deformGroups = offsetChannels / (2 * kernelH * kernelW);
        int groups = 1;

        int inputSize = batch * inChannels * inHeight * inWidth;

        using var gradOutputBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
        using var weightsBuffer = GetOrCacheWeightBuffer(backend, kernel.Data, PersistentTensorRole.Weights);
        using var offsetsBuffer = GetOrAllocateBuffer(backend, offsets.Data);
        using var gradInputBuffer = AllocateOutputBuffer(backend, inputSize);

        try
        {
            OwnedBuffer? maskBuffer = mask != null ? GetOrAllocateBuffer(backend, mask.Data) : null;
            try
            {
                backend.DeformableConv2DBackwardInput(
                    gradOutputBuffer.Buffer, weightsBuffer.Buffer, offsetsBuffer.Buffer, maskBuffer?.Buffer, gradInputBuffer.Buffer,
                    batch, inChannels, inHeight, inWidth,
                    outChannels, outHeight, outWidth,
                    kernelH, kernelW, stride[0], stride[1], padding[0], padding[1],
                    dilation[0], dilation[1], groups, deformGroups);

                float[] resultFloat = new float[inputSize];
                backend.DownloadBuffer(gradInputBuffer.Buffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, inputShape);
            }
            finally
            {
                maskBuffer?.Dispose();
            }
        }
        catch
        {
            return base.DeformableConv2DBackwardInput(gradOutput, input, kernel, offsets, mask, inputShape, stride, padding, dilation);
        }
    }

    /// <summary>
    /// GPU-accelerated deformable conv2D backward pass for kernel gradients.
    /// Falls back to CPU implementation if GPU is unavailable.
    /// </summary>
    public new Tensor<T> DeformableConv2DBackwardKernel<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> offsets,
        Tensor<T>? mask,
        int[] kernelShape,
        int[] stride,
        int[] padding,
        int[] dilation)
    {
        if (!TryGetBackend(out var backend))
            return base.DeformableConv2DBackwardKernel(gradOutput, input, offsets, mask, kernelShape, stride, padding, dilation);

        if (gradOutput.Rank != 4 || input.Rank != 4)
            return base.DeformableConv2DBackwardKernel(gradOutput, input, offsets, mask, kernelShape, stride, padding, dilation);

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int outChannels = kernelShape[0];
        int kernelH = kernelShape[2];
        int kernelW = kernelShape[3];

        int outHeight = gradOutput.Shape[2];
        int outWidth = gradOutput.Shape[3];

        int offsetChannels = offsets.Shape[1];
        int deformGroups = offsetChannels / (2 * kernelH * kernelW);
        int groups = 1;

        int kernelSize = kernelShape[0] * kernelShape[1] * kernelShape[2] * kernelShape[3];

        using var gradOutputBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        using var offsetsBuffer = GetOrAllocateBuffer(backend, offsets.Data);
        using var gradWeightsBuffer = AllocateOutputBuffer(backend, kernelSize);

        try
        {
            OwnedBuffer? maskBuffer = mask != null ? GetOrAllocateBuffer(backend, mask.Data) : null;
            try
            {
                backend.DeformableConv2DBackwardWeights(
                    inputBuffer.Buffer, gradOutputBuffer.Buffer, offsetsBuffer.Buffer, maskBuffer?.Buffer, gradWeightsBuffer.Buffer,
                    batch, inChannels, inHeight, inWidth,
                    outChannels, outHeight, outWidth,
                    kernelH, kernelW, stride[0], stride[1], padding[0], padding[1],
                    dilation[0], dilation[1], groups, deformGroups);

                float[] resultFloat = new float[kernelSize];
                backend.DownloadBuffer(gradWeightsBuffer.Buffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, kernelShape);
            }
            finally
            {
                maskBuffer?.Dispose();
            }
        }
        catch
        {
            return base.DeformableConv2DBackwardKernel(gradOutput, input, offsets, mask, kernelShape, stride, padding, dilation);
        }
    }

    /// <summary>
    /// GPU-accelerated deformable conv2D backward pass for offset gradients.
    /// Falls back to CPU implementation if GPU is unavailable.
    /// </summary>
    public new Tensor<T> DeformableConv2DBackwardOffset<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T> offsets,
        Tensor<T>? mask,
        int[] stride,
        int[] padding,
        int[] dilation)
    {
        if (!TryGetBackend(out var backend))
            return base.DeformableConv2DBackwardOffset(gradOutput, input, kernel, offsets, mask, stride, padding, dilation);

        if (gradOutput.Rank != 4 || input.Rank != 4 || kernel.Rank != 4)
            return base.DeformableConv2DBackwardOffset(gradOutput, input, kernel, offsets, mask, stride, padding, dilation);

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int outChannels = kernel.Shape[0];
        int kernelH = kernel.Shape[2];
        int kernelW = kernel.Shape[3];

        int outHeight = gradOutput.Shape[2];
        int outWidth = gradOutput.Shape[3];

        int offsetChannels = offsets.Shape[1];
        int deformGroups = offsetChannels / (2 * kernelH * kernelW);
        int groups = 1;

        int offsetSize = batch * offsetChannels * outHeight * outWidth;

        using var gradOutputBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        using var weightsBuffer = GetOrCacheWeightBuffer(backend, kernel.Data, PersistentTensorRole.Weights);
        using var offsetsBuffer = GetOrAllocateBuffer(backend, offsets.Data);
        using var gradOffsetBuffer = AllocateOutputBuffer(backend, offsetSize);

        try
        {
            OwnedBuffer? maskBuffer = mask != null ? GetOrAllocateBuffer(backend, mask.Data) : null;
            try
            {
                backend.DeformableConv2DBackwardOffset(
                    inputBuffer.Buffer, weightsBuffer.Buffer, gradOutputBuffer.Buffer, offsetsBuffer.Buffer, maskBuffer?.Buffer, gradOffsetBuffer.Buffer,
                    batch, inChannels, inHeight, inWidth,
                    outChannels, outHeight, outWidth,
                    kernelH, kernelW, stride[0], stride[1], padding[0], padding[1],
                    dilation[0], dilation[1], groups, deformGroups);

                float[] resultFloat = new float[offsetSize];
                backend.DownloadBuffer(gradOffsetBuffer.Buffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, offsets.Shape);
            }
            finally
            {
                maskBuffer?.Dispose();
            }
        }
        catch
        {
            return base.DeformableConv2DBackwardOffset(gradOutput, input, kernel, offsets, mask, stride, padding, dilation);
        }
    }

    /// <summary>
    /// GPU-accelerated deformable conv2D backward pass for mask gradients (DCNv2).
    /// Falls back to CPU implementation if GPU is unavailable.
    /// </summary>
    public new Tensor<T> DeformableConv2DBackwardMask<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T> offsets,
        Tensor<T>? mask,
        int[] stride,
        int[] padding,
        int[] dilation)
    {
        // Mask gradient computation requires a valid mask
        if (mask == null)
            throw new ArgumentNullException(nameof(mask), "Mask cannot be null when computing mask gradients");

        if (!TryGetBackend(out var backend))
            return base.DeformableConv2DBackwardMask(gradOutput, input, kernel, offsets, mask, stride, padding, dilation);

        if (gradOutput.Rank != 4 || input.Rank != 4 || kernel.Rank != 4 || mask.Rank != 4)
            return base.DeformableConv2DBackwardMask(gradOutput, input, kernel, offsets, mask, stride, padding, dilation);

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int outChannels = kernel.Shape[0];
        int kernelH = kernel.Shape[2];
        int kernelW = kernel.Shape[3];

        int outHeight = gradOutput.Shape[2];
        int outWidth = gradOutput.Shape[3];

        int offsetChannels = offsets.Shape[1];
        int deformGroups = offsetChannels / (2 * kernelH * kernelW);
        int maskChannels = deformGroups * kernelH * kernelW;
        int groups = 1;

        int maskSize = batch * maskChannels * outHeight * outWidth;

        using var gradOutputBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        using var weightsBuffer = GetOrCacheWeightBuffer(backend, kernel.Data, PersistentTensorRole.Weights);
        using var offsetsBuffer = GetOrAllocateBuffer(backend, offsets.Data);
        using var gradMaskBuffer = AllocateOutputBuffer(backend, maskSize);

        try
        {
            backend.DeformableConv2DBackwardMask(
                inputBuffer.Buffer, weightsBuffer.Buffer, gradOutputBuffer.Buffer, offsetsBuffer.Buffer, gradMaskBuffer.Buffer,
                batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth,
                kernelH, kernelW, stride[0], stride[1], padding[0], padding[1],
                dilation[0], dilation[1], groups, deformGroups);

            float[] resultFloat = new float[maskSize];
            backend.DownloadBuffer(gradMaskBuffer.Buffer, resultFloat);

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, mask.Shape);
        }
        catch
        {
            return base.DeformableConv2DBackwardMask(gradOutput, input, kernel, offsets, mask, stride, padding, dilation);
        }
    }

    /// <summary>
    /// GPU-accelerated deformable 2D convolution (DCNv2).
    /// Convolution with learnable offsets and optional modulation masks.
    /// </summary>
    /// <param name="input">Input tensor [batch, inChannels, inHeight, inWidth]</param>
    /// <param name="weights">Weight tensor [outChannels, inChannels/groups, kH, kW]</param>
    /// <param name="offsets">Offset tensor [batch, deformGroups*2*kH*kW, outH, outW]</param>
    /// <param name="mask">Optional mask tensor [batch, deformGroups*kH*kW, outH, outW] for DCNv2</param>
    /// <param name="bias">Optional bias tensor [outChannels]</param>
    /// <param name="strideH">Vertical stride</param>
    /// <param name="strideW">Horizontal stride</param>
    /// <param name="padH">Vertical padding</param>
    /// <param name="padW">Horizontal padding</param>
    /// <param name="dilationH">Vertical dilation</param>
    /// <param name="dilationW">Horizontal dilation</param>
    /// <param name="groups">Number of convolution groups</param>
    /// <param name="deformGroups">Number of deformable groups</param>
    /// <param name="activation">Fused activation type</param>
    /// <returns>Output GPU tensor [batch, outChannels, outHeight, outWidth]</returns>
    public IGpuTensor<T> DeformableConv2DGpu<T>(
        IGpuTensor<T> input,
        Tensor<T> weights,
        Tensor<T> offsets,
        Tensor<T>? mask,
        Tensor<T>? bias,
        int strideH, int strideW,
        int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups,
        FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for DeformableConv2DGpu");

        // Expected input shape: [batch, inChannels, height, width]
        // Expected weights shape: [outChannels, inChannels/groups, kH, kW]
        // Expected offsets shape: [batch, deformGroups*2*kH*kW, outH, outW]
        if (input.Shape.Length != 4)
            throw new ArgumentException("DeformableConv2DGpu requires 4D input tensor");
        if (weights.Rank != 4)
            throw new ArgumentException("DeformableConv2DGpu requires 4D weights tensor [outC, inC/groups, kH, kW]");

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int outChannels = weights.Shape[0];
        int kernelH = weights.Shape[2];
        int kernelW = weights.Shape[3];

        int outHeight = (inHeight + 2 * padH - dilationH * (kernelH - 1) - 1) / strideH + 1;
        int outWidth = (inWidth + 2 * padW - dilationW * (kernelW - 1) - 1) / strideW + 1;

        if (outHeight <= 0 || outWidth <= 0)
            throw new ArgumentException($"Invalid deformable convolution parameters result in non-positive output dimensions: {outHeight}x{outWidth}");

        int outputSize = batch * outChannels * outHeight * outWidth;

        using var weightsBuffer = GetOrCacheWeightBuffer(backend, weights.Data, PersistentTensorRole.Weights);
        using var offsetsBuffer = GetOrAllocateBuffer(backend, offsets.Data);
        using var maskBuffer = mask != null ? GetOrAllocateBuffer(backend, mask.Data) : default(OwnedBuffer?);
        using var biasBuffer = bias != null ? GetOrCacheWeightBuffer(backend, bias.Data, PersistentTensorRole.Biases) : default(OwnedBuffer?);
        var outputBuffer = backend.AllocateBuffer(outputSize);

        try
        {
            backend.DeformableConv2D(input.Buffer, weightsBuffer.Buffer, offsetsBuffer.Buffer, maskBuffer?.Buffer, outputBuffer,
                batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth,
                kernelH, kernelW, strideH, strideW, padH, padW,
                dilationH, dilationW, groups, deformGroups);

            // Add bias if present
            if (bias != null && biasBuffer.HasValue)
            {
                int spatialSize = outHeight * outWidth;
                var biasBuf = biasBuffer.Value.Buffer;
                backend.Conv2DBiasAdd(outputBuffer, biasBuf, batch, outChannels, spatialSize);
            }

            // Apply activation on GPU
            if (activation != FusedActivationType.None)
            {
                ApplyGpuActivation(backend, outputBuffer, outputSize, activation);
            }

            return new GpuTensor<T>(backend, outputBuffer, new[] { batch, outChannels, outHeight, outWidth },
                GpuTensorRole.Activation, ownsBuffer: true);
        }
        catch
        {
            outputBuffer.Dispose();
            throw;
        }
    }

    /// <summary>
    /// GPU-accelerated fused batch normalization with activation.
    /// Uses cached GPU buffers for registered persistent tensors (gamma/beta/running stats)
    /// to avoid redundant CPU→GPU transfers on every forward pass.
    /// </summary>
    public new Tensor<T> FusedBatchNorm<T>(
        Tensor<T> input,
        Tensor<T> gamma,
        Tensor<T> beta,
        Tensor<T> runningMean,
        Tensor<T> runningVar,
        double epsilon,
        double momentum,
        bool training,
        FusedActivationType activation,
        out Tensor<T> saveMean,
        out Tensor<T> saveVar)
    {
        if (!TryGetBackend(out var backend))
            return base.FusedBatchNorm(input, gamma, beta, runningMean, runningVar, epsilon, momentum, training, activation, out saveMean, out saveVar);

        if (input.Rank != 2)
            return base.FusedBatchNorm(input, gamma, beta, runningMean, runningVar, epsilon, momentum, training, activation, out saveMean, out saveVar);

        int batchSize = input.Shape[0];
        int features = input.Shape[1];

        // Use cache-aware buffer allocation (OwnedBuffer auto-disposes only if we allocated)
        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        using var outputBuffer = AllocateOutputBuffer(backend, batchSize * features);
        using var saveMeanBuffer = AllocateOutputBuffer(backend, features);
        using var saveVarBuffer = AllocateOutputBuffer(backend, features);
        using var gammaBuffer = GetOrCacheWeightBuffer(backend, gamma.Data, PersistentTensorRole.Weights);
        using var betaBuffer = GetOrCacheWeightBuffer(backend, beta.Data, PersistentTensorRole.Biases);
        using var runningMeanBuffer = GetOrCacheWeightBuffer(backend, runningMean.Data, PersistentTensorRole.NormalizationParams);
        using var runningVarBuffer = GetOrCacheWeightBuffer(backend, runningVar.Data, PersistentTensorRole.NormalizationParams);

        try
        {
            // Execute batch norm (spatialSize=1 for 2D tensors)
            backend.BatchNorm(inputBuffer.Buffer, outputBuffer.Buffer, gammaBuffer.Buffer, betaBuffer.Buffer,
                runningMeanBuffer.Buffer, runningVarBuffer.Buffer, saveMeanBuffer.Buffer, saveVarBuffer.Buffer,
                batchSize, features, 1, (float)epsilon, (float)momentum, training);

            // Apply activation if needed
            if (activation != FusedActivationType.None)
            {
                ApplyGpuActivation(backend, outputBuffer.Buffer, batchSize * features, activation);
            }

            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = new float[batchSize * features];
            float[] saveMeanFloat = new float[features];
            float[] saveVarFloat = new float[features];

            backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);
            backend.DownloadBuffer(saveMeanBuffer.Buffer, saveMeanFloat);
            backend.DownloadBuffer(saveVarBuffer.Buffer, saveVarFloat);

            // Convert back to T
            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            T[] saveMeanData = DirectGpuEngine.FromFloatArray<T>(saveMeanFloat);
            T[] saveVarData = DirectGpuEngine.FromFloatArray<T>(saveVarFloat);

            saveMean = new Tensor<T>(saveMeanData, new[] { features });
            saveVar = new Tensor<T>(saveVarData, new[] { features });
            return new Tensor<T>(resultData, input.Shape.ToArray());
        }
        catch
        {
            return base.FusedBatchNorm(input, gamma, beta, runningMean, runningVar, epsilon, momentum, training, activation, out saveMean, out saveVar);
        }
    }

    #endregion

    #region Attention Operations (GPU-accelerated)

    /// <summary>
    /// GPU-accelerated FlashAttention - memory-efficient O(N) attention algorithm.
    /// Uses cached GPU buffers for registered persistent tensors (e.g., KV cache) to avoid
    /// redundant CPU→GPU transfers on every forward pass.
    /// Falls back to CPU implementation when GPU is unavailable.
    /// </summary>
    public new Tensor<T> FlashAttention<T>(
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        double? scale,
        bool isCausal,
        out Tensor<T> softmaxStats)
    {
        if (!TryGetBackend(out var backend))
            return base.FlashAttention(query, key, value, scale, isCausal, out softmaxStats);

        // Validate tensor shapes [batch, heads, seq, head_dim]
        if (query.Rank != 4 || key.Rank != 4 || value.Rank != 4)
            return base.FlashAttention(query, key, value, scale, isCausal, out softmaxStats);

        int batch = query.Shape[0];
        int heads = query.Shape[1];
        int seqQ = query.Shape[2];
        int headDim = query.Shape[3];
        int seqK = key.Shape[2];

        // Compute scale if not provided
        float scaleFloat = (float)(scale ?? (1.0 / Math.Sqrt(headDim)));

        // Use cache-aware buffer allocation (especially important for KV cache)
        using var queryBuffer = GetOrAllocateBuffer(backend, query.Data);
        using var keyBuffer = GetOrAllocateBuffer(backend, key.Data);
        using var valueBuffer = GetOrAllocateBuffer(backend, value.Data);
        using var outputBuffer = AllocateOutputBuffer(backend, batch * heads * seqQ * headDim);
        using var statsBuffer = AllocateOutputBuffer(backend, batch * heads * seqQ);

        try
        {
            // Execute GPU FlashAttention
            backend.FlashAttentionV2(queryBuffer.Buffer, keyBuffer.Buffer, valueBuffer.Buffer, outputBuffer.Buffer, statsBuffer.Buffer,
                batch, heads, seqQ, seqK, headDim, scaleFloat, isCausal);

            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] outputFloat = new float[batch * heads * seqQ * headDim];
            float[] statsFloat = new float[batch * heads * seqQ];
            backend.DownloadBuffer(outputBuffer.Buffer, outputFloat);
            backend.DownloadBuffer(statsBuffer.Buffer, statsFloat);

            // Convert back to T
            T[] outputData = DirectGpuEngine.FromFloatArray<T>(outputFloat);
            T[] statsData = DirectGpuEngine.FromFloatArray<T>(statsFloat);

            softmaxStats = new Tensor<T>(statsData, new[] { batch, heads, seqQ });
            return new Tensor<T>(outputData, new[] { batch, heads, seqQ, headDim });
        }
        catch
        {
            // Fall back to CPU on any GPU error
            return base.FlashAttention(query, key, value, scale, isCausal, out softmaxStats);
        }
    }

    /// <summary>
    /// GPU-accelerated backward pass for FlashAttention.
    /// Uses cached GPU buffers for registered persistent tensors to avoid redundant transfers.
    /// </summary>
    public new Tensor<T> FlashAttentionBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        Tensor<T> output,
        Tensor<T> softmaxStats,
        double scale,
        bool isCausal,
        out Tensor<T> gradQuery,
        out Tensor<T> gradKey,
        out Tensor<T> gradValue)
    {
        if (!TryGetBackend(out var backend))
            return base.FlashAttentionBackward(gradOutput, query, key, value, output, softmaxStats, scale, isCausal,
                out gradQuery, out gradKey, out gradValue);

        if (query.Rank != 4)
            return base.FlashAttentionBackward(gradOutput, query, key, value, output, softmaxStats, scale, isCausal,
                out gradQuery, out gradKey, out gradValue);

        int batch = query.Shape[0];
        int heads = query.Shape[1];
        int seqQ = query.Shape[2];
        int headDim = query.Shape[3];
        int seqK = key.Shape[2];

        // Use cache-aware buffer allocation
        using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
        using var queryBuffer = GetOrAllocateBuffer(backend, query.Data);
        using var keyBuffer = GetOrAllocateBuffer(backend, key.Data);
        using var valueBuffer = GetOrAllocateBuffer(backend, value.Data);
        using var outputBuffer = GetOrAllocateBuffer(backend, output.Data);
        using var statsBuffer = GetOrAllocateBuffer(backend, softmaxStats.Data);
        using var gradQBuffer = AllocateOutputBuffer(backend, batch * heads * seqQ * headDim);
        using var gradKBuffer = AllocateOutputBuffer(backend, batch * heads * seqK * headDim);
        using var gradVBuffer = AllocateOutputBuffer(backend, batch * heads * seqK * headDim);

        try
        {
            // Execute GPU backward
            backend.FlashAttentionBackward(gradOutBuffer.Buffer, queryBuffer.Buffer, keyBuffer.Buffer, valueBuffer.Buffer,
                outputBuffer.Buffer, statsBuffer.Buffer, gradQBuffer.Buffer, gradKBuffer.Buffer, gradVBuffer.Buffer,
                batch, heads, seqQ, seqK, headDim, (float)scale, isCausal);

            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] gradQFloat = new float[batch * heads * seqQ * headDim];
            float[] gradKFloat = new float[batch * heads * seqK * headDim];
            float[] gradVFloat = new float[batch * heads * seqK * headDim];
            backend.DownloadBuffer(gradQBuffer.Buffer, gradQFloat);
            backend.DownloadBuffer(gradKBuffer.Buffer, gradKFloat);
            backend.DownloadBuffer(gradVBuffer.Buffer, gradVFloat);

            // Convert back to T
            gradQuery = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradQFloat), query.Shape.ToArray());
            gradKey = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradKFloat), key.Shape.ToArray());
            gradValue = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradVFloat), value.Shape.ToArray());

            return gradOutput;
        }
        catch
        {
            return base.FlashAttentionBackward(gradOutput, query, key, value, output, softmaxStats, scale, isCausal,
                out gradQuery, out gradKey, out gradValue);
        }
    }

    /// <summary>
    /// GPU-accelerated Grouped Query Attention for efficient inference.
    /// Uses cached GPU buffers for registered persistent tensors (e.g., KV cache) to avoid
    /// redundant CPU→GPU transfers on every forward pass.
    /// Falls back to CPU implementation when GPU is unavailable.
    /// </summary>
    public new Tensor<T> GroupedQueryAttention<T>(
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        int numQueriesPerKV,
        double? scale,
        bool isCausal,
        out Tensor<T> attentionWeights)
    {
        if (!TryGetBackend(out var backend))
            return base.GroupedQueryAttention(query, key, value, numQueriesPerKV, scale, isCausal, out attentionWeights);

        // Validate tensor shapes
        if (query.Rank != 4 || key.Rank != 4 || value.Rank != 4)
            return base.GroupedQueryAttention(query, key, value, numQueriesPerKV, scale, isCausal, out attentionWeights);

        int batch = query.Shape[0];
        int numQHeads = query.Shape[1];
        int seqQ = query.Shape[2];
        int headDim = query.Shape[3];
        int numKVHeads = key.Shape[1];
        int seqK = key.Shape[2];

        float scaleFloat = (float)(scale ?? (1.0 / Math.Sqrt(headDim)));

        // Use cache-aware buffer allocation (especially important for KV cache)
        using var queryBuffer = GetOrAllocateBuffer(backend, query.Data);
        using var keyBuffer = GetOrAllocateBuffer(backend, key.Data);
        using var valueBuffer = GetOrAllocateBuffer(backend, value.Data);
        using var outputBuffer = AllocateOutputBuffer(backend, batch * numQHeads * seqQ * headDim);
        using var attnWeightsBuffer = AllocateOutputBuffer(backend, batch * numQHeads * seqQ * seqK);

        try
        {
            // Execute GPU GQA
            backend.GroupedQueryAttention(queryBuffer.Buffer, keyBuffer.Buffer, valueBuffer.Buffer, outputBuffer.Buffer, attnWeightsBuffer.Buffer,
                batch, numQHeads, numKVHeads, seqQ, seqK, headDim, scaleFloat, isCausal);

            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] outputFloat = new float[batch * numQHeads * seqQ * headDim];
            float[] attnWeightsFloat = new float[batch * numQHeads * seqQ * seqK];
            backend.DownloadBuffer(outputBuffer.Buffer, outputFloat);
            backend.DownloadBuffer(attnWeightsBuffer.Buffer, attnWeightsFloat);

            // Convert back to T
            T[] outputData = DirectGpuEngine.FromFloatArray<T>(outputFloat);
            T[] attnWeightsData = DirectGpuEngine.FromFloatArray<T>(attnWeightsFloat);

            attentionWeights = new Tensor<T>(attnWeightsData, new[] { batch, numQHeads, seqQ, seqK });
            return new Tensor<T>(outputData, new[] { batch, numQHeads, seqQ, headDim });
        }
        catch
        {
            return base.GroupedQueryAttention(query, key, value, numQueriesPerKV, scale, isCausal, out attentionWeights);
        }
    }

    /// <summary>
    /// GPU-accelerated backward pass for Grouped Query Attention.
    /// Uses cached GPU buffers for registered persistent tensors to avoid redundant transfers.
    /// </summary>
    public new Tensor<T> GroupedQueryAttentionBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        Tensor<T> attentionWeights,
        int numQueriesPerKV,
        double scale,
        out Tensor<T> gradQuery,
        out Tensor<T> gradKey,
        out Tensor<T> gradValue)
    {
        if (!TryGetBackend(out var backend))
            return base.GroupedQueryAttentionBackward(gradOutput, query, key, value, attentionWeights, numQueriesPerKV, scale,
                out gradQuery, out gradKey, out gradValue);

        if (query.Rank != 4)
            return base.GroupedQueryAttentionBackward(gradOutput, query, key, value, attentionWeights, numQueriesPerKV, scale,
                out gradQuery, out gradKey, out gradValue);

        int batch = query.Shape[0];
        int numQHeads = query.Shape[1];
        int seqQ = query.Shape[2];
        int headDim = query.Shape[3];
        int numKVHeads = key.Shape[1];
        int seqK = key.Shape[2];

        // Use cache-aware buffer allocation
        using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
        using var queryBuffer = GetOrAllocateBuffer(backend, query.Data);
        using var keyBuffer = GetOrAllocateBuffer(backend, key.Data);
        using var valueBuffer = GetOrAllocateBuffer(backend, value.Data);
        using var attnWeightsBuffer = GetOrAllocateBuffer(backend, attentionWeights.Data);
        using var gradQBuffer = AllocateOutputBuffer(backend, batch * numQHeads * seqQ * headDim);
        using var gradKBuffer = AllocateOutputBuffer(backend, batch * numKVHeads * seqK * headDim);
        using var gradVBuffer = AllocateOutputBuffer(backend, batch * numKVHeads * seqK * headDim);

        try
        {
            // Execute GPU backward
            backend.GroupedQueryAttentionBackward(gradOutBuffer.Buffer, queryBuffer.Buffer, keyBuffer.Buffer, valueBuffer.Buffer,
                attnWeightsBuffer.Buffer, gradQBuffer.Buffer, gradKBuffer.Buffer, gradVBuffer.Buffer,
                batch, numQHeads, numKVHeads, seqQ, seqK, headDim, (float)scale);

            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] gradQFloat = new float[batch * numQHeads * seqQ * headDim];
            float[] gradKFloat = new float[batch * numKVHeads * seqK * headDim];
            float[] gradVFloat = new float[batch * numKVHeads * seqK * headDim];
            backend.DownloadBuffer(gradQBuffer.Buffer, gradQFloat);
            backend.DownloadBuffer(gradKBuffer.Buffer, gradKFloat);
            backend.DownloadBuffer(gradVBuffer.Buffer, gradVFloat);

            // Convert back to T
            gradQuery = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradQFloat), query.Shape.ToArray());
            gradKey = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradKFloat), key.Shape.ToArray());
            gradValue = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradVFloat), value.Shape.ToArray());

            return gradOutput;
        }
        catch
        {
            return base.GroupedQueryAttentionBackward(gradOutput, query, key, value, attentionWeights, numQueriesPerKV, scale,
                out gradQuery, out gradKey, out gradValue);
        }
    }

    /// <summary>
    /// GPU-resident Scaled Dot-Product Attention.
    /// Takes GPU-resident Q, K, V tensors in 4D shape [batch, heads, seq, head_dim]
    /// and returns GPU-resident attention output.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="query">GPU-resident query tensor [batch, heads, seqQ, headDim].</param>
    /// <param name="key">GPU-resident key tensor [batch, heads, seqK, headDim].</param>
    /// <param name="value">GPU-resident value tensor [batch, heads, seqK, headDim].</param>
    /// <param name="scale">Scaling factor (typically 1/sqrt(headDim)).</param>
    /// <param name="isCausal">If true, applies causal masking.</param>
    /// <returns>GPU-resident output tensor [batch, heads, seqQ, headDim].</returns>
    public IGpuTensor<T> ScaledDotProductAttentionGpu<T>(
        IGpuTensor<T> query,
        IGpuTensor<T> key,
        IGpuTensor<T> value,
        double scale,
        bool isCausal = false)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ScaledDotProductAttentionGpu");

        // Validate 4D tensor shapes
        if (query.Shape.Length != 4 || key.Shape.Length != 4 || value.Shape.Length != 4)
            throw new ArgumentException("Query, Key, Value must be 4D tensors [batch, heads, seq, headDim]");

        int batch = query.Shape[0];
        int heads = query.Shape[1];
        int seqQ = query.Shape[2];
        int headDim = query.Shape[3];
        int seqK = key.Shape[2];

        // Allocate output and attention weights buffers
        var outputBuffer = backend.AllocateBuffer(batch * heads * seqQ * headDim);
        var attnWeightsBuffer = backend.AllocateBuffer(batch * heads * seqQ * seqK);

        // Execute GPU ScaledDotProductAttention
        backend.ScaledDotProductAttention(
            query.Buffer, key.Buffer, value.Buffer,
            outputBuffer, attnWeightsBuffer, null,
            batch, heads, seqQ, headDim, (float)scale, isCausal);

        // Free attention weights buffer (not needed when not returning weights)
        attnWeightsBuffer.Dispose();

        // Return GPU-resident output
        return new GpuTensor<T>(backend, outputBuffer, new[] { batch, heads, seqQ, headDim },
            GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident scaled dot-product attention with attention weights output for training.
    /// Computes: softmax(Q @ K^T / scale) @ V, returning both output and attention weights.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="query">GPU-resident query tensor [batch, heads, seqQ, headDim].</param>
    /// <param name="key">GPU-resident key tensor [batch, heads, seqK, headDim].</param>
    /// <param name="value">GPU-resident value tensor [batch, heads, seqK, headDim].</param>
    /// <param name="scale">Scaling factor (typically 1/sqrt(headDim)).</param>
    /// <param name="attentionWeights">Output: GPU-resident attention weights tensor [batch, heads, seqQ, seqK].</param>
    /// <param name="isCausal">If true, applies causal masking.</param>
    /// <returns>GPU-resident output tensor [batch, heads, seqQ, headDim].</returns>
    public IGpuTensor<T> ScaledDotProductAttentionGpu<T>(
        IGpuTensor<T> query,
        IGpuTensor<T> key,
        IGpuTensor<T> value,
        double scale,
        out IGpuTensor<T> attentionWeights,
        bool isCausal = false)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ScaledDotProductAttentionGpu");

        // Validate 4D tensor shapes
        if (query.Shape.Length != 4 || key.Shape.Length != 4 || value.Shape.Length != 4)
            throw new ArgumentException("Query, Key, Value must be 4D tensors [batch, heads, seq, headDim]");

        int batch = query.Shape[0];
        int heads = query.Shape[1];
        int seqQ = query.Shape[2];
        int headDim = query.Shape[3];
        int seqK = key.Shape[2];

        // Allocate output and attention weights buffers
        var outputBuffer = backend.AllocateBuffer(batch * heads * seqQ * headDim);
        var attnWeightsBuffer = backend.AllocateBuffer(batch * heads * seqQ * seqK);

        // Execute GPU ScaledDotProductAttention
        backend.ScaledDotProductAttention(
            query.Buffer, key.Buffer, value.Buffer,
            outputBuffer, attnWeightsBuffer, null,
            batch, heads, seqQ, headDim, (float)scale, isCausal);

        // Return both output and attention weights as GPU-resident tensors
        attentionWeights = new GpuTensor<T>(backend, attnWeightsBuffer, new[] { batch, heads, seqQ, seqK },
            GpuTensorRole.Activation, ownsBuffer: true);

        return new GpuTensor<T>(backend, outputBuffer, new[] { batch, heads, seqQ, headDim },
            GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident backward pass for scaled dot-product attention.
    /// Computes gradients for query, key, and value tensors.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">Gradient of loss w.r.t. attention output [batch, heads, seqLen, headDim].</param>
    /// <param name="query">Query tensor from forward pass [batch, heads, seqLen, headDim].</param>
    /// <param name="key">Key tensor from forward pass [batch, heads, seqLen, headDim].</param>
    /// <param name="value">Value tensor from forward pass [batch, heads, seqLen, headDim].</param>
    /// <param name="attentionWeights">Attention weights from forward pass [batch, heads, seqLen, seqLen].</param>
    /// <param name="scale">Scale factor (typically 1/sqrt(headDim)).</param>
    /// <param name="isCausal">Whether to use causal masking.</param>
    /// <returns>Tuple of (gradQuery, gradKey, gradValue) GPU-resident tensors.</returns>
    public (IGpuTensor<T> GradQuery, IGpuTensor<T> GradKey, IGpuTensor<T> GradValue) ScaledDotProductAttentionBackwardGpu<T>(
        IGpuTensor<T> gradOutput,
        IGpuTensor<T> query,
        IGpuTensor<T> key,
        IGpuTensor<T> value,
        IGpuTensor<T> attentionWeights,
        double scale,
        bool isCausal = false)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ScaledDotProductAttentionBackwardGpu");

        // Extract dimensions from query tensor (all tensors share same shape except attn weights)
        int[] qShape = query.Shape;
        if (qShape.Length != 4)
            throw new ArgumentException("Query tensor must be 4D [batch, heads, seqLen, headDim]");

        int batch = qShape[0];
        int heads = qShape[1];
        int seqLen = qShape[2];
        int headDim = qShape[3];

        // Allocate output gradient buffers with exception-safe disposal
        int qkvSize = batch * heads * seqLen * headDim;
        IGpuBuffer? gradQueryBuffer = null;
        IGpuBuffer? gradKeyBuffer = null;
        IGpuBuffer? gradValueBuffer = null;

        try
        {
            gradQueryBuffer = backend.AllocateBuffer(qkvSize);
            gradKeyBuffer = backend.AllocateBuffer(qkvSize);
            gradValueBuffer = backend.AllocateBuffer(qkvSize);

            // Execute ScaledDotProductAttentionBackward on GPU
            backend.ScaledDotProductAttentionBackward(
                gradOutput.Buffer, query.Buffer, key.Buffer, value.Buffer,
                attentionWeights.Buffer, gradQueryBuffer, gradKeyBuffer, gradValueBuffer,
                batch, heads, seqLen, headDim, (float)scale, isCausal);

            // Return GPU-resident gradient tensors
            var gradQuery = new GpuTensor<T>(backend, gradQueryBuffer, qShape, GpuTensorRole.Gradient, ownsBuffer: true);
            var gradKey = new GpuTensor<T>(backend, gradKeyBuffer, qShape, GpuTensorRole.Gradient, ownsBuffer: true);
            var gradValue = new GpuTensor<T>(backend, gradValueBuffer, qShape, GpuTensorRole.Gradient, ownsBuffer: true);

            // Ownership transferred to tensors
            gradQueryBuffer = null;
            gradKeyBuffer = null;
            gradValueBuffer = null;

            return (gradQuery, gradKey, gradValue);
        }
        finally
        {
            // Dispose any buffers that weren't successfully transferred
            gradQueryBuffer?.Dispose();
            gradKeyBuffer?.Dispose();
            gradValueBuffer?.Dispose();
        }
    }

    /// <summary>
    /// GPU-resident tensor permutation (transpose with arbitrary dimension reordering).
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor.</param>
    /// <param name="permutation">Permutation of dimensions (e.g., [0, 2, 1, 3] for [B,H,S,D] -> [B,S,H,D]).</param>
    /// <returns>GPU-resident permuted tensor.</returns>
    public IGpuTensor<T> PermuteGpu<T>(IGpuTensor<T> input, int[] permutation)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for PermuteGpu");

        if (permutation.Length != input.Shape.Length)
            throw new ArgumentException("Permutation length must match input rank");

        // Compute output shape
        int[] outputShape = new int[input.Shape.Length];
        for (int i = 0; i < permutation.Length; i++)
            outputShape[i] = input.Shape[permutation[i]];

        int totalElements = 1;
        foreach (int dim in input.Shape)
            totalElements *= dim;

        var outputBuffer = backend.AllocateBuffer(totalElements);
        backend.Permute(input.Buffer, outputBuffer, input.Shape, permutation);

        return new GpuTensor<T>(backend, outputBuffer, outputShape,
            GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident batched matrix multiplication.
    /// Supports 3D inputs [batch, M, K] @ [K, N] -> [batch, M, N] for projections.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor [batch, seq, inputDim] or 2D [batch*seq, inputDim].</param>
    /// <param name="weights">Weight tensor [inputDim, outputDim].</param>
    /// <returns>GPU-resident output tensor.</returns>
    public IGpuTensor<T> BatchedMatMulGpu<T>(IGpuTensor<T> input, Tensor<T> weights)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for BatchedMatMulGpu");

        if (weights.Rank != 2)
            throw new ArgumentException("Weights must be 2D tensor [inputDim, outputDim]");

        int inputDim = weights.Shape[0];
        int outputDim = weights.Shape[1];

        // Flatten input to 2D for MatMul: [batch*seq, inputDim]
        int flatBatch = 1;
        for (int i = 0; i < input.Shape.Length - 1; i++)
            flatBatch *= input.Shape[i];
        int lastDim = input.Shape[^1];

        if (lastDim != inputDim)
            throw new ArgumentException($"Input last dimension {lastDim} doesn't match weight input dimension {inputDim}");

        // Upload weights
        using var weightsBuffer = GetOrCacheWeightBuffer(backend, weights.Data, PersistentTensorRole.Weights);

        // Execute MatMul
        var resultBuffer = backend.MatMul(input.Buffer, weightsBuffer.Buffer, flatBatch, outputDim, inputDim);

        // Compute output shape (same leading dimensions, last dim = outputDim)
        int[] outputShape = new int[input.Shape.Length];
        for (int i = 0; i < input.Shape.Length - 1; i++)
            outputShape[i] = input.Shape[i];
        outputShape[^1] = outputDim;

        return new GpuTensor<T>(backend, resultBuffer, outputShape,
            GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident tensor reshape (zero-copy view when possible).
    /// Creates a new GPU tensor with the same buffer but different shape interpretation.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor.</param>
    /// <param name="newShape">New shape (total elements must match).</param>
    /// <returns>GPU-resident reshaped tensor (shares buffer with input).</returns>
    public void CopyGpu<T>(IGpuTensor<T> source, IGpuTensor<T> destination, int size)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for CopyGpu");

        backend.Copy(source.Buffer, destination.Buffer, size);
    }

    public void CopyGpu<T>(IGpuTensor<T> source, int srcOffset, IGpuTensor<T> destination, int destOffset, int size)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for CopyGpu");

        backend.Copy(source.Buffer, srcOffset, destination.Buffer, destOffset, size);
    }

    public void FillGpu<T>(IGpuTensor<T> buffer, float value, int size)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for FillGpu");

        backend.Fill(buffer.Buffer, value, size);
    }

    /// <summary>
    /// Copies a 2D region from source to destination with different strides.
    /// Useful for concatenating features: dest[row, destColOffset:destColOffset+srcCols] = src[row, :]
    /// </summary>

    /// <summary>
    /// GPU-resident tensor bias addition with broadcasting.
    /// Adds bias to the last dimension of the input tensor.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor.</param>
    /// <param name="bias">Bias tensor (1D, length must match input's last dimension).</param>
    /// <returns>GPU-resident output tensor with bias added.</returns>
    public IGpuTensor<T> AddBiasGpu<T>(IGpuTensor<T> input, Tensor<T> bias)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for AddBiasGpu");

        if (bias.Rank != 1)
            throw new ArgumentException("Bias must be 1D tensor");

        int lastDim = input.Shape[^1];
        if (bias.Length != lastDim)
            throw new ArgumentException($"Bias length {bias.Length} doesn't match input last dimension {lastDim}");

        int totalElements = 1;
        foreach (int dim in input.Shape)
            totalElements *= dim;

        // Upload bias
        using var biasBuffer = GetOrCacheWeightBuffer(backend, bias.Data, PersistentTensorRole.Biases);

        // Allocate output
        var outputBuffer = backend.AllocateBuffer(totalElements);

        // Execute bias addition (broadcast along last dimension)
        // BiasAdd signature: BiasAdd(A, bias, C, M, N) where A is [M, N], bias is [N], C is output [M, N]
        int numVectors = totalElements / lastDim;
        backend.BiasAdd(input.Buffer, biasBuffer.Buffer, outputBuffer, numVectors, lastDim);

        return new GpuTensor<T>(backend, outputBuffer, input.Shape.ToArray(),
            GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident nearest-neighbor upsampling.
    /// Increases spatial dimensions (last two) by the specified scale factor.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor with shape [..., height, width].</param>
    /// <param name="scaleFactor">Scale factor for both height and width.</param>
    /// <returns>GPU-resident upsampled tensor.</returns>
    public IGpuTensor<T> UpsampleGpu<T>(IGpuTensor<T> input, int scaleFactor)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for UpsampleGpu");

        if (input.Shape.Length < 2)
            throw new ArgumentException("Input must have at least 2 dimensions for upsampling");

        // Parameter validation guard
        if (scaleFactor <= 0)
            throw new ArgumentOutOfRangeException(nameof(scaleFactor), "Scale factor must be positive");

        // Compute output shape (scale last two dimensions)
        int[] outputShape = new int[input.Shape.Length];
        for (int i = 0; i < input.Shape.Length - 2; i++)
            outputShape[i] = input.Shape[i];

        int inHeight = input.Shape[^2];
        int inWidth = input.Shape[^1];
        int outHeight = inHeight * scaleFactor;
        int outWidth = inWidth * scaleFactor;
        outputShape[^2] = outHeight;
        outputShape[^1] = outWidth;

        // Compute total elements
        int batchChannels = 1;
        for (int i = 0; i < input.Shape.Length - 2; i++)
            batchChannels *= input.Shape[i];

        int inputSize = batchChannels * inHeight * inWidth;
        int outputSize = batchChannels * outHeight * outWidth;

        var outputBuffer = backend.AllocateBuffer(outputSize);

        // Use NearestNeighborUpsample if available, otherwise implement manually
        backend.NearestNeighborUpsample(
            input.Buffer, outputBuffer,
            batchChannels, inHeight, inWidth,
            scaleFactor);

        return new GpuTensor<T>(backend, outputBuffer, outputShape,
            GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// Performs GPU-accelerated backward pass for nearest-neighbor upsampling (2D).
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="gradOutput">Gradient from the next layer with upsampled shape.</param>
    /// <param name="inputHeight">Original input height before upsampling.</param>
    /// <param name="inputWidth">Original input width before upsampling.</param>
    /// <param name="scaleFactor">Scale factor used during forward pass.</param>
    /// <returns>GPU-resident gradient input tensor with original input shape.</returns>
    public IGpuTensor<T> UpsampleBackwardGpu<T>(IGpuTensor<T> gradOutput, int inputHeight, int inputWidth, int scaleFactor)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for UpsampleBackwardGpu");

        if (gradOutput.Shape.Length < 2)
            throw new ArgumentException("Gradient output must have at least 2 dimensions for upsampling backward");

        // Parameter validation guards
        if (scaleFactor <= 0)
            throw new ArgumentOutOfRangeException(nameof(scaleFactor), "Scale factor must be positive");

        if (inputHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputHeight), "Input height must be positive");

        if (inputWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputWidth), "Input width must be positive");

        // Validate that gradOutput dimensions are consistent with scale factor
        int expectedOutputHeight = inputHeight * scaleFactor;
        int expectedOutputWidth = inputWidth * scaleFactor;
        int actualOutputHeight = gradOutput.Shape[^2];
        int actualOutputWidth = gradOutput.Shape[^1];

        if (actualOutputHeight != expectedOutputHeight)
            throw new ArgumentException(
                $"Gradient output height ({actualOutputHeight}) does not match expected height ({expectedOutputHeight}) based on inputHeight ({inputHeight}) and scaleFactor ({scaleFactor})");

        if (actualOutputWidth != expectedOutputWidth)
            throw new ArgumentException(
                $"Gradient output width ({actualOutputWidth}) does not match expected width ({expectedOutputWidth}) based on inputWidth ({inputWidth}) and scaleFactor ({scaleFactor})");

        // Compute input shape (original shape before upsampling)
        int[] inputShape = new int[gradOutput.Shape.Length];
        for (int i = 0; i < gradOutput.Shape.Length - 2; i++)
            inputShape[i] = gradOutput.Shape[i];

        inputShape[^2] = inputHeight;
        inputShape[^1] = inputWidth;

        // Compute total elements
        int batchChannels = 1;
        for (int i = 0; i < gradOutput.Shape.Length - 2; i++)
            batchChannels *= gradOutput.Shape[i];

        int inputSize = batchChannels * inputHeight * inputWidth;

        var gradInputBuffer = backend.AllocateBuffer(inputSize);

        // Zero initialize the gradient input buffer for accumulation
        backend.Fill(gradInputBuffer, 0.0f, inputSize);

        // Use NearestNeighborUpsampleBackward for gradient propagation
        backend.NearestNeighborUpsampleBackward(
            gradOutput.Buffer, gradInputBuffer,
            batchChannels, inputHeight, inputWidth,
            scaleFactor);

        return new GpuTensor<T>(backend, gradInputBuffer, inputShape,
            GpuTensorRole.Gradient, ownsBuffer: true);
    }

    #endregion

    #region Persistent Tensor Management

    /// <summary>
    /// Registers a tensor for GPU memory optimization by pre-allocating and uploading
    /// its data to GPU memory. This eliminates repeated CPU-GPU transfers for tensors
    /// that are reused across multiple operations (e.g., layer weights, biases).
    /// </summary>
    public new void RegisterPersistentTensor<T>(Tensor<T> tensor, PersistentTensorRole role)
    {
        base.RegisterPersistentTensor(tensor, role);

        if (!TryGetBackend(out var backend))
            return;

        // Use the tensor's data array as the cache key
        object key = tensor.Data;

        // Check if already registered
        if (_persistentBufferCache.ContainsKey(key))
            return;

        try
        {
            // Convert tensor data to float and upload to GPU
            float[] floatData = DirectGpuEngine.ToFloatArray(tensor.Data);
            IGpuBuffer gpuBuffer = backend.AllocateBuffer(floatData);
            backend.Synchronize();

            var entry = new GpuBufferCacheEntry(gpuBuffer, role);
            _persistentBufferCache.TryAdd(key, entry);
            _tensorVersions.TryAdd(key, 0);
        }
        catch
        {
            // Silently ignore GPU allocation failures - operations will fall back to CPU
        }
    }

    /// <summary>
    /// Unregisters a persistent tensor and releases its associated GPU memory.
    /// </summary>
    public new void UnregisterPersistentTensor<T>(Tensor<T> tensor)
    {
        base.UnregisterPersistentTensor(tensor);

        object key = tensor.Data;

        if (_persistentBufferCache.TryRemove(key, out var entry))
        {
            entry.Dispose();
        }
        _tensorVersions.TryRemove(key, out _);
    }

    /// <summary>
    /// Invalidates a persistent tensor's GPU buffer, triggering re-upload of its
    /// data to GPU memory. Call this after modifying the tensor's data on CPU.
    /// </summary>
    public new void InvalidatePersistentTensor<T>(Tensor<T> tensor)
    {
        base.InvalidatePersistentTensor(tensor);

        if (!TryGetBackend(out var backend))
            return;

        object key = tensor.Data;

        if (!_persistentBufferCache.TryGetValue(key, out var entry))
            return;

        try
        {
            // Dispose old buffer
            entry.Buffer.Dispose();

            // Upload new data
            float[] floatData = DirectGpuEngine.ToFloatArray(tensor.Data);
            IGpuBuffer newBuffer = backend.AllocateBuffer(floatData);
            backend.Synchronize();

            // Update cache entry with new buffer
            var newEntry = new GpuBufferCacheEntry(newBuffer, entry.Role);
            newEntry.Version = entry.Version + 1;

            _persistentBufferCache[key] = newEntry;
            _tensorVersions[key] = newEntry.Version;
        }
        catch
        {
            // On failure, remove from cache - operations will fall back to CPU
            _persistentBufferCache.TryRemove(key, out _);
            _tensorVersions.TryRemove(key, out _);
        }
    }

    /// <summary>
    /// Attempts to get a cached GPU buffer for a tensor.
    /// Returns null if the tensor is not registered as persistent.
    /// </summary>
    internal IGpuBuffer? TryGetCachedBuffer<T>(T[] tensorData)
    {
        if (_persistentBufferCache.TryGetValue(tensorData, out var entry))
        {
            return entry.Buffer;
        }
        return null;
    }

    /// <summary>
    /// Gets the number of tensors currently cached on GPU.
    /// </summary>
    public int CachedTensorCount => _persistentBufferCache.Count;

    #endregion

    #region FFT Operations (GPU-accelerated)

    /// <summary>
    /// GPU-accelerated 1D complex-to-complex FFT.
    /// </summary>
    void IEngine.FFT<T>(Tensor<T> inputReal, Tensor<T> inputImag, out Tensor<T> outputReal, out Tensor<T> outputImag)
    {
        if (!TryGetBackend(out var backend) || inputReal.Length != inputImag.Length)
        {
            base.FFT(inputReal, inputImag, out outputReal, out outputImag);
            return;
        }

        int n = inputReal.Shape[^1];
        if ((n & (n - 1)) != 0) // Not power of 2
        {
            base.FFT(inputReal, inputImag, out outputReal, out outputImag);
            return;
        }

        try
        {
            // Use cache-aware buffer allocation for inputs
            using var inputRealBuffer = GetOrAllocateBuffer(backend, inputReal.Data);
            using var inputImagBuffer = GetOrAllocateBuffer(backend, inputImag.Data);
            using var outputRealBuffer = AllocateOutputBuffer(backend, inputReal.Length);
            using var outputImagBuffer = AllocateOutputBuffer(backend, inputImag.Length);

            backend.FFT(inputRealBuffer.Buffer, inputImagBuffer.Buffer, outputRealBuffer.Buffer, outputImagBuffer.Buffer, n, inverse: false);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] outputRealFloat = new float[inputReal.Length];
            float[] outputImagFloat = new float[inputImag.Length];
            backend.DownloadBuffer(outputRealBuffer.Buffer, outputRealFloat);
            backend.DownloadBuffer(outputImagBuffer.Buffer, outputImagFloat);

            outputReal = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputRealFloat), inputReal.Shape.ToArray());
            outputImag = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputImagFloat), inputImag.Shape.ToArray());
        }
        catch
        {
            base.FFT(inputReal, inputImag, out outputReal, out outputImag);
        }
    }

    /// <summary>
    /// GPU-accelerated 1D complex-to-complex inverse FFT.
    /// </summary>
    void IEngine.IFFT<T>(Tensor<T> inputReal, Tensor<T> inputImag, out Tensor<T> outputReal, out Tensor<T> outputImag)
    {
        if (!TryGetBackend(out var backend) || inputReal.Length != inputImag.Length)
        {
            base.IFFT(inputReal, inputImag, out outputReal, out outputImag);
            return;
        }

        int n = inputReal.Shape[^1];
        if ((n & (n - 1)) != 0)
        {
            base.IFFT(inputReal, inputImag, out outputReal, out outputImag);
            return;
        }

        try
        {
            // Use cache-aware buffer allocation for inputs
            using var inputRealBuffer = GetOrAllocateBuffer(backend, inputReal.Data);
            using var inputImagBuffer = GetOrAllocateBuffer(backend, inputImag.Data);
            using var outputRealBuffer = AllocateOutputBuffer(backend, inputReal.Length);
            using var outputImagBuffer = AllocateOutputBuffer(backend, inputImag.Length);

            backend.FFT(inputRealBuffer.Buffer, inputImagBuffer.Buffer, outputRealBuffer.Buffer, outputImagBuffer.Buffer, n, inverse: true);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] outputRealFloat = new float[inputReal.Length];
            float[] outputImagFloat = new float[inputImag.Length];
            backend.DownloadBuffer(outputRealBuffer.Buffer, outputRealFloat);
            backend.DownloadBuffer(outputImagBuffer.Buffer, outputImagFloat);

            outputReal = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputRealFloat), inputReal.Shape.ToArray());
            outputImag = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputImagFloat), inputImag.Shape.ToArray());
        }
        catch
        {
            base.IFFT(inputReal, inputImag, out outputReal, out outputImag);
        }
    }

    /// <summary>
    /// GPU-accelerated 2D FFT.
    /// </summary>
    void IEngine.FFT2D<T>(Tensor<T> inputReal, Tensor<T> inputImag, out Tensor<T> outputReal, out Tensor<T> outputImag)
    {
        if (!TryGetBackend(out var backend) || inputReal.Rank < 2 || inputReal.Length != inputImag.Length)
        {
            base.FFT2D(inputReal, inputImag, out outputReal, out outputImag);
            return;
        }

        int height = inputReal.Shape[^2];
        int width = inputReal.Shape[^1];

        if ((height & (height - 1)) != 0 || (width & (width - 1)) != 0)
        {
            base.FFT2D(inputReal, inputImag, out outputReal, out outputImag);
            return;
        }

        try
        {
            // Use cache-aware buffer allocation for inputs
            using var inputRealBuffer = GetOrAllocateBuffer(backend, inputReal.Data);
            using var inputImagBuffer = GetOrAllocateBuffer(backend, inputImag.Data);
            using var outputRealBuffer = AllocateOutputBuffer(backend, inputReal.Length);
            using var outputImagBuffer = AllocateOutputBuffer(backend, inputImag.Length);

            backend.FFT2D(inputRealBuffer.Buffer, inputImagBuffer.Buffer, outputRealBuffer.Buffer, outputImagBuffer.Buffer, height, width, inverse: false);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] outputRealFloat = new float[inputReal.Length];
            float[] outputImagFloat = new float[inputImag.Length];
            backend.DownloadBuffer(outputRealBuffer.Buffer, outputRealFloat);
            backend.DownloadBuffer(outputImagBuffer.Buffer, outputImagFloat);

            outputReal = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputRealFloat), inputReal.Shape.ToArray());
            outputImag = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputImagFloat), inputImag.Shape.ToArray());
        }
        catch
        {
            base.FFT2D(inputReal, inputImag, out outputReal, out outputImag);
        }
    }

    /// <summary>
    /// GPU-accelerated 2D inverse FFT.
    /// </summary>
    void IEngine.IFFT2D<T>(Tensor<T> inputReal, Tensor<T> inputImag, out Tensor<T> outputReal, out Tensor<T> outputImag)
    {
        if (!TryGetBackend(out var backend) || inputReal.Rank < 2 || inputReal.Length != inputImag.Length)
        {
            base.IFFT2D(inputReal, inputImag, out outputReal, out outputImag);
            return;
        }

        int height = inputReal.Shape[^2];
        int width = inputReal.Shape[^1];

        if ((height & (height - 1)) != 0 || (width & (width - 1)) != 0)
        {
            base.IFFT2D(inputReal, inputImag, out outputReal, out outputImag);
            return;
        }

        try
        {
            // Use cache-aware buffer allocation for inputs
            using var inputRealBuffer = GetOrAllocateBuffer(backend, inputReal.Data);
            using var inputImagBuffer = GetOrAllocateBuffer(backend, inputImag.Data);
            using var outputRealBuffer = AllocateOutputBuffer(backend, inputReal.Length);
            using var outputImagBuffer = AllocateOutputBuffer(backend, inputImag.Length);

            backend.FFT2D(inputRealBuffer.Buffer, inputImagBuffer.Buffer, outputRealBuffer.Buffer, outputImagBuffer.Buffer, height, width, inverse: true);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] outputRealFloat = new float[inputReal.Length];
            float[] outputImagFloat = new float[inputImag.Length];
            backend.DownloadBuffer(outputRealBuffer.Buffer, outputRealFloat);
            backend.DownloadBuffer(outputImagBuffer.Buffer, outputImagFloat);

            outputReal = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputRealFloat), inputReal.Shape.ToArray());
            outputImag = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputImagFloat), inputImag.Shape.ToArray());
        }
        catch
        {
            base.IFFT2D(inputReal, inputImag, out outputReal, out outputImag);
        }
    }

    /// <summary>
    /// GPU-accelerated Short-Time Fourier Transform.
    /// </summary>
    void IEngine.STFT<T>(
        Tensor<T> input,
        int nFft,
        int hopLength,
        Tensor<T> window,
        bool center,
        out Tensor<T> magnitudeOut,
        out Tensor<T> phaseOut)
    {
        if (!TryGetBackend(out var backend) || (nFft & (nFft - 1)) != 0)
        {
            base.STFT(input, nFft, hopLength, window, center, out magnitudeOut, out phaseOut);
            return;
        }

        try
        {
            // For STFT, we need to process frame by frame
            // First, handle centering by padding the input
            T[] inputData = input.Data;
            if (center)
            {
                int padAmount = nFft / 2;
                T[] paddedData = new T[inputData.Length + 2 * padAmount];
                Array.Copy(inputData, 0, paddedData, padAmount, inputData.Length);
                inputData = paddedData;
            }

            int numSamples = inputData.Length;
            int numFrames = (numSamples - nFft) / hopLength + 1;
            int numFreqs = nFft / 2 + 1;

            if (numFrames <= 0)
            {
                base.STFT(input, nFft, hopLength, window, center, out magnitudeOut, out phaseOut);
                return;
            }

            float[] inputFloat = DirectGpuEngine.ToFloatArray(inputData);

            // Use cache-aware allocation for window (likely persistent)
            using var windowBuffer = GetOrAllocateBuffer(backend, window.Data);
            // Allocate working buffers
            using var frameBuffer = AllocateOutputBuffer(backend, nFft);
            using var windowedBuffer = AllocateOutputBuffer(backend, nFft);
            using var fftRealBuffer = AllocateOutputBuffer(backend, nFft);
            using var fftImagBuffer = AllocateOutputBuffer(backend, nFft);
            using var zeroBuffer = AllocateOutputBuffer(backend, nFft);

            float[] magnitudeData = new float[numFrames * numFreqs];
            float[] phaseData = new float[numFrames * numFreqs];

            for (int frame = 0; frame < numFrames; frame++)
            {
                int frameStart = frame * hopLength;

                // Extract frame from input
                float[] frameData = new float[nFft];
                Array.Copy(inputFloat, frameStart, frameData, 0, Math.Min(nFft, inputFloat.Length - frameStart));

                // Upload frame data
                using var currentFrameBuffer = backend.AllocateBuffer(frameData);

                // Apply window
                backend.ApplyWindow(currentFrameBuffer, windowBuffer.Buffer, windowedBuffer.Buffer, nFft);

                // Perform FFT (windowed signal as real input, zeros as imaginary)
                backend.FFT(windowedBuffer.Buffer, zeroBuffer.Buffer, fftRealBuffer.Buffer, fftImagBuffer.Buffer, nFft, inverse: false);

                // Download FFT results
                float[] fftReal = new float[nFft];
                float[] fftImag = new float[nFft];
                backend.DownloadBuffer(fftRealBuffer.Buffer, fftReal);
                backend.DownloadBuffer(fftImagBuffer.Buffer, fftImag);

                // Compute magnitude and phase for positive frequencies only
                for (int k = 0; k < numFreqs; k++)
                {
                    float real = fftReal[k];
                    float imag = fftImag[k];
                    magnitudeData[frame * numFreqs + k] = (float)Math.Sqrt(real * real + imag * imag);
                    phaseData[frame * numFreqs + k] = (float)Math.Atan2(imag, real);
                }
            }
            // Note: DownloadBuffer calls inside the loop are blocking, no need for Synchronize after

            int[] outputShape = input.Rank == 1
                ? new[] { numFrames, numFreqs }
                : new[] { input.Shape[0], numFrames, numFreqs };

            magnitudeOut = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(magnitudeData), outputShape);
            phaseOut = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(phaseData), outputShape);
        }
        catch
        {
            base.STFT(input, nFft, hopLength, window, center, out magnitudeOut, out phaseOut);
        }
    }

    /// <summary>
    /// GPU-accelerated inverse Short-Time Fourier Transform.
    /// </summary>
    Tensor<T> IEngine.ISTFT<T>(
        Tensor<T> magnitude,
        Tensor<T> phase,
        int nFft,
        int hopLength,
        Tensor<T> window,
        bool center,
        int? length)
    {
        if (!TryGetBackend(out var backend) || (nFft & (nFft - 1)) != 0)
        {
            return base.ISTFT(magnitude, phase, nFft, hopLength, window, center, length);
        }

        try
        {
            int numFrames = magnitude.Shape[^2];
            int numFreqs = magnitude.Shape[^1];

            float[] magnitudeFloat = DirectGpuEngine.ToFloatArray(magnitude.Data);
            float[] phaseFloat = DirectGpuEngine.ToFloatArray(phase.Data);
            float[] windowFloat = DirectGpuEngine.ToFloatArray(window.Data);

            // Reconstruct full spectrum (mirror for negative frequencies)
            int outputSamples = (numFrames - 1) * hopLength + nFft;
            float[] output = new float[outputSamples];
            float[] windowSum = new float[outputSamples];

            // Use cache-aware allocation for window (likely persistent)
            using var windowBuffer = GetOrAllocateBuffer(backend, window.Data);
            // Allocate working buffers
            using var outputRealBuffer = AllocateOutputBuffer(backend, nFft);
            using var outputImagBuffer = AllocateOutputBuffer(backend, nFft);

            for (int frame = 0; frame < numFrames; frame++)
            {
                // Convert polar to complex for full spectrum
                float[] frameReal = new float[nFft];
                float[] frameImag = new float[nFft];

                // Fill positive frequencies
                for (int k = 0; k < numFreqs; k++)
                {
                    float mag = magnitudeFloat[frame * numFreqs + k];
                    float ph = phaseFloat[frame * numFreqs + k];
                    frameReal[k] = mag * (float)Math.Cos(ph);
                    frameImag[k] = mag * (float)Math.Sin(ph);
                }

                // Mirror for negative frequencies (conjugate symmetry)
                for (int k = 1; k < nFft - numFreqs + 1; k++)
                {
                    int srcIdx = numFreqs - 1 - k;
                    if (srcIdx > 0 && srcIdx < numFreqs)
                    {
                        frameReal[nFft - k] = frameReal[srcIdx];
                        frameImag[nFft - k] = -frameImag[srcIdx];
                    }
                }

                using var frameRealBuffer = backend.AllocateBuffer(frameReal);
                using var frameImagBuffer = backend.AllocateBuffer(frameImag);

                // Perform inverse FFT
                backend.FFT(frameRealBuffer, frameImagBuffer, outputRealBuffer.Buffer, outputImagBuffer.Buffer, nFft, inverse: true);

                // Download result
                float[] ifftResult = new float[nFft];
                backend.DownloadBuffer(outputRealBuffer.Buffer, ifftResult);

                // Overlap-add with window
                int frameStart = frame * hopLength;
                for (int i = 0; i < nFft && frameStart + i < outputSamples; i++)
                {
                    float w = windowFloat[i];
                    output[frameStart + i] += ifftResult[i] * w;
                    windowSum[frameStart + i] += w * w;
                }
            }
            // Note: DownloadBuffer calls inside the loop are blocking, no need for Synchronize after

            // Normalize by window sum
            for (int i = 0; i < outputSamples; i++)
            {
                if (windowSum[i] > 1e-8f)
                {
                    output[i] /= windowSum[i];
                }
            }

            // Remove centering padding if needed
            if (center)
            {
                int padAmount = nFft / 2;
                int actualLength = length ?? (outputSamples - 2 * padAmount);
                float[] trimmed = new float[actualLength];
                Array.Copy(output, padAmount, trimmed, 0, Math.Min(actualLength, outputSamples - padAmount));
                output = trimmed;
            }
            else if (length.HasValue)
            {
                float[] trimmed = new float[length.Value];
                Array.Copy(output, 0, trimmed, 0, Math.Min(length.Value, output.Length));
                output = trimmed;
            }

            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(output), new[] { output.Length });
        }
        catch
        {
            return base.ISTFT(magnitude, phase, nFft, hopLength, window, center, length);
        }
    }

    /// <summary>
    /// GPU-accelerated Mel spectrogram computation.
    /// </summary>
    Tensor<T> IEngine.MelSpectrogram<T>(
        Tensor<T> input,
        int sampleRate,
        int nFft,
        int hopLength,
        int nMels,
        T fMin,
        T fMax,
        Tensor<T> window,
        bool powerToDb)
    {
        if (!TryGetBackend(out var backend) || (nFft & (nFft - 1)) != 0)
        {
            return base.MelSpectrogram(input, sampleRate, nFft, hopLength, nMels, fMin, fMax, window, powerToDb);
        }

        try
        {
            // First compute STFT
            ((IEngine)this).STFT(input, nFft, hopLength, window, center: true, out var magnitude, out var _);

            int numFrames = magnitude.Shape[^2];
            int numFreqs = magnitude.Shape[^1];

            // Create Mel filterbank
            var filterbank = ((IEngine)this).CreateMelFilterbank<T>(nMels, nFft, sampleRate, fMin, fMax);

            float[] magnitudeFloat = DirectGpuEngine.ToFloatArray(magnitude.Data);
            float[] filterbankFloat = DirectGpuEngine.ToFloatArray(filterbank.Data);

            // Compute power spectrum (magnitude squared)
            float[] powerSpec = new float[magnitudeFloat.Length];
            for (int i = 0; i < magnitudeFloat.Length; i++)
            {
                powerSpec[i] = magnitudeFloat[i] * magnitudeFloat[i];
            }

            // Use cache-aware allocation for filterbank (likely persistent)
            using var filterbankBuffer = GetOrAllocateBuffer(backend, filterbank.Data);
            // Allocate working buffers
            using var powerBuffer = backend.AllocateBuffer(powerSpec);
            using var melBuffer = AllocateOutputBuffer(backend, numFrames * nMels);

            // Apply Mel filterbank
            backend.ApplyMelFilterbank(powerBuffer, filterbankBuffer.Buffer, melBuffer.Buffer, numFrames, numFreqs, nMels);

            if (powerToDb)
            {
                using var dbBuffer = AllocateOutputBuffer(backend, numFrames * nMels);
                backend.PowerToDb(melBuffer.Buffer, dbBuffer.Buffer, numFrames * nMels, 1.0f, -80.0f);
                // DownloadBuffer uses blocking read, Synchronize() removed for performance
                float[] dbResult = new float[numFrames * nMels];
                backend.DownloadBuffer(dbBuffer.Buffer, dbResult);

                int[] outputShape = input.Rank == 1
                    ? new[] { numFrames, nMels }
                    : new[] { input.Shape[0], numFrames, nMels };

                return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(dbResult), outputShape);
            }
            else
            {
                // DownloadBuffer uses blocking read, Synchronize() removed for performance
                float[] melResult = new float[numFrames * nMels];
                backend.DownloadBuffer(melBuffer.Buffer, melResult);

                int[] outputShape = input.Rank == 1
                    ? new[] { numFrames, nMels }
                    : new[] { input.Shape[0], numFrames, nMels };

                return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(melResult), outputShape);
            }
        }
        catch
        {
            return base.MelSpectrogram(input, sampleRate, nFft, hopLength, nMels, fMin, fMax, window, powerToDb);
        }
    }

    /// <summary>
    /// GPU-accelerated Griffin-Lim algorithm for audio reconstruction from magnitude spectrogram.
    /// </summary>
    Tensor<T> IEngine.GriffinLim<T>(
        Tensor<T> magnitude,
        int nFft,
        int hopLength,
        Tensor<T> window,
        int iterations,
        double momentum,
        int? length)
    {
        if (!TryGetBackend(out var backend) || (nFft & (nFft - 1)) != 0)
        {
            return base.GriffinLim(magnitude, nFft, hopLength, window, iterations, momentum, length);
        }

        try
        {
            int numFrames = magnitude.Shape[^2];
            int numFreqs = magnitude.Shape[^1];

            float[] magnitudeFloat = DirectGpuEngine.ToFloatArray(magnitude.Data);

            // Initialize with random phase
            var random = new Random(42);
            float[] phase = new float[magnitudeFloat.Length];
            for (int i = 0; i < phase.Length; i++)
            {
                phase[i] = (float)(random.NextDouble() * 2 * Math.PI - Math.PI);
            }

            float[] prevPhase = new float[phase.Length];
            float momentumF = (float)momentum;

            for (int iter = 0; iter < iterations; iter++)
            {
                // Reconstruct signal using current phase estimate
                var phaseTensor = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(phase), magnitude.Shape.ToArray());
                var reconstructed = ((IEngine)this).ISTFT(magnitude, phaseTensor, nFft, hopLength, window, center: true, length);

                // Re-analyze to get new phase
                ((IEngine)this).STFT(reconstructed, nFft, hopLength, window, center: true, out var _, out var newPhaseTensor);

                float[] newPhase = DirectGpuEngine.ToFloatArray(newPhaseTensor.Data);

                // Apply momentum
                if (iter > 0 && momentumF > 0)
                {
                    for (int i = 0; i < phase.Length; i++)
                    {
                        // Unwrap phase difference for momentum
                        float diff = newPhase[i] - prevPhase[i];
                        while (diff > Math.PI) diff -= (float)(2 * Math.PI);
                        while (diff < -Math.PI) diff += (float)(2 * Math.PI);

                        float accelerated = prevPhase[i] + diff * (1 + momentumF);
                        phase[i] = accelerated;
                    }
                }
                else
                {
                    Array.Copy(newPhase, phase, phase.Length);
                }

                Array.Copy(newPhase, prevPhase, prevPhase.Length);
            }

            // Final reconstruction
            var finalPhaseTensor = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(phase), magnitude.Shape.ToArray());
            return ((IEngine)this).ISTFT(magnitude, finalPhaseTensor, nFft, hopLength, window, center: true, length);
        }
        catch
        {
            return base.GriffinLim(magnitude, nFft, hopLength, window, iterations, momentum, length);
        }
    }

    /// <summary>
    /// Creates a Mel filterbank matrix (CPU implementation, can be cached).
    /// </summary>
    Tensor<T> IEngine.CreateMelFilterbank<T>(int nMels, int nFft, int sampleRate, T fMin, T fMax)
    {
        // Filterbank creation is a one-time operation, use CPU base implementation
        return base.CreateMelFilterbank<T>(nMels, nFft, sampleRate, fMin, fMax);
    }

    /// <summary>
    /// Creates a window function (CPU implementation, can be cached).
    /// </summary>
    Tensor<T> IEngine.CreateWindow<T>(string windowType, int windowLength)
    {
        // Window creation is a one-time operation, use CPU base implementation
        return base.CreateWindow<T>(windowType, windowLength);
    }

    #endregion

    #region Normalization Operations (GPU Accelerated)

    /// <summary>
    /// GPU-accelerated Softmax operation.
    /// Supports arbitrary axes by treating the tensor as 2D (outerSize, features).
    /// </summary>
    Tensor<T> IEngine.Softmax<T>(Tensor<T> input, int axis)
    {
        if (!TryGetBackend(out var backend))
            return base.Softmax(input, axis);

        // Handle negative axis
        int rank = input.Rank;
        if (axis < 0) axis = rank + axis;
        if (axis < 0 || axis >= rank)
            return base.Softmax(input, axis);

        try
        {
            // For softmax over the last dimension, we can use GPU directly
            // by treating the tensor as 2D: (product of all dims except last) x (last dim)
            if (axis == rank - 1)
            {
                int features = input.Shape[rank - 1];
                int outerSize = input.Length / features;

                using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
                using var outputBuffer = AllocateOutputBuffer(backend, input.Length);

                backend.Softmax(inputBuffer.Buffer, outputBuffer.Buffer, outerSize, features);
                float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
                return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), input.Shape.ToArray());
            }

            // For softmax over a non-last axis, permute to move the axis to the end,
            // apply softmax, then permute back
            if (axis < rank - 1)
            {
                // Build permutation: move axis to end
                var permutation = new int[rank];
                int j = 0;
                for (int i = 0; i < rank; i++)
                {
                    if (i != axis) permutation[j++] = i;
                }
                permutation[rank - 1] = axis;

                // Permute input
                var permutedInput = PermuteImpl(input, permutation);

                // Now apply softmax over the last axis
                int features = permutedInput.Shape[rank - 1];
                int outerSize = permutedInput.Length / features;

                using var inputBuffer = GetOrAllocateBuffer(backend, permutedInput.Data);
                using var outputBuffer = AllocateOutputBuffer(backend, permutedInput.Length);

                backend.Softmax(inputBuffer.Buffer, outputBuffer.Buffer, outerSize, features);
                float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
                var permutedOutput = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), permutedInput.Shape.ToArray());

                // Build inverse permutation and permute back
                var inversePermutation = new int[rank];
                for (int i = 0; i < rank; i++)
                {
                    inversePermutation[permutation[i]] = i;
                }
                return PermuteImpl(permutedOutput, inversePermutation);
            }

            // Fall back to CPU for any other edge cases
            return base.Softmax(input, axis);
        }
        catch
        {
            return base.Softmax(input, axis);
        }
    }

    /// <summary>
    /// GPU-accelerated Softmax backward operation.
    /// Supports arbitrary axes by treating the tensor as 2D (outerSize, features).
    /// </summary>
    Tensor<T> IEngine.SoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> output, int axis)
    {
        if (!TryGetBackend(out var backend))
            return base.SoftmaxBackward(gradOutput, output, axis);

        int rank = output.Rank;
        if (axis < 0) axis = rank + axis;
        if (axis < 0 || axis >= rank)
            return base.SoftmaxBackward(gradOutput, output, axis);

        try
        {
            // For softmax backward over the last dimension
            if (axis == rank - 1)
            {
                int features = output.Shape[rank - 1];
                int outerSize = output.Length / features;

                using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
                using var outputBuffer = GetOrAllocateBuffer(backend, output.Data);
                using var gradInputBuffer = AllocateOutputBuffer(backend, output.Length);

                backend.SoftmaxBackward(gradOutBuffer.Buffer, outputBuffer.Buffer, gradInputBuffer.Buffer, outerSize, features);
                float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
                return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), output.Shape.ToArray());
            }

            // For softmax backward over a non-last axis, permute to move the axis to the end,
            // apply softmax backward, then permute back
            if (axis < rank - 1)
            {
                // Build permutation: move axis to end
                var permutation = new int[rank];
                int j = 0;
                for (int i = 0; i < rank; i++)
                {
                    if (i != axis) permutation[j++] = i;
                }
                permutation[rank - 1] = axis;

                // Permute inputs
                var permutedGradOutput = PermuteImpl(gradOutput, permutation);
                var permutedOutput = PermuteImpl(output, permutation);

                int features = permutedOutput.Shape[rank - 1];
                int outerSize = permutedOutput.Length / features;

                using var gradOutBuffer = GetOrAllocateBuffer(backend, permutedGradOutput.Data);
                using var outputBuffer = GetOrAllocateBuffer(backend, permutedOutput.Data);
                using var gradInputBuffer = AllocateOutputBuffer(backend, permutedOutput.Length);

                backend.SoftmaxBackward(gradOutBuffer.Buffer, outputBuffer.Buffer, gradInputBuffer.Buffer, outerSize, features);
                float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
                var permutedResult = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), permutedOutput.Shape.ToArray());

                // Build inverse permutation and permute back
                var inversePermutation = new int[rank];
                for (int i = 0; i < rank; i++)
                {
                    inversePermutation[permutation[i]] = i;
                }
                return PermuteImpl(permutedResult, inversePermutation);
            }

            return base.SoftmaxBackward(gradOutput, output, axis);
        }
        catch
        {
            return base.SoftmaxBackward(gradOutput, output, axis);
        }
    }

    /// <summary>
    /// GPU-accelerated Squash activation for capsule networks.
    /// squash(v) = ||v||² / (1 + ||v||²) × v / ||v||
    /// </summary>
    Tensor<T> IEngine.TensorSquash<T>(Tensor<T> tensor, int axis)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorSquash(tensor, axis);

        int rank = tensor.Rank;
        if (axis < 0) axis = rank + axis;
        if (axis < 0 || axis >= rank)
            return base.TensorSquash(tensor, axis);

        try
        {
            // For squash over the last dimension
            if (axis == rank - 1)
            {
                int capsuleDim = tensor.Shape[rank - 1];
                int numCapsules = tensor.Length / capsuleDim;

                using var inputBuffer = GetOrAllocateBuffer(backend, tensor.Data);
                using var outputBuffer = AllocateOutputBuffer(backend, tensor.Length);

                backend.Squash(inputBuffer.Buffer, outputBuffer.Buffer, numCapsules, capsuleDim, 1e-8f);
                float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
                return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), tensor.Shape.ToArray());
            }

            // For squash over a non-last axis, permute to move the axis to the end,
            // apply squash, then permute back
            if (axis < rank - 1)
            {
                // Build permutation: move axis to end
                var permutation = new int[rank];
                int j = 0;
                for (int i = 0; i < rank; i++)
                {
                    if (i != axis) permutation[j++] = i;
                }
                permutation[rank - 1] = axis;

                var permutedInput = PermuteImpl(tensor, permutation);

                int capsuleDim = permutedInput.Shape[rank - 1];
                int numCapsules = permutedInput.Length / capsuleDim;

                using var inputBuffer = GetOrAllocateBuffer(backend, permutedInput.Data);
                using var outputBuffer = AllocateOutputBuffer(backend, permutedInput.Length);

                backend.Squash(inputBuffer.Buffer, outputBuffer.Buffer, numCapsules, capsuleDim, 1e-8f);
                float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
                var permutedResult = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), permutedInput.Shape.ToArray());

                // Build inverse permutation and permute back
                var inversePermutation = new int[rank];
                for (int i = 0; i < rank; i++)
                {
                    inversePermutation[permutation[i]] = i;
                }
                return PermuteImpl(permutedResult, inversePermutation);
            }

            return base.TensorSquash(tensor, axis);
        }
        catch
        {
            return base.TensorSquash(tensor, axis);
        }
    }

    /// <summary>
    /// GPU-accelerated Squash backward operation.
    /// </summary>
    Tensor<T> IEngine.TensorSquashBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> output, int axis)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorSquashBackward(gradOutput, input, output, axis);

        int rank = input.Rank;
        if (axis < 0) axis = rank + axis;
        if (axis < 0 || axis >= rank)
            return base.TensorSquashBackward(gradOutput, input, output, axis);

        try
        {
            // For squash backward over the last dimension
            if (axis == rank - 1)
            {
                int capsuleDim = input.Shape[rank - 1];
                int numCapsules = input.Length / capsuleDim;

                using var gradOutputBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
                using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
                using var gradInputBuffer = AllocateOutputBuffer(backend, input.Length);

                backend.SquashBackward(gradOutputBuffer.Buffer, inputBuffer.Buffer, gradInputBuffer.Buffer, numCapsules, capsuleDim, 1e-8f);
                float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
                return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), input.Shape.ToArray());
            }

            // For squash backward over a non-last axis, permute to move the axis to the end,
            // apply squash backward, then permute back
            if (axis < rank - 1)
            {
                // Build permutation: move axis to end
                var permutation = new int[rank];
                int j = 0;
                for (int i = 0; i < rank; i++)
                {
                    if (i != axis) permutation[j++] = i;
                }
                permutation[rank - 1] = axis;

                var permutedGradOutput = PermuteImpl(gradOutput, permutation);
                var permutedInput = PermuteImpl(input, permutation);

                int capsuleDim = permutedInput.Shape[rank - 1];
                int numCapsules = permutedInput.Length / capsuleDim;

                using var gradOutputBuffer = GetOrAllocateBuffer(backend, permutedGradOutput.Data);
                using var inputBuffer = GetOrAllocateBuffer(backend, permutedInput.Data);
                using var gradInputBuffer = AllocateOutputBuffer(backend, permutedInput.Length);

                backend.SquashBackward(gradOutputBuffer.Buffer, inputBuffer.Buffer, gradInputBuffer.Buffer, numCapsules, capsuleDim, 1e-8f);
                float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
                var permutedResult = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), permutedInput.Shape.ToArray());

                // Build inverse permutation and permute back
                var inversePermutation = new int[rank];
                for (int i = 0; i < rank; i++)
                {
                    inversePermutation[permutation[i]] = i;
                }
                return PermuteImpl(permutedResult, inversePermutation);
            }

            return base.TensorSquashBackward(gradOutput, input, output, axis);
        }
        catch
        {
            return base.TensorSquashBackward(gradOutput, input, output, axis);
        }
    }

    /// <summary>
    /// GPU-resident tensor tiling (repeating) along the batch dimension.
    /// Tiles the input tensor to create a larger output tensor with the specified number of repeats.
    /// Output shape: [repeats * batchSize, ...] where input shape is [batchSize, ...].
    /// </summary>
    public IGpuTensor<T> TileBatchGpu<T>(IGpuTensor<T> input, int repeats)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for TileBatchGpu");

        if (repeats <= 0)
            throw new ArgumentOutOfRangeException(nameof(repeats), "Repeats must be positive");

        int batchSize = input.Shape[0];
        int innerSize = input.ElementCount / batchSize;
        int outputTotalSize = repeats * batchSize * innerSize;

        int[] outputShape = new int[input.Shape.Length];
        outputShape[0] = repeats * batchSize;
        for (int i = 1; i < input.Shape.Length; i++)
            outputShape[i] = input.Shape[i];

        var outputBuffer = backend.AllocateBuffer(outputTotalSize);
        backend.TileBatch(input.Buffer, outputBuffer, repeats * batchSize, innerSize);

        return new GpuTensor<T>(backend, outputBuffer, outputShape, GpuTensorRole.Intermediate, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident tensor tiling (repeating) along a specific axis.
    /// Tiles the input tensor by repeating elements along the specified axis.
    /// </summary>
    public IGpuTensor<T> TileAxisGpu<T>(IGpuTensor<T> input, int axis, int repeats)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for TileAxisGpu");

        if (repeats <= 0)
            throw new ArgumentOutOfRangeException(nameof(repeats), "Repeats must be positive");

        int rank = input.Shape.Length;
        if (axis < 0) axis = rank + axis;
        if (axis < 0 || axis >= rank)
            throw new ArgumentOutOfRangeException(nameof(axis), "Axis out of range");

        int outerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= input.Shape[i];
        int axisSize = input.Shape[axis];
        int innerSize = 1;
        for (int i = axis + 1; i < rank; i++) innerSize *= input.Shape[i];

        int outputTotalSize = outerSize * axisSize * repeats * innerSize;

        int[] outputShape = new int[rank];
        for (int i = 0; i < rank; i++)
            outputShape[i] = i == axis ? input.Shape[i] * repeats : input.Shape[i];

        var outputBuffer = backend.AllocateBuffer(outputTotalSize);
        backend.TileAxis(input.Buffer, outputBuffer, outerSize, axisSize, innerSize, repeats);

        return new GpuTensor<T>(backend, outputBuffer, outputShape, GpuTensorRole.Intermediate, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident global average pooling operation.
    /// Reduces spatial dimensions (all except batch and last) to 1 using mean.
    /// </summary>
    public IGpuTensor<T> GlobalMeanPoolGpu<T>(IGpuTensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for GlobalMeanPoolGpu");

        int rank = input.Shape.Length;

        // Get reduction axes (all except first and last)
        int[] axes = rank switch
        {
            4 => [1, 2],  // [batch, height, width, channels] -> reduce H, W
            3 => [1],     // [batch, seq_len, features] -> reduce seq_len
            2 => [],      // Nothing to reduce
            1 => [0],     // Reduce all
            _ when rank > 4 => Enumerable.Range(1, rank - 2).ToArray(),
            _ => []
        };

        if (axes.Length == 0)
        {
            // No reduction needed - return input with new shape
            return new GpuTensor<T>(backend, input.Buffer, input.Shape, GpuTensorRole.Intermediate, ownsBuffer: false);
        }

        // Calculate output shape and sizes
        int outerSize = 1;
        int reduceSize = 1;
        for (int i = 0; i < axes[0]; i++) outerSize *= input.Shape[i];
        foreach (int axis in axes) reduceSize *= input.Shape[axis];
        int innerSize = 1;
        for (int i = axes[^1] + 1; i < rank; i++) innerSize *= input.Shape[i];

        // For global pooling with innerSize, treat it as multiple reductions
        int totalOuter = outerSize * innerSize;
        int outputSize = totalOuter;

        int[] outputShape;
        if (rank == 4)
        {
            outputShape = [input.Shape[0], 1, 1, input.Shape[3]];
        }
        else if (rank == 3)
        {
            outputShape = [input.Shape[0], input.Shape[2]];
        }
        else
        {
            outputShape = [1];
        }

        var outputBuffer = backend.AllocateBuffer(outputSize);
        backend.MeanAxis(input.Buffer, outputBuffer, totalOuter, reduceSize);

        return new GpuTensor<T>(backend, outputBuffer, outputShape, GpuTensorRole.Intermediate, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident global max pooling operation.
    /// Reduces spatial dimensions (all except batch and last) to 1 using max.
    /// Returns CPU indices for backward pass compatibility.
    /// </summary>
    public IGpuTensor<T> GlobalMaxPoolGpu<T>(IGpuTensor<T> input, out int[] maxIndices)
    {
        var result = GlobalMaxPoolGpuWithGpuIndices(input, out var gpuIndices);

        // Download indices to CPU for backward pass
        if (gpuIndices is not null)
        {
            var indicesFloat = gpuIndices.GetCpuData();
            maxIndices = new int[indicesFloat.Length];
            for (int i = 0; i < indicesFloat.Length; i++)
                maxIndices[i] = (int)indicesFloat[i];
            gpuIndices.Dispose();
        }
        else
        {
            maxIndices = [];
        }

        return result;
    }

    /// <summary>
    /// GPU-resident global max pooling operation with GPU-resident indices.
    /// Keeps both max values and argmax indices on GPU for maximum performance.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="input">Input GPU tensor.</param>
    /// <param name="gpuIndices">GPU tensor containing argmax indices (as floats). Null if no reduction needed.</param>
    /// <returns>GPU tensor containing the max-pooled values.</returns>
    public IGpuTensor<T> GlobalMaxPoolGpuWithGpuIndices<T>(IGpuTensor<T> input, out IGpuTensor<float>? gpuIndices)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for GlobalMaxPoolGpu");

        int rank = input.Shape.Length;

        // Get reduction axes (all except first and last)
        int[] axes = rank switch
        {
            4 => [1, 2],  // [batch, height, width, channels] -> reduce H, W
            3 => [1],     // [batch, seq_len, features] -> reduce seq_len
            2 => [],      // Nothing to reduce
            1 => [0],     // Reduce all
            _ when rank > 4 => Enumerable.Range(1, rank - 2).ToArray(),
            _ => []
        };

        if (axes.Length == 0)
        {
            gpuIndices = null;
            return new GpuTensor<T>(backend, input.Buffer, input.Shape, GpuTensorRole.Intermediate, ownsBuffer: false);
        }

        // Calculate output shape and sizes
        int outerSize = 1;
        int reduceSize = 1;
        for (int i = 0; i < axes[0]; i++) outerSize *= input.Shape[i];
        foreach (int axis in axes) reduceSize *= input.Shape[axis];
        int innerSize = 1;
        for (int i = axes[^1] + 1; i < rank; i++) innerSize *= input.Shape[i];

        int totalOuter = outerSize * innerSize;
        int outputSize = totalOuter;

        int[] outputShape;
        if (rank == 4)
        {
            outputShape = [input.Shape[0], 1, 1, input.Shape[3]];
        }
        else if (rank == 3)
        {
            outputShape = [input.Shape[0], input.Shape[2]];
        }
        else
        {
            outputShape = [1];
        }

        var outputBuffer = backend.AllocateBuffer(outputSize);
        backend.MaxAxis(input.Buffer, outputBuffer, totalOuter, reduceSize);

        // Compute argmax indices on GPU and keep them GPU-resident
        var indicesBuffer = backend.AllocateBuffer(outputSize);
        backend.ArgMax(input.Buffer, indicesBuffer, totalOuter, reduceSize);
        gpuIndices = new GpuTensor<float>(backend, indicesBuffer, outputShape, GpuTensorRole.Intermediate, ownsBuffer: true);

        return new GpuTensor<T>(backend, outputBuffer, outputShape, GpuTensorRole.Intermediate, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident backward pass for global mean pooling.
    /// Broadcasts gradient to all spatial positions and divides by count.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="gradOutput">GPU-resident gradient of shape [batch, 1, 1, channels] or [batch, channels].</param>
    /// <param name="inputShape">Original input shape to broadcast to.</param>
    /// <returns>GPU-resident gradient of same shape as original input.</returns>
    public IGpuTensor<T> GlobalMeanPoolBackwardGpu<T>(IGpuTensor<T> gradOutput, int[] inputShape)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for GlobalMeanPoolBackwardGpu");

        // Parameter validation
        if (inputShape is null || inputShape.Length == 0)
            throw new ArgumentException("Input shape must not be null or empty", nameof(inputShape));

        foreach (int dim in inputShape)
        {
            if (dim <= 0)
                throw new ArgumentException("All input shape dimensions must be positive", nameof(inputShape));
        }

        int rank = inputShape.Length;

        // Get reduction axes (same as forward pass)
        int[] axes = rank switch
        {
            4 => [1, 2],  // [batch, height, width, channels]
            3 => [1],     // [batch, seq_len, features]
            2 => [],      // Nothing to reduce
            1 => [0],     // Reduce all
            _ when rank > 4 => Enumerable.Range(1, rank - 2).ToArray(),
            _ => []
        };

        if (axes.Length == 0)
        {
            // No reduction was done - return gradient as-is
            return new GpuTensor<T>(backend, gradOutput.Buffer, inputShape, GpuTensorRole.Gradient, ownsBuffer: false);
        }

        // Calculate sizes
        int reduceSize = 1;
        foreach (int axis in axes) reduceSize *= inputShape[axis];
        int totalSize = inputShape.Aggregate(1, (a, b) => a * b);
        int outerSize = gradOutput.ElementCount;

        // Allocate output buffer
        var outputBuffer = backend.AllocateBuffer(totalSize);

        // For mean pooling backward: broadcast and scale by 1/reduceSize
        // grad_input = grad_output tiled reduceSize times, then scaled
        float scale = 1.0f / reduceSize;

        // Use TileBatch to repeat gradient values
        // TileBatch(input, output, repeats, innerSize) tiles input[i] to output[i*repeats:(i+1)*repeats]
        backend.TileBatch(gradOutput.Buffer, outputBuffer, reduceSize, 1);

        // Scale the output by 1/reduceSize
        backend.Scale(outputBuffer, outputBuffer, scale, totalSize);

        return new GpuTensor<T>(backend, outputBuffer, inputShape, GpuTensorRole.Gradient, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident backward pass for global max pooling.
    /// Scatters gradient to the positions that had maximum values.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="gradOutput">GPU-resident gradient of shape [batch, 1, 1, channels] or [batch, channels].</param>
    /// <param name="maxIndices">CPU indices of max positions from forward pass.</param>
    /// <param name="inputShape">Original input shape.</param>
    /// <returns>GPU-resident gradient of same shape as original input.</returns>
    public IGpuTensor<T> GlobalMaxPoolBackwardGpu<T>(IGpuTensor<T> gradOutput, int[] maxIndices, int[] inputShape)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for GlobalMaxPoolBackwardGpu");

        // Parameter validation
        if (inputShape is null || inputShape.Length == 0)
            throw new ArgumentException("Input shape must not be null or empty", nameof(inputShape));

        foreach (int dim in inputShape)
        {
            if (dim <= 0)
                throw new ArgumentException("All input shape dimensions must be positive", nameof(inputShape));
        }

        if (maxIndices is null)
            throw new ArgumentNullException(nameof(maxIndices));

        int totalSize = inputShape.Aggregate(1, (a, b) => a * b);

        // Validate indices are within bounds
        foreach (int idx in maxIndices)
        {
            if (idx < 0 || idx >= totalSize)
                throw new ArgumentOutOfRangeException(nameof(maxIndices),
                    $"Index {idx} is out of bounds for input with total size {totalSize}");
        }

        // Allocate and zero-initialize output buffer
        var outputBuffer = backend.AllocateBuffer(totalSize);
        backend.Fill(outputBuffer, 0f, totalSize);

        // Upload indices
        using var indicesBuffer = backend.AllocateIntBuffer(maxIndices);

        // Scatter-add gradient to max positions: destination[indices[i]] += source[i]
        // sourceSize = number of gradients, destSize = total output size
        backend.ScatterAdd(gradOutput.Buffer, indicesBuffer, outputBuffer, maxIndices.Length, totalSize);

        return new GpuTensor<T>(backend, outputBuffer, inputShape, GpuTensorRole.Gradient, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident ArgMax operation. Returns indices of maximum values along an axis.
    /// Indices are returned as floats on GPU (cast to int when downloading).
    /// </summary>
    /// <typeparam name="T">Element type of input.</typeparam>
    /// <param name="input">Input GPU tensor with shape [outerSize, reduceSize].</param>
    /// <param name="axis">Axis along which to find argmax.</param>
    /// <returns>GPU tensor containing argmax indices as floats.</returns>
    public IGpuTensor<float> ArgMaxGpu<T>(IGpuTensor<T> input, int axis = -1)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ArgMaxGpu");

        int rank = input.Shape.Length;
        if (axis < 0) axis = rank + axis;
        if (axis < 0 || axis >= rank)
            throw new ArgumentOutOfRangeException(nameof(axis), $"Axis {axis} is out of range for tensor with {rank} dimensions");

        // Calculate sizes for reduction
        int outerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= input.Shape[i];
        int reduceSize = input.Shape[axis];
        int innerSize = 1;
        for (int i = axis + 1; i < rank; i++) innerSize *= input.Shape[i];

        int totalOuter = outerSize * innerSize;

        // Output shape removes the reduction axis
        var outputShape = new int[rank - 1];
        int outIdx = 0;
        for (int i = 0; i < rank; i++)
        {
            if (i != axis)
                outputShape[outIdx++] = input.Shape[i];
        }
        if (outputShape.Length == 0)
            outputShape = [1];

        var indicesBuffer = backend.AllocateBuffer(totalOuter);
        backend.ArgMax(input.Buffer, indicesBuffer, totalOuter, reduceSize);

        return new GpuTensor<float>(backend, indicesBuffer, outputShape, GpuTensorRole.Intermediate, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-accelerated LayerNorm operation.
    /// </summary>
    Tensor<T> IEngine.LayerNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance)
    {
        if (!TryGetBackend(out var backend))
            return base.LayerNorm(input, gamma, beta, epsilon, out mean, out variance);

        try
        {
            // Determine batch size and normalized size from gamma shape
            int normalizedSize = gamma.Length;
            int batchSize = input.Length / normalizedSize;

            if (batchSize * normalizedSize != input.Length)
                return base.LayerNorm(input, gamma, beta, epsilon, out mean, out variance);

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var gammaBuffer = GetOrCacheWeightBuffer(backend, gamma.Data, PersistentTensorRole.Weights);
            using var betaBuffer = GetOrCacheWeightBuffer(backend, beta.Data, PersistentTensorRole.Biases);
            using var outputBuffer = AllocateOutputBuffer(backend, input.Length);
            using var saveMeanBuffer = AllocateOutputBuffer(backend, batchSize);
            using var saveVarBuffer = AllocateOutputBuffer(backend, batchSize);

            backend.LayerNorm(inputBuffer.Buffer, outputBuffer.Buffer, gammaBuffer.Buffer, betaBuffer.Buffer,
                saveMeanBuffer.Buffer, saveVarBuffer.Buffer, batchSize, normalizedSize, (float)epsilon);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] outputFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            float[] meanFloat = backend.DownloadBuffer(saveMeanBuffer.Buffer);
            float[] varFloat = backend.DownloadBuffer(saveVarBuffer.Buffer);

            mean = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(meanFloat), new[] { batchSize });
            variance = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(varFloat), new[] { batchSize });
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.LayerNorm(input, gamma, beta, epsilon, out mean, out variance);
        }
    }

    /// <summary>
    /// GPU-accelerated LayerNorm backward operation.
    /// </summary>
    Tensor<T> IEngine.LayerNormBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gamma, Tensor<T> mean, Tensor<T> variance, double epsilon, out Tensor<T> gradGamma, out Tensor<T> gradBeta)
    {
        if (!TryGetBackend(out var backend))
            return base.LayerNormBackward(gradOutput, input, gamma, mean, variance, epsilon, out gradGamma, out gradBeta);

        try
        {
            int normalizedSize = gamma.Length;
            int batchSize = input.Length / normalizedSize;

            if (batchSize * normalizedSize != input.Length)
                return base.LayerNormBackward(gradOutput, input, gamma, mean, variance, epsilon, out gradGamma, out gradBeta);

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var gammaBuffer = GetOrCacheWeightBuffer(backend, gamma.Data, PersistentTensorRole.Weights);
            using var saveMeanBuffer = GetOrAllocateBuffer(backend, mean.Data);
            using var saveVarBuffer = GetOrAllocateBuffer(backend, variance.Data);
            using var gradInputBuffer = AllocateOutputBuffer(backend, input.Length);
            using var gradGammaBuffer = AllocateOutputBuffer(backend, normalizedSize);
            using var gradBetaBuffer = AllocateOutputBuffer(backend, normalizedSize);

            backend.LayerNormBackward(gradOutBuffer.Buffer, inputBuffer.Buffer, gammaBuffer.Buffer,
                saveMeanBuffer.Buffer, saveVarBuffer.Buffer, gradInputBuffer.Buffer, gradGammaBuffer.Buffer, gradBetaBuffer.Buffer,
                batchSize, normalizedSize, (float)epsilon);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] gradInputFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            float[] gradGammaFloat = backend.DownloadBuffer(gradGammaBuffer.Buffer);
            float[] gradBetaFloat = backend.DownloadBuffer(gradBetaBuffer.Buffer);

            gradGamma = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradGammaFloat), gamma.Shape.ToArray());
            gradBeta = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradBetaFloat), gamma.Shape.ToArray());
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradInputFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.LayerNormBackward(gradOutput, input, gamma, mean, variance, epsilon, out gradGamma, out gradBeta);
        }
    }

    /// <summary>
    /// GPU-accelerated RMSNorm operation.
    /// </summary>
    Tensor<T> IEngine.RMSNorm<T>(Tensor<T> input, Tensor<T> gamma, double epsilon, out Tensor<T> rms)
    {
        if (!TryGetBackend(out var backend))
            return base.RMSNorm(input, gamma, epsilon, out rms);

        try
        {
            int normalizedSize = gamma.Length;
            int batchSize = input.Length / normalizedSize;

            if (batchSize * normalizedSize != input.Length)
                return base.RMSNorm(input, gamma, epsilon, out rms);

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var gammaBuffer = GetOrCacheWeightBuffer(backend, gamma.Data, PersistentTensorRole.Weights);
            using var outputBuffer = AllocateOutputBuffer(backend, input.Length);
            using var saveRmsBuffer = AllocateOutputBuffer(backend, batchSize);

            backend.RmsNorm(inputBuffer.Buffer, outputBuffer.Buffer, gammaBuffer.Buffer, saveRmsBuffer.Buffer,
                batchSize, normalizedSize, (float)epsilon);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] outputFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            float[] rmsFloat = backend.DownloadBuffer(saveRmsBuffer.Buffer);

            rms = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(rmsFloat), new[] { batchSize });
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.RMSNorm(input, gamma, epsilon, out rms);
        }
    }

    /// <summary>
    /// GPU-accelerated RMSNorm backward operation.
    /// </summary>
    Tensor<T> IEngine.RMSNormBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gamma, Tensor<T> rms, double epsilon, out Tensor<T> gradGamma)
    {
        if (!TryGetBackend(out var backend))
            return base.RMSNormBackward(gradOutput, input, gamma, rms, epsilon, out gradGamma);

        try
        {
            int normalizedSize = gamma.Length;
            int batchSize = input.Length / normalizedSize;

            if (batchSize * normalizedSize != input.Length)
                return base.RMSNormBackward(gradOutput, input, gamma, rms, epsilon, out gradGamma);

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var gammaBuffer = GetOrCacheWeightBuffer(backend, gamma.Data, PersistentTensorRole.Weights);
            using var saveRmsBuffer = GetOrAllocateBuffer(backend, rms.Data);
            using var gradInputBuffer = AllocateOutputBuffer(backend, input.Length);
            using var gradGammaBuffer = AllocateOutputBuffer(backend, normalizedSize);

            backend.RmsNormBackward(gradOutBuffer.Buffer, inputBuffer.Buffer, gammaBuffer.Buffer, saveRmsBuffer.Buffer,
                gradInputBuffer.Buffer, gradGammaBuffer.Buffer, batchSize, normalizedSize, (float)epsilon);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] gradInputFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            float[] gradGammaFloat = backend.DownloadBuffer(gradGammaBuffer.Buffer);

            gradGamma = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradGammaFloat), gamma.Shape.ToArray());
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradInputFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.RMSNormBackward(gradOutput, input, gamma, rms, epsilon, out gradGamma);
        }
    }

    /// <summary>
    /// GPU-accelerated GroupNorm operation.
    /// </summary>
    Tensor<T> IEngine.GroupNorm<T>(Tensor<T> input, int numGroups, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance)
    {
        if (!TryGetBackend(out var backend))
            return base.GroupNorm(input, numGroups, gamma, beta, epsilon, out mean, out variance);

        try
        {
            // Input shape: [batch, channels, spatial...]
            if (input.Rank < 2)
                return base.GroupNorm(input, numGroups, gamma, beta, epsilon, out mean, out variance);

            int batch = input.Shape[0];
            int channels = input.Shape[1];
            int spatialSize = 1;
            for (int i = 2; i < input.Rank; i++)
                spatialSize *= input.Shape[i];

            if (channels % numGroups != 0)
                return base.GroupNorm(input, numGroups, gamma, beta, epsilon, out mean, out variance);

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var gammaBuffer = GetOrCacheWeightBuffer(backend, gamma.Data, PersistentTensorRole.Weights);
            using var betaBuffer = GetOrCacheWeightBuffer(backend, beta.Data, PersistentTensorRole.Biases);
            using var outputBuffer = AllocateOutputBuffer(backend, input.Length);
            using var saveMeanBuffer = AllocateOutputBuffer(backend, batch * numGroups);
            using var saveVarBuffer = AllocateOutputBuffer(backend, batch * numGroups);

            backend.GroupNorm(inputBuffer.Buffer, outputBuffer.Buffer, gammaBuffer.Buffer, betaBuffer.Buffer,
                saveMeanBuffer.Buffer, saveVarBuffer.Buffer, batch, numGroups, channels, spatialSize, (float)epsilon);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] outputFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            float[] meanFloat = backend.DownloadBuffer(saveMeanBuffer.Buffer);
            float[] varFloat = backend.DownloadBuffer(saveVarBuffer.Buffer);

            mean = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(meanFloat), new[] { batch, numGroups });
            variance = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(varFloat), new[] { batch, numGroups });
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.GroupNorm(input, numGroups, gamma, beta, epsilon, out mean, out variance);
        }
    }

    /// <summary>
    /// GPU-accelerated InstanceNorm operation.
    /// </summary>
    Tensor<T> IEngine.InstanceNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance)
    {
        if (!TryGetBackend(out var backend))
            return base.InstanceNorm(input, gamma, beta, epsilon, out mean, out variance);

        try
        {
            // Input shape: [batch, channels, spatial...]
            if (input.Rank < 2)
                return base.InstanceNorm(input, gamma, beta, epsilon, out mean, out variance);

            int batch = input.Shape[0];
            int channels = input.Shape[1];
            int spatialSize = 1;
            for (int i = 2; i < input.Rank; i++)
                spatialSize *= input.Shape[i];

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var gammaBuffer = GetOrCacheWeightBuffer(backend, gamma.Data, PersistentTensorRole.Weights);
            using var betaBuffer = GetOrCacheWeightBuffer(backend, beta.Data, PersistentTensorRole.Biases);
            using var outputBuffer = AllocateOutputBuffer(backend, input.Length);
            using var saveMeanBuffer = AllocateOutputBuffer(backend, batch * channels);
            using var saveVarBuffer = AllocateOutputBuffer(backend, batch * channels);

            backend.InstanceNorm(inputBuffer.Buffer, outputBuffer.Buffer, gammaBuffer.Buffer, betaBuffer.Buffer,
                saveMeanBuffer.Buffer, saveVarBuffer.Buffer, batch, channels, spatialSize, (float)epsilon);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] outputFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            float[] meanFloat = backend.DownloadBuffer(saveMeanBuffer.Buffer);
            float[] varFloat = backend.DownloadBuffer(saveVarBuffer.Buffer);

            mean = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(meanFloat), new[] { batch, channels });
            variance = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(varFloat), new[] { batch, channels });
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.InstanceNorm(input, gamma, beta, epsilon, out mean, out variance);
        }
    }

    /// <summary>
    /// GPU-resident batch normalization. Input and output remain on GPU, avoiding CPU round-trips.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor with shape [batch, features] or [batch, channels, H, W].</param>
    /// <param name="gamma">Scale parameters (from CPU, cached on GPU).</param>
    /// <param name="beta">Shift parameters (from CPU, cached on GPU).</param>
    /// <param name="runningMean">Running mean for inference (from CPU, will be updated during training).</param>
    /// <param name="runningVar">Running variance for inference (from CPU, will be updated during training).</param>
    /// <param name="epsilon">Numerical stability constant.</param>
    /// <param name="momentum">Momentum for running statistics update.</param>
    /// <param name="training">Whether in training mode.</param>
    /// <returns>GPU-resident output tensor with same shape as input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when GPU backend is not available.</exception>
    /// <remarks>
    /// <para>
    /// This method performs batch normalization entirely on GPU, returning a GPU-resident tensor.
    /// The running statistics are updated on GPU during training mode and then downloaded back
    /// to update the CPU-side tensors.
    /// </para>
    /// <para>
    /// For 4D input [batch, channels, H, W], spatialSize = H * W. For 2D input [batch, features], spatialSize = 1.
    /// </para>
    /// </remarks>
    public (IGpuTensor<T> Output, Tensor<T>? SaveMean, Tensor<T>? SaveVar) FusedBatchNormGpu<T>(
        IGpuTensor<T> input,
        Tensor<T> gamma,
        Tensor<T> beta,
        ref Tensor<T> runningMean,
        ref Tensor<T> runningVar,
        double epsilon,
        double momentum,
        bool training)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for FusedBatchNormGpu");

        int[] shape = input.Shape;
        int batch = shape[0];
        int channels = shape.Length > 1 ? shape[1] : shape[0];
        int spatialSize = 1;
        for (int i = 2; i < shape.Length; i++)
        {
            spatialSize *= shape[i];
        }

        // Upload parameters to GPU (these are typically cached)
        using var gammaBuffer = GetOrCacheWeightBuffer(backend, gamma.Data, PersistentTensorRole.Weights);
        using var betaBuffer = GetOrCacheWeightBuffer(backend, beta.Data, PersistentTensorRole.Biases);
        using var runningMeanBuffer = GetOrCacheWeightBuffer(backend, runningMean.Data, PersistentTensorRole.NormalizationParams);
        using var runningVarBuffer = GetOrCacheWeightBuffer(backend, runningVar.Data, PersistentTensorRole.NormalizationParams);

        // Allocate output and save buffers
        int outputSize = input.ElementCount;
        var outputBuffer = backend.AllocateBuffer(outputSize);
        using var saveMeanBuffer = AllocateOutputBuffer(backend, channels);
        using var saveVarBuffer = AllocateOutputBuffer(backend, channels);

        // Execute batch norm on GPU
        backend.BatchNorm(
            input.Buffer, outputBuffer, gammaBuffer.Buffer, betaBuffer.Buffer,
            runningMeanBuffer.Buffer, runningVarBuffer.Buffer,
            saveMeanBuffer.Buffer, saveVarBuffer.Buffer,
            batch, channels, spatialSize,
            (float)epsilon, (float)momentum, training);

        // If training, download updated running statistics back to CPU
        Tensor<T>? saveMean = null;
        Tensor<T>? saveVar = null;
        if (training)
        {
            float[] updatedRunningMean = backend.DownloadBuffer(runningMeanBuffer.Buffer);
            float[] updatedRunningVar = backend.DownloadBuffer(runningVarBuffer.Buffer);
            runningMean = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(updatedRunningMean), new[] { channels });
            runningVar = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(updatedRunningVar), new[] { channels });

            // Also return saveMean/saveVar for backward pass
            float[] saveMeanFloat = backend.DownloadBuffer(saveMeanBuffer.Buffer);
            float[] saveVarFloat = backend.DownloadBuffer(saveVarBuffer.Buffer);
            saveMean = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(saveMeanFloat), new[] { channels });
            saveVar = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(saveVarFloat), new[] { channels });
        }

        // Return GPU-resident output tensor
        var outputTensor = new GpuTensor<T>(backend, outputBuffer, shape, GpuTensorRole.Activation, ownsBuffer: true);
        return (outputTensor, saveMean, saveVar);
    }

    /// <summary>
    /// GPU-resident Layer Normalization forward pass.
    /// Normalizes input across the normalized (feature) dimension for each sample independently.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor [batch, normalizedSize].</param>
    /// <param name="gamma">Scale parameters [normalizedSize].</param>
    /// <param name="beta">Shift parameters [normalizedSize].</param>
    /// <param name="epsilon">Small constant for numerical stability.</param>
    /// <returns>
    /// A tuple containing:
    /// - Output: GPU-resident normalized tensor
    /// - SaveMean: Mean values per sample (for backward pass, downloaded to CPU)
    /// - SaveInvVar: Inverse variance per sample (for backward pass, downloaded to CPU)
    /// </returns>
    public (IGpuTensor<T> Output, Tensor<T> SaveMean, Tensor<T> SaveInvVar) LayerNormGpu<T>(
        IGpuTensor<T> input,
        Tensor<T> gamma,
        Tensor<T> beta,
        double epsilon)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for LayerNormGpu");

        int[] shape = input.Shape;
        int batchSize = shape[0];
        int normalizedSize = shape.Length > 1 ? shape[1] : shape[0];

        // For higher-rank tensors, flatten the normalized dimensions
        for (int i = 2; i < shape.Length; i++)
        {
            normalizedSize *= shape[i];
        }

        // Upload gamma and beta to GPU
        using var gammaBuffer = GetOrCacheWeightBuffer(backend, gamma.Data, PersistentTensorRole.Weights);
        using var betaBuffer = GetOrCacheWeightBuffer(backend, beta.Data, PersistentTensorRole.Biases);

        // Allocate output and save buffers
        int outputSize = input.ElementCount;
        var outputBuffer = backend.AllocateBuffer(outputSize);
        using var saveMeanBuffer = AllocateOutputBuffer(backend, batchSize);
        using var saveInvVarBuffer = AllocateOutputBuffer(backend, batchSize);

        // Execute LayerNorm on GPU
        backend.LayerNorm(
            input.Buffer, outputBuffer, gammaBuffer.Buffer, betaBuffer.Buffer,
            saveMeanBuffer.Buffer, saveInvVarBuffer.Buffer,
            batchSize, normalizedSize, (float)epsilon);

        // Download save buffers for backward pass (these are per-sample, so relatively small)
        float[] saveMeanFloat = backend.DownloadBuffer(saveMeanBuffer.Buffer);
        float[] saveInvVarFloat = backend.DownloadBuffer(saveInvVarBuffer.Buffer);
        var saveMean = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(saveMeanFloat), new[] { batchSize });
        var saveInvVar = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(saveInvVarFloat), new[] { batchSize });

        // Return GPU-resident output tensor
        var outputTensor = new GpuTensor<T>(backend, outputBuffer, shape, GpuTensorRole.Activation, ownsBuffer: true);
        return (outputTensor, saveMean, saveInvVar);
    }

    /// <summary>
    /// GPU-resident layer normalization backward pass.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident gradient of loss w.r.t. output.</param>
    /// <param name="input">GPU-resident input from forward pass.</param>
    /// <param name="gamma">Scale parameters.</param>
    /// <param name="saveMean">Saved mean from forward pass.</param>
    /// <param name="saveInvVar">Saved inverse variance from forward pass.</param>
    /// <param name="epsilon">Epsilon for numerical stability.</param>
    /// <returns>Tuple of (gradInput, gradGamma, gradBeta) GPU-resident tensors.</returns>
    public (IGpuTensor<T> GradInput, Tensor<T> GradGamma, Tensor<T> GradBeta) LayerNormBackwardGpu<T>(
        IGpuTensor<T> gradOutput,
        IGpuTensor<T> input,
        Tensor<T> gamma,
        Tensor<T> saveMean,
        Tensor<T> saveInvVar,
        double epsilon)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for LayerNormBackwardGpu");

        int[] shape = gradOutput.Shape;
        int batchSize = shape[0];
        int normalizedSize = shape.Length > 1 ? shape[1] : shape[0];

        // For higher-rank tensors, flatten the normalized dimensions
        for (int i = 2; i < shape.Length; i++)
        {
            normalizedSize *= shape[i];
        }

        // Validate parameter shapes to prevent out-of-bounds kernel access
        if (gamma.Length != normalizedSize)
            throw new ArgumentException($"gamma.Length ({gamma.Length}) must match normalizedSize ({normalizedSize}).", nameof(gamma));
        if (saveMean.Length != batchSize)
            throw new ArgumentException($"saveMean.Length ({saveMean.Length}) must match batchSize ({batchSize}).", nameof(saveMean));
        if (saveInvVar.Length != batchSize)
            throw new ArgumentException($"saveInvVar.Length ({saveInvVar.Length}) must match batchSize ({batchSize}).", nameof(saveInvVar));

        // Upload gamma, saveMean, saveInvVar to GPU
        using var gammaBuffer = GetOrCacheWeightBuffer(backend, gamma.Data, PersistentTensorRole.Weights);
        float[] saveMeanFloat = DirectGpuEngine.ToFloatArray(saveMean.Data);
        float[] saveInvVarFloat = DirectGpuEngine.ToFloatArray(saveInvVar.Data);

        // Allocate temporary and output buffers with exception-safe disposal
        IGpuBuffer? saveMeanBuffer = null;
        IGpuBuffer? saveInvVarBuffer = null;
        IGpuBuffer? gradInputBuffer = null;
        IGpuBuffer? gradGammaBuffer = null;
        IGpuBuffer? gradBetaBuffer = null;

        try
        {
            saveMeanBuffer = backend.AllocateBuffer(saveMeanFloat);
            saveInvVarBuffer = backend.AllocateBuffer(saveInvVarFloat);

            // Allocate output buffers
            int inputSize = gradOutput.ElementCount;
            gradInputBuffer = backend.AllocateBuffer(inputSize);
            gradGammaBuffer = backend.AllocateBuffer(normalizedSize);
            gradBetaBuffer = backend.AllocateBuffer(normalizedSize);

            // Execute LayerNormBackward on GPU
            backend.LayerNormBackward(
                gradOutput.Buffer, input.Buffer, gammaBuffer.Buffer,
                saveMeanBuffer, saveInvVarBuffer,
                gradInputBuffer, gradGammaBuffer, gradBetaBuffer,
                batchSize, normalizedSize, (float)epsilon);

            // Download gradGamma and gradBeta (these are small, same size as normalizedSize)
            float[] gradGammaFloat = backend.DownloadBuffer(gradGammaBuffer);
            float[] gradBetaFloat = backend.DownloadBuffer(gradBetaBuffer);

            // Dispose temporary buffers (not gradInputBuffer - it becomes part of returned tensor)
            saveMeanBuffer.Dispose();
            saveMeanBuffer = null;
            saveInvVarBuffer.Dispose();
            saveInvVarBuffer = null;
            gradGammaBuffer.Dispose();
            gradGammaBuffer = null;
            gradBetaBuffer.Dispose();
            gradBetaBuffer = null;

            var gradGamma = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradGammaFloat), new[] { normalizedSize });
            var gradBeta = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradBetaFloat), new[] { normalizedSize });

            // Return GPU-resident gradInput tensor
            var gradInputTensor = new GpuTensor<T>(backend, gradInputBuffer, shape, GpuTensorRole.Gradient, ownsBuffer: true);
            gradInputBuffer = null; // Ownership transferred to tensor
            return (gradInputTensor, gradGamma, gradBeta);
        }
        finally
        {
            // Dispose any buffers that weren't successfully transferred or already disposed
            saveMeanBuffer?.Dispose();
            saveInvVarBuffer?.Dispose();
            gradInputBuffer?.Dispose();
            gradGammaBuffer?.Dispose();
            gradBetaBuffer?.Dispose();
        }
    }

    /// <summary>
    /// GPU-resident element-wise addition: C = A + B
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="a">First GPU-resident input tensor.</param>
    /// <param name="b">Second GPU-resident input tensor.</param>
    /// <returns>A GPU-resident output tensor with the element-wise sum.</returns>
    public IGpuTensor<T> AddGpu<T>(IGpuTensor<T> a, IGpuTensor<T> b)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for AddGpu");

        int size = a.ElementCount;
        var outputBuffer = backend.AllocateBuffer(size);

        backend.Add(a.Buffer, b.Buffer, outputBuffer, size);

        return new GpuTensor<T>(backend, outputBuffer, a.Shape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident element-wise multiplication: C = A * B
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="a">First GPU-resident input tensor.</param>
    /// <param name="b">Second GPU-resident input tensor.</param>
    /// <returns>A GPU-resident output tensor with the element-wise product.</returns>
    public IGpuTensor<T> MultiplyGpu<T>(IGpuTensor<T> a, IGpuTensor<T> b)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for MultiplyGpu");

        int size = a.ElementCount;
        var outputBuffer = backend.AllocateBuffer(size);

        backend.Multiply(a.Buffer, b.Buffer, outputBuffer, size);

        return new GpuTensor<T>(backend, outputBuffer, a.Shape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident scalar multiplication: B = A * scalar
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor.</param>
    /// <param name="scalar">Scalar value to multiply by.</param>
    /// <returns>A GPU-resident output tensor with the scaled values.</returns>
    public IGpuTensor<T> ScaleGpu<T>(IGpuTensor<T> input, float scalar)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ScaleGpu");

        int size = input.ElementCount;
        var outputBuffer = backend.AllocateBuffer(size);

        backend.Scale(input.Buffer, outputBuffer, scalar, size);

        return new GpuTensor<T>(backend, outputBuffer, input.Shape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident softmax operation along the last axis.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor of shape [batch, features].</param>
    /// <returns>A GPU-resident output tensor with softmax applied.</returns>
    public IGpuTensor<T> SoftmaxGpu<T>(IGpuTensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SoftmaxGpu");

        // Assuming 2D input [batch, features]
        int batchSize = input.Shape[0];
        int features = input.Shape.Length > 1 ? input.Shape[1] : input.Shape[0];

        var outputBuffer = backend.AllocateBuffer(input.ElementCount);

        backend.Softmax(input.Buffer, outputBuffer, batchSize, features);

        return new GpuTensor<T>(backend, outputBuffer, input.Shape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident Top-K selection along the last axis.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor of shape [batch, features].</param>
    /// <param name="k">Number of top elements to select.</param>
    /// <param name="indices">Output GPU buffer containing the indices of top-k elements.</param>
    /// <param name="sorted">Whether to return sorted results (default true).</param>
    /// <returns>A GPU-resident output tensor with the top-k values.</returns>
    public IGpuTensor<T> TopKGpu<T>(IGpuTensor<T> input, int k, out IGpuTensor<int> indices, bool sorted = true)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for TopKGpu");

        // Assuming 2D input [batch, features]
        int outerSize = input.Shape[0];
        int reduceSize = input.Shape.Length > 1 ? input.Shape[1] : input.Shape[0];

        // Allocate output buffers
        var valuesBuffer = backend.AllocateBuffer(outerSize * k);
        var indicesBuffer = backend.AllocateBuffer(outerSize * k);

        backend.TopK(input.Buffer, valuesBuffer, indicesBuffer, outerSize, reduceSize, k, sorted);

        // Create output shape [batch, k]
        int[] outputShape = input.Shape.Length > 1 ? [outerSize, k] : [k];

        indices = new GpuTensor<int>(backend, indicesBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);
        return new GpuTensor<T>(backend, valuesBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident broadcast multiply: C[i,j] = A[i,j] * B[i,0]
    /// Broadcasts a column vector across the last dimension.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor [batch, features].</param>
    /// <param name="weights">GPU-resident weight tensor [batch, 1] to broadcast.</param>
    /// <returns>A GPU-resident output tensor with broadcast multiplication.</returns>
    public IGpuTensor<T> BroadcastMultiplyColumnGpu<T>(IGpuTensor<T> input, IGpuTensor<T> weights)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for BroadcastMultiplyColumnGpu");

        int outerSize = input.Shape[0];
        int innerSize = input.ElementCount / outerSize;

        var outputBuffer = backend.AllocateBuffer(input.ElementCount);

        backend.BroadcastMultiplyFirstAxis(input.Buffer, weights.Buffer, outputBuffer, outerSize, innerSize);

        return new GpuTensor<T>(backend, outputBuffer, input.Shape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident slice operation to extract a column from a 2D tensor.
    /// Uses gather with computed indices to extract strided elements.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor [batch, features].</param>
    /// <param name="columnIndex">Column index to extract.</param>
    /// <returns>A GPU-resident output tensor [batch, 1].</returns>
    public IGpuTensor<T> SliceColumnGpu<T>(IGpuTensor<T> input, int columnIndex)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SliceColumnGpu");

        int batchSize = input.Shape[0];
        int features = input.Shape.Length > 1 ? input.Shape[1] : 1;

        if (columnIndex < 0 || columnIndex >= features)
            throw new ArgumentOutOfRangeException(nameof(columnIndex));

        // Create indices for gathering: [columnIndex, features + columnIndex, 2*features + columnIndex, ...]
        int[] indices = new int[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            indices[i] = i * features + columnIndex;
        }

        var indicesBuffer = backend.AllocateIntBuffer(indices);
        var outputBuffer = backend.AllocateBuffer(batchSize);

        // Gather uses (source, indices, output, numIndices, featureSize)
        // With featureSize=1, it gathers individual elements at the specified indices
        backend.Gather(input.Buffer, indicesBuffer, outputBuffer, batchSize, 1);

        indicesBuffer.Dispose();

        return new GpuTensor<T>(backend, outputBuffer, [batchSize, 1], GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident slice operation along a specified axis.
    /// Extracts a contiguous slice from the input tensor.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor.</param>
    /// <param name="axis">The axis to slice along.</param>
    /// <param name="start">Starting index (inclusive).</param>
    /// <param name="end">Ending index (exclusive).</param>
    /// <returns>A GPU-resident sliced tensor.</returns>
    public IGpuTensor<T> SliceGpu<T>(IGpuTensor<T> input, int axis, int start, int end)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SliceGpu");

        int rank = input.Shape.Length;
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank)
            throw new ArgumentOutOfRangeException(nameof(axis));

        int sliceSize = end - start;
        if (sliceSize <= 0)
            throw new ArgumentException("End must be greater than start");

        // Calculate output shape
        int[] outputShape = new int[rank];
        Array.Copy(input.Shape, outputShape, rank);
        outputShape[axis] = sliceSize;

        int totalOutputSize = outputShape.Aggregate(1, (a, b) => a * b);
        var outputBuffer = backend.AllocateBuffer(totalOutputSize);

        // Use general gather approach for all axes
        // Calculate the stride pattern for the axis
        int beforeAxisSize = 1;
        for (int i = 0; i < axis; i++)
            beforeAxisSize *= input.Shape[i];

        int afterAxisSize = 1;
        for (int i = axis + 1; i < rank; i++)
            afterAxisSize *= input.Shape[i];

        int srcAxisSize = input.Shape[axis];

        // Build indices for gathering
        var indices = new int[totalOutputSize];
        int idx = 0;
        for (int b = 0; b < beforeAxisSize; b++)
        {
            for (int a = start; a < end; a++)
            {
                for (int s = 0; s < afterAxisSize; s++)
                {
                    indices[idx++] = b * srcAxisSize * afterAxisSize + a * afterAxisSize + s;
                }
            }
        }

        using var indicesBuffer = backend.AllocateIntBuffer(indices);
        backend.Gather(input.Buffer, indicesBuffer, outputBuffer, totalOutputSize, 1);

        return new GpuTensor<T>(backend, outputBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-accelerated power iteration for computing spectral norm (largest singular value).
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="weights">GPU-resident weight matrix [rows, cols].</param>
    /// <param name="u">Left singular vector [rows] - updated in-place.</param>
    /// <param name="v">Right singular vector [cols] - updated in-place.</param>
    /// <param name="numIterations">Number of power iterations.</param>
    /// <param name="epsilon">Small constant for numerical stability.</param>
    /// <returns>The estimated spectral norm (largest singular value).</returns>
    public float PowerIterationGpu<T>(
        IGpuTensor<T> weights,
        ref IGpuTensor<T> u,
        ref IGpuTensor<T> v,
        int numIterations,
        float epsilon = 1e-12f)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for PowerIterationGpu");

        int rows = weights.Shape[0];
        int cols = weights.Shape[1];

        // Allocate transpose buffer
        var wTransposeBuffer = backend.AllocateBuffer(rows * cols);

        // Compute W^T once
        backend.Transpose(weights.Buffer, wTransposeBuffer, rows, cols);

        for (int iter = 0; iter < numIterations; iter++)
        {
            // Step 1: v_new = W^T @ u
            // W^T is [cols, rows], u is [rows, 1], result is [cols, 1]
            var vNewBuffer = backend.AllocateBuffer(cols);
            backend.Gemm(wTransposeBuffer, u.Buffer, vNewBuffer, cols, 1, rows, 1.0f, 0.0f);

            // Normalize v_new
            float vNorm = backend.L2Norm(vNewBuffer, cols);
            float vNormSafe = Math.Max(vNorm, epsilon);
            backend.Scale(vNewBuffer, vNewBuffer, 1.0f / vNormSafe, cols);

            // Update v - the old buffer will be cleaned up by GC/finalizer
            v = new GpuTensor<T>(backend, vNewBuffer, [cols], GpuTensorRole.Activation, ownsBuffer: true);

            // Step 2: u_new = W @ v
            // W is [rows, cols], v is [cols, 1], result is [rows, 1]
            var uNewBuffer = backend.AllocateBuffer(rows);
            backend.Gemm(weights.Buffer, v.Buffer, uNewBuffer, rows, 1, cols, 1.0f, 0.0f);

            // Normalize u_new
            float uNorm = backend.L2Norm(uNewBuffer, rows);
            float uNormSafe = Math.Max(uNorm, epsilon);
            backend.Scale(uNewBuffer, uNewBuffer, 1.0f / uNormSafe, rows);

            // Update u - the old buffer will be cleaned up by GC/finalizer
            u = new GpuTensor<T>(backend, uNewBuffer, [rows], GpuTensorRole.Activation, ownsBuffer: true);
        }

        // Clean up transpose buffer
        wTransposeBuffer.Dispose();

        // Compute spectral norm: sigma = u^T @ W @ v
        // First compute Wv: W is [rows, cols], v is [cols, 1], Wv is [rows, 1]
        var wvBuffer = backend.AllocateBuffer(rows);
        backend.Gemm(weights.Buffer, v.Buffer, wvBuffer, rows, 1, cols, 1.0f, 0.0f);

        // Then compute u^T @ Wv (dot product)
        // Element-wise multiply u and Wv, then sum
        var productBuffer = backend.AllocateBuffer(rows);
        backend.Multiply(u.Buffer, wvBuffer, productBuffer, rows);

        // Sum reduction to get scalar
        var sumBuffer = backend.AllocateBuffer(1);
        backend.SumAxis(productBuffer, sumBuffer, 1, rows);

        // Download the scalar result
        float[] sumResult = backend.DownloadBuffer(sumBuffer);
        float spectralNorm = sumResult[0];

        // Clean up temporary buffers
        wvBuffer.Dispose();
        productBuffer.Dispose();
        sumBuffer.Dispose();

        return Math.Max(spectralNorm, epsilon);
    }

    /// <summary>
    /// GPU-resident scalar division: B = A / scalar
    /// </summary>
    public IGpuTensor<T> DivideScalarGpu<T>(IGpuTensor<T> input, float scalar)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for DivideScalarGpu");

        int size = input.ElementCount;
        var outputBuffer = backend.AllocateBuffer(size);

        // Use Scale with 1/scalar
        float invScalar = 1.0f / scalar;
        backend.Scale(input.Buffer, outputBuffer, invScalar, size);

        return new GpuTensor<T>(backend, outputBuffer, input.Shape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident affine grid generation for spatial transformers.
    /// Given affine transformation matrices, generates a sampling grid.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="theta">GPU-resident affine transformation matrices [batch, 2, 3] flattened to [batch * 6].</param>
    /// <param name="batch">Batch size.</param>
    /// <param name="outputHeight">Height of the output grid.</param>
    /// <param name="outputWidth">Width of the output grid.</param>
    /// <returns>A GPU-resident output grid [batch, outputHeight, outputWidth, 2].</returns>
    public IGpuTensor<T> AffineGridGpu<T>(IGpuTensor<T> theta, int batch, int outputHeight, int outputWidth)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for AffineGridGpu");

        // Output shape: [batch, outputHeight, outputWidth, 2]
        int gridSize = batch * outputHeight * outputWidth * 2;
        var gridBuffer = backend.AllocateBuffer(gridSize);

        backend.AffineGrid(theta.Buffer, gridBuffer, batch, outputHeight, outputWidth);

        return new GpuTensor<T>(backend, gridBuffer, [batch, outputHeight, outputWidth, 2], GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident grid sampling with bilinear interpolation for spatial transformers.
    /// Samples from input using a sampling grid.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor [batch, channels, inHeight, inWidth].</param>
    /// <param name="grid">GPU-resident sampling grid [batch, outHeight, outWidth, 2].</param>
    /// <param name="paddingMode">Padding mode: 0=zeros, 1=border, 2=reflection.</param>
    /// <param name="alignCorners">If true, [-1, 1] maps to corner pixels.</param>
    /// <returns>A GPU-resident output tensor [batch, channels, outHeight, outWidth].</returns>
    public IGpuTensor<T> GridSampleGpu<T>(IGpuTensor<T> input, IGpuTensor<T> grid, int paddingMode = 0, bool alignCorners = false)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for GridSampleGpu");

        // Input: [batch, channels, inHeight, inWidth]
        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        // Grid: [batch, outHeight, outWidth, 2]
        int outHeight = grid.Shape[1];
        int outWidth = grid.Shape[2];

        // Output shape: [batch, channels, outHeight, outWidth]
        int outputSize = batch * channels * outHeight * outWidth;
        var outputBuffer = backend.AllocateBuffer(outputSize);

        backend.GridSample(input.Buffer, grid.Buffer, outputBuffer,
            batch, channels, inHeight, inWidth, outHeight, outWidth,
            paddingMode, alignCorners);

        return new GpuTensor<T>(backend, outputBuffer, [batch, channels, outHeight, outWidth], GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident backward pass for grid sampling.
    /// Computes gradients for both input and grid.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident gradient from upstream [batch, channels, outHeight, outWidth].</param>
    /// <param name="input">GPU-resident original input [batch, channels, inHeight, inWidth].</param>
    /// <param name="grid">GPU-resident sampling grid [batch, outHeight, outWidth, 2].</param>
    /// <param name="gradInput">Output: GPU-resident gradient w.r.t. input.</param>
    /// <param name="gradGrid">Output: GPU-resident gradient w.r.t. grid.</param>
    /// <param name="paddingMode">Padding mode: 0=zeros, 1=border, 2=reflection.</param>
    /// <param name="alignCorners">If true, [-1, 1] maps to corner pixels.</param>
    public void GridSampleBackwardGpu<T>(
        IGpuTensor<T> gradOutput,
        IGpuTensor<T> input,
        IGpuTensor<T> grid,
        out IGpuTensor<T> gradInput,
        out IGpuTensor<T> gradGrid,
        int paddingMode = 0,
        bool alignCorners = false)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for GridSampleBackwardGpu");

        // Input: [batch, channels, inHeight, inWidth]
        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        // Grid: [batch, outHeight, outWidth, 2]
        int outHeight = grid.Shape[1];
        int outWidth = grid.Shape[2];

        // Allocate gradient buffers
        var gradInputBuffer = backend.AllocateBuffer(input.ElementCount);
        var gradGridBuffer = backend.AllocateBuffer(grid.ElementCount);

        // Initialize gradInput to zero
        backend.Fill(gradInputBuffer, 0f, input.ElementCount);

        backend.GridSampleBackward(gradOutput.Buffer, input.Buffer, grid.Buffer,
            gradInputBuffer, gradGridBuffer,
            batch, channels, inHeight, inWidth, outHeight, outWidth,
            paddingMode, alignCorners);

        gradInput = new GpuTensor<T>(backend, gradInputBuffer, input.Shape, GpuTensorRole.Gradient, ownsBuffer: true);
        gradGrid = new GpuTensor<T>(backend, gradGridBuffer, grid.Shape, GpuTensorRole.Gradient, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident ReLU activation: y = max(0, x)
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor.</param>
    /// <returns>A GPU-resident output tensor with ReLU applied.</returns>
    public IGpuTensor<T> ReluGpu<T>(IGpuTensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ReluGpu");

        int size = input.ElementCount;
        var outputBuffer = backend.AllocateBuffer(size);

        backend.Relu(input.Buffer, outputBuffer, size);

        return new GpuTensor<T>(backend, outputBuffer, input.Shape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident Tanh activation: y = tanh(x)
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor.</param>
    /// <returns>A GPU-resident output tensor with Tanh applied.</returns>
    public IGpuTensor<T> TanhGpu<T>(IGpuTensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for TanhGpu");

        int size = input.ElementCount;
        var outputBuffer = backend.AllocateBuffer(size);

        backend.Tanh(input.Buffer, outputBuffer, size);

        return new GpuTensor<T>(backend, outputBuffer, input.Shape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident sum reduction along a specified axis.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor.</param>
    /// <param name="axis">Axis to reduce (0 for sum over rows, 1 for sum over columns).</param>
    /// <returns>A GPU-resident output tensor with reduced dimensions.</returns>
    public IGpuTensor<T> SumAxisGpu<T>(IGpuTensor<T> input, int axis)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SumAxisGpu");

        // Validate axis for 2D tensors - only axis 0 and 1 are supported
        if (axis < 0 || axis > 1)
            throw new ArgumentOutOfRangeException(nameof(axis), axis, "SumAxisGpu only supports axis 0 (sum over rows) or axis 1 (sum over columns) for 2D tensors.");

        if (input.Shape.Length < 1)
            throw new ArgumentException("Input tensor must have at least one dimension.", nameof(input));

        int outerSize = input.Shape[0];
        int innerSize = input.Shape.Length > 1 ? input.Shape[1] : 1;

        int outputSize;
        int[] outputShape;

        if (axis == 0)
        {
            // Sum over rows -> output shape [1, innerSize]
            outputSize = innerSize;
            outputShape = [1, innerSize];
        }
        else // axis == 1 (validated above)
        {
            // Sum over columns -> output shape [outerSize, 1]
            outputSize = outerSize;
            outputShape = [outerSize, 1];
        }

        var outputBuffer = backend.AllocateBuffer(outputSize);
        backend.SumAxis(input.Buffer, outputBuffer, outerSize, innerSize);

        return new GpuTensor<T>(backend, outputBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident gather operation: gathers feature vectors from source at specified indices.
    /// Each index selects a feature vector of size featureSize from the source.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="source">GPU-resident source tensor [vocabSize, featureSize] or flat.</param>
    /// <param name="indices">GPU-resident indices buffer containing indices to gather.</param>
    /// <param name="numIndices">Number of indices to gather.</param>
    /// <param name="featureSize">Size of each feature vector.</param>
    /// <returns>A GPU-resident output tensor [numIndices, featureSize] with gathered values.</returns>
    public IGpuTensor<T> GatherGpu<T>(IGpuTensor<T> source, IGpuBuffer indices, int numIndices, int featureSize)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for GatherGpu");

        var outputBuffer = backend.AllocateBuffer(numIndices * featureSize);

        backend.Gather(source.Buffer, indices, outputBuffer, numIndices, featureSize);

        return new GpuTensor<T>(backend, outputBuffer, [numIndices, featureSize], GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident scatter-add operation: accumulates source values into destination at specified indices.
    /// destination[indices[i]] += source[i]
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="source">GPU-resident source values tensor.</param>
    /// <param name="indices">GPU-resident indices buffer.</param>
    /// <param name="destSize">Size of the destination buffer.</param>
    /// <returns>A GPU-resident output tensor with scattered values.</returns>
    public IGpuTensor<T> ScatterAddGpu<T>(IGpuTensor<T> source, IGpuBuffer indices, int destSize)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ScatterAddGpu");

        int sourceSize = source.ElementCount;

        var outputBuffer = backend.AllocateBuffer(destSize);

        // Initialize to zero
        backend.Fill(outputBuffer, 0f, destSize);

        backend.ScatterAdd(source.Buffer, indices, outputBuffer, sourceSize, destSize);

        return new GpuTensor<T>(backend, outputBuffer, [destSize], GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// Creates a GPU-resident tensor filled with zeros.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="shape">Shape of the tensor to create.</param>
    /// <returns>A GPU-resident tensor filled with zeros.</returns>
    public IGpuTensor<T> ZerosGpu<T>(int[] shape)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ZerosGpu");

        int size = 1;
        foreach (var dim in shape)
            size *= dim;

        var outputBuffer = backend.AllocateBuffer(size);
        backend.Fill(outputBuffer, 0f, size);

        return new GpuTensor<T>(backend, outputBuffer, shape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident element-wise division with broadcast: C[i,j] = A[i,j] / B[i,0]
    /// Divides each element by the corresponding element in the first column of B.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="a">GPU-resident input tensor [batchSize, features].</param>
    /// <param name="b">GPU-resident divisor tensor [batchSize, 1] to broadcast.</param>
    /// <returns>A GPU-resident output tensor with element-wise division.</returns>
    public IGpuTensor<T> DivideByBroadcastGpu<T>(IGpuTensor<T> a, IGpuTensor<T> b)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for DivideByBroadcastGpu");

        int outerSize = a.Shape[0];
        int innerSize = a.ElementCount / outerSize;
        int bSize = b.ElementCount;

        // Compute reciprocal of b: 1/b
        var reciprocalBuffer = backend.AllocateBuffer(bSize);
        backend.Reciprocal(b.Buffer, reciprocalBuffer, bSize);

        // Multiply a by broadcast reciprocal: a * (1/b) = a / b
        var outputBuffer = backend.AllocateBuffer(a.ElementCount);
        backend.BroadcastMultiplyFirstAxis(a.Buffer, reciprocalBuffer, outputBuffer, outerSize, innerSize);

        // Clean up intermediate buffer
        reciprocalBuffer.Dispose();

        return new GpuTensor<T>(backend, outputBuffer, a.Shape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    #endregion

    #region Dropout Operations (GPU Accelerated)

    /// <summary>
    /// GPU-accelerated Dropout operation.
    /// </summary>
    Tensor<T> IEngine.Dropout<T>(Tensor<T> input, double dropoutRate, bool training, out Tensor<T> mask)
    {
        if (!TryGetBackend(out var backend) || !training)
            return base.Dropout(input, dropoutRate, training, out mask);

        try
        {
            int size = input.Length;
            ulong seed = (ulong)DateTime.UtcNow.Ticks;

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var outputBuffer = AllocateOutputBuffer(backend, size);
            using var maskBuffer = AllocateOutputBuffer(backend, size);

            backend.Dropout(inputBuffer.Buffer, outputBuffer.Buffer, maskBuffer.Buffer, size, (float)dropoutRate, seed, training);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] outputFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            float[] maskFloat = backend.DownloadBuffer(maskBuffer.Buffer);

            mask = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(maskFloat), input.Shape.ToArray());
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.Dropout(input, dropoutRate, training, out mask);
        }
    }

    /// <summary>
    /// GPU-accelerated Dropout backward operation.
    /// </summary>
    Tensor<T> IEngine.DropoutBackward<T>(Tensor<T> gradOutput, Tensor<T> mask, double dropoutRate)
    {
        if (!TryGetBackend(out var backend))
            return base.DropoutBackward(gradOutput, mask, dropoutRate);

        try
        {
            int size = gradOutput.Length;

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
            using var maskBuffer = GetOrAllocateBuffer(backend, mask.Data);
            using var gradInputBuffer = AllocateOutputBuffer(backend, size);

            backend.DropoutBackward(gradOutBuffer.Buffer, maskBuffer.Buffer, gradInputBuffer.Buffer, size, (float)dropoutRate);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), gradOutput.Shape.ToArray());
        }
        catch
        {
            return base.DropoutBackward(gradOutput, mask, dropoutRate);
        }
    }

    #endregion

    #region Embedding Operations (GPU Accelerated)

    /// <summary>
    /// GPU-accelerated Embedding lookup operation.
    /// </summary>
    Tensor<T> IEngine.Embedding<T>(Tensor<int> indices, Tensor<T> embeddingTable)
    {
        if (!TryGetBackend(out var backend))
            return base.Embedding(indices, embeddingTable);

        try
        {
            int numIndices = indices.Length;
            int embeddingDim = embeddingTable.Shape[^1];

            using var indicesBuffer = backend.AllocateIntBuffer(indices.Data);
            using var tableBuffer = GetOrCacheWeightBuffer(backend, embeddingTable.Data, PersistentTensorRole.Embeddings);
            using var outputBuffer = AllocateOutputBuffer(backend, numIndices * embeddingDim);

            backend.Embedding(indicesBuffer, tableBuffer.Buffer, outputBuffer.Buffer, numIndices, embeddingDim);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] outputFloat = backend.DownloadBuffer(outputBuffer.Buffer);

            // Output shape: indices.Shape + [embeddingDim]
            int[] outputShape = new int[indices.Shape.Length + 1];
            for (int i = 0; i < indices.Shape.Length; i++)
                outputShape[i] = indices.Shape[i];
            outputShape[^1] = embeddingDim;

            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputFloat), outputShape);
        }
        catch
        {
            return base.Embedding(indices, embeddingTable);
        }
    }

    /// <summary>
    /// GPU-accelerated Embedding backward operation.
    /// </summary>
    Tensor<T> IEngine.EmbeddingBackward<T>(Tensor<T> gradOutput, Tensor<int> indices, int vocabSize, int embeddingDim)
    {
        if (!TryGetBackend(out var backend))
            return base.EmbeddingBackward(gradOutput, indices, vocabSize, embeddingDim);

        try
        {
            int numIndices = indices.Length;

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
            using var indicesBuffer = backend.AllocateIntBuffer(indices.Data);
            using var gradEmbeddingBuffer = AllocateOutputBuffer(backend, vocabSize * embeddingDim);

            // Initialize to zero
            backend.Fill(gradEmbeddingBuffer.Buffer, 0f, vocabSize * embeddingDim);

            backend.EmbeddingBackward(gradOutBuffer.Buffer, indicesBuffer, gradEmbeddingBuffer.Buffer, numIndices, embeddingDim, vocabSize);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(gradEmbeddingBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), new[] { vocabSize, embeddingDim });
        }
        catch
        {
            return base.EmbeddingBackward(gradOutput, indices, vocabSize, embeddingDim);
        }
    }

    /// <summary>
    /// GPU-resident embedding lookup operation.
    /// Performs embedding lookup on GPU and returns a GPU-resident tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of the embedding tensor.</typeparam>
    /// <param name="embeddingTable">The embedding table tensor (either CPU Tensor or already on GPU).</param>
    /// <param name="indices">The token indices to look up.</param>
    /// <returns>A GPU-resident tensor containing the embeddings for the given indices.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no GPU backend is available.</exception>
    /// <remarks>
    /// <para>
    /// This method performs embedding lookup entirely on GPU, returning a GPU-resident tensor
    /// that can be passed to subsequent GPU operations without downloading to CPU.
    /// </para>
    /// <para>
    /// The output shape is: indices.Shape + [embeddingDim]
    /// For example, if indices has shape [batch, seqLen] and embeddingDim is 512,
    /// the output will have shape [batch, seqLen, 512].
    /// </para>
    /// </remarks>
    public IGpuTensor<T> EmbeddingLookupGpu<T>(Tensor<T> embeddingTable, Tensor<int> indices)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for EmbeddingLookupGpu");

        int numIndices = indices.Length;
        int embeddingDim = embeddingTable.Shape[^1];

        // Upload indices and embedding table to GPU
        using var indicesBuffer = backend.AllocateIntBuffer(indices.Data);
        using var tableBuffer = GetOrCacheWeightBuffer(backend, embeddingTable.Data, PersistentTensorRole.Embeddings);

        // Allocate output buffer (stays on GPU)
        var outputBuffer = backend.AllocateBuffer(numIndices * embeddingDim);

        // Perform embedding lookup on GPU
        backend.Embedding(indicesBuffer, tableBuffer.Buffer, outputBuffer, numIndices, embeddingDim);

        // Calculate output shape: indices.Shape + [embeddingDim]
        int[] outputShape = new int[indices.Shape.Length + 1];
        for (int i = 0; i < indices.Shape.Length; i++)
            outputShape[i] = indices.Shape[i];
        outputShape[^1] = embeddingDim;

        return new GpuTensor<T>(backend, outputBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident embedding lookup operation with GPU-resident embedding table.
    /// Both input embedding table and output remain on GPU.
    /// </summary>
    /// <typeparam name="T">The numeric type of the embedding tensor.</typeparam>
    /// <param name="embeddingTableGpu">The GPU-resident embedding table.</param>
    /// <param name="indices">The token indices to look up.</param>
    /// <param name="embeddingDim">The dimension of each embedding vector.</param>
    /// <returns>A GPU-resident tensor containing the embeddings for the given indices.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no GPU backend is available.</exception>
    public IGpuTensor<T> EmbeddingLookupGpu<T>(IGpuTensor<T> embeddingTableGpu, Tensor<int> indices, int embeddingDim)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for EmbeddingLookupGpu");

        int numIndices = indices.Length;

        // Upload indices to GPU (embedding table is already on GPU)
        using var indicesBuffer = backend.AllocateIntBuffer(indices.Data);

        // Allocate output buffer (stays on GPU)
        var outputBuffer = backend.AllocateBuffer(numIndices * embeddingDim);

        // Perform embedding lookup on GPU
        backend.Embedding(indicesBuffer, embeddingTableGpu.Buffer, outputBuffer, numIndices, embeddingDim);

        // Calculate output shape: indices.Shape + [embeddingDim]
        int[] outputShape = new int[indices.Shape.Length + 1];
        for (int i = 0; i < indices.Shape.Length; i++)
            outputShape[i] = indices.Shape[i];
        outputShape[^1] = embeddingDim;

        return new GpuTensor<T>(backend, outputBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident embedding backward operation.
    /// Computes gradients for the embedding table on GPU.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="gradOutput">The GPU-resident gradient of the loss w.r.t. output.</param>
    /// <param name="indices">The indices that were used in the forward pass.</param>
    /// <param name="vocabSize">The vocabulary size (number of embeddings).</param>
    /// <param name="embeddingDim">The dimension of each embedding vector.</param>
    /// <returns>A GPU-resident gradient tensor for the embedding table.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no GPU backend is available.</exception>
    public IGpuTensor<T> EmbeddingBackwardGpu<T>(IGpuTensor<T> gradOutput, Tensor<int> indices, int vocabSize, int embeddingDim)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for EmbeddingBackwardGpu");

        int numIndices = indices.Length;

        // Upload indices to GPU
        using var indicesBuffer = backend.AllocateIntBuffer(indices.Data);

        // Allocate gradient embedding buffer and initialize to zero
        var gradEmbeddingBuffer = backend.AllocateBuffer(vocabSize * embeddingDim);
        backend.Fill(gradEmbeddingBuffer, 0f, vocabSize * embeddingDim);

        // Perform scatter-add for gradient accumulation
        backend.EmbeddingBackward(gradOutput.Buffer, indicesBuffer, gradEmbeddingBuffer, numIndices, embeddingDim, vocabSize);

        return new GpuTensor<T>(backend, gradEmbeddingBuffer, [vocabSize, embeddingDim], GpuTensorRole.Gradient, ownsBuffer: true);
    }

    #endregion

    #region Loss Functions (GPU Accelerated)

    /// <summary>
    /// GPU-accelerated CrossEntropy loss computation.
    /// </summary>
    T IEngine.CrossEntropyLoss<T>(Tensor<T> predictions, Tensor<T> targets)
    {
        if (!TryGetBackend(out var backend))
            return base.CrossEntropyLoss(predictions, targets);

        try
        {
            // Assume predictions: [batch, numClasses], targets: [batch] or [batch, numClasses]
            if (predictions.Rank != 2)
                return base.CrossEntropyLoss(predictions, targets);

            int batchSize = predictions.Shape[0];
            int numClasses = predictions.Shape[1];

            using var predBuffer = GetOrAllocateBuffer(backend, predictions.Data);
            using var targetBuffer = GetOrAllocateBuffer(backend, targets.Data);

            float loss = backend.CrossEntropyLoss(predBuffer.Buffer, targetBuffer.Buffer, batchSize, numClasses);
            return DirectGpuEngine.FromFloatArray<T>(new[] { loss })[0];
        }
        catch
        {
            return base.CrossEntropyLoss(predictions, targets);
        }
    }

    /// <summary>
    /// GPU-accelerated CrossEntropy backward computation.
    /// </summary>
    Tensor<T> IEngine.CrossEntropyBackward<T>(Tensor<T> predictions, Tensor<T> targets)
    {
        if (!TryGetBackend(out var backend))
            return base.CrossEntropyBackward(predictions, targets);

        try
        {
            if (predictions.Rank != 2)
                return base.CrossEntropyBackward(predictions, targets);

            int batchSize = predictions.Shape[0];
            int numClasses = predictions.Shape[1];

            using var predBuffer = GetOrAllocateBuffer(backend, predictions.Data);
            using var targetBuffer = GetOrAllocateBuffer(backend, targets.Data);
            using var gradInputBuffer = AllocateOutputBuffer(backend, predictions.Length);

            backend.CrossEntropyBackward(predBuffer.Buffer, targetBuffer.Buffer, gradInputBuffer.Buffer, batchSize, numClasses);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), predictions.Shape.ToArray());
        }
        catch
        {
            return base.CrossEntropyBackward(predictions, targets);
        }
    }

    /// <summary>
    /// GPU-accelerated MSE loss computation.
    /// </summary>
    T IEngine.MseLoss<T>(Tensor<T> predictions, Tensor<T> targets)
    {
        if (!TryGetBackend(out var backend))
            return base.MseLoss(predictions, targets);

        try
        {
            int size = predictions.Length;

            using var predBuffer = GetOrAllocateBuffer(backend, predictions.Data);
            using var targetBuffer = GetOrAllocateBuffer(backend, targets.Data);

            float loss = backend.MseLoss(predBuffer.Buffer, targetBuffer.Buffer, size);
            return DirectGpuEngine.FromFloatArray<T>(new[] { loss })[0];
        }
        catch
        {
            return base.MseLoss(predictions, targets);
        }
    }

    /// <summary>
    /// GPU-accelerated MSE backward computation.
    /// </summary>
    Tensor<T> IEngine.MseBackward<T>(Tensor<T> predictions, Tensor<T> targets)
    {
        if (!TryGetBackend(out var backend))
            return base.MseBackward(predictions, targets);

        try
        {
            int size = predictions.Length;

            using var predBuffer = GetOrAllocateBuffer(backend, predictions.Data);
            using var targetBuffer = GetOrAllocateBuffer(backend, targets.Data);
            using var gradInputBuffer = AllocateOutputBuffer(backend, size);

            backend.MseBackward(predBuffer.Buffer, targetBuffer.Buffer, gradInputBuffer.Buffer, size);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), predictions.Shape.ToArray());
        }
        catch
        {
            return base.MseBackward(predictions, targets);
        }
    }

    #endregion

    #region Activation Backward Operations (GPU Accelerated)

    /// <summary>
    /// GPU-accelerated ReLU backward operation.
    /// </summary>
    Tensor<T> IEngine.ReluBackward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            return base.ReluBackward(gradOutput, input);

        try
        {
            int size = gradOutput.Length;

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var gradInputBuffer = AllocateOutputBuffer(backend, size);

            backend.ReluBackward(gradOutBuffer.Buffer, inputBuffer.Buffer, gradInputBuffer.Buffer, size);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), gradOutput.Shape.ToArray());
        }
        catch
        {
            return base.ReluBackward(gradOutput, input);
        }
    }

    /// <summary>
    /// GPU-accelerated Sigmoid backward operation.
    /// </summary>
    Tensor<T> IEngine.SigmoidBackward<T>(Tensor<T> gradOutput, Tensor<T> output)
    {
        if (!TryGetBackend(out var backend))
            return base.SigmoidBackward(gradOutput, output);

        try
        {
            int size = gradOutput.Length;

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
            using var outputBuffer = GetOrAllocateBuffer(backend, output.Data);
            using var gradInputBuffer = AllocateOutputBuffer(backend, size);

            backend.SigmoidBackward(gradOutBuffer.Buffer, outputBuffer.Buffer, gradInputBuffer.Buffer, size);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), gradOutput.Shape.ToArray());
        }
        catch
        {
            return base.SigmoidBackward(gradOutput, output);
        }
    }

    /// <summary>
    /// GPU-accelerated Tanh backward operation.
    /// </summary>
    Tensor<T> IEngine.TanhBackward<T>(Tensor<T> gradOutput, Tensor<T> output)
    {
        if (!TryGetBackend(out var backend))
            return base.TanhBackward(gradOutput, output);

        try
        {
            int size = gradOutput.Length;

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
            using var outputBuffer = GetOrAllocateBuffer(backend, output.Data);
            using var gradInputBuffer = AllocateOutputBuffer(backend, size);

            backend.TanhBackward(gradOutBuffer.Buffer, outputBuffer.Buffer, gradInputBuffer.Buffer, size);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), gradOutput.Shape.ToArray());
        }
        catch
        {
            return base.TanhBackward(gradOutput, output);
        }
    }

    /// <summary>
    /// GPU-accelerated GELU backward operation.
    /// </summary>
    Tensor<T> IEngine.GeluBackward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            return base.GeluBackward(gradOutput, input);

        try
        {
            int size = gradOutput.Length;

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var gradInputBuffer = AllocateOutputBuffer(backend, size);

            backend.GeluBackward(gradOutBuffer.Buffer, inputBuffer.Buffer, gradInputBuffer.Buffer, size);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), gradOutput.Shape.ToArray());
        }
        catch
        {
            return base.GeluBackward(gradOutput, input);
        }
    }

    /// <summary>
    /// GPU-accelerated LeakyReLU activation.
    /// </summary>
    Tensor<T> IEngine.LeakyReLU<T>(Tensor<T> input, T alpha)
    {
        if (!TryGetBackend(out var backend))
            return base.LeakyReLU(input, alpha);

        try
        {
            int size = input.Length;
            var numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            float negativeSlope = (float)numOps.ToDouble(alpha);

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var outputBuffer = AllocateOutputBuffer(backend, size);

            backend.LeakyRelu(inputBuffer.Buffer, outputBuffer.Buffer, negativeSlope, size);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.LeakyReLU(input, alpha);
        }
    }

    /// <summary>
    /// GPU-accelerated LeakyReLU backward operation.
    /// </summary>
    Tensor<T> IEngine.LeakyReluBackward<T>(Tensor<T> gradOutput, Tensor<T> input, double negativeSlope)
    {
        if (!TryGetBackend(out var backend))
            return base.LeakyReluBackward(gradOutput, input, negativeSlope);

        try
        {
            int size = gradOutput.Length;

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var gradInputBuffer = AllocateOutputBuffer(backend, size);

            backend.LeakyReluBackward(gradOutBuffer.Buffer, inputBuffer.Buffer, gradInputBuffer.Buffer, (float)negativeSlope, size);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), gradOutput.Shape.ToArray());
        }
        catch
        {
            return base.LeakyReluBackward(gradOutput, input, negativeSlope);
        }
    }

    /// <summary>
    /// GPU-accelerated ELU activation.
    /// </summary>
    Tensor<T> IEngine.ELU<T>(Tensor<T> input, double alpha)
    {
        if (!TryGetBackend(out var backend))
            return base.ELU(input, alpha);

        try
        {
            int size = input.Length;

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var outputBuffer = AllocateOutputBuffer(backend, size);

            backend.Elu(inputBuffer.Buffer, outputBuffer.Buffer, (float)alpha, size);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.ELU(input, alpha);
        }
    }

    /// <summary>
    /// GPU-accelerated Swish activation.
    /// </summary>
    Tensor<T> IEngine.Swish<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            return base.Swish(input);

        try
        {
            int size = input.Length;

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var outputBuffer = AllocateOutputBuffer(backend, size);

            backend.Swish(inputBuffer.Buffer, outputBuffer.Buffer, size);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.Swish(input);
        }
    }

    /// <summary>
    /// GPU-accelerated Mish activation.
    /// </summary>
    Tensor<T> IEngine.Mish<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            return base.Mish(input);

        try
        {
            int size = input.Length;

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var outputBuffer = AllocateOutputBuffer(backend, size);

            backend.Mish(inputBuffer.Buffer, outputBuffer.Buffer, size);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.Mish(input);
        }
    }

    /// <summary>
    /// GPU-accelerated Softplus activation.
    /// </summary>
    Tensor<T> IEngine.Softplus<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            return base.Softplus(input);

        try
        {
            int size = input.Length;

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var outputBuffer = AllocateOutputBuffer(backend, size);

            backend.Softplus(inputBuffer.Buffer, outputBuffer.Buffer, size);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.Softplus(input);
        }
    }

    /// <summary>
    /// GPU-accelerated HardSwish activation.
    /// </summary>
    Tensor<T> IEngine.HardSwish<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            return base.HardSwish(input);

        try
        {
            int size = input.Length;

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var outputBuffer = AllocateOutputBuffer(backend, size);

            backend.Hardswish(inputBuffer.Buffer, outputBuffer.Buffer, size);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.HardSwish(input);
        }
    }

    #endregion

    #region Convolution Backward Operations (GPU Accelerated)

    /// <summary>
    /// GPU-accelerated Conv2D backward for input gradients.
    /// </summary>
    Tensor<T> IEngine.Conv2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape,
        int[] stride, int[] padding, int[] dilation)
    {
        if (!TryGetBackend(out var backend))
            return base.Conv2DBackwardInput(gradOutput, kernel, inputShape, stride, padding, dilation);

        try
        {
            if (gradOutput.Rank != 4 || kernel.Rank != 4)
                return base.Conv2DBackwardInput(gradOutput, kernel, inputShape, stride, padding, dilation);

            int strideH = stride.Length > 0 ? stride[0] : 1;
            int strideW = stride.Length > 1 ? stride[1] : strideH;
            int padH = padding.Length > 0 ? padding[0] : 0;
            int padW = padding.Length > 1 ? padding[1] : padH;
            int dilationH = dilation.Length > 0 ? dilation[0] : 1;
            int dilationW = dilation.Length > 1 ? dilation[1] : dilationH;

            int batch = gradOutput.Shape[0];
            int outChannels = gradOutput.Shape[1];
            int outHeight = gradOutput.Shape[2];
            int outWidth = gradOutput.Shape[3];

            int inChannels = inputShape[1];
            int inHeight = inputShape[2];
            int inWidth = inputShape[3];

            int kernelH = kernel.Shape[2];
            int kernelW = kernel.Shape[3];

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
            using var kernelBuffer = GetOrCacheWeightBuffer(backend, kernel.Data, PersistentTensorRole.Weights);
            using var gradInputBuffer = AllocateOutputBuffer(backend, batch * inChannels * inHeight * inWidth);

            backend.Conv2DBackwardInput(gradOutBuffer.Buffer, kernelBuffer.Buffer, gradInputBuffer.Buffer,
                batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth,
                kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), inputShape);
        }
        catch
        {
            return base.Conv2DBackwardInput(gradOutput, kernel, inputShape, stride, padding, dilation);
        }
    }

    /// <summary>
    /// GPU-accelerated Conv2D backward for kernel gradients.
    /// </summary>
    Tensor<T> IEngine.Conv2DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape,
        int[] stride, int[] padding, int[] dilation)
    {
        if (!TryGetBackend(out var backend))
            return base.Conv2DBackwardKernel(gradOutput, input, kernelShape, stride, padding, dilation);

        try
        {
            if (input.Rank != 4 || gradOutput.Rank != 4)
                return base.Conv2DBackwardKernel(gradOutput, input, kernelShape, stride, padding, dilation);

            int strideH = stride.Length > 0 ? stride[0] : 1;
            int strideW = stride.Length > 1 ? stride[1] : strideH;
            int padH = padding.Length > 0 ? padding[0] : 0;
            int padW = padding.Length > 1 ? padding[1] : padH;
            int dilationH = dilation.Length > 0 ? dilation[0] : 1;
            int dilationW = dilation.Length > 1 ? dilation[1] : dilationH;

            int batch = input.Shape[0];
            int inChannels = input.Shape[1];
            int inHeight = input.Shape[2];
            int inWidth = input.Shape[3];

            int outChannels = gradOutput.Shape[1];
            int outHeight = gradOutput.Shape[2];
            int outWidth = gradOutput.Shape[3];

            int kernelH = kernelShape[2];
            int kernelW = kernelShape[3];

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
            using var gradKernelBuffer = AllocateOutputBuffer(backend, outChannels * inChannels * kernelH * kernelW);

            backend.Conv2DBackwardKernel(inputBuffer.Buffer, gradOutBuffer.Buffer, gradKernelBuffer.Buffer,
                batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth,
                kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(gradKernelBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), kernelShape);
        }
        catch
        {
            return base.Conv2DBackwardKernel(gradOutput, input, kernelShape, stride, padding, dilation);
        }
    }

    #endregion

    #region Global Pooling Operations (GPU Accelerated)

    /// <summary>
    /// GPU-accelerated Global Average Pooling.
    /// </summary>
    Tensor<T> IEngine.GlobalAvgPool2D<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            return base.GlobalAvgPool2D(input);

        try
        {
            if (input.Rank != 4)
                return base.GlobalAvgPool2D(input);

            int batch = input.Shape[0];
            int channels = input.Shape[1];
            int height = input.Shape[2];
            int width = input.Shape[3];

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var outputBuffer = AllocateOutputBuffer(backend, batch * channels);

            backend.GlobalAvgPool2D(inputBuffer.Buffer, outputBuffer.Buffer, batch, channels, height, width);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), new[] { batch, channels, 1, 1 });
        }
        catch
        {
            return base.GlobalAvgPool2D(input);
        }
    }

    /// <summary>
    /// GPU-accelerated Global Max Pooling.
    /// </summary>
    Tensor<T> IEngine.GlobalMaxPool2D<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            return base.GlobalMaxPool2D(input);

        try
        {
            if (input.Rank != 4)
                return base.GlobalMaxPool2D(input);

            int batch = input.Shape[0];
            int channels = input.Shape[1];
            int height = input.Shape[2];
            int width = input.Shape[3];

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var outputBuffer = AllocateOutputBuffer(backend, batch * channels);

            backend.GlobalMaxPool2D(inputBuffer.Buffer, outputBuffer.Buffer, batch, channels, height, width);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), new[] { batch, channels, 1, 1 });
        }
        catch
        {
            return base.GlobalMaxPool2D(input);
        }
    }

    /// <summary>
    /// GPU-accelerated Adaptive Average Pooling.
    /// </summary>
    Tensor<T> IEngine.AdaptiveAvgPool2D<T>(Tensor<T> input, int outputHeight, int outputWidth)
    {
        if (!TryGetBackend(out var backend))
            return base.AdaptiveAvgPool2D(input, outputHeight, outputWidth);

        try
        {
            if (input.Rank != 4)
                return base.AdaptiveAvgPool2D(input, outputHeight, outputWidth);

            int batch = input.Shape[0];
            int channels = input.Shape[1];
            int inHeight = input.Shape[2];
            int inWidth = input.Shape[3];

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var outputBuffer = AllocateOutputBuffer(backend, batch * channels * outputHeight * outputWidth);

            backend.AdaptiveAvgPool2D(inputBuffer.Buffer, outputBuffer.Buffer, batch, channels, inHeight, inWidth, outputHeight, outputWidth);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), new[] { batch, channels, outputHeight, outputWidth });
        }
        catch
        {
            return base.AdaptiveAvgPool2D(input, outputHeight, outputWidth);
        }
    }

    #endregion

    #region GPU-Accelerated Reduction Operations

    /// <summary>
    /// GPU-accelerated ReduceMean operation.
    /// </summary>
    public new Tensor<T> ReduceMean<T>(Tensor<T> input, int[] axes, bool keepDims)
    {
        var safeAxes = axes ?? Array.Empty<int>();
        if (!TryGetBackend(out var backend))
            return base.ReduceMean(input, safeAxes, keepDims);

        // Validate and normalize axes
        if (safeAxes.Length == 0)
            return base.ReduceMean(input, safeAxes, keepDims);

        // Normalize negative axes
        var normalizedAxes = new int[safeAxes.Length];
        for (int i = 0; i < safeAxes.Length; i++)
        {
            normalizedAxes[i] = safeAxes[i] < 0 ? safeAxes[i] + input.Rank : safeAxes[i];
        }
        Array.Sort(normalizedAxes);

        try
        {
            return ReduceAxisGpu(input, normalizedAxes, keepDims, backend, ReduceOperation.Mean);
        }
        catch
        {
            return base.ReduceMean(input, safeAxes, keepDims);
        }
    }

    /// <summary>
    /// GPU-accelerated ReduceMax operation.
    /// </summary>
    public new Tensor<T> ReduceMax<T>(Tensor<T> input, int[] axes, bool keepDims, out int[] maxIndices)
    {
        var safeAxes = axes ?? Array.Empty<int>();
        if (!TryGetBackend(out var backend))
            return base.ReduceMax(input, safeAxes, keepDims, out maxIndices);

        // Validate and normalize axes
        if (safeAxes.Length == 0)
            return base.ReduceMax(input, safeAxes, keepDims, out maxIndices);

        // Normalize negative axes
        var normalizedAxes = new int[safeAxes.Length];
        for (int i = 0; i < safeAxes.Length; i++)
        {
            normalizedAxes[i] = safeAxes[i] < 0 ? safeAxes[i] + input.Rank : safeAxes[i];
        }
        Array.Sort(normalizedAxes);

        try
        {
            // For now, indices are computed on CPU for simplicity
            // TODO: Add GPU ArgMax to get indices efficiently
            var result = ReduceAxisGpu(input, normalizedAxes, keepDims, backend, ReduceOperation.Max);

            // Compute indices on CPU (fallback for now)
            var cpuResult = base.ReduceMax(input, safeAxes, keepDims, out maxIndices);
            return result;
        }
        catch
        {
            return base.ReduceMax(input, safeAxes, keepDims, out maxIndices);
        }
    }

    /// <summary>
    /// GPU-accelerated ReduceSum operation.
    /// </summary>
    public new Tensor<T> ReduceSum<T>(Tensor<T> tensor, int[]? axes = null, bool keepDims = false)
    {
        if (!TryGetBackend(out var backend))
            return base.ReduceSum(tensor, axes, keepDims);

        // If axes is null, reduce all dimensions
        if (axes == null || axes.Length == 0)
        {
            // Full reduction - use existing Sum implementation
            return base.ReduceSum(tensor, axes, keepDims);
        }

        // Normalize negative axes
        var normalizedAxes = new int[axes.Length];
        for (int i = 0; i < axes.Length; i++)
        {
            normalizedAxes[i] = axes[i] < 0 ? axes[i] + tensor.Rank : axes[i];
        }
        Array.Sort(normalizedAxes);

        try
        {
            return ReduceAxisGpu(tensor, normalizedAxes, keepDims, backend, ReduceOperation.Sum);
        }
        catch
        {
            return base.ReduceSum(tensor, axes, keepDims);
        }
    }

    private enum ReduceOperation { Sum, Mean, Max }

    /// <summary>
    /// Internal GPU reduction implementation that handles arbitrary axes.
    /// </summary>
    private Tensor<T> ReduceAxisGpu<T>(Tensor<T> input, int[] normalizedAxes, bool keepDims,
        IDirectGpuBackend backend, ReduceOperation op)
    {
        var inputShape = input.Shape;
        int inputRank = inputShape.Length;

        // Compute output shape
        var outputShapeList = new List<int>();
        int reduceSize = 1;
        int outerSize = 1;

        // For single axis reduction at the end, we can use backend directly
        // For other cases, we need to reshape/permute
        if (normalizedAxes.Length == 1 && normalizedAxes[0] == inputRank - 1)
        {
            // Reduction over last axis - optimal case
            for (int i = 0; i < inputRank - 1; i++)
            {
                outerSize *= inputShape[i];
                outputShapeList.Add(inputShape[i]);
            }
            reduceSize = inputShape[^1];
            if (keepDims) outputShapeList.Add(1);
        }
        else
        {
            // General case: permute axes so reduction axes are at the end
            // Then reshape to 2D [outerSize, reduceSize]
            var permutation = new List<int>();
            var reduceDims = new HashSet<int>(normalizedAxes);

            // First add non-reduce dimensions
            for (int i = 0; i < inputRank; i++)
            {
                if (!reduceDims.Contains(i))
                {
                    permutation.Add(i);
                    outerSize *= inputShape[i];
                    outputShapeList.Add(inputShape[i]);
                }
            }

            // Then add reduce dimensions
            foreach (int axis in normalizedAxes)
            {
                permutation.Add(axis);
                reduceSize *= inputShape[axis];
                if (keepDims) outputShapeList.Add(1);
            }

            // Permute the input tensor
            input = PermuteImpl(input, permutation.ToArray());
        }

        if (outputShapeList.Count == 0)
            outputShapeList.Add(1);

        var outputShape = outputShapeList.ToArray();

        // Upload input
        float[] inputFloat = DirectGpuEngine.ToFloatArray(input.Data);
        using var inputBuffer = GetOrAllocateBuffer(backend, inputFloat);
        using var outputBuffer = AllocateOutputBuffer(backend, outerSize);

        // Execute the appropriate reduction
        switch (op)
        {
            case ReduceOperation.Sum:
                backend.SumAxis(inputBuffer.Buffer, outputBuffer.Buffer, outerSize, reduceSize);
                break;
            case ReduceOperation.Mean:
                backend.MeanAxis(inputBuffer.Buffer, outputBuffer.Buffer, outerSize, reduceSize);
                break;
            case ReduceOperation.Max:
                backend.MaxAxis(inputBuffer.Buffer, outputBuffer.Buffer, outerSize, reduceSize);
                break;
        }

        // Download result
        float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
        T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);

        return new Tensor<T>(outputShape, new Vector<T>(resultData));
    }

    /// <summary>
    /// Internal tensor permutation helper.
    /// </summary>
    private static Tensor<T> PermuteImpl<T>(Tensor<T> input, int[] permutation)
    {
        var inputShape = input.Shape;
        int rank = inputShape.Length;

        // Compute output shape
        var outputShape = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            outputShape[i] = inputShape[permutation[i]];
        }

        // Compute strides
        var inputStrides = new int[rank];
        var outputStrides = new int[rank];
        inputStrides[rank - 1] = 1;
        outputStrides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--)
        {
            inputStrides[i] = inputStrides[i + 1] * inputShape[i + 1];
            outputStrides[i] = outputStrides[i + 1] * outputShape[i + 1];
        }

        var inputData = input.ToArray();
        var outputData = new T[inputData.Length];

        // Permute data
        for (int i = 0; i < inputData.Length; i++)
        {
            // Convert flat index to multi-index
            var multiIndex = new int[rank];
            int remaining = i;
            for (int d = 0; d < rank; d++)
            {
                multiIndex[d] = remaining / inputStrides[d];
                remaining %= inputStrides[d];
            }

            // Apply permutation and compute output index
            int outputIdx = 0;
            for (int d = 0; d < rank; d++)
            {
                outputIdx += multiIndex[permutation[d]] * outputStrides[d];
            }

            outputData[outputIdx] = inputData[i];
        }

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    #endregion

    #region Broadcast Operations

    /// <summary>
    /// GPU-accelerated TensorBroadcastMultiply operation.
    /// Performs element-wise multiplication with NumPy-style broadcasting.
    /// </summary>
    public new Tensor<T> TensorBroadcastMultiply<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorBroadcastMultiply(a, b);

        // Fast path: same shape - use element-wise multiply
        if (a.Shape.SequenceEqual(b.Shape))
        {
            try
            {
                using var bufferA = GetOrAllocateBuffer(backend, a.Data);
                using var bufferB = GetOrAllocateBuffer(backend, b.Data);
                using var bufferC = AllocateOutputBuffer(backend, a.Length);

                backend.Multiply(bufferA.Buffer, bufferB.Buffer, bufferC.Buffer, a.Length);

                float[] resultFloat = new float[a.Length];
                backend.DownloadBuffer(bufferC.Buffer, resultFloat);
                return new Tensor<T>(a.Shape, new Vector<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat)));
            }
            catch
            {
                return base.TensorBroadcastMultiply(a, b);
            }
        }

        // Check for common broadcast patterns that we can accelerate
        try
        {
            // Pattern 1: (outer, inner) * (inner,) -> broadcast along last axis
            if (b.Rank == 1 && a.Shape[a.Rank - 1] == b.Shape[0])
            {
                int innerSize = b.Shape[0];
                int outerSize = a.Length / innerSize;

                using var bufferA = GetOrAllocateBuffer(backend, a.Data);
                using var bufferB = GetOrAllocateBuffer(backend, b.Data);
                using var bufferC = AllocateOutputBuffer(backend, a.Length);

                backend.BroadcastMultiplyLastAxis(bufferA.Buffer, bufferB.Buffer, bufferC.Buffer, outerSize, innerSize);

                float[] resultFloat = new float[a.Length];
                backend.DownloadBuffer(bufferC.Buffer, resultFloat);
                return new Tensor<T>(a.Shape, new Vector<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat)));
            }

            // Pattern 2: (outer, inner) * (outer, 1) -> broadcast along first axis (column broadcast)
            if (a.Rank == 2 && b.Rank == 2 && b.Shape[0] == a.Shape[0] && b.Shape[1] == 1)
            {
                int outerSize = a.Shape[0];
                int innerSize = a.Shape[1];

                // Extract first column from b as 1D array
                T[] bFlatData = new T[outerSize];
                for (int i = 0; i < outerSize; i++)
                    bFlatData[i] = b.Data[i];

                using var bufferA = GetOrAllocateBuffer(backend, a.Data);
                using var bufferB = GetOrAllocateBuffer(backend, bFlatData);
                using var bufferC = AllocateOutputBuffer(backend, a.Length);

                backend.BroadcastMultiplyFirstAxis(bufferA.Buffer, bufferB.Buffer, bufferC.Buffer, outerSize, innerSize);

                float[] resultFloat = new float[a.Length];
                backend.DownloadBuffer(bufferC.Buffer, resultFloat);
                return new Tensor<T>(a.Shape, new Vector<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat)));
            }

            // Pattern 3: (batch, seq, features) * (1, 1, features) -> common in attention/normalization
            if (a.Rank >= 2 && b.Rank == a.Rank)
            {
                // Check if broadcasting along all but last axis
                bool isLastAxisBroadcast = true;
                for (int i = 0; i < a.Rank - 1; i++)
                {
                    if (b.Shape[i] != 1)
                    {
                        isLastAxisBroadcast = false;
                        break;
                    }
                }
                if (isLastAxisBroadcast && a.Shape[a.Rank - 1] == b.Shape[b.Rank - 1])
                {
                    int innerSize = a.Shape[a.Rank - 1];
                    int outerSize = a.Length / innerSize;

                    // Extract last dimension from b as 1D array
                    T[] bFlatData = new T[innerSize];
                    for (int i = 0; i < innerSize; i++)
                        bFlatData[i] = b.Data[i];

                    using var bufferA = GetOrAllocateBuffer(backend, a.Data);
                    using var bufferB = GetOrAllocateBuffer(backend, bFlatData);
                    using var bufferC = AllocateOutputBuffer(backend, a.Length);

                    backend.BroadcastMultiplyLastAxis(bufferA.Buffer, bufferB.Buffer, bufferC.Buffer, outerSize, innerSize);

                    float[] resultFloat = new float[a.Length];
                    backend.DownloadBuffer(bufferC.Buffer, resultFloat);
                    return new Tensor<T>(a.Shape, new Vector<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat)));
                }
            }

            // Fallback to CPU for complex broadcast patterns
            return base.TensorBroadcastMultiply(a, b);
        }
        catch
        {
            return base.TensorBroadcastMultiply(a, b);
        }
    }

    #endregion

    #region GPU Sparse Matrix Operations

    /// <summary>
    /// GPU-resident sparse-dense matrix multiplication using CSR format.
    /// Computes: C[M,N] = A[M,K] * B[K,N] where A is in CSR sparse format.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="sparseA">CSR sparse tensor A [M, K].</param>
    /// <param name="denseB">GPU-resident dense tensor B [K, N].</param>
    /// <returns>GPU-resident dense output tensor C [M, N].</returns>
    public IGpuTensor<T> SparseDenseMatMulGpu<T>(ICsrGpuTensor<T> sparseA, IGpuTensor<T> denseB)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SparseDenseMatMulGpu");

        if (sparseA is null) throw new ArgumentNullException(nameof(sparseA));
        if (denseB is null) throw new ArgumentNullException(nameof(denseB));

        // Validate dimensions: A[M,K] @ B[K,N] -> C[M,N]
        if (denseB.Shape.Length != 2)
            throw new ArgumentException("Dense tensor B must be 2D [K, N]");

        int M = sparseA.Rows;
        int K = sparseA.Cols;
        int N = denseB.Shape[1];

        if (denseB.Shape[0] != K)
            throw new ArgumentException($"Dimension mismatch: sparse A has {K} columns, but dense B has {denseB.Shape[0]} rows");

        // Allocate output buffer
        var outputBuffer = backend.AllocateBuffer(M * N);

        // Execute CSR SpMM
        backend.CsrSpMM(
            sparseA.Values,
            sparseA.ColumnIndices,
            sparseA.RowPointers,
            denseB.Buffer,
            outputBuffer,
            M, K, N, sparseA.Nnz);

        return new GpuTensor<T>(backend, outputBuffer, [M, N],
            GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident sparse-dense matrix multiplication with bias using CSR format.
    /// Computes: C[M,N] = A[M,K] * B[K,N] + bias[N] where A is in CSR sparse format.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="sparseA">CSR sparse tensor A [M, K].</param>
    /// <param name="denseB">GPU-resident dense tensor B [K, N].</param>
    /// <param name="bias">Bias tensor [N].</param>
    /// <returns>GPU-resident dense output tensor C [M, N].</returns>
    public IGpuTensor<T> SparseDenseMatMulBiasGpu<T>(ICsrGpuTensor<T> sparseA, IGpuTensor<T> denseB, Tensor<T> bias)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SparseDenseMatMulBiasGpu");

        if (sparseA is null) throw new ArgumentNullException(nameof(sparseA));
        if (denseB is null) throw new ArgumentNullException(nameof(denseB));
        if (bias is null) throw new ArgumentNullException(nameof(bias));

        // Validate dimensions
        if (denseB.Shape.Length != 2)
            throw new ArgumentException("Dense tensor B must be 2D [K, N]");

        int M = sparseA.Rows;
        int K = sparseA.Cols;
        int N = denseB.Shape[1];

        if (denseB.Shape[0] != K)
            throw new ArgumentException($"Dimension mismatch: sparse A has {K} columns, but dense B has {denseB.Shape[0]} rows");

        if (bias.Length != N)
            throw new ArgumentException($"Bias length {bias.Length} must match output columns {N}");

        // Upload bias
        using var biasBuffer = GetOrCacheWeightBuffer(backend, bias.Data, PersistentTensorRole.Biases);

        // Allocate output buffer
        var outputBuffer = backend.AllocateBuffer(M * N);

        // Execute CSR SpMM with bias
        backend.CsrSpMMBias(
            sparseA.Values,
            sparseA.ColumnIndices,
            sparseA.RowPointers,
            denseB.Buffer,
            biasBuffer.Buffer,
            outputBuffer,
            M, K, N, sparseA.Nnz);

        return new GpuTensor<T>(backend, outputBuffer, [M, N],
            GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU scatter-add operation for graph neural network message passing.
    /// For each edge (source -> target), adds source features weighted by edge values to target.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="nodeFeatures">GPU-resident node feature tensor [numNodes, features].</param>
    /// <param name="sourceIndices">Source node indices for each edge [numEdges].</param>
    /// <param name="targetIndices">Target node indices for each edge [numEdges].</param>
    /// <param name="edgeValues">Optional edge weights [numEdges]. If null, uses 1.0 for all edges.</param>
    /// <returns>GPU-resident aggregated node features [numNodes, features].</returns>
    public IGpuTensor<T> ScatterAddGpu<T>(
        IGpuTensor<T> nodeFeatures,
        int[] sourceIndices,
        int[] targetIndices,
        float[]? edgeValues = null)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ScatterAddGpu");

        if (nodeFeatures is null) throw new ArgumentNullException(nameof(nodeFeatures));
        if (sourceIndices is null) throw new ArgumentNullException(nameof(sourceIndices));
        if (targetIndices is null) throw new ArgumentNullException(nameof(targetIndices));

        if (nodeFeatures.Shape.Length != 2)
            throw new ArgumentException("Node features must be 2D [numNodes, features]");

        if (sourceIndices.Length != targetIndices.Length)
            throw new ArgumentException("Source and target indices must have the same length");

        int numNodes = nodeFeatures.Shape[0];
        int features = nodeFeatures.Shape[1];
        int numEdges = sourceIndices.Length;

        // Upload indices as float buffers (GPU kernels use float for everything)
        float[] srcFloat = new float[numEdges];
        float[] tgtFloat = new float[numEdges];
        for (int i = 0; i < numEdges; i++)
        {
            srcFloat[i] = sourceIndices[i];
            tgtFloat[i] = targetIndices[i];
        }

        using var srcBuffer = GetOrAllocateBuffer(backend, srcFloat);
        using var tgtBuffer = GetOrAllocateBuffer(backend, tgtFloat);
        OwnedBuffer? edgeBuffer = edgeValues is not null ? GetOrAllocateBuffer(backend, edgeValues) : null;

        try
        {
            // Allocate output buffer and zero it
            var outputBuffer = backend.AllocateBuffer(numNodes * features);
            backend.Fill(outputBuffer, 0.0f, numNodes * features);

            // Execute scatter add
            backend.ScatterAddEdges(
                nodeFeatures.Buffer,
                srcBuffer.Buffer,
                tgtBuffer.Buffer,
                edgeBuffer?.Buffer,
                outputBuffer,
                numNodes, numEdges, features);

            return new GpuTensor<T>(backend, outputBuffer, [numNodes, features],
                GpuTensorRole.Activation, ownsBuffer: true);
        }
        finally
        {
            edgeBuffer?.Dispose();
        }
    }

    /// <summary>
    /// Creates a CSR GPU tensor from edge indices for graph operations.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="sourceIndices">Source node indices for each edge.</param>
    /// <param name="targetIndices">Target node indices for each edge.</param>
    /// <param name="values">Edge values (weights). If null, uses 1.0 for all edges.</param>
    /// <param name="numNodes">Number of nodes in the graph.</param>
    /// <returns>CSR GPU tensor representing the adjacency matrix.</returns>
    public CsrGpuTensor<T> CreateCsrFromEdges<T>(
        int[] sourceIndices,
        int[] targetIndices,
        float[]? values,
        int numNodes)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for CreateCsrFromEdges");

        return CsrGpuTensorFactory.FromEdgeIndices<T>(backend, sourceIndices, targetIndices, values, numNodes);
    }

    /// <summary>
    /// Creates a CSR GPU tensor from a dense tensor by extracting non-zero elements.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="denseTensor">Dense tensor to convert (must be 2D).</param>
    /// <param name="threshold">Values with absolute value below this are treated as zero.</param>
    /// <returns>CSR GPU tensor.</returns>
    public CsrGpuTensor<T> CreateCsrFromDense<T>(Tensor<T> denseTensor, float threshold = 1e-6f)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for CreateCsrFromDense");

        return CsrGpuTensorFactory.FromDenseTensor(backend, denseTensor, threshold);
    }

    #region Element-wise Operations (GPU)

    public IGpuTensor<T> ExpGpu<T>(IGpuTensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ExpGpu");

        int size = input.ElementCount;
        var outputBuffer = backend.AllocateBuffer(size);
        backend.Exp(input.Buffer, outputBuffer, size);

        return new GpuTensor<T>(backend, outputBuffer, input.Shape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    public IGpuTensor<T> SubtractGpu<T>(IGpuTensor<T> a, IGpuTensor<T> b)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SubtractGpu");

        int size = a.ElementCount;
        var outputBuffer = backend.AllocateBuffer(size);
        backend.Subtract(a.Buffer, b.Buffer, outputBuffer, size);

        return new GpuTensor<T>(backend, outputBuffer, a.Shape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    public IGpuTensor<T> BroadcastMultiplyRowGpu<T>(IGpuTensor<T> input, IGpuTensor<T> weights)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for BroadcastMultiplyRowGpu");

        int outerSize = input.Shape[0];
        int innerSize = input.ElementCount / outerSize;

        var outputBuffer = backend.AllocateBuffer(input.ElementCount);
        backend.BroadcastMultiplyLastAxis(input.Buffer, weights.Buffer, outputBuffer, outerSize, innerSize);

        return new GpuTensor<T>(backend, outputBuffer, input.Shape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    public IGpuTensor<T> SinGpu<T>(IGpuTensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SinGpu");

        int size = input.ElementCount;
        var outputBuffer = backend.AllocateBuffer(size);
        backend.Sin(input.Buffer, outputBuffer, size);

        return new GpuTensor<T>(backend, outputBuffer, input.Shape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    public IGpuTensor<T> CosGpu<T>(IGpuTensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for CosGpu");

        int size = input.ElementCount;
        var outputBuffer = backend.AllocateBuffer(size);
        backend.Cos(input.Buffer, outputBuffer, size);

        return new GpuTensor<T>(backend, outputBuffer, input.Shape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    public IGpuTensor<T> GreaterThanScalarGpu<T>(IGpuTensor<T> input, float scalar)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for GreaterThanScalarGpu");

        int size = input.ElementCount;
        var outputBuffer = backend.AllocateBuffer(size);

        using var scalarBuffer = backend.AllocateBuffer(size);
        backend.Fill(scalarBuffer, scalar, size);

        backend.GreaterThan(input.Buffer, scalarBuffer, outputBuffer, size);

        return new GpuTensor<T>(backend, outputBuffer, input.Shape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    public IGpuTensor<T> ConcatGpu<T>(IGpuTensor<T>[] inputs, int axis)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ConcatGpu");

        if (inputs.Length == 0) throw new ArgumentException("No inputs to concatenate");

        var input0 = inputs[0];
        int rank = input0.Shape.Length;
        int actualAxis = axis < 0 ? rank + axis : axis;

        int[] outputShape = input0.Shape.ToArray();
        outputShape[actualAxis] = 0;
        foreach (var input in inputs)
        {
            outputShape[actualAxis] += input.Shape[actualAxis];
        }

        // 1. Move concatenation axis to last dimension via permutation if needed
        bool needsPermute = actualAxis != rank - 1;
        int[]? permutation = null;
        int[]? invPermutation = null;
        IGpuTensor<T>[] processedInputs = inputs;

        if (needsPermute)
        {
            permutation = new int[rank];
            invPermutation = new int[rank];
            int j = 0;
            for (int i = 0; i < rank; i++)
            {
                if (i != actualAxis) permutation[j++] = i;
            }
            permutation[rank - 1] = actualAxis;
            for (int i = 0; i < rank; i++) invPermutation[permutation[i]] = i;

            processedInputs = new IGpuTensor<T>[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                processedInputs[i] = PermuteGpu(inputs[i], permutation);
            }
        }

        // 2. Flatten to 2D [Outer, AxisDim] for strided copy
        long outerSize = 1;
        for (int i = 0; i < rank - 1; i++)
            outerSize *= (needsPermute ? inputs[0].Shape[permutation![i]] : inputs[0].Shape[i]);

        int totalAxisDim = outputShape[actualAxis];
        int totalSize = (int)(outerSize * totalAxisDim);
        var outputBuffer = backend.AllocateBuffer(totalSize);

        // 3. Copy inputs into concatenated buffer at specific offsets
        int currentOffset = 0;
        foreach (var input in processedInputs)
        {
            int axisDim = input.Shape[rank - 1];
            backend.Copy2DStrided(input.Buffer, outputBuffer, (int)outerSize, axisDim, totalAxisDim, currentOffset);
            currentOffset += axisDim;
        }

        // 4. Construct output tensor and restore original axis order if permuted
        int[] tempShape = new int[rank];
        if (needsPermute)
        {
            for (int i = 0; i < rank - 1; i++) tempShape[i] = outputShape[permutation![i]];
            tempShape[rank - 1] = totalAxisDim;
        }
        else
        {
            Array.Copy(outputShape, tempShape, rank);
        }

        var result = new GpuTensor<T>(backend, outputBuffer, tempShape, GpuTensorRole.Activation, ownsBuffer: true);

        if (needsPermute)
        {
            var permutedResult = PermuteGpu(result, invPermutation!);
            result.Dispose();
            result = (GpuTensor<T>)permutedResult;

            foreach (var pInput in processedInputs) pInput.Dispose();
        }

        return result;
    }

    #endregion

    public IGpuTensor<T> ArgMaxAxisGpu<T>(IGpuTensor<T> input, int axis)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ArgMaxAxisGpu");

        // Similar logic to ReduceAxisGpu for arbitrary axis
        // For CRF, axis is usually 1 (after reshape).
        // Let's implement generic axis handling via Permute if needed.

        var inputShape = input.Shape;
        int inputRank = inputShape.Length;
        int outerSize = 1;
        int reduceSize = inputShape[axis];

        // If axis is last, optimal.
        // If not, Permute.
        IGpuTensor<T> processedInput = input;
        bool needsPermute = axis != inputRank - 1;

        if (needsPermute)
        {
            var perm = new int[inputRank];
            int j = 0;
            for (int i = 0; i < inputRank; i++)
                if (i != axis) perm[j++] = i;
            perm[inputRank - 1] = axis;
            processedInput = PermuteGpu(input, perm);
        }

        // Calculate outer size (product of all dims except axis)
        outerSize = processedInput.ElementCount / reduceSize;

        var outputBuffer = backend.AllocateBuffer(outerSize);
        backend.ArgMaxAxis(processedInput.Buffer, outputBuffer, outerSize, reduceSize);

        if (needsPermute)
        {
            processedInput.Dispose();
        }

        // Output shape is input shape with axis removed (or set to 1? ArgMax usually reduces rank).
        // Let's keep rank for compatibility with Torch-like ArgMax, or reduce.
        // ReduceAxisGpu kept dims optionally.
        // For CRF Viterbi, we want [B, C] from [B, C, C].
        // Output shape construction:
        var outputShapeList = new List<int>();
        for (int i = 0; i < inputRank; i++)
        {
            if (i != axis) outputShapeList.Add(inputShape[i]);
        }
        if (outputShapeList.Count == 0) outputShapeList.Add(1);

        return new GpuTensor<T>(backend, outputBuffer, outputShapeList.ToArray(), GpuTensorRole.Activation, ownsBuffer: true);
    }

    public IGpuTensor<T> MaxAxisGpu<T>(IGpuTensor<T> input, int axis)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for MaxAxisGpu");

        var inputShape = input.Shape;
        int inputRank = inputShape.Length;
        int outerSize = 1;
        int reduceSize = inputShape[axis];

        IGpuTensor<T> processedInput = input;
        bool needsPermute = axis != inputRank - 1;

        if (needsPermute)
        {
            var perm = new int[inputRank];
            int j = 0;
            for (int i = 0; i < inputRank; i++)
                if (i != axis) perm[j++] = i;
            perm[inputRank - 1] = axis;
            processedInput = PermuteGpu(input, perm);
        }

        outerSize = processedInput.ElementCount / reduceSize;
        var outputBuffer = backend.AllocateBuffer(outerSize);
        backend.MaxAxis(processedInput.Buffer, outputBuffer, outerSize, reduceSize);

        if (needsPermute) processedInput.Dispose();

        var outputShapeList = new List<int>();
        for (int i = 0; i < inputRank; i++)
            if (i != axis) outputShapeList.Add(inputShape[i]);
        if (outputShapeList.Count == 0) outputShapeList.Add(1);

        return new GpuTensor<T>(backend, outputBuffer, outputShapeList.ToArray(), GpuTensorRole.Activation, ownsBuffer: true);
    }

    public IGpuTensor<T> BroadcastAddGpu<T>(IGpuTensor<T> a, IGpuTensor<T> b)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for BroadcastAddGpu");

        // Support full broadcasting logic like NumPy?
        // Or specific patterns?
        // Implementing full broadcast requires analyzing shapes and tiling.
        // For CRF: [B, C, 1] + [B, C, C] (tiled from [C, C]).
        // Actually, if we use TileAxisGpu manually, we just need element-wise AddGpu.
        // So we don't strictly need BroadcastAddGpu if the caller tiles.
        // But a helper is nice.
        // Let's implement generic AddGpu that handles simple broadcasts or falls back to Tile+Add.
        // But for now, let's expose explicit Tile ops and let caller handle shape matching.
        // It's more predictable.
        // So I will just implement AddGpu (which I assume exists? No, I saw 'Add' in backend).
        // I need to expose AddGpu (element-wise).

        // Wait, AddGpu probably exists?
        // I'll check "Element-wise Operations (GPU)" region.
        // I see SinGpu, CosGpu... AddGpu might be missing from public API in this file?
        // I will add AddGpu just in case.

        int size = a.ElementCount;
        if (size != b.ElementCount)
            throw new ArgumentException($"AddGpu requires matching sizes: {size} vs {b.ElementCount}");

        var outputBuffer = backend.AllocateBuffer(size);
        backend.Add(a.Buffer, b.Buffer, outputBuffer, size);
        return new GpuTensor<T>(backend, outputBuffer, a.Shape, GpuTensorRole.Activation, ownsBuffer: true);
    }



    #endregion

    #region Random Number Generation

    /// <summary>
    /// Generates a GPU-resident tensor with uniformly distributed random numbers.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="min">Minimum value (inclusive).</param>
    /// <param name="max">Maximum value (exclusive).</param>
    /// <param name="seed">Random seed.</param>
    /// <returns>A GPU-resident tensor.</returns>
    public IGpuTensor<T> RandomUniformGpu<T>(int[] shape, float min, float max, ulong seed)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for RandomUniformGpu");

        int size = 1;
        foreach (var dim in shape) size *= dim;

        var outputBuffer = backend.AllocateBuffer(size);
        backend.GenerateRandomUniform(outputBuffer, size, min, max, seed);

        return new GpuTensor<T>(backend, outputBuffer, shape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// Generates a GPU-resident tensor with normally distributed (Gaussian) random numbers.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="mean">Mean of the distribution.</param>
    /// <param name="stdDev">Standard deviation of the distribution.</param>
    /// <param name="seed">Random seed.</param>
    /// <returns>A GPU-resident tensor.</returns>
    public IGpuTensor<T> RandomNormalGpu<T>(int[] shape, float mean, float stdDev, ulong seed)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for RandomNormalGpu");

        int size = 1;
        foreach (var dim in shape) size *= dim;

        var outputBuffer = backend.AllocateBuffer(size);
        backend.GenerateRandomNormal(outputBuffer, size, mean, stdDev, seed);

        return new GpuTensor<T>(backend, outputBuffer, shape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    public IGpuTensor<T> ReshapeGpu<T>(IGpuTensor<T> input, int[] newShape)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ReshapeGpu");

        // Validate size
        int newSize = 1;
        foreach (var dim in newShape) newSize *= dim;

        if (newSize != input.ElementCount)
            throw new ArgumentException($"Reshape total size mismatch: {input.ElementCount} vs {newSize}");

        // Check input type
        if (input is GpuTensor<T> gpuTensor)
        {
            return gpuTensor.CreateView(0, newShape);
        }

        // Fallback: create wrapper
        return new GpuTensor<T>(backend, input.Buffer, newShape, GpuTensorRole.Activation, ownsBuffer: false);
    }

    #endregion

    #region Optimizer Operations

    public void SgdMomentumUpdateGpu<T>(Tensor<T> param, Tensor<T> gradient, Tensor<T> velocity, float learningRate, float momentum, float weightDecay)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SgdMomentumUpdateGpu");

        using var paramBuffer = GetOrAllocateBuffer(backend, param.Data);
        using var gradBuffer = GetOrAllocateBuffer(backend, gradient.Data);
        using var velocityBuffer = GetOrAllocateBuffer(backend, velocity.Data);

        backend.SgdMomentumUpdate(paramBuffer.Buffer, gradBuffer.Buffer, velocityBuffer.Buffer,
            learningRate, momentum, weightDecay, param.Length);
    }

    #endregion

    #region Specialized Layer Operations

    public IGpuTensor<T> RbfKernelGpu<T>(IGpuTensor<T> input, Tensor<T> centers, Tensor<T> widths)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for RbfKernelGpu");

        int batch = input.Shape[0];
        int inputDim = input.Shape.Length > 1 ? input.Shape[1] : 1;
        int numCenters = centers.Shape[0];

        // Compute epsilons on CPU (small calculation)
        // epsilon = 1 / (2 * width^2)
        var ops = MathHelper.GetNumericOperations<T>();
        var epsilons = new Tensor<T>(widths.Shape);
        var two = ops.FromDouble(2.0);
        for (int i = 0; i < numCenters; i++)
        {
            var w = widths[i];
            epsilons[i] = ops.Divide(ops.One, ops.Multiply(two, ops.Multiply(w, w)));
        }

        // Upload persistent tensors (using cache if registered)
        using var centersBuffer = GetOrAllocateBuffer(backend, centers.Data);
        using var epsilonsBuffer = GetOrAllocateBuffer(backend, epsilons.Data);

        var outputBuffer = backend.AllocateBuffer(batch * numCenters);

        backend.RbfForward(
            input.Buffer,
            centersBuffer.Buffer,
            epsilonsBuffer.Buffer,
            outputBuffer,
            batch, numCenters, inputDim);

        return new GpuTensor<T>(backend, outputBuffer, [batch, numCenters], GpuTensorRole.Activation, ownsBuffer: true);
    }

    public void UpdateTracesGpu<T>(IGpuTensor<T> traces, IGpuTensor<T> spikes, IGpuTensor<T> input, float decay, float threshold)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for UpdateTracesGpu");

        backend.UpdateTraces(traces.Buffer, spikes.Buffer, input.Buffer, decay, threshold, input.ElementCount);
    }

    public void StdpUpdateGpu<T>(
        Tensor<T> weights,
        IGpuTensor<T> preTrace,
        IGpuTensor<T> postTrace,
        IGpuTensor<T> preSpike,
        IGpuTensor<T> postSpike,
        double ltpRate,
        double ltdRate,
        double homeostasisRate,
        double minWeight,
        double maxWeight)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for StdpUpdateGpu");

        int numPre = weights.Shape[0];
        int numPost = weights.Shape[1]; // Correct shape for fully connected?
                                        // SynapticPlasticityLayer weights are [size, size] so Pre x Post.

        // Weights are modified in-place on GPU, then need to be invalidating CPU cache or vice-versa.
        // We assume weights are persistent GPU tensors.
        // But here we accept Tensor<T> weights.
        // We should check if it's cached.

        // This operation modifies weights in-place on GPU.
        // If we only have CPU weights, we must upload, modify, download.
        // But for training loop, weights should stay on GPU.
        // We use RegisterPersistentTensor mechanism.

        // Get buffer without allocating new one if possible, but we need writable access.
        // GetOrAllocateBuffer returns OwnedBuffer which might be cached.
        // If cached, we modify it in place.
        // We must ensure CPU side knows it's dirty if we download later.

        using var weightsBuffer = GetOrCacheWeightBuffer(backend, weights.Data, PersistentTensorRole.Weights);
        backend.StdpUpdate(
            weightsBuffer.Buffer,
            preTrace.Buffer,
            postTrace.Buffer,
            preSpike.Buffer,
            postSpike.Buffer,
            (float)ltpRate, (float)ltdRate, (float)homeostasisRate,
            (float)minWeight, (float)maxWeight,
            numPre, numPost);

        // Mark as modified on GPU so next Download syncs it
        // BUT our current system doesn't track dirty state for download.
        // We assume explicit download or automatic handling.
        // For now, let's assume the user will keep using GPU path.
        // Ideally we should download if we are done, but for training loop we keep it there.
        // To be safe, we can download back to CPU tensor immediately if this isn't fully persistent-managed.
        // Or we rely on the fact that weights.Data is the key.
        // If we modify GPU buffer, CPU array is stale.
        // DirectGpuTensorEngine doesn't have "MarkDirty".
        // We will download to keep consistency for now, or assume layer manages it.
        // Given UpdateParameters returns void, we should update the CPU tensor too.

        backend.DownloadBuffer(weightsBuffer.Buffer, DirectGpuEngine.ToFloatArray(weights.Data));
    }

    #endregion

    #region GPU-Resident Linear Layer Backward Operations

    /// <summary>
    /// GPU-resident 2D matrix transpose.
    /// Transposes input tensor from [rows, cols] to [cols, rows].
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident 2D input tensor [rows, cols].</param>
    /// <returns>GPU-resident transposed tensor [cols, rows].</returns>
    public IGpuTensor<T> TransposeGpu<T>(IGpuTensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for TransposeGpu");

        if (input.Shape.Length != 2)
            throw new ArgumentException("TransposeGpu requires 2D tensor [rows, cols]");

        int rows = input.Shape[0];
        int cols = input.Shape[1];

        var outputBuffer = backend.AllocateBuffer(rows * cols);
        backend.Transpose(input.Buffer, outputBuffer, rows, cols);

        return new GpuTensor<T>(backend, outputBuffer, [cols, rows],
            GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident matrix multiplication between two GPU tensors.
    /// Computes: C = A @ B where A is [M, K] and B is [K, N].
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="A">GPU-resident tensor A [M, K].</param>
    /// <param name="B">GPU-resident tensor B [K, N].</param>
    /// <returns>GPU-resident output tensor C [M, N].</returns>
    public IGpuTensor<T> MatMulGpuTensors<T>(IGpuTensor<T> A, IGpuTensor<T> B)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for MatMulGpuTensors");

        if (A.Shape.Length != 2 || B.Shape.Length != 2)
            throw new ArgumentException("MatMulGpuTensors requires 2D tensors");

        int M = A.Shape[0];
        int K = A.Shape[1];
        int N = B.Shape[1];

        if (B.Shape[0] != K)
            throw new ArgumentException($"Dimension mismatch: A has {K} columns, but B has {B.Shape[0]} rows");

        var resultBuffer = backend.MatMul(A.Buffer, B.Buffer, M, N, K);

        return new GpuTensor<T>(backend, resultBuffer, [M, N],
            GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident ReLU backward operation.
    /// Computes: gradInput = gradOutput * (input > 0 ? 1 : 0)
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient.</param>
    /// <param name="input">GPU-resident input from forward pass (pre-activation).</param>
    /// <returns>GPU-resident input gradient.</returns>
    public IGpuTensor<T> ReluBackwardGpu<T>(IGpuTensor<T> gradOutput, IGpuTensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ReluBackwardGpu");

        int size = gradOutput.ElementCount;
        var outputBuffer = backend.AllocateBuffer(size);

        backend.ReluBackward(gradOutput.Buffer, input.Buffer, outputBuffer, size);

        return new GpuTensor<T>(backend, outputBuffer, gradOutput.Shape,
            GpuTensorRole.Gradient, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident Sigmoid backward operation.
    /// Computes: gradInput = gradOutput * sigmoid(x) * (1 - sigmoid(x))
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient.</param>
    /// <param name="output">GPU-resident output from forward pass (post-activation sigmoid output).</param>
    /// <returns>GPU-resident input gradient.</returns>
    public IGpuTensor<T> SigmoidBackwardGpu<T>(IGpuTensor<T> gradOutput, IGpuTensor<T> output)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SigmoidBackwardGpu");

        int size = gradOutput.ElementCount;
        var outputBuffer = backend.AllocateBuffer(size);

        backend.SigmoidBackward(gradOutput.Buffer, output.Buffer, outputBuffer, size);

        return new GpuTensor<T>(backend, outputBuffer, gradOutput.Shape,
            GpuTensorRole.Gradient, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident Tanh backward operation.
    /// Computes: gradInput = gradOutput * (1 - tanh(x)^2)
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient.</param>
    /// <param name="output">GPU-resident output from forward pass (post-activation tanh output).</param>
    /// <returns>GPU-resident input gradient.</returns>
    public IGpuTensor<T> TanhBackwardGpu<T>(IGpuTensor<T> gradOutput, IGpuTensor<T> output)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for TanhBackwardGpu");

        int size = gradOutput.ElementCount;
        var outputBuffer = backend.AllocateBuffer(size);

        backend.TanhBackward(gradOutput.Buffer, output.Buffer, outputBuffer, size);

        return new GpuTensor<T>(backend, outputBuffer, gradOutput.Shape,
            GpuTensorRole.Gradient, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident LeakyReLU backward operation.
    /// Computes: gradInput = gradOutput * (input > 0 ? 1 : alpha)
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient.</param>
    /// <param name="input">GPU-resident input from forward pass (pre-activation).</param>
    /// <param name="alpha">Negative slope parameter.</param>
    /// <returns>GPU-resident input gradient.</returns>
    public IGpuTensor<T> LeakyReluBackwardGpu<T>(IGpuTensor<T> gradOutput, IGpuTensor<T> input, float alpha = 0.01f)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for LeakyReluBackwardGpu");

        int size = gradOutput.ElementCount;
        var outputBuffer = backend.AllocateBuffer(size);

        backend.LeakyReluBackward(gradOutput.Buffer, input.Buffer, outputBuffer, alpha, size);

        return new GpuTensor<T>(backend, outputBuffer, gradOutput.Shape,
            GpuTensorRole.Gradient, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident GELU backward operation.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient.</param>
    /// <param name="input">GPU-resident input from forward pass (pre-activation).</param>
    /// <returns>GPU-resident input gradient.</returns>
    public IGpuTensor<T> GeluBackwardGpu<T>(IGpuTensor<T> gradOutput, IGpuTensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for GeluBackwardGpu");

        int size = gradOutput.ElementCount;
        var outputBuffer = backend.AllocateBuffer(size);

        backend.GeluBackward(gradOutput.Buffer, input.Buffer, outputBuffer, size);

        return new GpuTensor<T>(backend, outputBuffer, gradOutput.Shape,
            GpuTensorRole.Gradient, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident Softmax backward operation.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient.</param>
    /// <param name="output">GPU-resident output from forward pass (post-activation softmax output).</param>
    /// <returns>GPU-resident input gradient.</returns>
    public IGpuTensor<T> SoftmaxBackwardGpu<T>(IGpuTensor<T> gradOutput, IGpuTensor<T> output)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SoftmaxBackwardGpu");

        int batchSize = gradOutput.Shape[0];
        int features = gradOutput.Shape.Length > 1 ? gradOutput.Shape[1] : 1;
        int size = gradOutput.ElementCount;
        var outputBuffer = backend.AllocateBuffer(size);

        backend.SoftmaxBackward(gradOutput.Buffer, output.Buffer, outputBuffer, batchSize, features);

        return new GpuTensor<T>(backend, outputBuffer, gradOutput.Shape,
            GpuTensorRole.Gradient, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident Swish backward operation.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient.</param>
    /// <param name="input">GPU-resident input from forward pass (pre-activation).</param>
    /// <returns>GPU-resident input gradient.</returns>
    public IGpuTensor<T> SwishBackwardGpu<T>(IGpuTensor<T> gradOutput, IGpuTensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SwishBackwardGpu");

        int size = gradOutput.ElementCount;
        var outputBuffer = backend.AllocateBuffer(size);

        backend.SwishBackward(gradOutput.Buffer, input.Buffer, outputBuffer, size);

        return new GpuTensor<T>(backend, outputBuffer, gradOutput.Shape,
            GpuTensorRole.Gradient, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident ELU backward operation.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient.</param>
    /// <param name="input">GPU-resident input from forward pass (pre-activation).</param>
    /// <param name="output">GPU-resident output from forward pass (post-activation).</param>
    /// <param name="alpha">ELU alpha parameter.</param>
    /// <returns>GPU-resident input gradient.</returns>
    public IGpuTensor<T> EluBackwardGpu<T>(IGpuTensor<T> gradOutput, IGpuTensor<T> input, IGpuTensor<T> output, float alpha = 1.0f)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for EluBackwardGpu");

        int size = gradOutput.ElementCount;
        var outputBuffer = backend.AllocateBuffer(size);

        backend.EluBackward(gradOutput.Buffer, input.Buffer, output.Buffer, outputBuffer, alpha, size);

        return new GpuTensor<T>(backend, outputBuffer, gradOutput.Shape,
            GpuTensorRole.Gradient, ownsBuffer: true);
    }

    #endregion

    #region GPU-Resident BatchNorm Backward Operations

    /// <summary>
    /// GPU-resident batch normalization backward pass.
    /// Computes gradients for input, gamma (scale), and beta (shift).
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient [B, C, H, W] or [B, C].</param>
    /// <param name="input">GPU-resident input from forward pass.</param>
    /// <param name="gamma">Scale parameter [C].</param>
    /// <param name="saveMean">Running mean saved from forward pass [C].</param>
    /// <param name="saveInvVar">Running inverse variance saved from forward pass [C].</param>
    /// <param name="epsilon">Epsilon for numerical stability.</param>
    /// <returns>Tuple of (gradInput, gradGamma, gradBeta).</returns>
    public (IGpuTensor<T> gradInput, IGpuTensor<T> gradGamma, IGpuTensor<T> gradBeta) BatchNormBackwardGpu<T>(
        IGpuTensor<T> gradOutput,
        IGpuTensor<T> input,
        Tensor<T> gamma,
        IGpuTensor<T> saveMean,
        IGpuTensor<T> saveInvVar,
        float epsilon)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for BatchNormBackwardGpu");

        // Determine dimensions
        int batch, channels, spatialSize;
        if (gradOutput.Shape.Length == 2)
        {
            // [B, C] - fully connected
            batch = gradOutput.Shape[0];
            channels = gradOutput.Shape[1];
            spatialSize = 1;
        }
        else if (gradOutput.Shape.Length == 4)
        {
            // [B, C, H, W] - convolutional
            batch = gradOutput.Shape[0];
            channels = gradOutput.Shape[1];
            spatialSize = gradOutput.Shape[2] * gradOutput.Shape[3];
        }
        else
        {
            throw new ArgumentException($"BatchNormBackwardGpu expects 2D [B, C] or 4D [B, C, H, W] tensor, got {gradOutput.Shape.Length}D");
        }

        // Validate parameter lengths match channels to prevent out-of-bounds kernel access
        if (gamma.Length != channels)
            throw new ArgumentException($"gamma.Length ({gamma.Length}) must match channels ({channels}).", nameof(gamma));
        if (saveMean.ElementCount != channels)
            throw new ArgumentException($"saveMean.ElementCount ({saveMean.ElementCount}) must match channels ({channels}).", nameof(saveMean));
        if (saveInvVar.ElementCount != channels)
            throw new ArgumentException($"saveInvVar.ElementCount ({saveInvVar.ElementCount}) must match channels ({channels}).", nameof(saveInvVar));

        // Allocate output buffers
        var gradInputBuffer = backend.AllocateBuffer(gradOutput.ElementCount);
        var gradGammaBuffer = backend.AllocateBuffer(channels);
        var gradBetaBuffer = backend.AllocateBuffer(channels);

        // Upload gamma
        using var gammaBuffer = GetOrCacheWeightBuffer(backend, gamma.Data, PersistentTensorRole.Weights);

        backend.BatchNormBackward(
            gradOutput.Buffer, input.Buffer, gammaBuffer.Buffer,
            saveMean.Buffer, saveInvVar.Buffer,
            gradInputBuffer, gradGammaBuffer, gradBetaBuffer,
            batch, channels, spatialSize, epsilon);

        return (
            new GpuTensor<T>(backend, gradInputBuffer, gradOutput.Shape, GpuTensorRole.Gradient, ownsBuffer: true),
            new GpuTensor<T>(backend, gradGammaBuffer, [channels], GpuTensorRole.Gradient, ownsBuffer: true),
            new GpuTensor<T>(backend, gradBetaBuffer, [channels], GpuTensorRole.Gradient, ownsBuffer: true)
        );
    }

    #endregion

    #region GPU-Resident Conv2D Backward Operations

    /// <summary>
    /// GPU-resident Conv2D backward pass for input gradients.
    /// Computes gradient with respect to input.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient [B, outC, outH, outW].</param>
    /// <param name="kernel">Kernel weights [outC, inC, kH, kW].</param>
    /// <param name="inputShape">Shape of the original input [B, inC, inH, inW].</param>
    /// <param name="stride">Convolution stride [strideH, strideW].</param>
    /// <param name="padding">Padding [padH, padW].</param>
    /// <param name="dilation">Dilation [dilationH, dilationW].</param>
    /// <returns>GPU-resident gradient with respect to input.</returns>
    public IGpuTensor<T> Conv2DBackwardInputGpu<T>(
        IGpuTensor<T> gradOutput,
        Tensor<T> kernel,
        int[] inputShape,
        int[] stride,
        int[] padding,
        int[] dilation)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for Conv2DBackwardInputGpu");

        // Validate shape lengths to prevent index out of bounds
        if (inputShape.Length != 4)
            throw new ArgumentException($"inputShape must be 4D [B, inC, inH, inW], got {inputShape.Length}D.", nameof(inputShape));
        if (kernel.Rank != 4)
            throw new ArgumentException($"kernel must be 4D [outC, inC, kH, kW], got {kernel.Rank}D.", nameof(kernel));
        if (gradOutput.Shape.Length != 4)
            throw new ArgumentException($"gradOutput must be 4D [B, outC, outH, outW], got {gradOutput.Shape.Length}D.", nameof(gradOutput));

        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int inHeight = inputShape[2];
        int inWidth = inputShape[3];

        int outChannels = kernel.Shape[0];
        int kernelH = kernel.Shape[2];
        int kernelW = kernel.Shape[3];

        int outHeight = gradOutput.Shape[2];
        int outWidth = gradOutput.Shape[3];

        // Allocate output buffer for input gradient
        var gradInputBuffer = backend.AllocateBuffer(batch * inChannels * inHeight * inWidth);

        // Upload kernel
        using var kernelBuffer = GetOrCacheWeightBuffer(backend, kernel.Data, PersistentTensorRole.Weights);

        backend.Conv2DBackwardInput(
            gradOutput.Buffer, kernelBuffer.Buffer, gradInputBuffer,
            batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth,
            kernelH, kernelW,
            stride[0], stride[1], padding[0], padding[1],
            dilation[0], dilation[1]);

        return new GpuTensor<T>(backend, gradInputBuffer, inputShape,
            GpuTensorRole.Gradient, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident Conv2D backward pass for kernel gradients.
    /// Computes gradient with respect to kernel weights.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient [B, outC, outH, outW].</param>
    /// <param name="input">GPU-resident input from forward pass [B, inC, inH, inW].</param>
    /// <param name="kernelShape">Shape of the kernel [outC, inC, kH, kW].</param>
    /// <param name="stride">Convolution stride [strideH, strideW].</param>
    /// <param name="padding">Padding [padH, padW].</param>
    /// <param name="dilation">Dilation [dilationH, dilationW].</param>
    /// <returns>GPU-resident gradient with respect to kernels.</returns>
    public IGpuTensor<T> Conv2DBackwardKernelGpu<T>(
        IGpuTensor<T> gradOutput,
        IGpuTensor<T> input,
        int[] kernelShape,
        int[] stride,
        int[] padding,
        int[] dilation)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for Conv2DBackwardKernelGpu");

        // Validate shape lengths to prevent index out of bounds
        if (input.Shape.Length != 4)
            throw new ArgumentException($"input must be 4D [B, inC, inH, inW], got {input.Shape.Length}D.", nameof(input));
        if (kernelShape.Length != 4)
            throw new ArgumentException($"kernelShape must be 4D [outC, inC, kH, kW], got {kernelShape.Length}D.", nameof(kernelShape));
        if (gradOutput.Shape.Length != 4)
            throw new ArgumentException($"gradOutput must be 4D [B, outC, outH, outW], got {gradOutput.Shape.Length}D.", nameof(gradOutput));

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int outChannels = kernelShape[0];
        int kernelH = kernelShape[2];
        int kernelW = kernelShape[3];

        int outHeight = gradOutput.Shape[2];
        int outWidth = gradOutput.Shape[3];

        // Allocate output buffer for kernel gradient
        int kernelSize = kernelShape[0] * kernelShape[1] * kernelShape[2] * kernelShape[3];
        var gradKernelBuffer = backend.AllocateBuffer(kernelSize);

        backend.Conv2DBackwardKernel(
            input.Buffer, gradOutput.Buffer, gradKernelBuffer,
            batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth,
            kernelH, kernelW,
            stride[0], stride[1], padding[0], padding[1],
            dilation[0], dilation[1]);

        return new GpuTensor<T>(backend, gradKernelBuffer, kernelShape,
            GpuTensorRole.Gradient, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident Conv2D backward pass for bias gradients.
    /// Computes gradient with respect to bias by summing over batch and spatial dimensions.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient [B, outC, outH, outW].</param>
    /// <returns>GPU-resident gradient with respect to bias [outC].</returns>
    public IGpuTensor<T> Conv2DBackwardBiasGpu<T>(IGpuTensor<T> gradOutput)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for Conv2DBackwardBiasGpu");

        // Validate shape length to prevent index out of bounds
        if (gradOutput.Shape.Length != 4)
            throw new ArgumentException($"gradOutput must be 4D [B, outC, outH, outW], got {gradOutput.Shape.Length}D.", nameof(gradOutput));

        int batch = gradOutput.Shape[0];
        int outChannels = gradOutput.Shape[1];
        int outHeight = gradOutput.Shape[2];
        int outWidth = gradOutput.Shape[3];

        // Bias gradient = sum over batch and spatial dimensions
        // gradOutput: [B, outC, outH, outW] -> gradBias: [outC]
        // Sum over batch and spatial dimensions for each channel
        // Note: For large tensors, this could be optimized with GPU reduction kernels
        // (e.g., reshape to [B*H*W, C] and use SumAxis). Current implementation downloads
        // to CPU for simplicity and correctness.
        float[] gradOutData = backend.DownloadBuffer(gradOutput.Buffer);
        float[] gradBiasData = new float[outChannels];

        int spatialSize = outHeight * outWidth;
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < outChannels; c++)
            {
                int baseIdx = b * outChannels * spatialSize + c * spatialSize;
                for (int s = 0; s < spatialSize; s++)
                {
                    gradBiasData[c] += gradOutData[baseIdx + s];
                }
            }
        }

        // Create new GPU buffer with the computed bias gradients
        var gradBiasBuffer = backend.AllocateBuffer(gradBiasData);

        return new GpuTensor<T>(backend, gradBiasBuffer, [outChannels],
            GpuTensorRole.Gradient, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident ConvTranspose2D backward pass for input gradients.
    /// Computes gradient with respect to input.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient [B, outC, outH, outW].</param>
    /// <param name="kernel">Kernel weights [inC, outC, kH, kW].</param>
    /// <param name="inputShape">Shape of the input [B, inC, inH, inW].</param>
    /// <param name="stride">Convolution stride [strideH, strideW].</param>
    /// <param name="padding">Padding [padH, padW].</param>
    /// <param name="outputPadding">Output padding [outputPadH, outputPadW].</param>
    /// <returns>GPU-resident gradient with respect to input.</returns>
    public IGpuTensor<T> ConvTranspose2DBackwardInputGpu<T>(
        IGpuTensor<T> gradOutput,
        Tensor<T> kernel,
        int[] inputShape,
        int[] stride,
        int[] padding,
        int[] outputPadding)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ConvTranspose2DBackwardInputGpu");

        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int inHeight = inputShape[2];
        int inWidth = inputShape[3];

        // For transposed conv: kernel shape is [inC, outC, kH, kW]
        int outChannels = kernel.Shape[1];
        int kernelH = kernel.Shape[2];
        int kernelW = kernel.Shape[3];

        int outHeight = gradOutput.Shape[2];
        int outWidth = gradOutput.Shape[3];

        // Allocate output buffer for input gradient
        var gradInputBuffer = backend.AllocateBuffer(batch * inChannels * inHeight * inWidth);

        // Upload kernel
        using var kernelBuffer = GetOrCacheWeightBuffer(backend, kernel.Data, PersistentTensorRole.Weights);

        backend.ConvTranspose2DBackwardInput(
            gradOutput.Buffer, kernelBuffer.Buffer, gradInputBuffer,
            batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth,
            kernelH, kernelW,
            stride[0], stride[1], padding[0], padding[1],
            outputPadding[0], outputPadding[1]);

        return new GpuTensor<T>(backend, gradInputBuffer, inputShape,
            GpuTensorRole.Gradient, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident ConvTranspose2D backward pass for kernel gradients.
    /// Computes gradient with respect to kernel weights.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient [B, outC, outH, outW].</param>
    /// <param name="input">GPU-resident input from forward pass [B, inC, inH, inW].</param>
    /// <param name="kernelShape">Shape of the kernel [inC, outC, kH, kW].</param>
    /// <param name="stride">Convolution stride [strideH, strideW].</param>
    /// <param name="padding">Padding [padH, padW].</param>
    /// <param name="outputPadding">Output padding [outputPadH, outputPadW].</param>
    /// <returns>GPU-resident gradient with respect to kernels.</returns>
    public IGpuTensor<T> ConvTranspose2DBackwardKernelGpu<T>(
        IGpuTensor<T> gradOutput,
        IGpuTensor<T> input,
        int[] kernelShape,
        int[] stride,
        int[] padding,
        int[] outputPadding)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ConvTranspose2DBackwardKernelGpu");

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        // For transposed conv: kernel shape is [inC, outC, kH, kW]
        int outChannels = kernelShape[1];
        int kernelH = kernelShape[2];
        int kernelW = kernelShape[3];

        int outHeight = gradOutput.Shape[2];
        int outWidth = gradOutput.Shape[3];

        // Allocate output buffer for kernel gradient
        int kernelSize = kernelShape[0] * kernelShape[1] * kernelShape[2] * kernelShape[3];
        var gradKernelBuffer = backend.AllocateBuffer(kernelSize);

        backend.ConvTranspose2DBackwardKernel(
            input.Buffer, gradOutput.Buffer, gradKernelBuffer,
            batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth,
            kernelH, kernelW,
            stride[0], stride[1], padding[0], padding[1],
            outputPadding[0], outputPadding[1]);

        return new GpuTensor<T>(backend, gradKernelBuffer, kernelShape,
            GpuTensorRole.Gradient, ownsBuffer: true);
    }

    #endregion

    public void Dispose()
    {
        // Clear activation cache to free GPU memory from cached activations
        ClearActivationCache();

        // Dispose all cached GPU buffers
        foreach (var entry in _persistentBufferCache.Values)
        {
            entry.Dispose();
        }
        _persistentBufferCache.Clear();
        _tensorVersions.Clear();

        if (_ownsDirectGpu)
            _directGpu?.Dispose();
    }
}
