using System.Collections.Concurrent;

namespace AiDotNet.Memory;

/// <summary>
/// A high-performance, thread-safe memory pool for reusing tensors during neural network operations.
/// Reduces memory allocations and garbage collection pressure by pooling tensor buffers.
/// </summary>
/// <typeparam name="T">The numeric element type of tensors in this pool (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The tensor pool maintains buckets of pre-allocated tensors grouped by shape.
/// When a tensor is requested via <see cref="Rent"/>, the pool returns an existing tensor
/// from the appropriate bucket if available, otherwise allocates a new one.
/// </para>
/// <para>
/// When tensors are returned via <see cref="Return"/>, they are cleared and added
/// back to the pool for future reuse, up to the configured memory limits.
/// </para>
/// <para>
/// Basic usage example:
/// <code>
/// using var pool = new TensorPool&lt;float&gt;(maxPoolSizeMB: 256);
///
/// // Rent a tensor from the pool
/// var tensor = pool.Rent(new[] { 32, 784 });
///
/// // Use the tensor for computations...
///
/// // Return it to the pool when done
/// pool.Return(tensor);
/// </code>
/// </para>
/// <para>
/// For automatic lifetime management, use <see cref="RentPooled"/> which returns
/// a <see cref="PooledTensor{T}"/> that automatically returns itself when disposed:
/// <code>
/// using var pooled = pool.RentPooled(new[] { 32, 784 });
/// // Use pooled.Tensor for computations...
/// // Tensor is automatically returned when 'using' block exits
/// </code>
/// </para>
/// </remarks>
public class TensorPool<T> : IDisposable
{
    private readonly ConcurrentDictionary<int, ConcurrentBag<TensorEntry>> _tensorPools;
    private readonly PoolingOptions _options;
    private long _currentMemoryBytes;
    private int _totalPooledTensors;
    private bool _disposed;

    /// <summary>
    /// Gets the current memory usage of all pooled tensors, in bytes.
    /// </summary>
    public long CurrentMemoryBytes => Interlocked.Read(ref _currentMemoryBytes);

    /// <summary>
    /// Gets the current size of the pool in bytes. Alias for <see cref="CurrentMemoryBytes"/>.
    /// </summary>
    public long CurrentPoolSizeBytes => Interlocked.Read(ref _currentMemoryBytes);

    /// <summary>
    /// Gets the maximum allowed memory size for the pool, in bytes.
    /// </summary>
    public long MaxPoolSizeBytes => _options.MaxPoolSizeBytes;

    /// <summary>
    /// Gets the total number of tensors currently held in the pool across all buckets.
    /// </summary>
    public int TotalPooledTensors => _totalPooledTensors;

    /// <summary>
    /// Gets the pooling options configured for this pool.
    /// </summary>
    public PoolingOptions Options => _options;

    /// <summary>
    /// Initializes a new instance of the <see cref="TensorPool{T}"/> class with default options.
    /// Default max pool size is 256 MB.
    /// </summary>
    public TensorPool() : this(new PoolingOptions()) { }

    /// <summary>
    /// Initializes a new instance of the <see cref="TensorPool{T}"/> class with the specified maximum pool size.
    /// </summary>
    /// <param name="maxPoolSizeMB">The maximum memory size for the pool in megabytes.</param>
    public TensorPool(int maxPoolSizeMB) : this(new PoolingOptions { MaxPoolSizeMB = maxPoolSizeMB }) { }

    /// <summary>
    /// Initializes a new instance of the <see cref="TensorPool{T}"/> class with the specified options.
    /// </summary>
    /// <param name="options">The pooling configuration options. If null, default options are used.</param>
    public TensorPool(PoolingOptions options)
    {
        _options = options ?? new PoolingOptions();
        _tensorPools = new ConcurrentDictionary<int, ConcurrentBag<TensorEntry>>();
    }

    /// <summary>
    /// Rents a tensor with the specified shape from the pool.
    /// If a matching tensor is available in the pool, it is returned after being cleared.
    /// Otherwise, a new tensor is allocated.
    /// </summary>
    /// <param name="shape">The shape of the tensor to rent (e.g., [32, 784] for a 2D tensor).</param>
    /// <returns>A tensor with the requested shape, either from the pool or newly allocated.</returns>
    /// <exception cref="ObjectDisposedException">Thrown if the pool has been disposed.</exception>
    /// <exception cref="ArgumentException">Thrown if shape is null, empty, or contains non-positive dimensions.</exception>
    public Tensor<T> Rent(int[] shape)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(TensorPool<T>));

        if (shape == null || shape.Length == 0)
            throw new ArgumentException("Shape must be non-empty.", nameof(shape));

        var totalElements = 1;
        foreach (var dim in shape)
        {
            if (dim <= 0)
                throw new ArgumentException("All dimensions must be positive.", nameof(shape));
            checked { totalElements *= dim; }
        }

        if (!_options.Enabled || totalElements > _options.MaxElementsToPool)
            return new Tensor<T>(shape);

        var key = GetTensorPoolKey(shape);

        if (_tensorPools.TryGetValue(key, out var pool))
        {
            while (pool.TryTake(out var entry))
            {
                UpdateMemoryUsage(-entry.SizeBytes);
                Interlocked.Decrement(ref _totalPooledTensors);

                if (entry.Tensor is not null && ShapeMatches(entry.Tensor.Shape, shape))
                {
                    ClearTensor(entry.Tensor);
                    return entry.Tensor;
                }
            }
        }

        return new Tensor<T>(shape);
    }

    /// <summary>
    /// Returns a tensor to the pool for future reuse.
    /// The tensor is cleared and added to the appropriate shape bucket if memory limits allow.
    /// </summary>
    /// <param name="tensor">The tensor to return to the pool.</param>
    /// <exception cref="ObjectDisposedException">Thrown if the pool has been disposed.</exception>
    /// <remarks>
    /// If the tensor is null, too large for pooling, or the pool is at capacity,
    /// the tensor is not pooled and will be garbage collected normally.
    /// </remarks>
    public void Return(Tensor<T> tensor)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(TensorPool<T>));

        if (tensor == null || tensor.Length > _options.MaxElementsToPool)
            return;

        var sizeBytes = GetTensorSizeBytes(tensor.Length);
        if (!TryReserveMemory(sizeBytes))
            return;

        ClearTensor(tensor);

        var key = GetTensorPoolKey(tensor.Shape);
        var pool = _tensorPools.GetOrAdd(key, _ => new ConcurrentBag<TensorEntry>());

        if (pool.Count < _options.MaxItemsPerBucket)
        {
            pool.Add(new TensorEntry { Tensor = tensor, SizeBytes = sizeBytes });
            Interlocked.Increment(ref _totalPooledTensors);
        }
        else
        {
            UpdateMemoryUsage(-sizeBytes);
        }
    }

    /// <summary>
    /// Rents a tensor wrapped in a <see cref="PooledTensor{T}"/> for automatic pool return on disposal.
    /// </summary>
    /// <param name="shape">The shape of the tensor to rent.</param>
    /// <returns>A pooled tensor wrapper that returns the tensor to the pool when disposed.</returns>
    /// <exception cref="ObjectDisposedException">Thrown if the pool has been disposed.</exception>
    /// <example>
    /// <code>
    /// using var pooled = pool.RentPooled(new[] { 32, 32 });
    /// var tensor = pooled.Tensor;
    /// // Use tensor... it's automatically returned when block exits
    /// </code>
    /// </example>
    public PooledTensor<T> RentPooled(int[] shape)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(TensorPool<T>));

        return new PooledTensor<T>(this, Rent(shape));
    }

    /// <summary>
    /// Removes all tensors from the pool and resets memory tracking.
    /// </summary>
    /// <remarks>
    /// After calling Clear, all pooled tensors become eligible for garbage collection.
    /// New tensors can still be rented and returned after clearing.
    /// </remarks>
    public void Clear()
    {
        foreach (var pool in _tensorPools.Values)
            while (pool.TryTake(out _)) { }
        Interlocked.Exchange(ref _currentMemoryBytes, 0);
        Interlocked.Exchange(ref _totalPooledTensors, 0);
    }

    /// <summary>
    /// Gets current statistics about the pool's state.
    /// </summary>
    /// <returns>A <see cref="PoolStatistics"/> object containing pool metrics.</returns>
    /// <example>
    /// <code>
    /// var stats = pool.GetStatistics();
    /// Console.WriteLine($"Pool utilization: {stats.MemoryUtilizationPercent:F1}%");
    /// Console.WriteLine($"Tensors in pool: {stats.PooledTensorCount}");
    /// </code>
    /// </example>
    public PoolStatistics GetStatistics()
    {
        var tensorCount = 0;
        foreach (var pool in _tensorPools.Values)
            tensorCount += pool.Count;

        return new PoolStatistics
        {
            PooledTensorCount = tensorCount,
            CurrentMemoryBytes = CurrentMemoryBytes,
            MaxMemoryBytes = _options.MaxPoolSizeBytes,
            TensorBuckets = _tensorPools.Count
        };
    }

    private bool TryReserveMemory(long bytes)
    {
        while (true)
        {
            var current = Interlocked.Read(ref _currentMemoryBytes);
            var newValue = current + bytes;
            if (newValue > _options.MaxPoolSizeBytes)
                return false;
            if (Interlocked.CompareExchange(ref _currentMemoryBytes, newValue, current) == current)
                return true;
        }
    }

    private void UpdateMemoryUsage(long delta) => Interlocked.Add(ref _currentMemoryBytes, delta);

    private static int GetTensorPoolKey(int[] shape)
    {
        unchecked
        {
            var hash = 17;
            hash = hash * 31 + shape.Length;
            foreach (var dim in shape)
                hash = hash * 31 + dim;
            return hash;
        }
    }

    private static long GetTensorSizeBytes(int length) => length * GetElementSize() + 64;

    private static int GetElementSize() => typeof(T) switch
    {
        Type t when t == typeof(float) => 4,
        Type t when t == typeof(double) => 8,
        Type t when t == typeof(int) => 4,
        Type t when t == typeof(long) => 8,
        Type t when t == typeof(Half) => 2,
        Type t when t == typeof(byte) => 1,
        Type t when t == typeof(short) => 2,
        _ => 8
    };

    private static bool ShapeMatches(int[] shape1, int[] shape2)
    {
        if (shape1.Length != shape2.Length)
            return false;
        for (int i = 0; i < shape1.Length; i++)
            if (shape1[i] != shape2[i])
                return false;
        return true;
    }

    private static void ClearTensor(Tensor<T> tensor)
    {
        var zero = MathHelper.GetNumericOperations<T>().Zero;
        for (int i = 0; i < tensor.Length; i++)
            tensor.Data[i] = zero;
    }

    private struct TensorEntry
    {
        public Tensor<T>? Tensor;
        public long SizeBytes;
    }

    /// <summary>
    /// Disposes the tensor pool and releases all pooled tensors.
    /// After disposal, the pool cannot be used.
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            Clear();
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}
