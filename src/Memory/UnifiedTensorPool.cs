using System.Collections.Concurrent;

namespace AiDotNet.Memory;

/// <summary>
/// A high-performance, thread-safe memory pool for tensors and raw arrays.
/// </summary>
/// <remarks>
/// <para>
/// UnifiedTensorPool provides efficient memory pooling to reduce allocations and GC pressure
/// during neural network operations. It combines the best features of tensor pooling and
/// array pooling with configurable memory limits and pooling strategies.
/// </para>
/// <para><b>For Beginners:</b> Think of this as a "rental service" for memory.
///
/// Instead of constantly allocating and freeing memory (which is slow), you:
/// 1. Borrow a buffer from the pool when you need it
/// 2. Return it when you're done
/// 3. The pool recycles it for the next user
///
/// This dramatically improves performance for repeated operations like training loops.
/// </para>
/// <para>
/// <b>Features:</b>
/// <list type="bullet">
///   <item>Thread-safe concurrent access</item>
///   <item>Configurable memory limits</item>
///   <item>Power-of-2 size buckets for efficient reuse</item>
///   <item>Weak references option for graceful degradation under memory pressure</item>
///   <item>Support for both Tensor&lt;T&gt; and raw array pooling</item>
///   <item>RAII pattern support via PooledTensor&lt;T&gt; and PooledArray&lt;T&gt;</item>
/// </list>
/// </para>
/// </remarks>
public class UnifiedTensorPool : IDisposable
{
    private readonly ConcurrentDictionary<int, ConcurrentBag<PoolEntry>> _arrayPools;
    private readonly ConcurrentDictionary<int, ConcurrentBag<TensorEntry>> _tensorPools;
    private readonly PoolingOptions _options;
    private long _currentMemoryBytes;
    private bool _disposed;

    /// <summary>
    /// Gets the current memory usage of the pool in bytes.
    /// </summary>
    public long CurrentMemoryBytes => Interlocked.Read(ref _currentMemoryBytes);

    /// <summary>
    /// Gets the pooling options.
    /// </summary>
    public PoolingOptions Options => _options;

    /// <summary>
    /// Creates a new unified tensor pool with default options.
    /// </summary>
    public UnifiedTensorPool() : this(new PoolingOptions())
    {
    }

    /// <summary>
    /// Creates a new unified tensor pool with the specified options.
    /// </summary>
    /// <param name="options">The pooling configuration options.</param>
    public UnifiedTensorPool(PoolingOptions options)
    {
        _options = options ?? new PoolingOptions();
        _arrayPools = new ConcurrentDictionary<int, ConcurrentBag<PoolEntry>>();
        _tensorPools = new ConcurrentDictionary<int, ConcurrentBag<TensorEntry>>();
    }

    #region Array Pooling

    /// <summary>
    /// Rents a raw array buffer from the pool.
    /// </summary>
    /// <typeparam name="T">The element type of the array.</typeparam>
    /// <param name="minimumLength">The minimum required length.</param>
    /// <returns>An array with at least the requested length.</returns>
    /// <remarks>
    /// <para>
    /// The returned array may be larger than requested if a suitable buffer exists in the pool.
    /// The array contents are NOT cleared - you must initialize values before use.
    /// </para>
    /// </remarks>
    public T[] RentArray<T>(int minimumLength)
    {
        if (minimumLength <= 0)
        {
            return Array.Empty<T>();
        }

        // Don't pool arrays larger than the threshold
        if (minimumLength > _options.MaxElementsToPool)
        {
            return new T[minimumLength];
        }

        var bucketLength = GetBucketSize(minimumLength);
        var key = GetArrayPoolKey<T>(bucketLength);

        if (_arrayPools.TryGetValue(key, out var pool))
        {
            while (pool.TryTake(out var entry))
            {
                // Always decrement memory usage when removing entry from pool
                // This prevents memory accounting leaks when entries are invalid
                UpdateMemoryUsage(-entry.SizeBytes);

                if (_options.UseWeakReferences)
                {
                    if (entry.WeakArray is not null &&
                        entry.WeakArray.TryGetTarget(out var array) &&
                        array is T[] typedArray &&
                        typedArray.Length >= minimumLength)
                    {
                        return typedArray;
                    }
                    // Entry was invalid (weak ref dead) - memory already decremented, continue
                }
                else
                {
                    if (entry.StrongArray is T[] typedArray && typedArray.Length >= minimumLength)
                    {
                        return typedArray;
                    }
                    // Entry was invalid (wrong type) - memory already decremented, continue
                }
            }
        }

        // No suitable buffer found - allocate new
        return new T[bucketLength];
    }

    /// <summary>
    /// Returns an array buffer to the pool for reuse.
    /// </summary>
    /// <typeparam name="T">The element type of the array.</typeparam>
    /// <param name="array">The array to return.</param>
    /// <param name="clearArray">If true, clears the array before pooling (security).</param>
    public void ReturnArray<T>(T[] array, bool clearArray = false)
    {
        if (array == null || array.Length == 0 || array.Length > _options.MaxElementsToPool)
        {
            return;
        }

        var sizeBytes = GetArraySizeBytes<T>(array.Length);
        if (!TryReserveMemory(sizeBytes))
        {
            return; // Pool is full
        }

        if (clearArray)
        {
            Array.Clear(array, 0, array.Length);
        }

        var key = GetArrayPoolKey<T>(array.Length);
        var pool = _arrayPools.GetOrAdd(key, _ => new ConcurrentBag<PoolEntry>());

        if (pool.Count < _options.MaxItemsPerBucket)
        {
            var entry = _options.UseWeakReferences
                ? new PoolEntry { WeakArray = new WeakReference<Array>(array), SizeBytes = sizeBytes }
                : new PoolEntry { StrongArray = array, SizeBytes = sizeBytes };
            pool.Add(entry);
        }
        else
        {
            // Pool is full for this bucket - release the reserved memory
            UpdateMemoryUsage(-sizeBytes);
        }
    }

    /// <summary>
    /// Rents an array and returns a disposable wrapper that automatically returns it.
    /// </summary>
    /// <typeparam name="T">The element type of the array.</typeparam>
    /// <param name="minimumLength">The minimum required length.</param>
    /// <returns>A PooledArray that returns the array to the pool when disposed.</returns>
    public PooledArray<T> RentArrayPooled<T>(int minimumLength)
    {
        var array = RentArray<T>(minimumLength);
        return new PooledArray<T>(this, array);
    }

    #endregion

    #region Tensor Pooling

    /// <summary>
    /// Rents a tensor with the specified shape from the pool.
    /// </summary>
    /// <typeparam name="T">The element type of the tensor.</typeparam>
    /// <param name="shape">The desired tensor shape.</param>
    /// <returns>A tensor with the requested shape.</returns>
    public Tensor<T> RentTensor<T>(int[] shape)
    {
        if (shape == null || shape.Length == 0)
        {
            throw new ArgumentException("Shape must be non-empty.", nameof(shape));
        }

        var totalElements = 1;
        foreach (var dim in shape)
        {
            if (dim <= 0)
            {
                throw new ArgumentException("All dimensions must be positive.", nameof(shape));
            }
            checked { totalElements *= dim; }
        }

        if (totalElements > _options.MaxElementsToPool)
        {
            return new Tensor<T>(shape);
        }

        var key = GetTensorPoolKey<T>(shape);

        if (_tensorPools.TryGetValue(key, out var pool))
        {
            while (pool.TryTake(out var entry))
            {
                // Always decrement memory usage when removing entry from pool
                // This prevents memory accounting leaks when entries are invalid
                UpdateMemoryUsage(-entry.SizeBytes);

                if (entry.Tensor is Tensor<T> tensor && ShapeMatches(tensor.Shape, shape))
                {
                    ClearTensor(tensor);
                    return tensor;
                }
                // Entry was invalid (wrong type or shape mismatch) - memory already decremented, continue
            }
        }

        // No suitable tensor found - create new
        return new Tensor<T>(shape);
    }

    /// <summary>
    /// Returns a tensor to the pool for reuse.
    /// </summary>
    /// <typeparam name="T">The element type of the tensor.</typeparam>
    /// <param name="tensor">The tensor to return.</param>
    public void ReturnTensor<T>(Tensor<T> tensor)
    {
        if (tensor == null || tensor.Length > _options.MaxElementsToPool)
        {
            return;
        }

        var sizeBytes = GetTensorSizeBytes<T>(tensor.Length);
        if (!TryReserveMemory(sizeBytes))
        {
            return; // Pool is full
        }

        ClearTensor(tensor);

        var key = GetTensorPoolKey<T>(tensor.Shape);
        var pool = _tensorPools.GetOrAdd(key, _ => new ConcurrentBag<TensorEntry>());

        if (pool.Count < _options.MaxItemsPerBucket)
        {
            pool.Add(new TensorEntry { Tensor = tensor, SizeBytes = sizeBytes });
        }
        else
        {
            UpdateMemoryUsage(-sizeBytes);
        }
    }

    /// <summary>
    /// Rents a tensor and returns a disposable wrapper that automatically returns it.
    /// </summary>
    /// <typeparam name="T">The element type of the tensor.</typeparam>
    /// <param name="shape">The desired tensor shape.</param>
    /// <returns>A PooledTensor that returns the tensor to the pool when disposed.</returns>
    public PooledTensor<T> RentTensorPooled<T>(int[] shape)
    {
        var tensor = RentTensor<T>(shape);
        return new PooledTensor<T>(this, tensor);
    }

    #endregion

    #region Pool Management

    /// <summary>
    /// Clears all pooled items, releasing memory.
    /// </summary>
    public void Clear()
    {
        foreach (var pool in _arrayPools.Values)
        {
            while (pool.TryTake(out _)) { }
        }
        foreach (var pool in _tensorPools.Values)
        {
            while (pool.TryTake(out _)) { }
        }
        Interlocked.Exchange(ref _currentMemoryBytes, 0);
    }

    /// <summary>
    /// Gets statistics about the current pool state.
    /// </summary>
    /// <returns>Pool statistics.</returns>
    public PoolStatistics GetStatistics()
    {
        var arrayCount = 0;
        var tensorCount = 0;

        foreach (var pool in _arrayPools.Values)
        {
            arrayCount += pool.Count;
        }

        foreach (var pool in _tensorPools.Values)
        {
            tensorCount += pool.Count;
        }

        return new PoolStatistics
        {
            PooledArrayCount = arrayCount,
            PooledTensorCount = tensorCount,
            CurrentMemoryBytes = CurrentMemoryBytes,
            MaxMemoryBytes = _options.MaxPoolSizeBytes,
            ArrayBuckets = _arrayPools.Count,
            TensorBuckets = _tensorPools.Count
        };
    }

    /// <summary>
    /// Trims the pool by removing weak references that have been collected.
    /// </summary>
    /// <remarks>
    /// This is only effective when UseWeakReferences is enabled.
    /// </remarks>
    public void Trim()
    {
        if (!_options.UseWeakReferences)
        {
            return;
        }

        foreach (var pool in _arrayPools.Values)
        {
            var itemsToKeep = new List<PoolEntry>();

            while (pool.TryTake(out var entry))
            {
                if (entry.WeakArray is not null && entry.WeakArray.TryGetTarget(out _))
                {
                    itemsToKeep.Add(entry);
                }
                else
                {
                    UpdateMemoryUsage(-entry.SizeBytes);
                }
            }

            foreach (var item in itemsToKeep)
            {
                pool.Add(item);
            }
        }
    }

    #endregion

    #region Private Helpers

    private bool TryReserveMemory(long bytes)
    {
        while (true)
        {
            var current = Interlocked.Read(ref _currentMemoryBytes);
            var newValue = current + bytes;

            if (newValue > _options.MaxPoolSizeBytes)
            {
                return false;
            }

            if (Interlocked.CompareExchange(ref _currentMemoryBytes, newValue, current) == current)
            {
                return true;
            }
        }
    }

    private void UpdateMemoryUsage(long delta)
    {
        Interlocked.Add(ref _currentMemoryBytes, delta);
    }

    private static int GetBucketSize(int size)
    {
        // Round up to nearest power of 2 for better bucket reuse
        if (size <= 16) return 16;

        size--;
        size |= size >> 1;
        size |= size >> 2;
        size |= size >> 4;
        size |= size >> 8;
        size |= size >> 16;
        return size + 1;
    }

    private static int GetArrayPoolKey<T>(int length)
    {
        unchecked
        {
            var hash = 17;
            hash = hash * 31 + typeof(T).GetHashCode();
            hash = hash * 31 + length;
            return hash;
        }
    }

    private static int GetTensorPoolKey<T>(int[] shape)
    {
        unchecked
        {
            var hash = 17;
            hash = hash * 31 + typeof(T).GetHashCode();
            hash = hash * 31 + shape.Length;
            foreach (var dim in shape)
            {
                hash = hash * 31 + dim;
            }
            return hash;
        }
    }

    private static long GetArraySizeBytes<T>(int length)
    {
        return length * GetElementSize<T>();
    }

    private static long GetTensorSizeBytes<T>(int length)
    {
        return length * GetElementSize<T>() + 64; // Add overhead for tensor metadata
    }

    private static int GetElementSize<T>()
    {
        return typeof(T) switch
        {
            Type t when t == typeof(float) => 4,
            Type t when t == typeof(double) => 8,
            Type t when t == typeof(int) => 4,
            Type t when t == typeof(long) => 8,
            Type t when t == typeof(Half) => 2,
            Type t when t == typeof(byte) => 1,
            Type t when t == typeof(short) => 2,
            _ => 8 // Default estimate
        };
    }

    private static bool ShapeMatches(int[] shape1, int[] shape2)
    {
        if (shape1.Length != shape2.Length)
        {
            return false;
        }

        for (int i = 0; i < shape1.Length; i++)
        {
            if (shape1[i] != shape2[i])
            {
                return false;
            }
        }

        return true;
    }

    private static void ClearTensor<T>(Tensor<T> tensor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var zero = numOps.Zero;

        for (int i = 0; i < tensor.Length; i++)
        {
            tensor.Data[i] = zero;
        }
    }

    #endregion

    #region Nested Types

    private struct PoolEntry
    {
        public WeakReference<Array>? WeakArray;
        public Array? StrongArray;
        public long SizeBytes;
    }

    private struct TensorEntry
    {
        public object? Tensor; // Stored as object to avoid generic type in struct
        public long SizeBytes;
    }

    #endregion

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

/// <summary>
/// Configuration options for the unified tensor pool.
/// </summary>
public class PoolingOptions
{
    /// <summary>
    /// Gets or sets the maximum pool size in bytes. Default is 256 MB.
    /// </summary>
    public long MaxPoolSizeBytes { get; set; } = 256L * 1024 * 1024;

    /// <summary>
    /// Gets or sets the maximum pool size in megabytes.
    /// </summary>
    public int MaxPoolSizeMB
    {
        get => (int)(MaxPoolSizeBytes / (1024 * 1024));
        set => MaxPoolSizeBytes = value * 1024L * 1024;
    }

    /// <summary>
    /// Gets or sets the maximum number of elements in a single buffer to pool.
    /// Larger buffers are allocated but not pooled. Default is 10 million.
    /// </summary>
    public int MaxElementsToPool { get; set; } = 10_000_000;

    /// <summary>
    /// Gets or sets the maximum number of items per size bucket. Default is 10.
    /// </summary>
    public int MaxItemsPerBucket { get; set; } = 10;

    /// <summary>
    /// Gets or sets whether to use weak references for pooled arrays.
    /// When true, the GC can reclaim pooled memory under pressure.
    /// Default is false (use strong references for predictable pooling).
    /// </summary>
    public bool UseWeakReferences { get; set; } = false;

    /// <summary>
    /// Gets or sets whether pooling is enabled. Default is true.
    /// </summary>
    public bool Enabled { get; set; } = true;
}

/// <summary>
/// Statistics about the pool state.
/// </summary>
public class PoolStatistics
{
    /// <summary>
    /// Gets or sets the number of pooled arrays.
    /// </summary>
    public int PooledArrayCount { get; set; }

    /// <summary>
    /// Gets or sets the number of pooled tensors.
    /// </summary>
    public int PooledTensorCount { get; set; }

    /// <summary>
    /// Gets or sets the current memory usage in bytes.
    /// </summary>
    public long CurrentMemoryBytes { get; set; }

    /// <summary>
    /// Gets or sets the maximum allowed memory in bytes.
    /// </summary>
    public long MaxMemoryBytes { get; set; }

    /// <summary>
    /// Gets or sets the number of array size buckets.
    /// </summary>
    public int ArrayBuckets { get; set; }

    /// <summary>
    /// Gets or sets the number of tensor shape buckets.
    /// </summary>
    public int TensorBuckets { get; set; }

    /// <summary>
    /// Gets the memory utilization as a percentage.
    /// </summary>
    public double MemoryUtilizationPercent => MaxMemoryBytes > 0
        ? (CurrentMemoryBytes * 100.0 / MaxMemoryBytes)
        : 0;
}

/// <summary>
/// A wrapper that returns a pooled array when disposed.
/// </summary>
/// <typeparam name="T">The element type of the array.</typeparam>
public readonly struct PooledArray<T> : IDisposable
{
    private readonly UnifiedTensorPool _pool;

    /// <summary>
    /// Gets the pooled array.
    /// </summary>
    public T[] Array { get; }

    /// <summary>
    /// Gets the length of the array.
    /// </summary>
    public int Length => Array.Length;

    /// <summary>
    /// Creates a new pooled array wrapper.
    /// </summary>
    public PooledArray(UnifiedTensorPool pool, T[] array)
    {
        _pool = pool ?? throw new ArgumentNullException(nameof(pool));
        Array = array ?? throw new ArgumentNullException(nameof(array));
    }

    /// <summary>
    /// Returns the array to the pool.
    /// </summary>
    public void Dispose()
    {
        _pool.ReturnArray(Array);
    }

    /// <summary>
    /// Implicit conversion to the underlying array.
    /// </summary>
    public static implicit operator T[](PooledArray<T> pooled) => pooled.Array;

    /// <summary>
    /// Array indexer.
    /// </summary>
    public T this[int index]
    {
        get => Array[index];
        set => Array[index] = value;
    }
}

/// <summary>
/// A wrapper that returns a pooled tensor when disposed.
/// </summary>
/// <typeparam name="T">The element type of the tensor.</typeparam>
public readonly struct PooledTensor<T> : IDisposable
{
    private readonly UnifiedTensorPool _pool;

    /// <summary>
    /// Gets the pooled tensor.
    /// </summary>
    public Tensor<T> Tensor { get; }

    /// <summary>
    /// Creates a new pooled tensor wrapper.
    /// </summary>
    public PooledTensor(UnifiedTensorPool pool, Tensor<T> tensor)
    {
        _pool = pool ?? throw new ArgumentNullException(nameof(pool));
        Tensor = tensor ?? throw new ArgumentNullException(nameof(tensor));
    }

    /// <summary>
    /// Returns the tensor to the pool.
    /// </summary>
    public void Dispose()
    {
        _pool.ReturnTensor(Tensor);
    }

    /// <summary>
    /// Implicit conversion to the underlying tensor.
    /// </summary>
    public static implicit operator Tensor<T>(PooledTensor<T> pooled) => pooled.Tensor;
}

/// <summary>
/// Provides a global/shared tensor pool instance.
/// </summary>
public static class TensorPoolManager
{
    private static UnifiedTensorPool? _shared;
    private static readonly object _lock = new();

    /// <summary>
    /// Gets or creates the shared tensor pool instance.
    /// </summary>
    public static UnifiedTensorPool Shared
    {
        get
        {
            if (_shared is null)
            {
                lock (_lock)
                {
                    _shared ??= new UnifiedTensorPool();
                }
            }
            return _shared;
        }
    }

    /// <summary>
    /// Configures the shared pool with custom options.
    /// </summary>
    /// <param name="options">The pooling options.</param>
    /// <remarks>
    /// This replaces the existing shared pool. Any buffers in the old pool will be lost.
    /// </remarks>
    public static void Configure(PoolingOptions options)
    {
        lock (_lock)
        {
            _shared?.Dispose();
            _shared = new UnifiedTensorPool(options);
        }
    }

    /// <summary>
    /// Resets the shared pool, clearing all pooled items.
    /// </summary>
    public static void Reset()
    {
        lock (_lock)
        {
            _shared?.Clear();
        }
    }
}
