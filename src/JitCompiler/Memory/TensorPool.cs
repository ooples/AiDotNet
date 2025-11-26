using System.Collections.Concurrent;
using AiDotNet.Autodiff;

namespace AiDotNet.JitCompiler.Memory;

/// <summary>
/// Provides efficient tensor memory pooling to reduce allocations and GC pressure during JIT execution.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This is like a "rental service" for tensor memory.
///
/// Creating and destroying large tensors repeatedly is expensive because:
/// 1. Memory allocation takes time
/// 2. Garbage collection causes pauses
/// 3. Memory fragmentation reduces performance
///
/// The TensorPool keeps frequently-used tensor buffers around and recycles them:
/// 1. When you need a tensor, borrow one from the pool
/// 2. When you're done, return it to the pool
/// 3. Next time someone needs a tensor of that size, they get your recycled one
///
/// This dramatically improves performance for repeated computations like training loops.
/// </para>
/// </remarks>
public class TensorPool : IDisposable
{
    private readonly ConcurrentDictionary<int, ConcurrentBag<WeakReference<Array>>> _pools = new();
    private readonly int _maxPoolSizePerShape;
    private readonly int _maxElementsToPool;
    private bool _disposed;

    /// <summary>
    /// Creates a new tensor pool with default settings.
    /// </summary>
    public TensorPool() : this(maxPoolSizePerShape: 10, maxElementsToPool: 10_000_000)
    {
    }

    /// <summary>
    /// Creates a new tensor pool with custom settings.
    /// </summary>
    /// <param name="maxPoolSizePerShape">Maximum number of tensors to keep per shape.</param>
    /// <param name="maxElementsToPool">Maximum total elements in a tensor to pool (larger tensors won't be pooled).</param>
    public TensorPool(int maxPoolSizePerShape, int maxElementsToPool)
    {
        _maxPoolSizePerShape = maxPoolSizePerShape;
        _maxElementsToPool = maxElementsToPool;
    }

    /// <summary>
    /// Rents a tensor buffer of the specified size.
    /// </summary>
    /// <typeparam name="T">The element type of the tensor.</typeparam>
    /// <param name="totalElements">Total number of elements needed.</param>
    /// <returns>An array that may be recycled from the pool or newly allocated.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Gets a buffer for your tensor data.
    ///
    /// The buffer might be recycled from a previous tensor, so it may contain old data.
    /// You should initialize or overwrite all values before using the tensor.
    ///
    /// Example:
    ///   var buffer = pool.Rent&lt;float&gt;(1000);
    ///   // Use buffer for computation...
    ///   pool.Return(buffer);
    /// </para>
    /// </remarks>
    public T[] Rent<T>(int totalElements)
    {
        if (totalElements > _maxElementsToPool)
        {
            // Too large to pool - allocate directly
            return new T[totalElements];
        }

        var key = GetPoolKey<T>(totalElements);
        if (_pools.TryGetValue(key, out var pool))
        {
            while (pool.TryTake(out var weakRef))
            {
                if (weakRef.TryGetTarget(out var array) && array is T[] typedArray && typedArray.Length >= totalElements)
                {
                    return typedArray;
                }
            }
        }

        // No suitable buffer found - allocate new
        return new T[totalElements];
    }

    /// <summary>
    /// Returns a tensor buffer to the pool for reuse.
    /// </summary>
    /// <typeparam name="T">The element type of the tensor.</typeparam>
    /// <param name="buffer">The buffer to return.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Gives back a buffer you're done using.
    ///
    /// After returning a buffer, you must not use it anymore!
    /// The buffer might be given to someone else immediately.
    ///
    /// Important:
    /// - Always return buffers you rented
    /// - Never use a buffer after returning it
    /// - Don't return buffers you didn't rent from this pool
    /// </para>
    /// </remarks>
    public void Return<T>(T[] buffer)
    {
        if (buffer == null || buffer.Length > _maxElementsToPool)
        {
            return; // Don't pool null or oversized buffers
        }

        var key = GetPoolKey<T>(buffer.Length);
        var pool = _pools.GetOrAdd(key, _ => new ConcurrentBag<WeakReference<Array>>());

        // Only add if pool isn't too full
        if (pool.Count < _maxPoolSizePerShape)
        {
            pool.Add(new WeakReference<Array>(buffer));
        }
    }

    /// <summary>
    /// Clears all pooled buffers, allowing them to be garbage collected.
    /// </summary>
    public void Clear()
    {
        foreach (var pool in _pools.Values)
        {
            while (pool.TryTake(out _)) { }
        }
    }

    /// <summary>
    /// Gets statistics about the current pool state.
    /// </summary>
    /// <returns>Pool statistics including buffer counts and estimated memory usage.</returns>
    public TensorPoolStats GetStats()
    {
        int totalBuffers = 0;
        long estimatedBytes = 0;

        foreach (var kvp in _pools)
        {
            var count = kvp.Value.Count;
            totalBuffers += count;
            // Rough estimate: assume 4 bytes per element average
            estimatedBytes += count * (kvp.Key % 1_000_000) * 4;
        }

        return new TensorPoolStats
        {
            TotalPooledBuffers = totalBuffers,
            EstimatedMemoryBytes = estimatedBytes,
            UniqueShapes = _pools.Count
        };
    }

    private static int GetPoolKey<T>(int elements)
    {
        // Combine type hash and element count for pool key
        // Round up to nearest power of 2 for better reuse
        int roundedElements = NextPowerOfTwo(elements);
        return HashCode.Combine(typeof(T).GetHashCode(), roundedElements);
    }

    private static int NextPowerOfTwo(int n)
    {
        if (n <= 0) return 1;
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        return n + 1;
    }

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
/// Statistics about the tensor pool state.
/// </summary>
public class TensorPoolStats
{
    /// <summary>
    /// Total number of buffers currently in the pool.
    /// </summary>
    public int TotalPooledBuffers { get; set; }

    /// <summary>
    /// Estimated memory usage of pooled buffers in bytes.
    /// </summary>
    public long EstimatedMemoryBytes { get; set; }

    /// <summary>
    /// Number of unique tensor shapes being pooled.
    /// </summary>
    public int UniqueShapes { get; set; }
}

/// <summary>
/// Provides a scoped rental of a tensor buffer that automatically returns to the pool.
/// </summary>
/// <typeparam name="T">The element type of the tensor.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A convenient way to use pooled buffers with automatic cleanup.
///
/// Instead of manually calling Rent and Return, use this with a 'using' statement:
///
/// Example:
///   using (var rental = new TensorRental&lt;float&gt;(pool, 1000))
///   {
///       // Use rental.Buffer for computation
///       // Buffer is automatically returned when leaving this block
///   }
/// </para>
/// </remarks>
public readonly struct TensorRental<T> : IDisposable
{
    private readonly TensorPool _pool;

    /// <summary>
    /// The rented buffer.
    /// </summary>
    public T[] Buffer { get; }

    /// <summary>
    /// Creates a new tensor rental.
    /// </summary>
    /// <param name="pool">The pool to rent from.</param>
    /// <param name="totalElements">Number of elements needed.</param>
    public TensorRental(TensorPool pool, int totalElements)
    {
        _pool = pool;
        Buffer = pool.Rent<T>(totalElements);
    }

    /// <summary>
    /// Returns the buffer to the pool.
    /// </summary>
    public void Dispose()
    {
        _pool?.Return(Buffer);
    }
}
