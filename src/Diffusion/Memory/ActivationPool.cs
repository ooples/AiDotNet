using System.Collections.Concurrent;
using AiDotNet.Interfaces;

namespace AiDotNet.Diffusion.Memory;

/// <summary>
/// Memory pool for tensor activations during diffusion model forward/backward passes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Diffusion models process large tensors through many layers, creating significant
/// memory pressure from intermediate activations. This pool reduces allocations
/// by recycling tensor buffers.
/// </para>
/// <para>
/// <b>For Beginners:</b> When running diffusion models, temporary data (activations)
/// is created at each layer.
///
/// Without pooling:
/// - Layer 1 creates activation A (allocate new memory)
/// - Layer 2 creates activation B (allocate new memory)
/// - Layer 1's activation A becomes garbage (GC must clean up)
/// - This creates memory pressure and GC pauses
///
/// With pooling:
/// - Layer 1 borrows a buffer from the pool
/// - Layer 1 returns the buffer when done
/// - Layer 2 reuses the same buffer
/// - No garbage, no GC pauses, faster inference
///
/// Usage:
/// ```csharp
/// using var pool = new ActivationPool&lt;float&gt;(maxMemoryMB: 2048);
///
/// // During forward pass
/// var activation = pool.Rent(new[] { 1, 256, 64, 64 });
/// // ... use activation ...
/// pool.Return(activation);
/// ```
/// </para>
/// </remarks>
public class ActivationPool<T> : IDisposable
{
    /// <summary>
    /// Maximum memory to use for pooling in bytes.
    /// </summary>
    private readonly long _maxMemoryBytes;

    /// <summary>
    /// Current estimated memory usage.
    /// </summary>
    private long _currentMemoryBytes;

    /// <summary>
    /// Size class buckets for tensor pooling.
    /// Each bucket holds tensors of similar total element counts.
    /// </summary>
    private readonly ConcurrentDictionary<int, ConcurrentBag<PooledTensor<T>>> _pools;

    /// <summary>
    /// Lock for memory accounting.
    /// </summary>
    private readonly object _memoryLock = new();

    /// <summary>
    /// Whether the pool has been disposed.
    /// </summary>
    private bool _disposed;

    /// <summary>
    /// Statistics about pool usage.
    /// </summary>
    public ActivationPoolStats Stats { get; } = new();

    /// <summary>
    /// Initializes a new activation pool with specified memory limit.
    /// </summary>
    /// <param name="maxMemoryMB">Maximum memory in megabytes (default: 4096 MB).</param>
    /// <remarks>
    /// <para>
    /// The pool will evict older tensors when approaching the memory limit.
    /// Set this based on your available GPU/CPU memory minus what the model weights use.
    /// </para>
    /// </remarks>
    public ActivationPool(long maxMemoryMB = 4096)
    {
        _maxMemoryBytes = maxMemoryMB * 1024 * 1024;
        _pools = new ConcurrentDictionary<int, ConcurrentBag<PooledTensor<T>>>();
    }

    /// <summary>
    /// Rents a tensor from the pool or creates a new one.
    /// </summary>
    /// <param name="shape">Desired tensor shape.</param>
    /// <returns>A tensor ready for use (contents may be uninitialized).</returns>
    /// <remarks>
    /// <para>
    /// <b>Important:</b> The returned tensor may contain data from previous uses.
    /// Always initialize the tensor before reading from it.
    /// </para>
    /// </remarks>
    public Tensor<T> Rent(int[] shape)
    {
        if (shape == null || shape.Length == 0)
            throw new ArgumentException("Shape cannot be null or empty.", nameof(shape));

        var totalElements = CalculateTotalElements(shape);
        var sizeClass = GetSizeClass(totalElements);
        var memorySize = GetMemorySize(totalElements);

        // Try to get from pool
        if (_pools.TryGetValue(sizeClass, out var pool))
        {
            // Collect non-matching tensors to return after the loop
            var toReturn = new List<PooledTensor<T>>();

            while (pool.TryTake(out var pooledTensor))
            {
                if (ShapeMatches(pooledTensor.Tensor.Shape, shape))
                {
                    // Return non-matching tensors back to pool before returning
                    foreach (var pt in toReturn)
                    {
                        pool.Add(pt);
                    }
                    Stats.IncrementCacheHits();
                    return pooledTensor.Tensor;
                }

                // Wrong shape - keep for returning to pool after the loop
                if (pooledTensor.TotalElements >= totalElements)
                {
                    toReturn.Add(pooledTensor);
                }
            }

            // Return non-matching tensors back to pool
            foreach (var pt in toReturn)
            {
                pool.Add(pt);
            }
        }

        // Need to allocate new
        Stats.IncrementCacheMisses();

        // Check if we need to evict
        lock (_memoryLock)
        {
            if (_currentMemoryBytes + memorySize > _maxMemoryBytes)
            {
                EvictOldest(memorySize);
            }
            _currentMemoryBytes += memorySize;
        }

        var newTensor = new Tensor<T>(shape);
        return newTensor;
    }

    /// <summary>
    /// Returns a tensor to the pool for reuse.
    /// </summary>
    /// <param name="tensor">The tensor to return.</param>
    /// <remarks>
    /// <para>
    /// <b>Important:</b> Do not use the tensor after returning it to the pool.
    /// The pool may immediately give it to another caller.
    /// </para>
    /// </remarks>
    public void Return(Tensor<T> tensor)
    {
        if (tensor == null)
            return;

        var totalElements = CalculateTotalElements(tensor.Shape);
        var sizeClass = GetSizeClass(totalElements);

        var pool = _pools.GetOrAdd(sizeClass, _ => new ConcurrentBag<PooledTensor<T>>());

        pool.Add(new PooledTensor<T>
        {
            Tensor = tensor,
            TotalElements = totalElements,
            LastUsed = DateTime.UtcNow
        });

        Stats.IncrementReturns();
    }

    /// <summary>
    /// Clears all pooled tensors and resets memory accounting.
    /// </summary>
    public void Clear()
    {
        foreach (var pool in _pools.Values)
        {
            while (pool.TryTake(out _)) { }
        }

        lock (_memoryLock)
        {
            _currentMemoryBytes = 0;
        }

        Stats.Evictions += Stats.PooledTensors;
    }

    /// <summary>
    /// Gets current memory usage statistics.
    /// </summary>
    /// <returns>Current memory usage in bytes.</returns>
    public long GetMemoryUsage()
    {
        lock (_memoryLock)
        {
            return _currentMemoryBytes;
        }
    }

    /// <summary>
    /// Evicts oldest tensors to make room for new allocation.
    /// </summary>
    private void EvictOldest(long requiredBytes)
    {
        long freedBytes = 0;
        var now = DateTime.UtcNow;

        // First pass: evict tensors older than 30 seconds
        foreach (var pool in _pools.Values)
        {
            var toKeep = new List<PooledTensor<T>>();

            while (pool.TryTake(out var tensor))
            {
                if ((now - tensor.LastUsed).TotalSeconds > 30)
                {
                    // Evict
                    freedBytes += GetMemorySize(tensor.TotalElements);
                    Stats.IncrementEvictions();
                }
                else
                {
                    toKeep.Add(tensor);
                }
            }

            // Put back the ones we're keeping
            foreach (var t in toKeep)
            {
                pool.Add(t);
            }

            if (freedBytes >= requiredBytes)
                break;
        }

        // Second pass if still not enough: evict by LRU
        if (freedBytes < requiredBytes)
        {
            var allTensors = new List<(int SizeClass, PooledTensor<T> Tensor)>();

            foreach (var kvp in _pools)
            {
                while (kvp.Value.TryTake(out var tensor))
                {
                    allTensors.Add((kvp.Key, tensor));
                }
            }

            // Sort by last used (oldest first)
            allTensors.Sort((a, b) => a.Tensor.LastUsed.CompareTo(b.Tensor.LastUsed));

            foreach (var (sizeClass, tensor) in allTensors)
            {
                if (freedBytes >= requiredBytes)
                {
                    // Keep this one
                    _pools[sizeClass].Add(tensor);
                }
                else
                {
                    // Evict
                    freedBytes += GetMemorySize(tensor.TotalElements);
                    Stats.IncrementEvictions();
                }
            }
        }

        lock (_memoryLock)
        {
            _currentMemoryBytes = Math.Max(0, _currentMemoryBytes - freedBytes);
        }
    }

    /// <summary>
    /// Calculates total elements in a shape.
    /// </summary>
    private static long CalculateTotalElements(int[] shape)
    {
        long total = 1;
        foreach (var dim in shape)
        {
            total *= dim;
        }
        return total;
    }

    /// <summary>
    /// Gets the size class bucket for a given element count.
    /// Size classes are powers of 2 for efficient reuse.
    /// </summary>
    private static int GetSizeClass(long elements)
    {
        // Group by powers of 2 for better reuse
        if (elements <= 0) return 0;
        int log2 = (int)Math.Ceiling(Math.Log(elements) / Math.Log(2));
        return Math.Max(0, Math.Min(log2, 31));
    }

    /// <summary>
    /// Estimates memory size for a given element count.
    /// </summary>
    private static long GetMemorySize(long elements)
    {
        // Assume average of 4 bytes per element (float/int)
        // For double, this would be 8 bytes
        return elements * 4;
    }

    /// <summary>
    /// Checks if two shapes match exactly.
    /// </summary>
    private static bool ShapeMatches(int[] shape1, int[] shape2)
    {
        if (shape1.Length != shape2.Length)
            return false;

        for (int i = 0; i < shape1.Length; i++)
        {
            if (shape1[i] != shape2[i])
                return false;
        }

        return true;
    }

    /// <summary>
    /// Disposes the pool and releases all tensors.
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

/// <summary>
/// Wrapper for pooled tensors with metadata.
/// </summary>
internal class PooledTensor<T>
{
    /// <summary>
    /// The tensor being pooled.
    /// </summary>
    public Tensor<T> Tensor { get; set; } = null!;

    /// <summary>
    /// Total element count for quick matching.
    /// </summary>
    public long TotalElements { get; set; }

    /// <summary>
    /// When the tensor was last used (for LRU eviction).
    /// </summary>
    public DateTime LastUsed { get; set; }
}

/// <summary>
/// Statistics about activation pool usage.
/// Thread-safe counters for concurrent access.
/// </summary>
public class ActivationPoolStats
{
    private long _cacheHits;
    private long _cacheMisses;
    private long _returns;
    private long _evictions;

    /// <summary>
    /// Number of times a tensor was found in the pool.
    /// </summary>
    public long CacheHits
    {
        get => Interlocked.Read(ref _cacheHits);
        set => Interlocked.Exchange(ref _cacheHits, value);
    }

    /// <summary>
    /// Number of times a new tensor had to be allocated.
    /// </summary>
    public long CacheMisses
    {
        get => Interlocked.Read(ref _cacheMisses);
        set => Interlocked.Exchange(ref _cacheMisses, value);
    }

    /// <summary>
    /// Number of tensors returned to the pool.
    /// </summary>
    public long Returns
    {
        get => Interlocked.Read(ref _returns);
        set => Interlocked.Exchange(ref _returns, value);
    }

    /// <summary>
    /// Number of tensors evicted due to memory pressure.
    /// </summary>
    public long Evictions
    {
        get => Interlocked.Read(ref _evictions);
        set => Interlocked.Exchange(ref _evictions, value);
    }

    /// <summary>
    /// Thread-safe increment of cache hits counter.
    /// </summary>
    public void IncrementCacheHits() => Interlocked.Increment(ref _cacheHits);

    /// <summary>
    /// Thread-safe increment of cache misses counter.
    /// </summary>
    public void IncrementCacheMisses() => Interlocked.Increment(ref _cacheMisses);

    /// <summary>
    /// Thread-safe increment of returns counter.
    /// </summary>
    public void IncrementReturns() => Interlocked.Increment(ref _returns);

    /// <summary>
    /// Thread-safe increment of evictions counter.
    /// </summary>
    public void IncrementEvictions() => Interlocked.Increment(ref _evictions);

    /// <summary>
    /// Current number of tensors in the pool.
    /// </summary>
    public long PooledTensors => Returns - CacheHits - Evictions;

    /// <summary>
    /// Cache hit ratio (0-1).
    /// </summary>
    public double HitRatio => CacheHits + CacheMisses > 0
        ? (double)CacheHits / (CacheHits + CacheMisses)
        : 0;

    /// <inheritdoc />
    public override string ToString()
    {
        return $"Hits: {CacheHits}, Misses: {CacheMisses}, " +
               $"HitRatio: {HitRatio:P1}, Evictions: {Evictions}";
    }
}
