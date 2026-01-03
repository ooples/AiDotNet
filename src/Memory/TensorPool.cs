using System.Collections.Concurrent;

namespace AiDotNet.Memory;

/// <summary>
/// A thread-safe object pool for reusing tensors to reduce garbage collection pressure.
/// </summary>
/// <remarks>
/// <para>
/// TensorPool maintains collections of pre-allocated tensors organized by size class.
/// When a tensor is needed, it can be rented from the pool. When no longer needed,
/// it should be returned to the pool for reuse by other operations.
/// </para>
/// <para><b>For Beginners:</b> Think of this like a library for tensors.
/// Instead of creating new tensors (which costs memory and triggers garbage collection),
/// you can "borrow" a tensor, use it, and return it when done. The next operation
/// can then reuse that same tensor, saving memory allocations.
/// </para>
/// <para>
/// <b>Thread Safety:</b> This class is thread-safe. Multiple threads can rent and return
/// tensors concurrently without synchronization.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for tensor elements.</typeparam>
public class TensorPool<T> : IDisposable
{
    private readonly ConcurrentBag<Tensor<T>>[] _pools;
    private readonly long _maxPoolSizeBytes;
    private long _currentPoolSizeBytes;
    private readonly int _numSizeClasses;
    private bool _disposed;

    /// <summary>
    /// Gets the current total size of pooled tensors in bytes.
    /// </summary>
    public long CurrentPoolSizeBytes => Interlocked.Read(ref _currentPoolSizeBytes);

    /// <summary>
    /// Gets the maximum allowed pool size in bytes.
    /// </summary>
    public long MaxPoolSizeBytes => _maxPoolSizeBytes;

    /// <summary>
    /// Gets the number of tensors currently in the pool across all size classes.
    /// </summary>
    public int TotalPooledTensors
    {
        get
        {
            int count = 0;
            for (int i = 0; i < _numSizeClasses; i++)
            {
                count += _pools[i].Count;
            }
            return count;
        }
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="TensorPool{T}"/> class.
    /// </summary>
    /// <param name="maxPoolSizeMB">Maximum total size of pooled tensors in megabytes. Default is 256 MB.</param>
    /// <param name="numSizeClasses">Number of size classes for organizing tensors. Default is 32.</param>
    /// <remarks>
    /// <para>
    /// The pool organizes tensors into size classes based on their total element count.
    /// This allows efficient lookup of tensors with matching or similar sizes.
    /// </para>
    /// </remarks>
    public TensorPool(int maxPoolSizeMB = 256, int numSizeClasses = 32)
    {
        if (maxPoolSizeMB <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxPoolSizeMB), "Max pool size must be positive.");
        }

        if (numSizeClasses <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numSizeClasses), "Number of size classes must be positive.");
        }

        _maxPoolSizeBytes = maxPoolSizeMB * 1024L * 1024L;
        _numSizeClasses = numSizeClasses;
        _pools = new ConcurrentBag<Tensor<T>>[numSizeClasses];

        for (int i = 0; i < numSizeClasses; i++)
        {
            _pools[i] = new ConcurrentBag<Tensor<T>>();
        }
    }

    /// <summary>
    /// Rents a tensor with the specified shape from the pool.
    /// </summary>
    /// <param name="shape">The desired shape of the tensor.</param>
    /// <returns>
    /// A tensor with at least the specified capacity. The tensor is cleared before return.
    /// </returns>
    /// <remarks>
    /// <para>
    /// If a suitable tensor exists in the pool, it is returned. Otherwise, a new tensor
    /// is allocated. The returned tensor may have a larger capacity than requested if
    /// a larger tensor was available in the pool.
    /// </para>
    /// <para><b>For Beginners:</b> This is like checking out a book from the library.
    /// You ask for a specific size, and if one is available, you get it. Otherwise,
    /// the library orders a new one for you.
    /// </para>
    /// </remarks>
    public Tensor<T> Rent(int[] shape)
    {
        if (shape == null || shape.Length == 0)
        {
            throw new ArgumentException("Shape must be non-empty.", nameof(shape));
        }

        int totalElements = 1;
        foreach (int dim in shape)
        {
            if (dim <= 0)
            {
                throw new ArgumentException("All dimensions must be positive.", nameof(shape));
            }
            totalElements *= dim;
        }

        int sizeClass = GetSizeClass(totalElements);

        // Try to find a tensor with matching or larger size
        for (int i = sizeClass; i < _numSizeClasses; i++)
        {
            if (_pools[i].TryTake(out var tensor))
            {
                // Check if shape matches exactly
                if (ShapeMatches(tensor.Shape, shape))
                {
                    ClearTensor(tensor);
                    Interlocked.Add(ref _currentPoolSizeBytes, -GetTensorSizeBytes(tensor));
                    return tensor;
                }

                // Return to pool if shape doesn't match, continue searching
                _pools[i].Add(tensor);
            }
        }

        // No suitable tensor found, create new one
        return new Tensor<T>(shape);
    }

    /// <summary>
    /// Returns a tensor to the pool for future reuse.
    /// </summary>
    /// <param name="tensor">The tensor to return to the pool.</param>
    /// <remarks>
    /// <para>
    /// The tensor is cleared before being added to the pool to prevent data leakage.
    /// If the pool is at capacity, the tensor may be discarded instead of pooled.
    /// </para>
    /// <para><b>For Beginners:</b> This is like returning a book to the library.
    /// The content is "erased" (cleared) so the next borrower gets a clean tensor.
    /// If the library is full, the book might be discarded instead.
    /// </para>
    /// </remarks>
    public void Return(Tensor<T> tensor)
    {
        if (tensor == null)
        {
            return;
        }

        long tensorSize = GetTensorSizeBytes(tensor);

        // Check if we have room in the pool
        if (Interlocked.Read(ref _currentPoolSizeBytes) + tensorSize > _maxPoolSizeBytes)
        {
            // Pool is full, let GC handle this tensor
            return;
        }

        // Clear tensor data for security
        ClearTensor(tensor);

        int sizeClass = GetSizeClass(tensor.Length);
        _pools[sizeClass].Add(tensor);
        Interlocked.Add(ref _currentPoolSizeBytes, tensorSize);
    }

    /// <summary>
    /// Clears all tensors from the pool.
    /// </summary>
    public void Clear()
    {
        for (int i = 0; i < _numSizeClasses; i++)
        {
            while (_pools[i].TryTake(out _))
            {
                // Discard all tensors
            }
        }
        Interlocked.Exchange(ref _currentPoolSizeBytes, 0);
    }

    /// <summary>
    /// Gets the size class index for a given element count.
    /// </summary>
    private int GetSizeClass(int elementCount)
    {
        if (elementCount <= 0)
        {
            return 0;
        }

        // Use log2 to distribute sizes into classes
        // This gives roughly power-of-2 buckets
        int log2 = (int)Math.Log2(elementCount);
        return Math.Min(log2, _numSizeClasses - 1);
    }

    /// <summary>
    /// Gets the size of a tensor in bytes.
    /// </summary>
    private static long GetTensorSizeBytes(Tensor<T> tensor)
    {
        // Estimate size based on element count and type
        int elementSize = typeof(T) switch
        {
            Type t when t == typeof(float) => 4,
            Type t when t == typeof(double) => 8,
            Type t when t == typeof(int) => 4,
            Type t when t == typeof(long) => 8,
            Type t when t == typeof(Half) => 2,
            _ => 8 // Default estimate
        };

        return tensor.Length * elementSize;
    }

    /// <summary>
    /// Checks if two shapes are equal.
    /// </summary>
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

    /// <summary>
    /// Clears all elements of a tensor to the default value.
    /// </summary>
    private static void ClearTensor(Tensor<T> tensor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var zero = numOps.Zero;

        for (int i = 0; i < tensor.Length; i++)
        {
            tensor.Data[i] = zero;
        }
    }

    /// <summary>
    /// Releases all resources used by the pool.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Releases resources used by the pool.
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                Clear();
            }
            _disposed = true;
        }
    }
}

/// <summary>
/// A wrapper that returns a rented tensor to the pool when disposed.
/// </summary>
/// <remarks>
/// <para>
/// This wrapper enables using pooled tensors with the using statement pattern,
/// ensuring tensors are automatically returned to the pool when no longer needed.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for tensor elements.</typeparam>
public readonly struct PooledTensor<T> : IDisposable
{
    private readonly TensorPool<T> _pool;

    /// <summary>
    /// Gets the underlying tensor.
    /// </summary>
    public Tensor<T> Tensor { get; }

    /// <summary>
    /// Initializes a new pooled tensor wrapper.
    /// </summary>
    /// <param name="pool">The pool to return the tensor to.</param>
    /// <param name="tensor">The rented tensor.</param>
    public PooledTensor(TensorPool<T> pool, Tensor<T> tensor)
    {
        _pool = pool ?? throw new ArgumentNullException(nameof(pool));
        Tensor = tensor ?? throw new ArgumentNullException(nameof(tensor));
    }

    /// <summary>
    /// Returns the tensor to the pool.
    /// </summary>
    public void Dispose()
    {
        _pool.Return(Tensor);
    }

    /// <summary>
    /// Implicit conversion to the underlying tensor.
    /// </summary>
    public static implicit operator Tensor<T>(PooledTensor<T> pooled) => pooled.Tensor;
}

/// <summary>
/// Extension methods for TensorPool.
/// </summary>
public static class TensorPoolExtensions
{
    /// <summary>
    /// Rents a tensor and returns a disposable wrapper that automatically returns it to the pool.
    /// </summary>
    /// <param name="pool">The tensor pool.</param>
    /// <param name="shape">The desired tensor shape.</param>
    /// <returns>A PooledTensor that will return the tensor to the pool when disposed.</returns>
    public static PooledTensor<T> RentPooled<T>(this TensorPool<T> pool, int[] shape)
    {
        var tensor = pool.Rent(shape);
        return new PooledTensor<T>(pool, tensor);
    }
}
