namespace AiDotNet.Memory;

/// <summary>
/// A thread-safe typed wrapper for pooling tensors of a specific element type.
/// </summary>
/// <remarks>
/// <para>
/// TensorPool&lt;T&gt; provides a convenient typed interface for tensor pooling,
/// backed by the UnifiedTensorPool. This allows type-safe renting and returning
/// of tensors without mixing element types.
/// </para>
/// <para><b>For Beginners:</b> This is like a specialized section of the tensor "library"
/// that only handles tensors of one type (like all books being in English).
/// You borrow a tensor, use it, and return it when done for others to reuse.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for tensor elements.</typeparam>
public class TensorPool<T> : IDisposable
{
    private readonly UnifiedTensorPool _backingPool;
    private readonly bool _ownsPool;
    private bool _disposed;

    /// <summary>
    /// Gets the current total size of pooled tensors in bytes.
    /// </summary>
    public long CurrentPoolSizeBytes => _backingPool.CurrentMemoryBytes;

    /// <summary>
    /// Gets the maximum allowed pool size in bytes.
    /// </summary>
    public long MaxPoolSizeBytes => _backingPool.Options.MaxPoolSizeBytes;

    /// <summary>
    /// Gets the number of tensors currently pooled.
    /// </summary>
    public int TotalPooledTensors => _backingPool.GetStatistics().PooledTensorCount;

    /// <summary>
    /// Gets the underlying unified pool.
    /// </summary>
    public UnifiedTensorPool UnderlyingPool => _backingPool;

    /// <summary>
    /// Initializes a new instance of the <see cref="TensorPool{T}"/> class with default settings.
    /// </summary>
    /// <param name="maxPoolSizeMB">Maximum total size of pooled tensors in megabytes. Default is 256 MB.</param>
    public TensorPool(int maxPoolSizeMB = 256)
    {
        _backingPool = new UnifiedTensorPool(new PoolingOptions
        {
            MaxPoolSizeMB = maxPoolSizeMB
        });
        _ownsPool = true;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="TensorPool{T}"/> class using an existing pool.
    /// </summary>
    /// <param name="backingPool">The underlying pool to use.</param>
    public TensorPool(UnifiedTensorPool backingPool)
    {
        _backingPool = backingPool ?? throw new ArgumentNullException(nameof(backingPool));
        _ownsPool = false;
    }

    /// <summary>
    /// Rents a tensor with the specified shape from the pool.
    /// </summary>
    /// <param name="shape">The desired shape of the tensor.</param>
    /// <returns>
    /// A tensor with the specified shape. The tensor is cleared before return.
    /// </returns>
    /// <remarks>
    /// <para>
    /// If a suitable tensor exists in the pool, it is returned. Otherwise, a new tensor
    /// is allocated.
    /// </para>
    /// </remarks>
    public Tensor<T> Rent(int[] shape)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(TensorPool<T>));
        }

        return _backingPool.RentTensor<T>(shape);
    }

    /// <summary>
    /// Rents a 1D tensor with the specified length.
    /// </summary>
    /// <param name="length">The length of the tensor.</param>
    /// <returns>A 1D tensor with the specified length.</returns>
    public Tensor<T> Rent1D(int length) => Rent(new[] { length });

    /// <summary>
    /// Rents a 2D tensor with the specified dimensions.
    /// </summary>
    /// <param name="rows">Number of rows.</param>
    /// <param name="cols">Number of columns.</param>
    /// <returns>A 2D tensor with the specified shape.</returns>
    public Tensor<T> Rent2D(int rows, int cols) => Rent(new[] { rows, cols });

    /// <summary>
    /// Returns a tensor to the pool for future reuse.
    /// </summary>
    /// <param name="tensor">The tensor to return to the pool.</param>
    public void Return(Tensor<T> tensor)
    {
        if (_disposed || tensor == null)
        {
            return;
        }

        _backingPool.ReturnTensor(tensor);
    }

    /// <summary>
    /// Rents a tensor and returns a disposable wrapper that automatically returns it.
    /// </summary>
    /// <param name="shape">The desired tensor shape.</param>
    /// <returns>A PooledTensorHandle that returns the tensor to the pool when disposed.</returns>
    public PooledTensorHandle<T> RentPooled(int[] shape)
    {
        var tensor = Rent(shape);
        return new PooledTensorHandle<T>(this, tensor);
    }

    /// <summary>
    /// Clears all tensors from the pool.
    /// </summary>
    public void Clear()
    {
        _backingPool.Clear();
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
            if (disposing && _ownsPool)
            {
                _backingPool.Dispose();
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
public readonly struct PooledTensorHandle<T> : IDisposable
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
    public PooledTensorHandle(TensorPool<T> pool, Tensor<T> tensor)
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
    public static implicit operator Tensor<T>(PooledTensorHandle<T> handle) => handle.Tensor;
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
    /// <returns>A PooledTensorHandle that will return the tensor to the pool when disposed.</returns>
    public static PooledTensorHandle<T> RentPooled<T>(this TensorPool<T> pool, int[] shape)
    {
        var tensor = pool.Rent(shape);
        return new PooledTensorHandle<T>(pool, tensor);
    }
}
