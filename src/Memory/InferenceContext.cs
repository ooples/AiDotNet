using System.Collections.Concurrent;

namespace AiDotNet.Memory;

/// <summary>
/// Provides a scoped context for inference operations with automatic tensor pooling.
/// </summary>
/// <remarks>
/// <para>
/// InferenceContext manages temporary tensors during inference, automatically returning
/// them to the pool when the context is disposed. This reduces GC pressure during
/// repeated inference calls by reusing tensors instead of allocating new ones.
/// </para>
/// <para><b>For Beginners:</b> Think of this as a "workspace" for running neural networks.
/// When you open a workspace (create context), all temporary tensors used during inference
/// are tracked. When you close the workspace (dispose context), all those tensors are
/// automatically cleaned up and returned to the pool for reuse.
/// </para>
/// <para>
/// <b>Thread Safety:</b> Each thread should use its own InferenceContext.
/// The underlying TensorPool is thread-safe, but the context itself tracks tensors
/// per-scope and should not be shared across threads.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for tensor elements.</typeparam>
/// <example>
/// <code>
/// var pool = new TensorPool&lt;float&gt;(maxPoolSizeMB: 256);
///
/// // During inference loop
/// for (int i = 0; i &lt; batchCount; i++)
/// {
///     using var context = new InferenceContext&lt;float&gt;(pool);
///     var result = network.Forward(input, context);
///     // Process result...
///     // All temporary tensors are automatically returned to pool here
/// }
/// </code>
/// </example>
public class InferenceContext<T> : IDisposable
{
    private readonly TensorPool<T> _pool;
    private readonly ConcurrentDictionary<Tensor<T>, byte> _rentedTensors;
    private bool _disposed;

    /// <summary>
    /// Gets the underlying tensor pool.
    /// </summary>
    public TensorPool<T> Pool => _pool;

    /// <summary>
    /// Gets the number of tensors currently rented in this context.
    /// </summary>
    public int RentedTensorCount => _rentedTensors.Count;

    /// <summary>
    /// Gets or sets whether this context is active for pooling.
    /// When false, Rent() creates new tensors instead of pooling.
    /// </summary>
    public bool IsPoolingEnabled { get; set; } = true;

    /// <summary>
    /// Initializes a new instance of the <see cref="InferenceContext{T}"/> class.
    /// </summary>
    /// <param name="pool">The tensor pool to use for allocation. If null, a default pool is created.</param>
    /// <param name="maxPoolSizeMB">Maximum pool size in MB when creating default pool. Default is 256 MB.</param>
    public InferenceContext(TensorPool<T>? pool = null, int maxPoolSizeMB = 256)
    {
        _pool = pool ?? new TensorPool<T>(maxPoolSizeMB);
        _rentedTensors = new ConcurrentDictionary<Tensor<T>, byte>();
    }

    /// <summary>
    /// Rents a tensor with the specified shape from the pool.
    /// </summary>
    /// <param name="shape">The desired shape of the tensor.</param>
    /// <returns>A tensor with the specified shape.</returns>
    /// <remarks>
    /// The returned tensor is tracked by this context and will be automatically
    /// returned to the pool when the context is disposed.
    /// </remarks>
    public Tensor<T> Rent(int[] shape)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(InferenceContext<T>));
        }

        var tensor = IsPoolingEnabled ? _pool.Rent(shape) : new Tensor<T>(shape);
        _rentedTensors.TryAdd(tensor, 0);
        return tensor;
    }

    /// <summary>
    /// Rents a 1D tensor with the specified length from the pool.
    /// </summary>
    /// <param name="length">The length of the 1D tensor.</param>
    /// <returns>A 1D tensor with the specified length.</returns>
    public Tensor<T> Rent1D(int length) => Rent(new[] { length });

    /// <summary>
    /// Rents a 2D tensor with the specified dimensions from the pool.
    /// </summary>
    /// <param name="rows">Number of rows.</param>
    /// <param name="cols">Number of columns.</param>
    /// <returns>A 2D tensor with the specified shape.</returns>
    public Tensor<T> Rent2D(int rows, int cols) => Rent(new[] { rows, cols });

    /// <summary>
    /// Rents a 3D tensor with the specified dimensions from the pool.
    /// </summary>
    /// <param name="dim0">Size of the first dimension.</param>
    /// <param name="dim1">Size of the second dimension.</param>
    /// <param name="dim2">Size of the third dimension.</param>
    /// <returns>A 3D tensor with the specified shape.</returns>
    public Tensor<T> Rent3D(int dim0, int dim1, int dim2) => Rent(new[] { dim0, dim1, dim2 });

    /// <summary>
    /// Rents a 4D tensor with the specified dimensions from the pool.
    /// </summary>
    /// <param name="batch">Batch size (first dimension).</param>
    /// <param name="channels">Number of channels (second dimension).</param>
    /// <param name="height">Height (third dimension).</param>
    /// <param name="width">Width (fourth dimension).</param>
    /// <returns>A 4D tensor with the specified shape.</returns>
    public Tensor<T> Rent4D(int batch, int channels, int height, int width)
        => Rent(new[] { batch, channels, height, width });

    /// <summary>
    /// Rents a tensor with the same shape as an existing tensor.
    /// </summary>
    /// <param name="template">The tensor whose shape to match.</param>
    /// <returns>A tensor with the same shape as the template.</returns>
    public Tensor<T> RentLike(Tensor<T> template)
    {
        if (template == null)
        {
            throw new ArgumentNullException(nameof(template));
        }

        return Rent(template.Shape);
    }

    /// <summary>
    /// Marks a tensor as no longer needed and returns it to the pool immediately.
    /// </summary>
    /// <param name="tensor">The tensor to return.</param>
    /// <remarks>
    /// This is optional - all tensors are automatically returned when the context
    /// is disposed. Use this when you know a tensor is no longer needed to free
    /// it for reuse earlier.
    /// </remarks>
    public void Release(Tensor<T> tensor)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(InferenceContext<T>));
        }

        if (tensor != null && _rentedTensors.TryRemove(tensor, out _))
        {
            // Only return to pool if we successfully removed from tracking
            // This prevents double-return if Release is called multiple times
            // or if Dispose is called after Release
            if (IsPoolingEnabled)
            {
                _pool.Return(tensor);
            }
        }
    }

    /// <summary>
    /// Returns all rented tensors to the pool and releases resources.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Returns all rented tensors to the pool and releases resources.
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing && IsPoolingEnabled)
            {
                // Return all remaining rented tensors to the pool
                // Tensors that were already released via Release() won't be in the dictionary
                foreach (var kvp in _rentedTensors)
                {
                    _pool.Return(kvp.Key);
                }
                _rentedTensors.Clear();
            }
            _disposed = true;
        }
    }
}

/// <summary>
/// Static class providing ambient context support for InferenceContext.
/// </summary>
/// <remarks>
/// <para>
/// This class provides thread-local storage for an ambient InferenceContext,
/// allowing layers to access pooling without explicit parameter passing.
/// </para>
/// <para><b>For Beginners:</b> This is like a "shared workspace" that layers
/// can automatically use. Instead of passing the context to every method,
/// you set it once and layers can find it automatically.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for tensor elements.</typeparam>
public static class InferenceScope<T>
{
    [ThreadStatic]
    private static InferenceContext<T>? _current;

    /// <summary>
    /// Gets or sets the current inference context for this thread.
    /// </summary>
    /// <value>
    /// The current inference context, or null if no context is active.
    /// </value>
    public static InferenceContext<T>? Current
    {
        get => _current;
        set => _current = value;
    }

    /// <summary>
    /// Gets a value indicating whether an inference context is currently active.
    /// </summary>
    public static bool IsActive => _current != null;

    /// <summary>
    /// Begins a new inference scope with the specified context.
    /// </summary>
    /// <param name="context">The inference context to use.</param>
    /// <returns>A disposable scope that restores the previous context when disposed.</returns>
    /// <example>
    /// <code>
    /// using var ctx = new InferenceContext&lt;float&gt;(pool);
    /// using (InferenceScope&lt;float&gt;.Begin(ctx))
    /// {
    ///     // Layers can now access InferenceScope&lt;float&gt;.Current
    ///     var result = network.Forward(input);
    /// }
    /// </code>
    /// </example>
    public static InferenceScopeHandle<T> Begin(InferenceContext<T> context)
    {
        var previous = _current;
        _current = context;
        return new InferenceScopeHandle<T>(previous);
    }

    /// <summary>
    /// Rents a tensor from the current context, or creates a new one if no context is active.
    /// </summary>
    /// <param name="shape">The desired shape of the tensor.</param>
    /// <returns>A tensor with the specified shape.</returns>
    /// <remarks>
    /// If an inference context is active, the tensor is rented from the pool.
    /// Otherwise, a new tensor is allocated. This allows layers to use pooling
    /// when available without requiring explicit context parameters.
    /// </remarks>
    public static Tensor<T> RentOrCreate(int[] shape)
    {
        if (_current != null && _current.IsPoolingEnabled)
        {
            return _current.Rent(shape);
        }
        return new Tensor<T>(shape);
    }

    /// <summary>
    /// Rents a tensor with the same shape as an existing tensor.
    /// </summary>
    /// <param name="template">The tensor whose shape to match.</param>
    /// <returns>A tensor with the same shape as the template.</returns>
    public static Tensor<T> RentOrCreateLike(Tensor<T> template)
    {
        if (template == null)
        {
            throw new ArgumentNullException(nameof(template));
        }

        return RentOrCreate(template.Shape);
    }
}

/// <summary>
/// A disposable handle that restores the previous inference context when disposed.
/// </summary>
/// <typeparam name="T">The numeric type for tensor elements.</typeparam>
public readonly struct InferenceScopeHandle<T> : IDisposable
{
    private readonly InferenceContext<T>? _previous;

    internal InferenceScopeHandle(InferenceContext<T>? previous)
    {
        _previous = previous;
    }

    /// <summary>
    /// Restores the previous inference context.
    /// </summary>
    public void Dispose()
    {
        InferenceScope<T>.Current = _previous;
    }
}
