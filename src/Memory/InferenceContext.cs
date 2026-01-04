using System.Collections.Concurrent;

namespace AiDotNet.Memory;

/// <summary>
/// Provides a scoped context for inference operations with automatic tensor pooling and lifecycle management.
/// All tensors rented through this context are tracked and automatically returned to the pool when disposed.
/// </summary>
/// <typeparam name="T">The numeric type for tensor elements (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// InferenceContext simplifies memory management during neural network inference by:
/// - Tracking all rented tensors automatically
/// - Returning all tensors to the pool on disposal (even if Release wasn't called)
/// - Providing convenient rent methods for common tensor shapes
/// </para>
/// <para>
/// Basic usage example:
/// <code>
/// using var context = new InferenceContext&lt;float&gt;();
///
/// var input = context.Rent2D(32, 784);    // Rent a batch of inputs
/// var hidden = context.Rent2D(32, 256);   // Rent hidden layer buffer
/// var output = context.Rent2D(32, 10);    // Rent output buffer
///
/// // Perform inference operations...
///
/// // All tensors automatically returned when context is disposed
/// </code>
/// </para>
/// <para>
/// For ambient context support (avoiding parameter threading), use <see cref="InferenceScope{T}"/>:
/// <code>
/// using var context = new InferenceContext&lt;float&gt;();
/// using var scope = InferenceScope&lt;float&gt;.Begin(context);
///
/// // Now any code can access the context via InferenceScope&lt;float&gt;.Current
/// var tensor = InferenceScope&lt;float&gt;.RentOrCreate(new[] { 32, 784 });
/// </code>
/// </para>
/// </remarks>
public class InferenceContext<T> : IDisposable
{
    private readonly TensorPool<T> _pool;
    private readonly ConcurrentDictionary<Tensor<T>, byte> _rentedTensors;
    private readonly bool _ownsPool;
    private bool _disposed;

    /// <summary>
    /// Gets the underlying tensor pool used by this context.
    /// </summary>
    public TensorPool<T> Pool => _pool;

    /// <summary>
    /// Gets the number of tensors currently rented from this context.
    /// </summary>
    public int RentedTensorCount => _rentedTensors.Count;

    /// <summary>
    /// Gets or sets whether pooling is enabled. When disabled, tensors are allocated
    /// directly without pool reuse, which can be useful for debugging.
    /// </summary>
    public bool IsPoolingEnabled { get; set; } = true;

    /// <summary>
    /// Initializes a new instance of the <see cref="InferenceContext{T}"/> class.
    /// </summary>
    /// <param name="pool">
    /// An existing tensor pool to use. If null, a new pool is created and owned by this context.
    /// </param>
    /// <param name="maxPoolSizeMB">
    /// The maximum pool size in MB if creating a new pool. Ignored if pool is provided.
    /// </param>
    /// <remarks>
    /// When a pool is provided, the context does not own it and will not dispose it.
    /// When no pool is provided, the context creates and owns its own pool.
    /// </remarks>
    public InferenceContext(TensorPool<T>? pool = null, int maxPoolSizeMB = 256)
    {
        if (pool is not null)
        {
            _pool = pool;
            _ownsPool = false;
        }
        else
        {
            _pool = new TensorPool<T>(maxPoolSizeMB);
            _ownsPool = true;
        }
        _rentedTensors = new ConcurrentDictionary<Tensor<T>, byte>();
    }

    /// <summary>
    /// Rents a tensor with the specified shape from the pool.
    /// The tensor is tracked and will be automatically returned when this context is disposed.
    /// </summary>
    /// <param name="shape">The shape of the tensor to rent.</param>
    /// <returns>A tensor with the requested shape.</returns>
    /// <exception cref="ObjectDisposedException">Thrown if the context has been disposed.</exception>
    public Tensor<T> Rent(int[] shape)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(InferenceContext<T>));

        var tensor = IsPoolingEnabled ? _pool.Rent(shape) : new Tensor<T>(shape);
        _rentedTensors.TryAdd(tensor, 0);
        return tensor;
    }

    /// <summary>
    /// Rents a 1D tensor (vector) with the specified length.
    /// </summary>
    /// <param name="length">The number of elements in the tensor.</param>
    /// <returns>A 1D tensor with shape [length].</returns>
    public Tensor<T> Rent1D(int length) => Rent(new[] { length });

    /// <summary>
    /// Rents a 2D tensor (matrix) with the specified dimensions.
    /// </summary>
    /// <param name="rows">The number of rows.</param>
    /// <param name="cols">The number of columns.</param>
    /// <returns>A 2D tensor with shape [rows, cols].</returns>
    public Tensor<T> Rent2D(int rows, int cols) => Rent(new[] { rows, cols });

    /// <summary>
    /// Rents a 3D tensor with the specified dimensions.
    /// </summary>
    /// <param name="dim0">The first dimension (e.g., batch size).</param>
    /// <param name="dim1">The second dimension (e.g., sequence length).</param>
    /// <param name="dim2">The third dimension (e.g., feature size).</param>
    /// <returns>A 3D tensor with shape [dim0, dim1, dim2].</returns>
    public Tensor<T> Rent3D(int dim0, int dim1, int dim2) => Rent(new[] { dim0, dim1, dim2 });

    /// <summary>
    /// Rents a 4D tensor with the specified dimensions (typically for image data).
    /// </summary>
    /// <param name="batch">The batch size.</param>
    /// <param name="channels">The number of channels (e.g., RGB = 3).</param>
    /// <param name="height">The image height.</param>
    /// <param name="width">The image width.</param>
    /// <returns>A 4D tensor with shape [batch, channels, height, width].</returns>
    public Tensor<T> Rent4D(int batch, int channels, int height, int width) => Rent(new[] { batch, channels, height, width });

    /// <summary>
    /// Rents a tensor with the same shape as the template tensor.
    /// </summary>
    /// <param name="template">The tensor whose shape should be matched.</param>
    /// <returns>A tensor with the same shape as the template.</returns>
    /// <exception cref="ArgumentNullException">Thrown if template is null.</exception>
    public Tensor<T> RentLike(Tensor<T> template)
    {
        if (template == null)
            throw new ArgumentNullException(nameof(template));
        return Rent(template.Shape);
    }

    /// <summary>
    /// Releases a tensor back to the pool before the context is disposed.
    /// Use this for early release of tensors that are no longer needed.
    /// </summary>
    /// <param name="tensor">The tensor to release.</param>
    /// <exception cref="ObjectDisposedException">Thrown if the context has been disposed.</exception>
    /// <remarks>
    /// Releasing tensors early can reduce peak memory usage during long inference operations.
    /// Tensors not explicitly released will be automatically returned when the context is disposed.
    /// </remarks>
    public void Release(Tensor<T> tensor)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(InferenceContext<T>));

        if (tensor != null && _rentedTensors.TryRemove(tensor, out _))
        {
            if (IsPoolingEnabled)
                _pool.Return(tensor);
        }
    }

    /// <summary>
    /// Disposes the context and returns all rented tensors to the pool.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes the context resources.
    /// </summary>
    /// <param name="disposing">True if called from Dispose(), false if called from finalizer.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing && IsPoolingEnabled)
            {
                foreach (var kvp in _rentedTensors)
                    _pool.Return(kvp.Key);
                _rentedTensors.Clear();

                if (_ownsPool)
                    _pool.Dispose();
            }
            _disposed = true;
        }
    }
}

/// <summary>
/// Provides ambient (thread-local) context support for <see cref="InferenceContext{T}"/>.
/// Allows code to access the current inference context without parameter threading.
/// </summary>
/// <typeparam name="T">The numeric type for tensor elements.</typeparam>
/// <remarks>
/// <para>
/// InferenceScope enables the ambient context pattern, where a context is set once
/// and then accessible throughout the call stack without passing it as a parameter.
/// </para>
/// <para>
/// Usage example:
/// <code>
/// using var context = new InferenceContext&lt;float&gt;();
/// using var scope = InferenceScope&lt;float&gt;.Begin(context);
///
/// // Deep in the call stack, any code can now do:
/// if (InferenceScope&lt;float&gt;.IsActive)
/// {
///     var tensor = InferenceScope&lt;float&gt;.RentOrCreate(shape);
/// }
/// </code>
/// </para>
/// <para>
/// Scopes can be nested. When disposed, the previous scope is automatically restored:
/// <code>
/// using var outer = InferenceScope&lt;float&gt;.Begin(outerContext);
/// using var inner = InferenceScope&lt;float&gt;.Begin(innerContext);
/// // Current is now innerContext
/// // When inner is disposed, Current becomes outerContext again
/// </code>
/// </para>
/// </remarks>
public static class InferenceScope<T>
{
    [ThreadStatic]
    private static InferenceContext<T>? _current;

    /// <summary>
    /// Gets or sets the current inference context for this thread.
    /// Returns null if no scope is active.
    /// </summary>
    public static InferenceContext<T>? Current
    {
        get => _current;
        set => _current = value;
    }

    /// <summary>
    /// Gets whether an inference scope is currently active on this thread.
    /// </summary>
    public static bool IsActive => _current != null;

    /// <summary>
    /// Begins a new inference scope with the specified context.
    /// The previous context is saved and restored when the returned handle is disposed.
    /// </summary>
    /// <param name="context">The inference context to make current.</param>
    /// <returns>A handle that restores the previous context when disposed.</returns>
    public static InferenceScopeHandle<T> Begin(InferenceContext<T> context)
    {
        var previous = _current;
        _current = context;
        return new InferenceScopeHandle<T>(previous);
    }

    /// <summary>
    /// Rents a tensor from the current context if active, otherwise creates a new tensor.
    /// </summary>
    /// <param name="shape">The shape of the tensor to rent or create.</param>
    /// <returns>
    /// A pooled tensor if a scope is active and pooling is enabled;
    /// otherwise a newly allocated tensor.
    /// </returns>
    public static Tensor<T> RentOrCreate(int[] shape)
    {
        if (_current != null && _current.IsPoolingEnabled)
            return _current.Rent(shape);
        return new Tensor<T>(shape);
    }

    /// <summary>
    /// Rents or creates a tensor with the same shape as the template.
    /// </summary>
    /// <param name="template">The tensor whose shape should be matched.</param>
    /// <returns>A tensor with the same shape as the template.</returns>
    /// <exception cref="ArgumentNullException">Thrown if template is null.</exception>
    public static Tensor<T> RentOrCreateLike(Tensor<T> template)
    {
        if (template == null)
            throw new ArgumentNullException(nameof(template));
        return RentOrCreate(template.Shape);
    }
}

/// <summary>
/// A disposable handle that restores the previous inference context when disposed.
/// Returned by <see cref="InferenceScope{T}.Begin"/> to enable proper scope nesting.
/// </summary>
/// <typeparam name="T">The numeric type for tensor elements.</typeparam>
/// <remarks>
/// This struct should be used with a using statement to ensure the previous context
/// is properly restored even if an exception occurs.
/// </remarks>
public readonly struct InferenceScopeHandle<T> : IDisposable
{
    private readonly InferenceContext<T>? _previous;

    /// <summary>
    /// Initializes a new handle that will restore the specified context on disposal.
    /// </summary>
    /// <param name="previous">The context to restore when this handle is disposed.</param>
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
