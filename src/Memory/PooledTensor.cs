namespace AiDotNet.Memory;

/// <summary>
/// A RAII wrapper that automatically returns a pooled tensor to its pool when disposed.
/// This class ensures tensors are properly returned to the pool even if an exception occurs.
/// </summary>
/// <typeparam name="T">The element type of the tensor (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// PooledTensor provides a safe way to use pooled tensors with the using statement,
/// ensuring the tensor is returned to the pool when it goes out of scope.
/// </para>
/// <para>
/// This is implemented as a class (not struct) to ensure reference semantics.
/// Multiple references to the same PooledTensor share disposal state, preventing
/// double-dispose issues that could corrupt the pool.
/// </para>
/// <para>
/// Example usage:
/// <code>
/// var pool = new TensorPool&lt;float&gt;();
/// using (var pooled = pool.RentPooled(new[] { 32, 32 }))
/// {
///     var tensor = pooled.Tensor;
///     // Use tensor for computations...
///     // Tensor is automatically returned to pool when block exits
/// }
/// </code>
/// </para>
/// <para>
/// Alternative explicit cast usage:
/// <code>
/// using var pooled = pool.RentPooled(new[] { 32, 32 });
/// Tensor&lt;float&gt; tensor = (Tensor&lt;float&gt;)pooled;
/// </code>
/// </para>
/// <para>
/// <b>Thread Safety:</b> The Dispose method is idempotent - calling it multiple times
/// is safe and will only return the tensor to the pool once.
/// </para>
/// </remarks>
public sealed class PooledTensor<T> : IDisposable
{
    private readonly TensorPool<T> _pool;
    private int _disposed; // 0 = not disposed, 1 = disposed (int for Interlocked)

    /// <summary>
    /// Gets the underlying tensor managed by this wrapper.
    /// </summary>
    /// <value>
    /// The pooled tensor instance.
    /// </value>
    /// <remarks>
    /// Access the tensor through this property for computations.
    /// Do not hold a reference to this tensor after the PooledTensor is disposed.
    /// </remarks>
    public Tensor<T> Tensor { get; }

    /// <summary>
    /// Gets whether this wrapper has been disposed and the tensor returned to the pool.
    /// </summary>
    public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

    /// <summary>
    /// Initializes a new instance of the <see cref="PooledTensor{T}"/> class.
    /// </summary>
    /// <param name="pool">The tensor pool that owns this tensor and will receive it back on disposal.</param>
    /// <param name="tensor">The tensor being wrapped.</param>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="pool"/> or <paramref name="tensor"/> is null.
    /// </exception>
    /// <remarks>
    /// Typically you should not create PooledTensor instances directly.
    /// Instead, use <see cref="TensorPool{T}.RentPooled"/> to obtain a properly initialized wrapper.
    /// </remarks>
    public PooledTensor(TensorPool<T> pool, Tensor<T> tensor)
    {
        _pool = pool ?? throw new ArgumentNullException(nameof(pool));
        Tensor = tensor ?? throw new ArgumentNullException(nameof(tensor));
        _disposed = 0;
    }

    /// <summary>
    /// Returns the tensor to the pool.
    /// After disposal, the tensor should not be used as it may be rented by another operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method is idempotent - calling it multiple times is safe and will only
    /// return the tensor to the pool on the first call.
    /// </para>
    /// <para>
    /// Thread-safe: Uses atomic compare-and-swap to ensure only one thread
    /// can successfully dispose and return the tensor, even under concurrent calls.
    /// </para>
    /// </remarks>
    public void Dispose()
    {
        // Atomically transition from 0 (not disposed) to 1 (disposed)
        // Only the thread that successfully makes this transition returns the tensor
        if (Interlocked.CompareExchange(ref _disposed, 1, 0) == 0)
        {
            _pool.Return(Tensor);
        }
    }

    /// <summary>
    /// Explicitly converts a PooledTensor to its underlying Tensor.
    /// </summary>
    /// <param name="pooled">The pooled tensor wrapper.</param>
    /// <returns>The underlying tensor.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="pooled"/> is null.</exception>
    /// <remarks>
    /// This cast is explicit to remind users that they are responsible for the tensor's lifecycle.
    /// The returned tensor reference should not outlive the PooledTensor wrapper.
    /// </remarks>
    public static explicit operator Tensor<T>(PooledTensor<T> pooled)
    {
        if (pooled == null)
            throw new ArgumentNullException(nameof(pooled));
        return pooled.Tensor;
    }
}
