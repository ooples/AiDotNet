namespace AiDotNet.Memory;

/// <summary>
/// A RAII wrapper that automatically returns a pooled tensor to its pool when disposed.
/// This struct ensures tensors are properly returned to the pool even if an exception occurs.
/// </summary>
/// <typeparam name="T">The element type of the tensor (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// PooledTensor provides a safe way to use pooled tensors with the using statement,
/// ensuring the tensor is returned to the pool when it goes out of scope.
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
/// </remarks>
public readonly struct PooledTensor<T> : IDisposable
{
    private readonly TensorPool<T> _pool;

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
    /// Initializes a new instance of the <see cref="PooledTensor{T}"/> struct.
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
    }

    /// <summary>
    /// Returns the tensor to the pool.
    /// After disposal, the tensor should not be used as it may be rented by another operation.
    /// </summary>
    public void Dispose() => _pool.Return(Tensor);

    /// <summary>
    /// Explicitly converts a PooledTensor to its underlying Tensor.
    /// </summary>
    /// <param name="pooled">The pooled tensor wrapper.</param>
    /// <returns>The underlying tensor.</returns>
    /// <remarks>
    /// This cast is explicit to remind users that they are responsible for the tensor's lifecycle.
    /// The returned tensor reference should not outlive the PooledTensor wrapper.
    /// </remarks>
    public static explicit operator Tensor<T>(PooledTensor<T> pooled) => pooled.Tensor;
}
