namespace AiDotNet.JitCompiler.Memory;

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
