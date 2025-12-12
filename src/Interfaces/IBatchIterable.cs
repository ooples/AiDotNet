namespace AiDotNet.Interfaces;

/// <summary>
/// Defines capability to iterate through data in batches.
/// </summary>
/// <typeparam name="TBatch">The type of batch returned by iteration.</typeparam>
/// <remarks>
/// <para>
/// Data loaders that implement this interface can provide data in batches,
/// which is the standard way to process data during model training.
/// </para>
/// <para><b>For Beginners:</b> Instead of feeding your model one example at a time,
/// batching groups multiple examples together. Training in batches is faster
/// (more efficient GPU usage) and often leads to better learning (smoother gradients).
/// </para>
/// </remarks>
public interface IBatchIterable<TBatch>
{
    /// <summary>
    /// Gets or sets the number of samples per batch.
    /// </summary>
    int BatchSize { get; set; }

    /// <summary>
    /// Gets whether there are more batches available in the current iteration.
    /// </summary>
    bool HasNext { get; }

    /// <summary>
    /// Gets the next batch of data.
    /// </summary>
    /// <returns>The next batch of data.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no more batches are available.</exception>
    TBatch GetNextBatch();

    /// <summary>
    /// Attempts to get the next batch without throwing if unavailable.
    /// </summary>
    /// <param name="batch">The batch if available, default otherwise.</param>
    /// <returns>True if a batch was available, false if iteration is complete.</returns>
    /// <remarks>
    /// When false is returned, batch contains the default value for TBatch.
    /// Callers should check the return value before using batch.
    /// </remarks>
    bool TryGetNextBatch(out TBatch batch);
}
