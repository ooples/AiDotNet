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

    /// <summary>
    /// Iterates through all batches in the dataset using lazy evaluation.
    /// </summary>
    /// <param name="batchSize">Optional batch size override. Uses default BatchSize if null.</param>
    /// <param name="shuffle">Whether to shuffle data before batching. Default is true.</param>
    /// <param name="dropLast">Whether to drop the last incomplete batch. Default is false.</param>
    /// <param name="seed">Optional random seed for reproducible shuffling.</param>
    /// <returns>An enumerable sequence of batches using yield return for memory efficiency.</returns>
    /// <remarks>
    /// <para>
    /// This method provides a PyTorch-style iteration pattern using IEnumerable and yield return
    /// for memory-efficient lazy evaluation. Each call creates a fresh iteration, automatically
    /// handling reset and shuffle operations.
    /// </para>
    /// <para><b>For Beginners:</b> This is the recommended way to iterate through your data:
    ///
    /// <code>
    /// foreach (var (xBatch, yBatch) in dataLoader.GetBatches(batchSize: 32, shuffle: true))
    /// {
    ///     // Train on this batch
    ///     model.TrainOnBatch(xBatch, yBatch);
    /// }
    /// </code>
    ///
    /// Unlike GetNextBatch(), you don't need to call Reset() - each GetBatches() call
    /// starts fresh. The yield return pattern means batches are generated on-demand,
    /// not all loaded into memory at once.
    /// </para>
    /// </remarks>
    IEnumerable<TBatch> GetBatches(int? batchSize = null, bool shuffle = true, bool dropLast = false, int? seed = null);

    /// <summary>
    /// Asynchronously iterates through all batches with prefetching support.
    /// </summary>
    /// <param name="batchSize">Optional batch size override. Uses default BatchSize if null.</param>
    /// <param name="shuffle">Whether to shuffle data before batching. Default is true.</param>
    /// <param name="dropLast">Whether to drop the last incomplete batch. Default is false.</param>
    /// <param name="seed">Optional random seed for reproducible shuffling.</param>
    /// <param name="prefetchCount">Number of batches to prefetch ahead. Default is 2.</param>
    /// <param name="cancellationToken">Token to cancel the iteration.</param>
    /// <returns>An async enumerable sequence of batches.</returns>
    /// <remarks>
    /// <para>
    /// This method enables async batch iteration with configurable prefetching, similar to
    /// PyTorch's num_workers or TensorFlow's prefetch(). Batches are prepared in the background
    /// while the current batch is being processed.
    /// </para>
    /// <para><b>For Beginners:</b> Use this for large datasets or when batch preparation is slow:
    ///
    /// <code>
    /// await foreach (var (xBatch, yBatch) in dataLoader.GetBatchesAsync(prefetchCount: 2))
    /// {
    ///     // While training on this batch, the next 2 batches are being prepared
    ///     await model.TrainOnBatchAsync(xBatch, yBatch);
    /// }
    /// </code>
    ///
    /// Prefetching helps hide data loading latency, especially useful for:
    /// - Large images that need decoding
    /// - Data that requires preprocessing
    /// - Slow storage (network drives, cloud storage)
    /// </para>
    /// </remarks>
    IAsyncEnumerable<TBatch> GetBatchesAsync(
        int? batchSize = null,
        bool shuffle = true,
        bool dropLast = false,
        int? seed = null,
        int prefetchCount = 2,
        CancellationToken cancellationToken = default);
}
