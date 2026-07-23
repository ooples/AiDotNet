namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for streaming data loaders that process data on-demand without loading all data into memory.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The input data type for each sample.</typeparam>
/// <typeparam name="TOutput">The output/label data type for each sample.</typeparam>
/// <remarks>
/// <para>
/// IStreamingDataLoader is designed for datasets that are too large to fit in memory.
/// Unlike IInputOutputDataLoader which provides Features and Labels properties for all data,
/// streaming loaders read data on-demand and yield batches through iteration.
/// </para>
/// <para><b>For Beginners:</b> When your dataset is too large to fit in RAM (like millions
/// of images or text documents), you can't load it all at once. Streaming data loaders
/// solve this by reading data piece by piece as needed during training.
///
/// Example usage:
/// <code>
/// var loader = DataLoaders.FromCsv&lt;float&gt;("huge_dataset.csv", parseRow);
/// loader.BatchSize = 32; // Set batch size before iteration
///
/// await foreach (var (inputs, labels) in loader.GetBatchesAsync())
/// {
///     await model.TrainOnBatchAsync(inputs, labels);
/// }
/// </code>
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("StreamingDataLoader")]
public interface IStreamingDataLoader<T, TInput, TOutput> : IDataLoader<T>
{
    /// <summary>
    /// Gets the total number of samples in the dataset.
    /// </summary>
    /// <remarks>
    /// This may be known upfront (e.g., from file metadata) or estimated.
    /// For truly streaming sources where the count is unknown, this may return -1.
    /// </remarks>
    int SampleCount { get; }

    /// <summary>
    /// Gets or sets the batch size for iteration.
    /// </summary>
    int BatchSize { get; set; }

    /// <summary>
    /// Gets the number of batches to prefetch for improved throughput.
    /// </summary>
    int PrefetchCount { get; }

    /// <summary>
    /// Gets the number of parallel workers for sample loading.
    /// </summary>
    int NumWorkers { get; }

    /// <summary>
    /// Iterates through the dataset in batches synchronously.
    /// </summary>
    /// <param name="shuffle">Whether to shuffle the data before batching.</param>
    /// <param name="dropLast">Whether to drop the last incomplete batch.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>An enumerable of batches, each containing arrays of inputs and outputs.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this when you want simple, synchronous iteration.
    /// Each iteration gives you a batch of inputs and their corresponding outputs.
    /// </para>
    /// </remarks>
    IEnumerable<(TInput[] Inputs, TOutput[] Outputs)> GetBatches(
        bool shuffle = true,
        bool dropLast = false,
        int? seed = null);

    /// <summary>
    /// Iterates through the dataset in batches asynchronously with prefetching.
    /// </summary>
    /// <param name="shuffle">Whether to shuffle the data before batching.</param>
    /// <param name="dropLast">Whether to drop the last incomplete batch.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <param name="cancellationToken">Token to cancel the iteration.</param>
    /// <returns>An async enumerable of batches, each containing arrays of inputs and outputs.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the recommended method for training.
    /// It uses async/await to overlap data loading with model training, keeping
    /// your GPU busy while the next batch is being prepared.
    ///
    /// Example:
    /// <code>
    /// await foreach (var batch in loader.GetBatchesAsync(shuffle: true))
    /// {
    ///     await model.TrainOnBatchAsync(batch.Inputs, batch.Outputs);
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    IAsyncEnumerable<(TInput[] Inputs, TOutput[] Outputs)> GetBatchesAsync(
        bool shuffle = true,
        bool dropLast = false,
        int? seed = null,
        CancellationToken cancellationToken = default);
}
