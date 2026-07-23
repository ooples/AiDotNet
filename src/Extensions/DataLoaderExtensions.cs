using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Extensions;

/// <summary>
/// Provides extension methods for data loaders to enhance batch iteration capabilities.
/// </summary>
/// <remarks>
/// <para>
/// These extension methods provide a fluent API for batch iteration, enabling
/// PyTorch-style and TensorFlow-style data loading patterns.
/// </para>
/// <para><b>For Beginners:</b> Extension methods add new capabilities to existing types.
/// These methods make it easy to iterate through your data in batches with a clean syntax:
///
/// <code>
/// // Fluent API for batch iteration
/// foreach (var batch in dataLoader.CreateBatches(batchSize: 32).Shuffled().DropLast())
/// {
///     model.TrainOnBatch(batch);
/// }
/// </code>
/// </para>
/// </remarks>
public static class DataLoaderExtensions
{
    /// <summary>
    /// Creates a batch configuration builder for fluent batch iteration.
    /// </summary>
    /// <typeparam name="TBatch">The type of batch returned by the data loader.</typeparam>
    /// <param name="source">The batch iterable source.</param>
    /// <param name="batchSize">Optional batch size override.</param>
    /// <returns>A batch configuration builder for fluent configuration.</returns>
    /// <remarks>
    /// <para>
    /// This method starts a fluent chain for configuring batch iteration.
    /// Call methods like Shuffled(), DropLast(), WithSeed() to configure,
    /// then iterate using foreach or ToList().
    /// </para>
    /// <para><b>For Beginners:</b> This is the starting point for creating batches:
    ///
    /// <code>
    /// // Basic usage
    /// var batches = dataLoader.CreateBatches(batchSize: 32);
    ///
    /// // With configuration
    /// var batches = dataLoader.CreateBatches(batchSize: 32)
    ///     .Shuffled()
    ///     .DropLast()
    ///     .WithSeed(42);
    ///
    /// // Iterate
    /// foreach (var batch in batches)
    /// {
    ///     // Process batch
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public static BatchConfigurationBuilder<TBatch> CreateBatches<TBatch>(
        this IBatchIterable<TBatch> source,
        int? batchSize = null)
    {
        return new BatchConfigurationBuilder<TBatch>(source, batchSize);
    }

    /// <summary>
    /// Creates an async batch configuration builder for fluent async batch iteration.
    /// </summary>
    /// <typeparam name="TBatch">The type of batch returned by the data loader.</typeparam>
    /// <param name="source">The batch iterable source.</param>
    /// <param name="batchSize">Optional batch size override.</param>
    /// <param name="prefetchCount">Number of batches to prefetch. Default is 2.</param>
    /// <returns>An async batch configuration builder for fluent configuration.</returns>
    /// <remarks>
    /// <para>
    /// This method starts a fluent chain for configuring async batch iteration with prefetching.
    /// Prefetching prepares batches in the background while the current batch is being processed.
    /// </para>
    /// <para><b>For Beginners:</b> Use this for async iteration with prefetching:
    ///
    /// <code>
    /// // Async iteration with prefetching
    /// await foreach (var batch in dataLoader.CreateBatchesAsync(batchSize: 32, prefetchCount: 3))
    /// {
    ///     await model.TrainOnBatchAsync(batch);
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public static AsyncBatchConfigurationBuilder<TBatch> CreateBatchesAsync<TBatch>(
        this IBatchIterable<TBatch> source,
        int? batchSize = null,
        int prefetchCount = 2)
    {
        return new AsyncBatchConfigurationBuilder<TBatch>(source, batchSize, prefetchCount);
    }
}

/// <summary>
/// Builder for configuring batch iteration with a fluent API.
/// </summary>
/// <typeparam name="TBatch">The type of batch returned by iteration.</typeparam>
/// <remarks>
/// <para>
/// This builder allows chaining configuration methods before iteration.
/// Implements IEnumerable to support foreach loops and LINQ operations.
/// </para>
/// </remarks>
public class BatchConfigurationBuilder<TBatch> : IEnumerable<TBatch>
{
    private readonly IBatchIterable<TBatch> _source;
    private readonly int? _batchSize;
    private bool _shuffle = true;
    private bool _dropLast = false;
    private int? _seed = null;

    /// <summary>
    /// Initializes a new instance of the BatchConfigurationBuilder.
    /// </summary>
    /// <param name="source">The batch iterable source.</param>
    /// <param name="batchSize">Optional batch size override.</param>
    public BatchConfigurationBuilder(IBatchIterable<TBatch> source, int? batchSize)
    {
        Guard.NotNull(source);
        _source = source;
        _batchSize = batchSize;
    }

    /// <summary>
    /// Enables shuffling of data before batching.
    /// </summary>
    /// <returns>This builder for method chaining.</returns>
    /// <remarks>
    /// Shuffling is enabled by default. Call this explicitly for clarity or after NoShuffle().
    /// </remarks>
    public BatchConfigurationBuilder<TBatch> Shuffled()
    {
        _shuffle = true;
        return this;
    }

    /// <summary>
    /// Disables shuffling of data before batching.
    /// </summary>
    /// <returns>This builder for method chaining.</returns>
    /// <remarks>
    /// Use this when you need deterministic iteration order, such as during evaluation.
    /// </remarks>
    public BatchConfigurationBuilder<TBatch> NoShuffle()
    {
        _shuffle = false;
        return this;
    }

    /// <summary>
    /// Drops the last incomplete batch if the dataset doesn't divide evenly.
    /// </summary>
    /// <returns>This builder for method chaining.</returns>
    /// <remarks>
    /// Useful when your model requires consistent batch sizes (e.g., batch normalization).
    /// </remarks>
    public BatchConfigurationBuilder<TBatch> DropLast()
    {
        _dropLast = true;
        return this;
    }

    /// <summary>
    /// Keeps the last batch even if incomplete.
    /// </summary>
    /// <returns>This builder for method chaining.</returns>
    /// <remarks>
    /// This is the default behavior. Call this explicitly for clarity or after DropLast().
    /// </remarks>
    public BatchConfigurationBuilder<TBatch> KeepLast()
    {
        _dropLast = false;
        return this;
    }

    /// <summary>
    /// Sets a random seed for reproducible shuffling.
    /// </summary>
    /// <param name="seed">The random seed value.</param>
    /// <returns>This builder for method chaining.</returns>
    /// <remarks>
    /// Setting a seed ensures the same shuffle order each time for reproducibility.
    /// </remarks>
    public BatchConfigurationBuilder<TBatch> WithSeed(int seed)
    {
        _seed = seed;
        return this;
    }

    /// <summary>
    /// Returns an enumerator that iterates through the batches.
    /// </summary>
    /// <returns>An enumerator for the batch sequence.</returns>
    public IEnumerator<TBatch> GetEnumerator()
    {
        return _source.GetBatches(_batchSize, _shuffle, _dropLast, _seed).GetEnumerator();
    }

    /// <summary>
    /// Returns an enumerator that iterates through the batches.
    /// </summary>
    /// <returns>An enumerator for the batch sequence.</returns>
    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }
}

/// <summary>
/// Builder for configuring async batch iteration with a fluent API.
/// </summary>
/// <typeparam name="TBatch">The type of batch returned by iteration.</typeparam>
/// <remarks>
/// <para>
/// This builder allows chaining configuration methods before async iteration.
/// Implements IAsyncEnumerable to support await foreach loops.
/// </para>
/// </remarks>
public class AsyncBatchConfigurationBuilder<TBatch> : IAsyncEnumerable<TBatch>
{
    private readonly IBatchIterable<TBatch> _source;
    private readonly int? _batchSize;
    private readonly int _prefetchCount;
    private bool _shuffle = true;
    private bool _dropLast = false;
    private int? _seed = null;

    /// <summary>
    /// Initializes a new instance of the AsyncBatchConfigurationBuilder.
    /// </summary>
    /// <param name="source">The batch iterable source.</param>
    /// <param name="batchSize">Optional batch size override.</param>
    /// <param name="prefetchCount">Number of batches to prefetch.</param>
    public AsyncBatchConfigurationBuilder(IBatchIterable<TBatch> source, int? batchSize, int prefetchCount)
    {
        Guard.NotNull(source);
        _source = source;
        _batchSize = batchSize;
        _prefetchCount = prefetchCount;
    }

    /// <summary>
    /// Enables shuffling of data before batching.
    /// </summary>
    /// <returns>This builder for method chaining.</returns>
    public AsyncBatchConfigurationBuilder<TBatch> Shuffled()
    {
        _shuffle = true;
        return this;
    }

    /// <summary>
    /// Disables shuffling of data before batching.
    /// </summary>
    /// <returns>This builder for method chaining.</returns>
    public AsyncBatchConfigurationBuilder<TBatch> NoShuffle()
    {
        _shuffle = false;
        return this;
    }

    /// <summary>
    /// Drops the last incomplete batch if the dataset doesn't divide evenly.
    /// </summary>
    /// <returns>This builder for method chaining.</returns>
    public AsyncBatchConfigurationBuilder<TBatch> DropLast()
    {
        _dropLast = true;
        return this;
    }

    /// <summary>
    /// Keeps the last batch even if incomplete.
    /// </summary>
    /// <returns>This builder for method chaining.</returns>
    public AsyncBatchConfigurationBuilder<TBatch> KeepLast()
    {
        _dropLast = false;
        return this;
    }

    /// <summary>
    /// Sets a random seed for reproducible shuffling.
    /// </summary>
    /// <param name="seed">The random seed value.</param>
    /// <returns>This builder for method chaining.</returns>
    public AsyncBatchConfigurationBuilder<TBatch> WithSeed(int seed)
    {
        _seed = seed;
        return this;
    }

    /// <summary>
    /// Returns an async enumerator that iterates through the batches.
    /// </summary>
    /// <param name="cancellationToken">Token to cancel the iteration.</param>
    /// <returns>An async enumerator for the batch sequence.</returns>
    public IAsyncEnumerator<TBatch> GetAsyncEnumerator(CancellationToken cancellationToken = default)
    {
        return _source.GetBatchesAsync(_batchSize, _shuffle, _dropLast, _seed, _prefetchCount, cancellationToken)
            .GetAsyncEnumerator(cancellationToken);
    }
}
