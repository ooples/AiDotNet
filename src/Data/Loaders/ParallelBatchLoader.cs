using System.Collections.Concurrent;
using System.Threading.Channels;
using AiDotNet.Interfaces;

namespace AiDotNet.Data.Loaders;

/// <summary>
/// Provides parallel batch loading with multiple workers for improved throughput.
/// </summary>
/// <typeparam name="TBatch">The type of batch produced.</typeparam>
/// <remarks>
/// <para>
/// ParallelBatchLoader uses multiple worker threads to prepare batches in parallel,
/// similar to PyTorch's DataLoader with num_workers > 0. This can significantly
/// improve training throughput when batch preparation is CPU-bound.
/// </para>
/// <para><b>For Beginners:</b> Training neural networks often involves:
/// 1. Loading data from disk
/// 2. Preprocessing (augmentation, normalization)
/// 3. GPU training
///
/// With single-threaded loading, the GPU waits while data is prepared.
/// With parallel loading, multiple workers prepare batches simultaneously,
/// keeping the GPU constantly fed with data.
///
/// Example:
/// <code>
/// var parallelLoader = new ParallelBatchLoader&lt;(Matrix&lt;float&gt;, Vector&lt;float&gt;)&gt;(
///     batchProvider: dataLoader.GetBatches(32),
///     numWorkers: 4,
///     prefetchCount: 2
/// );
///
/// await foreach (var batch in parallelLoader.GetBatchesAsync())
/// {
///     await model.TrainOnBatchAsync(batch);
/// }
/// </code>
/// </para>
/// </remarks>
public class ParallelBatchLoader<TBatch> : IDisposable
{
    private readonly Func<IEnumerable<int>> _indexProvider;
    private readonly Func<int[], TBatch> _batchFactory;
    private readonly int _numWorkers;
    private readonly int _prefetchCount;
    private readonly int _batchSize;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the ParallelBatchLoader class.
    /// </summary>
    /// <param name="indexProvider">Function that provides indices for each epoch.</param>
    /// <param name="batchFactory">
    /// Function that creates a batch from an array of sample indices.
    /// The factory receives all indices for a single batch and should aggregate them
    /// into a single batch (e.g., by stacking samples into a matrix).
    /// </param>
    /// <param name="batchSize">Number of samples per batch.</param>
    /// <param name="numWorkers">Number of parallel workers. Default is processor count.</param>
    /// <param name="prefetchCount">Number of batches to prefetch. Default is 2 * numWorkers.</param>
    /// <remarks>
    /// <para>
    /// The batchFactory function is responsible for:
    /// <list type="bullet">
    /// <item><description>Loading samples at the given indices from the dataset</description></item>
    /// <item><description>Aggregating them into a single batch structure</description></item>
    /// <item><description>Applying any preprocessing or augmentation</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// Example batchFactory for Matrix/Vector data:
    /// <code>
    /// batchFactory: indices => {
    ///     var xBatch = new Matrix&lt;float&gt;(indices.Length, numFeatures);
    ///     var yBatch = new Vector&lt;float&gt;(indices.Length);
    ///     for (int i = 0; i &lt; indices.Length; i++) {
    ///         int idx = indices[i];
    ///         for (int j = 0; j &lt; numFeatures; j++)
    ///             xBatch[i, j] = dataset.X[idx, j];
    ///         yBatch[i] = dataset.Y[idx];
    ///     }
    ///     return (xBatch, yBatch);
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public ParallelBatchLoader(
        Func<IEnumerable<int>> indexProvider,
        Func<int[], TBatch> batchFactory,
        int batchSize,
        int? numWorkers = null,
        int? prefetchCount = null)
    {
        _indexProvider = indexProvider ?? throw new ArgumentNullException(nameof(indexProvider));
        _batchFactory = batchFactory ?? throw new ArgumentNullException(nameof(batchFactory));
        _batchSize = batchSize > 0 ? batchSize : throw new ArgumentOutOfRangeException(nameof(batchSize));

        _numWorkers = numWorkers ?? Environment.ProcessorCount;
        _numWorkers = Math.Max(1, _numWorkers);

        _prefetchCount = prefetchCount ?? (_numWorkers * 2);
        _prefetchCount = Math.Max(1, _prefetchCount);
    }

    /// <summary>
    /// Gets the number of parallel workers.
    /// </summary>
    public int NumWorkers => _numWorkers;

    /// <summary>
    /// Gets the prefetch count.
    /// </summary>
    public int PrefetchCount => _prefetchCount;

    /// <summary>
    /// Iterates through batches using parallel workers.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Async enumerable of batches.</returns>
    public async IAsyncEnumerable<TBatch> GetBatchesAsync(
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(ParallelBatchLoader<TBatch>));
        }

        // Create bounded channel for output batches
        var outputChannel = Channel.CreateBounded<TBatch>(
            new BoundedChannelOptions(_prefetchCount)
            {
                FullMode = BoundedChannelFullMode.Wait,
                SingleReader = true,
                SingleWriter = false
            });

        // Create concurrent queue for work items
        var workQueue = new ConcurrentQueue<int[]>();

        // Prepare work items (batch indices)
        var indices = _indexProvider().ToArray();
        int numBatches = (indices.Length + _batchSize - 1) / _batchSize;

        for (int b = 0; b < numBatches; b++)
        {
            int start = b * _batchSize;
            int end = Math.Min(start + _batchSize, indices.Length);
            int[] batchIndices = new int[end - start];
            Array.Copy(indices, start, batchIndices, 0, batchIndices.Length);
            workQueue.Enqueue(batchIndices);
        }

        // Track completed workers and capture any worker exceptions
        int completedWorkers = 0;
        Exception? workerException = null;

        // Start worker tasks
        var workerTasks = new Task[_numWorkers];
        for (int w = 0; w < _numWorkers; w++)
        {
            workerTasks[w] = Task.Run(async () =>
            {
                try
                {
                    while (workQueue.TryDequeue(out var batchIndices))
                    {
                        cancellationToken.ThrowIfCancellationRequested();

                        // Pass all indices to the batch factory so it can properly
                        // aggregate all samples into a single batch
                        if (batchIndices.Length > 0)
                        {
                            var batch = _batchFactory(batchIndices);
                            await outputChannel.Writer.WriteAsync(batch, cancellationToken);
                        }
                    }
                }
                catch (OperationCanceledException)
                {
                    // Cancellation is expected, don't treat as error
                    throw;
                }
                catch (Exception ex)
                {
                    // Capture the first worker exception to propagate
                    Interlocked.CompareExchange(ref workerException, ex, null);
                }
                finally
                {
                    // Track worker completion - complete channel when last worker finishes
                    if (Interlocked.Increment(ref completedWorkers) == _numWorkers)
                    {
                        // Complete the channel, passing any captured exception
                        outputChannel.Writer.Complete(workerException);
                    }
                }
            }, cancellationToken);
        }

        // Consume batches (net471 compatible - no ReadAllAsync)
        while (await outputChannel.Reader.WaitToReadAsync(cancellationToken))
        {
            while (outputChannel.Reader.TryRead(out var batch))
            {
                yield return batch;
            }
        }

        // Wait for all workers and propagate any exceptions
        await Task.WhenAll(workerTasks);

        // If a worker exception was captured but not propagated through the channel,
        // throw it now to ensure no silent failures
        if (workerException != null)
        {
            throw new AggregateException("One or more parallel batch workers failed.", workerException);
        }
    }

    /// <summary>
    /// Disposes the parallel batch loader.
    /// </summary>
    public void Dispose()
    {
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// Configuration for parallel batch loading.
/// </summary>
public class ParallelBatchLoaderConfig
{
    /// <summary>
    /// Gets or sets the number of worker threads.
    /// </summary>
    /// <remarks>
    /// More workers can improve throughput but use more CPU and memory.
    /// Recommended: 2-8 workers depending on CPU cores and batch preparation time.
    /// </remarks>
    public int NumWorkers { get; set; } = Environment.ProcessorCount;

    /// <summary>
    /// Gets or sets the number of batches to prefetch.
    /// </summary>
    /// <remarks>
    /// Higher prefetch counts reduce GPU idle time but increase memory usage.
    /// Recommended: 2-4 batches per worker.
    /// </remarks>
    public int PrefetchCount { get; set; } = 4;

    /// <summary>
    /// Gets or sets whether to pin memory for faster CPU-to-GPU transfer.
    /// </summary>
    /// <remarks>
    /// Only applicable when using GPU acceleration.
    /// </remarks>
    public bool PinMemory { get; set; }

    /// <summary>
    /// Gets or sets the timeout for worker operations in milliseconds.
    /// </summary>
    public int WorkerTimeoutMs { get; set; } = 30000;

    /// <summary>
    /// Gets or sets whether to use persistent workers that stay alive between epochs.
    /// </summary>
    /// <remarks>
    /// Persistent workers avoid thread creation overhead but use more memory.
    /// </remarks>
    public bool PersistentWorkers { get; set; }
}

/// <summary>
/// Provides extension methods for parallel batch loading.
/// </summary>
public static class ParallelBatchLoaderExtensions
{
    /// <summary>
    /// Wraps a batch iterable with parallel loading support.
    /// </summary>
    /// <typeparam name="TBatch">The batch type.</typeparam>
    /// <param name="source">The source batch iterable.</param>
    /// <param name="batchFactory">
    /// Function to create a batch from an array of sample indices.
    /// The factory receives all indices for a single batch and should aggregate them.
    /// </param>
    /// <param name="batchSize">The batch size (number of samples per batch).</param>
    /// <param name="numWorkers">Number of parallel workers. Default is processor count.</param>
    /// <param name="prefetchCount">Number of batches to prefetch. Default is 2 * numWorkers.</param>
    /// <returns>A parallel batch loader configured with the specified parameters.</returns>
    /// <example>
    /// <code>
    /// var parallelLoader = dataLoader.WithParallelLoading(
    ///     batchFactory: indices => {
    ///         var xBatch = new Matrix&lt;float&gt;(indices.Length, numFeatures);
    ///         var yBatch = new Vector&lt;float&gt;(indices.Length);
    ///         for (int i = 0; i &lt; indices.Length; i++) {
    ///             int idx = indices[i];
    ///             // Copy sample data at idx to position i in batch
    ///             for (int j = 0; j &lt; numFeatures; j++)
    ///                 xBatch[i, j] = fullDataset.X[idx, j];
    ///             yBatch[i] = fullDataset.Y[idx];
    ///         }
    ///         return (xBatch, yBatch);
    ///     },
    ///     batchSize: 32,
    ///     numWorkers: 4
    /// );
    /// </code>
    /// </example>
    public static ParallelBatchLoader<TBatch> WithParallelLoading<TBatch>(
        this IBatchIterable<TBatch> source,
        Func<int[], TBatch> batchFactory,
        int batchSize,
        int? numWorkers = null,
        int? prefetchCount = null)
    {
        Func<IEnumerable<int>> indexProvider = () =>
            Enumerable.Range(0, source.TotalCount());

        return new ParallelBatchLoader<TBatch>(
            indexProvider,
            batchFactory,
            batchSize,
            numWorkers,
            prefetchCount);
    }
}

/// <summary>
/// Extension interface for types that expose TotalCount.
/// </summary>
internal static class BatchIterableExtensions
{
    /// <summary>
    /// Gets the total count from a batch iterable if available.
    /// </summary>
    public static int TotalCount<TBatch>(this IBatchIterable<TBatch> source)
    {
        // Check if source implements ICountable
        if (source is ICountable countable)
        {
            return countable.TotalCount;
        }

        // Fallback: count by iterating (expensive)
        return source.GetBatches(shuffle: false).Count();
    }
}
