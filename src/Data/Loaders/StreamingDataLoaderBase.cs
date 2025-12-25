using System.Runtime.CompilerServices;
using System.Threading.Channels;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Loaders;

/// <summary>
/// Abstract base class for streaming data loaders that process data on-demand.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The input data type for each sample.</typeparam>
/// <typeparam name="TOutput">The output/label data type for each sample.</typeparam>
/// <remarks>
/// <para>
/// StreamingDataLoaderBase provides the foundation for data loaders that read data on-demand
/// rather than loading everything into memory. This is essential for:
/// - Large datasets that don't fit in RAM
/// - Real-time data streams
/// - Memory-efficient training pipelines
/// </para>
/// <para><b>For Beginners:</b> When working with huge datasets (millions of images,
/// terabytes of text), you can't load everything into memory at once. This base class
/// handles the complexity of streaming data efficiently while you focus on implementing
/// the actual data reading logic.
/// </para>
/// </remarks>
public abstract class StreamingDataLoaderBase<T, TInput, TOutput> :
    DataLoaderBase<T>,
    IStreamingDataLoader<T, TInput, TOutput>
{
    private readonly int _prefetchCount;
    private readonly int _numWorkers;

    /// <summary>
    /// Initializes a new instance of the StreamingDataLoaderBase class.
    /// </summary>
    /// <param name="batchSize">Number of samples per batch.</param>
    /// <param name="prefetchCount">Number of batches to prefetch for improved throughput. Default is 2.</param>
    /// <param name="numWorkers">Number of parallel workers for sample loading. Default is 4.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when batchSize is not positive.</exception>
    protected StreamingDataLoaderBase(int batchSize, int prefetchCount = 2, int numWorkers = 4)
        : base(batchSize > 0 ? batchSize : throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be positive."))
    {
        _prefetchCount = Math.Max(1, prefetchCount);
        _numWorkers = Math.Max(1, numWorkers);
    }

    /// <inheritdoc/>
    public abstract int SampleCount { get; }

    /// <inheritdoc/>
    public override int TotalCount => SampleCount;

    /// <inheritdoc/>
    public int PrefetchCount => _prefetchCount;

    /// <inheritdoc/>
    public int NumWorkers => _numWorkers;

    /// <summary>
    /// Reads a single sample by index.
    /// </summary>
    /// <param name="index">The index of the sample to read.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A tuple containing the input and output for the sample.</returns>
    /// <remarks>
    /// Derived classes must implement this to read a single sample from the data source.
    /// This method is called by the batching infrastructure to build batches.
    /// </remarks>
    protected abstract Task<(TInput Input, TOutput Output)> ReadSampleAsync(
        int index,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Aggregates multiple samples into a batch.
    /// </summary>
    /// <param name="samples">The individual samples to aggregate.</param>
    /// <returns>A tuple of arrays containing the batched inputs and outputs.</returns>
    /// <remarks>
    /// Override this method if you need custom batching logic (e.g., padding sequences
    /// to the same length, or stacking tensors along a new dimension).
    /// </remarks>
    protected virtual (TInput[] Inputs, TOutput[] Outputs) AggregateSamples(
        IList<(TInput Input, TOutput Output)> samples)
    {
        var inputs = new TInput[samples.Count];
        var outputs = new TOutput[samples.Count];

        for (int i = 0; i < samples.Count; i++)
        {
            inputs[i] = samples[i].Input;
            outputs[i] = samples[i].Output;
        }

        return (inputs, outputs);
    }

    /// <inheritdoc/>
    public virtual IEnumerable<(TInput[] Inputs, TOutput[] Outputs)> GetBatches(
        bool shuffle = true,
        bool dropLast = false,
        int? seed = null)
    {
        EnsureLoaded();

        int[] indices = GetShuffledIndices(shuffle, seed);
        int numBatches = dropLast
            ? SampleCount / BatchSize
            : (SampleCount + BatchSize - 1) / BatchSize;

        for (int b = 0; b < numBatches; b++)
        {
            int startIdx = b * BatchSize;
            int endIdx = Math.Min(startIdx + BatchSize, indices.Length);

            if (dropLast && endIdx - startIdx < BatchSize)
            {
                continue;
            }

            // Read samples for this batch
            var samples = new List<(TInput Input, TOutput Output)>(endIdx - startIdx);
            for (int i = startIdx; i < endIdx; i++)
            {
                // Synchronously read the sample (for sync iteration)
                var sample = ReadSampleAsync(indices[i], CancellationToken.None)
                    .GetAwaiter().GetResult();
                samples.Add(sample);
            }

            yield return AggregateSamples(samples);
        }
    }

    /// <inheritdoc/>
    public virtual async IAsyncEnumerable<(TInput[] Inputs, TOutput[] Outputs)> GetBatchesAsync(
        bool shuffle = true,
        bool dropLast = false,
        int? seed = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        EnsureLoaded();

        int[] indices = GetShuffledIndices(shuffle, seed);
        int numBatches = dropLast
            ? SampleCount / BatchSize
            : (SampleCount + BatchSize - 1) / BatchSize;

        // Create bounded channel for prefetching
        var channel = Channel.CreateBounded<(TInput[] Inputs, TOutput[] Outputs)>(
            new BoundedChannelOptions(_prefetchCount)
            {
                FullMode = BoundedChannelFullMode.Wait,
                SingleReader = true,
                SingleWriter = true
            });

        // Start producer task with bounded parallelism
        var producerTask = Task.Run(async () =>
        {
            // Use semaphore to limit concurrent sample reads to _numWorkers
            using var semaphore = new SemaphoreSlim(_numWorkers, _numWorkers);

            try
            {
                for (int b = 0; b < numBatches; b++)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    int startIdx = b * BatchSize;
                    int endIdx = Math.Min(startIdx + BatchSize, indices.Length);

                    if (dropLast && endIdx - startIdx < BatchSize)
                    {
                        continue;
                    }

                    // Read samples for this batch using parallel workers with bounded concurrency
                    var samples = new (TInput Input, TOutput Output)[endIdx - startIdx];
                    var tasks = new Task[endIdx - startIdx];

                    for (int i = startIdx; i < endIdx; i++)
                    {
                        int localIdx = i - startIdx;
                        int sampleIdx = indices[i];

                        // Wait for semaphore slot before starting task
                        await semaphore.WaitAsync(cancellationToken);

                        tasks[localIdx] = Task.Run(async () =>
                        {
                            try
                            {
                                samples[localIdx] = await ReadSampleAsync(sampleIdx, cancellationToken);
                            }
                            finally
                            {
                                semaphore.Release();
                            }
                        }, cancellationToken);
                    }

                    await Task.WhenAll(tasks);

                    var batch = AggregateSamples(samples);
                    await channel.Writer.WriteAsync(batch, cancellationToken);
                }
            }
            finally
            {
                channel.Writer.Complete();
            }
        }, cancellationToken);

        // Consume batches from channel
        while (await channel.Reader.WaitToReadAsync(cancellationToken))
        {
            while (channel.Reader.TryRead(out var batch))
            {
                yield return batch;
            }
        }

        await producerTask;
    }

    /// <summary>
    /// Gets indices for iteration, optionally shuffled.
    /// </summary>
    /// <param name="shuffle">Whether to shuffle the indices.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>An array of indices in the desired order.</returns>
    protected int[] GetShuffledIndices(bool shuffle, int? seed)
    {
        var indices = new int[SampleCount];
        for (int i = 0; i < SampleCount; i++)
        {
            indices[i] = i;
        }

        if (shuffle)
        {
            var random = seed.HasValue
                ? RandomHelper.CreateSeededRandom(seed.Value)
                : RandomHelper.CreateSecureRandom();

            // Fisher-Yates shuffle
            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }

        return indices;
    }

    /// <inheritdoc/>
    protected override Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        // For streaming loaders, "loading" typically just validates the source
        // and prepares metadata. Actual data reading happens during iteration.
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        // For streaming loaders, unloading typically just releases any cached
        // resources. Override if specific cleanup is needed.
    }
}
