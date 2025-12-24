using System.Collections.Concurrent;
using System.Threading.Channels;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Pipeline;

/// <summary>
/// Provides TensorFlow-style data pipeline operations for transforming and processing data.
/// </summary>
/// <typeparam name="T">The type of data in the pipeline.</typeparam>
/// <remarks>
/// <para>
/// DataPipeline provides a fluent API for building data processing pipelines similar to
/// TensorFlow's tf.data API. Operations are lazily evaluated and can be chained together.
/// </para>
/// <para><b>For Beginners:</b> Data pipelines let you build complex data processing
/// workflows step by step. Each operation transforms the data in some way:
///
/// <code>
/// var pipeline = DataPipeline.From(dataLoader)
///     .Map(x => Normalize(x))           // Transform each sample
///     .Filter(x => IsValid(x))           // Keep only valid samples
///     .Cache()                           // Cache in memory
///     .Shuffle(1000)                     // Shuffle with buffer
///     .Batch(32)                         // Create batches
///     .Prefetch(2);                      // Prefetch batches
///
/// foreach (var batch in pipeline)
/// {
///     model.Train(batch);
/// }
/// </code>
/// </para>
/// </remarks>
public class DataPipeline<T> : IEnumerable<T>
{
    private readonly Func<IEnumerable<T>> _sourceFactory;

    /// <summary>
    /// Initializes a new instance of the DataPipeline class.
    /// </summary>
    /// <param name="sourceFactory">Factory function that creates the source enumerable.</param>
    public DataPipeline(Func<IEnumerable<T>> sourceFactory)
    {
        _sourceFactory = sourceFactory ?? throw new ArgumentNullException(nameof(sourceFactory));
    }

    /// <summary>
    /// Gets the source factory for this pipeline.
    /// </summary>
    internal Func<IEnumerable<T>> SourceFactory => _sourceFactory;

    /// <summary>
    /// Creates a new DataPipeline from an enumerable source.
    /// </summary>
    /// <param name="source">The source enumerable.</param>
    /// <returns>A new DataPipeline.</returns>
    public static DataPipeline<T> From(IEnumerable<T> source)
    {
        return new DataPipeline<T>(() => source);
    }

    /// <summary>
    /// Creates a new DataPipeline from a batch iterable.
    /// </summary>
    /// <param name="source">The batch iterable source.</param>
    /// <returns>A new DataPipeline.</returns>
    public static DataPipeline<T> From(IBatchIterable<T> source)
    {
        return new DataPipeline<T>(() => source.GetBatches());
    }

    /// <summary>
    /// Applies a transformation function to each element.
    /// </summary>
    /// <typeparam name="TResult">The type of the transformed elements.</typeparam>
    /// <param name="selector">The transformation function.</param>
    /// <returns>A new DataPipeline with transformed elements.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Map transforms each element. For example,
    /// normalizing pixel values or converting data types.
    /// </para>
    /// </remarks>
    public DataPipeline<TResult> Map<TResult>(Func<T, TResult> selector)
    {
        var source = _sourceFactory;
        return new DataPipeline<TResult>(() => source().Select(selector));
    }

    /// <summary>
    /// Applies an async transformation function to each element.
    /// </summary>
    /// <typeparam name="TResult">The type of the transformed elements.</typeparam>
    /// <param name="selector">The async transformation function.</param>
    /// <param name="maxConcurrency">Maximum concurrent operations. Default is 4.</param>
    /// <returns>A new AsyncDataPipeline with transformed elements.</returns>
    public AsyncDataPipeline<TResult> MapAsync<TResult>(
        Func<T, CancellationToken, Task<TResult>> selector,
        int maxConcurrency = 4)
    {
        var source = _sourceFactory;
        return new AsyncDataPipeline<TResult>(
            new MapAsyncIterator<T, TResult>(source, selector, maxConcurrency));
    }

    /// <summary>
    /// Filters elements based on a predicate.
    /// </summary>
    /// <param name="predicate">The filter predicate.</param>
    /// <returns>A new DataPipeline with filtered elements.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Filter keeps only elements that match a condition.
    /// For example, removing corrupted samples or samples below a quality threshold.
    /// </para>
    /// </remarks>
    public DataPipeline<T> Filter(Func<T, bool> predicate)
    {
        var source = _sourceFactory;
        return new DataPipeline<T>(() => source().Where(predicate));
    }

    /// <summary>
    /// Caches elements in memory for faster subsequent iterations.
    /// </summary>
    /// <returns>A new DataPipeline with cached elements.</returns>
    /// <remarks>
    /// <para>
    /// Cache stores all elements in memory after the first iteration.
    /// Subsequent iterations read from the cache instead of recomputing.
    /// </para>
    /// <para><b>For Beginners:</b> Caching is useful when you iterate through
    /// the same data multiple times (like training for many epochs) and the
    /// data fits in memory. The first epoch loads/computes data, and subsequent
    /// epochs read from RAM.
    /// </para>
    /// </remarks>
    public DataPipeline<T> Cache()
    {
        List<T>? cachedData = null;
        object cacheLock = new();
        var source = _sourceFactory;

        return new DataPipeline<T>(() =>
        {
            lock (cacheLock)
            {
                if (cachedData == null)
                {
                    cachedData = source().ToList();
                }
                return cachedData;
            }
        });
    }

    /// <summary>
    /// Shuffles elements using a buffer.
    /// </summary>
    /// <param name="bufferSize">Size of the shuffle buffer.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>A new DataPipeline with shuffled elements.</returns>
    /// <remarks>
    /// <para>
    /// Shuffle maintains a buffer of elements and randomly samples from it.
    /// Larger buffers provide more randomness but use more memory.
    /// </para>
    /// <para><b>For Beginners:</b> Shuffling data helps the model learn better
    /// by preventing it from memorizing the order. A shuffle buffer of 1000
    /// means 1000 samples are kept in memory and randomly selected from.
    /// </para>
    /// </remarks>
    public DataPipeline<T> Shuffle(int bufferSize, int? seed = null)
    {
        var source = _sourceFactory;
        return new DataPipeline<T>(() => ShuffleIterator(source(), bufferSize, seed));
    }

    private static IEnumerable<T> ShuffleIterator(IEnumerable<T> source, int bufferSize, int? seed)
    {
        Random random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        var buffer = new List<T>(bufferSize);

        foreach (var item in source)
        {
            if (buffer.Count < bufferSize)
            {
                buffer.Add(item);
            }
            else
            {
                int idx = random.Next(bufferSize);
                yield return buffer[idx];
                buffer[idx] = item;
            }
        }

        // Shuffle and yield remaining buffer
        for (int i = buffer.Count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (buffer[i], buffer[j]) = (buffer[j], buffer[i]);
        }

        foreach (var item in buffer)
        {
            yield return item;
        }
    }

    /// <summary>
    /// Groups elements into batches.
    /// </summary>
    /// <param name="batchSize">Number of elements per batch.</param>
    /// <param name="dropRemainder">Whether to drop the last incomplete batch.</param>
    /// <returns>A new DataPipeline of batched elements.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Batch groups individual samples together
    /// for more efficient processing. A batch size of 32 means 32 samples
    /// are processed together.
    /// </para>
    /// </remarks>
    public DataPipeline<T[]> Batch(int batchSize, bool dropRemainder = false)
    {
        var source = _sourceFactory;
        return new DataPipeline<T[]>(() => BatchIterator(source(), batchSize, dropRemainder));
    }

    private static IEnumerable<T[]> BatchIterator(IEnumerable<T> source, int batchSize, bool dropRemainder)
    {
        var batch = new List<T>(batchSize);

        foreach (var item in source)
        {
            batch.Add(item);
            if (batch.Count == batchSize)
            {
                yield return batch.ToArray();
                batch.Clear();
            }
        }

        if (batch.Count > 0 && !dropRemainder)
        {
            yield return batch.ToArray();
        }
    }

    /// <summary>
    /// Takes only the first N elements.
    /// </summary>
    /// <param name="count">Number of elements to take.</param>
    /// <returns>A new DataPipeline with limited elements.</returns>
    public DataPipeline<T> Take(int count)
    {
        var source = _sourceFactory;
        return new DataPipeline<T>(() => source().Take(count));
    }

    /// <summary>
    /// Skips the first N elements.
    /// </summary>
    /// <param name="count">Number of elements to skip.</param>
    /// <returns>A new DataPipeline with skipped elements.</returns>
    public DataPipeline<T> Skip(int count)
    {
        var source = _sourceFactory;
        return new DataPipeline<T>(() => source().Skip(count));
    }

    /// <summary>
    /// Repeats the pipeline indefinitely or a specified number of times.
    /// </summary>
    /// <param name="count">Number of times to repeat. Null for infinite repeat.</param>
    /// <returns>A new DataPipeline that repeats.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Repeat allows iterating through the data
    /// multiple times. With count=null, it repeats forever (useful with Take
    /// to limit total samples).
    /// </para>
    /// </remarks>
    public DataPipeline<T> Repeat(int? count = null)
    {
        var source = _sourceFactory;
        return new DataPipeline<T>(() => RepeatIterator(source, count));
    }

    private static IEnumerable<T> RepeatIterator(Func<IEnumerable<T>> sourceFactory, int? count)
    {
        if (count == null)
        {
            while (true)
            {
                foreach (var item in sourceFactory())
                {
                    yield return item;
                }
            }
        }
        else
        {
            for (int i = 0; i < count.Value; i++)
            {
                foreach (var item in sourceFactory())
                {
                    yield return item;
                }
            }
        }
    }

    /// <summary>
    /// Concatenates another pipeline to this one.
    /// </summary>
    /// <param name="other">The pipeline to concatenate.</param>
    /// <returns>A new DataPipeline with concatenated elements.</returns>
    public DataPipeline<T> Concat(DataPipeline<T> other)
    {
        var source = _sourceFactory;
        var otherSource = other._sourceFactory;
        return new DataPipeline<T>(() => source().Concat(otherSource()));
    }

    /// <summary>
    /// Interleaves multiple pipelines.
    /// </summary>
    /// <param name="pipelines">The pipelines to interleave with.</param>
    /// <returns>A new DataPipeline with interleaved elements.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Interleave mixes elements from multiple sources.
    /// This is useful when you have data from different domains and want to
    /// mix them during training.
    /// </para>
    /// </remarks>
    public DataPipeline<T> Interleave(params DataPipeline<T>[] pipelines)
    {
        var allPipelines = new[] { this }.Concat(pipelines).ToArray();
        return new DataPipeline<T>(() => InterleaveIterator(allPipelines));
    }

    private static IEnumerable<T> InterleaveIterator(DataPipeline<T>[] pipelines)
    {
        var enumerators = pipelines.Select(p => p.GetEnumerator()).ToList();
        try
        {
            bool hasAny = true;
            while (hasAny)
            {
                hasAny = false;
                // Use explicit index iteration to avoid implicit filtering pattern
                // Each enumerator is checked for remaining items in round-robin fashion
                for (int i = 0; i < enumerators.Count; i++)
                {
                    var enumerator = enumerators[i];
                    if (enumerator.MoveNext())
                    {
                        hasAny = true;
                        yield return enumerator.Current;
                    }
                }
            }
        }
        finally
        {
            // Dispose all enumerators with exception aggregation to ensure cleanup
            List<Exception>? disposeExceptions = null;
            for (int i = 0; i < enumerators.Count; i++)
            {
                try
                {
                    enumerators[i].Dispose();
                }
                catch (Exception ex)
                {
                    disposeExceptions ??= new List<Exception>();
                    disposeExceptions.Add(ex);
                }
            }

            if (disposeExceptions is { Count: > 0 })
            {
                throw new AggregateException("One or more enumerators failed to dispose.", disposeExceptions);
            }
        }
    }

    /// <summary>
    /// Zips this pipeline with another to create pairs.
    /// </summary>
    /// <typeparam name="TOther">The type of the other pipeline.</typeparam>
    /// <param name="other">The other pipeline.</param>
    /// <returns>A new DataPipeline of tuples.</returns>
    public DataPipeline<(T, TOther)> Zip<TOther>(DataPipeline<TOther> other)
    {
        var source = _sourceFactory;
        var otherSource = other._sourceFactory;
        return new DataPipeline<(T, TOther)>(() =>
            source().Zip(otherSource(), (a, b) => (a, b)));
    }

    /// <summary>
    /// Flattens a pipeline by extracting enumerables from each element.
    /// </summary>
    /// <typeparam name="TElement">The element type within the enumerables.</typeparam>
    /// <param name="selector">Function to extract the enumerable from each element.</param>
    /// <returns>A flattened DataPipeline.</returns>
    public DataPipeline<TElement> Flatten<TElement>(Func<T, IEnumerable<TElement>> selector)
    {
        var source = _sourceFactory;
        return new DataPipeline<TElement>(() => source().SelectMany(selector));
    }

    /// <summary>
    /// Applies a side effect to each element without modifying it.
    /// </summary>
    /// <param name="action">The action to perform.</param>
    /// <returns>A new DataPipeline that performs the action.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> ForEach is useful for logging, debugging,
    /// or collecting statistics without changing the data.
    /// </para>
    /// </remarks>
    public DataPipeline<T> ForEach(Action<T> action)
    {
        var source = _sourceFactory;
        return new DataPipeline<T>(() => source().Select(x =>
        {
            action(x);
            return x;
        }));
    }

    /// <summary>
    /// Prefetches elements in the background for improved performance.
    /// </summary>
    /// <param name="bufferSize">Number of elements to prefetch.</param>
    /// <returns>An async DataPipeline with prefetching.</returns>
    /// <remarks>
    /// <para>
    /// Prefetch prepares the next elements while the current ones are being processed.
    /// This hides data loading latency.
    /// </para>
    /// <para><b>For Beginners:</b> Prefetching is like having someone prepare
    /// the next ingredients while you're cooking. It keeps the pipeline flowing
    /// smoothly without waiting for data.
    /// </para>
    /// </remarks>
    public AsyncDataPipeline<T> Prefetch(int bufferSize = 2)
    {
        var source = _sourceFactory;
        return new AsyncDataPipeline<T>(new PrefetchIterator<T>(source, bufferSize));
    }

    /// <inheritdoc/>
    public IEnumerator<T> GetEnumerator()
    {
        return _sourceFactory().GetEnumerator();
    }

    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }
}

/// <summary>
/// Internal iterator for async map operations.
/// </summary>
internal class MapAsyncIterator<TSource, TResult> : IAsyncEnumerableProvider<TResult>
{
    private readonly Func<IEnumerable<TSource>> _source;
    private readonly Func<TSource, CancellationToken, Task<TResult>> _selector;
    private readonly int _maxConcurrency;

    public MapAsyncIterator(
        Func<IEnumerable<TSource>> source,
        Func<TSource, CancellationToken, Task<TResult>> selector,
        int maxConcurrency)
    {
        _source = source;
        _selector = selector;
        _maxConcurrency = maxConcurrency;
    }

    public async IAsyncEnumerable<TResult> GetAsyncEnumerable(
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct)
    {
        // Use a bounded channel for output with capacity = maxConcurrency to limit memory
        var outputChannel = Channel.CreateBounded<TResult>(
            new BoundedChannelOptions(_maxConcurrency)
            {
                FullMode = BoundedChannelFullMode.Wait,
                SingleReader = true,
                SingleWriter = false
            });

        // Create a work queue from the source
        var workQueue = new ConcurrentQueue<TSource>();
        foreach (var item in _source())
        {
            workQueue.Enqueue(item);
        }

        // Track completed workers
        int completedWorkers = 0;
        Exception? workerException = null;

        // Start worker tasks
        var workerTasks = new Task[_maxConcurrency];
        for (int w = 0; w < _maxConcurrency; w++)
        {
            workerTasks[w] = Task.Run(async () =>
            {
                try
                {
                    while (workQueue.TryDequeue(out var item))
                    {
                        ct.ThrowIfCancellationRequested();
                        var result = await _selector(item, ct);
                        await outputChannel.Writer.WriteAsync(result, ct);
                    }
                }
                catch (OperationCanceledException)
                {
                    throw;
                }
                catch (Exception ex)
                {
                    Interlocked.CompareExchange(ref workerException, ex, null);
                }
                finally
                {
                    if (Interlocked.Increment(ref completedWorkers) == _maxConcurrency)
                    {
                        outputChannel.Writer.Complete(workerException);
                    }
                }
            }, ct);
        }

        // Consume results from channel (net471 compatible - no ReadAllAsync)
        while (await outputChannel.Reader.WaitToReadAsync(ct))
        {
            while (outputChannel.Reader.TryRead(out var result))
            {
                yield return result;
            }
        }

        // Wait for all workers and propagate any exceptions
        await Task.WhenAll(workerTasks);

        if (workerException is not null)
        {
            throw new AggregateException("One or more parallel map workers failed.", workerException);
        }
    }
}

/// <summary>
/// Internal iterator for prefetch operations.
/// </summary>
internal class PrefetchIterator<T> : IAsyncEnumerableProvider<T>
{
    private readonly Func<IEnumerable<T>> _source;
    private readonly int _bufferSize;

    public PrefetchIterator(Func<IEnumerable<T>> source, int bufferSize)
    {
        _source = source;
        _bufferSize = bufferSize;
    }

    public async IAsyncEnumerable<T> GetAsyncEnumerable(
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct)
    {
        var channel = Channel.CreateBounded<T>(new BoundedChannelOptions(_bufferSize)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = true,
            SingleWriter = true
        });

        var producerTask = Task.Run(async () =>
        {
            try
            {
                foreach (var item in _source())
                {
                    ct.ThrowIfCancellationRequested();
                    await channel.Writer.WriteAsync(item, ct);
                }
            }
            finally
            {
                channel.Writer.Complete();
            }
        }, ct);

        // Consume items (net471 compatible)
        while (await channel.Reader.WaitToReadAsync(ct))
        {
            while (channel.Reader.TryRead(out var item))
            {
                yield return item;
            }
        }

        await producerTask;
    }
}

/// <summary>
/// Interface for types that provide async enumerables.
/// </summary>
internal interface IAsyncEnumerableProvider<T>
{
    IAsyncEnumerable<T> GetAsyncEnumerable(CancellationToken ct);
}

/// <summary>
/// Provides async data pipeline operations with prefetching support.
/// </summary>
/// <typeparam name="T">The type of data in the pipeline.</typeparam>
public class AsyncDataPipeline<T> : IAsyncEnumerable<T>
{
    private readonly IAsyncEnumerableProvider<T> _provider;

    /// <summary>
    /// Initializes a new instance of the AsyncDataPipeline class.
    /// </summary>
    /// <param name="provider">Provider that creates the async source.</param>
    internal AsyncDataPipeline(IAsyncEnumerableProvider<T> provider)
    {
        _provider = provider ?? throw new ArgumentNullException(nameof(provider));
    }

    /// <summary>
    /// Applies a transformation function to each element.
    /// </summary>
    /// <typeparam name="TResult">The type of the transformed elements.</typeparam>
    /// <param name="selector">The transformation function.</param>
    /// <returns>A new AsyncDataPipeline with transformed elements.</returns>
    public AsyncDataPipeline<TResult> Map<TResult>(Func<T, TResult> selector)
    {
        return new AsyncDataPipeline<TResult>(new AsyncMapIterator<T, TResult>(_provider, selector));
    }

    /// <summary>
    /// Applies an async transformation function to each element.
    /// </summary>
    /// <typeparam name="TResult">The type of the transformed elements.</typeparam>
    /// <param name="selector">The async transformation function.</param>
    /// <returns>A new AsyncDataPipeline with transformed elements.</returns>
    public AsyncDataPipeline<TResult> MapAsync<TResult>(Func<T, CancellationToken, Task<TResult>> selector)
    {
        return new AsyncDataPipeline<TResult>(new AsyncMapAsyncIterator<T, TResult>(_provider, selector));
    }

    /// <summary>
    /// Filters elements based on a predicate.
    /// </summary>
    /// <param name="predicate">The filter predicate.</param>
    /// <returns>A new AsyncDataPipeline with filtered elements.</returns>
    public AsyncDataPipeline<T> Filter(Func<T, bool> predicate)
    {
        return new AsyncDataPipeline<T>(new AsyncFilterIterator<T>(_provider, predicate));
    }

    /// <summary>
    /// Groups elements into batches.
    /// </summary>
    /// <param name="batchSize">Number of elements per batch.</param>
    /// <param name="dropRemainder">Whether to drop the last incomplete batch.</param>
    /// <returns>A new AsyncDataPipeline of batched elements.</returns>
    public AsyncDataPipeline<T[]> Batch(int batchSize, bool dropRemainder = false)
    {
        return new AsyncDataPipeline<T[]>(new AsyncBatchIterator<T>(_provider, batchSize, dropRemainder));
    }

    /// <summary>
    /// Takes only the first N elements.
    /// </summary>
    /// <param name="count">Number of elements to take.</param>
    /// <returns>A new AsyncDataPipeline with limited elements.</returns>
    public AsyncDataPipeline<T> Take(int count)
    {
        return new AsyncDataPipeline<T>(new AsyncTakeIterator<T>(_provider, count));
    }

    /// <summary>
    /// Prefetches elements in the background for improved performance.
    /// </summary>
    /// <param name="bufferSize">Number of elements to prefetch.</param>
    /// <returns>A new AsyncDataPipeline with prefetching.</returns>
    public AsyncDataPipeline<T> Prefetch(int bufferSize = 2)
    {
        return new AsyncDataPipeline<T>(new AsyncPrefetchIterator<T>(_provider, bufferSize));
    }

    /// <inheritdoc/>
    public IAsyncEnumerator<T> GetAsyncEnumerator(CancellationToken cancellationToken = default)
    {
        return _provider.GetAsyncEnumerable(cancellationToken).GetAsyncEnumerator(cancellationToken);
    }
}

internal class AsyncMapIterator<TSource, TResult> : IAsyncEnumerableProvider<TResult>
{
    private readonly IAsyncEnumerableProvider<TSource> _source;
    private readonly Func<TSource, TResult> _selector;

    public AsyncMapIterator(IAsyncEnumerableProvider<TSource> source, Func<TSource, TResult> selector)
    {
        _source = source;
        _selector = selector;
    }

    public async IAsyncEnumerable<TResult> GetAsyncEnumerable(
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct)
    {
        await foreach (var item in _source.GetAsyncEnumerable(ct).WithCancellation(ct))
        {
            yield return _selector(item);
        }
    }
}

internal class AsyncMapAsyncIterator<TSource, TResult> : IAsyncEnumerableProvider<TResult>
{
    private readonly IAsyncEnumerableProvider<TSource> _source;
    private readonly Func<TSource, CancellationToken, Task<TResult>> _selector;

    public AsyncMapAsyncIterator(
        IAsyncEnumerableProvider<TSource> source,
        Func<TSource, CancellationToken, Task<TResult>> selector)
    {
        _source = source;
        _selector = selector;
    }

    public async IAsyncEnumerable<TResult> GetAsyncEnumerable(
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct)
    {
        await foreach (var item in _source.GetAsyncEnumerable(ct).WithCancellation(ct))
        {
            yield return await _selector(item, ct);
        }
    }
}

internal class AsyncFilterIterator<T> : IAsyncEnumerableProvider<T>
{
    private readonly IAsyncEnumerableProvider<T> _source;
    private readonly Func<T, bool> _predicate;

    public AsyncFilterIterator(IAsyncEnumerableProvider<T> source, Func<T, bool> predicate)
    {
        _source = source;
        _predicate = predicate;
    }

    public async IAsyncEnumerable<T> GetAsyncEnumerable(
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct)
    {
        await foreach (var item in _source.GetAsyncEnumerable(ct).WithCancellation(ct))
        {
            if (_predicate(item))
            {
                yield return item;
            }
        }
    }
}

internal class AsyncBatchIterator<T> : IAsyncEnumerableProvider<T[]>
{
    private readonly IAsyncEnumerableProvider<T> _source;
    private readonly int _batchSize;
    private readonly bool _dropRemainder;

    public AsyncBatchIterator(IAsyncEnumerableProvider<T> source, int batchSize, bool dropRemainder)
    {
        _source = source;
        _batchSize = batchSize;
        _dropRemainder = dropRemainder;
    }

    public async IAsyncEnumerable<T[]> GetAsyncEnumerable(
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct)
    {
        var batch = new List<T>(_batchSize);

        await foreach (var item in _source.GetAsyncEnumerable(ct).WithCancellation(ct))
        {
            batch.Add(item);
            if (batch.Count == _batchSize)
            {
                yield return batch.ToArray();
                batch.Clear();
            }
        }

        if (batch.Count > 0 && !_dropRemainder)
        {
            yield return batch.ToArray();
        }
    }
}

internal class AsyncTakeIterator<T> : IAsyncEnumerableProvider<T>
{
    private readonly IAsyncEnumerableProvider<T> _source;
    private readonly int _count;

    public AsyncTakeIterator(IAsyncEnumerableProvider<T> source, int count)
    {
        _source = source;
        _count = count;
    }

    public async IAsyncEnumerable<T> GetAsyncEnumerable(
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct)
    {
        int taken = 0;
        await foreach (var item in _source.GetAsyncEnumerable(ct).WithCancellation(ct))
        {
            if (taken >= _count) yield break;
            yield return item;
            taken++;
        }
    }
}

internal class AsyncPrefetchIterator<T> : IAsyncEnumerableProvider<T>
{
    private readonly IAsyncEnumerableProvider<T> _source;
    private readonly int _bufferSize;

    public AsyncPrefetchIterator(IAsyncEnumerableProvider<T> source, int bufferSize)
    {
        _source = source;
        _bufferSize = bufferSize;
    }

    public async IAsyncEnumerable<T> GetAsyncEnumerable(
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct)
    {
        var channel = Channel.CreateBounded<T>(new BoundedChannelOptions(_bufferSize)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = true,
            SingleWriter = true
        });

        var producerTask = Task.Run(async () =>
        {
            try
            {
                await foreach (var item in _source.GetAsyncEnumerable(ct).WithCancellation(ct))
                {
                    await channel.Writer.WriteAsync(item, ct);
                }
            }
            finally
            {
                channel.Writer.Complete();
            }
        }, ct);

        // Consume items (net471 compatible)
        while (await channel.Reader.WaitToReadAsync(ct))
        {
            while (channel.Reader.TryRead(out var item))
            {
                yield return item;
            }
        }

        await producerTask;
    }
}

/// <summary>
/// Extension methods for creating data pipelines from various sources.
/// </summary>
public static class DataPipelineExtensions
{
    /// <summary>
    /// Creates a DataPipeline from a batch iterable.
    /// </summary>
    /// <typeparam name="TBatch">The batch type.</typeparam>
    /// <param name="source">The batch iterable source.</param>
    /// <returns>A new DataPipeline.</returns>
    public static DataPipeline<TBatch> ToPipeline<TBatch>(this IBatchIterable<TBatch> source)
    {
        return DataPipeline<TBatch>.From(source);
    }

    /// <summary>
    /// Creates a DataPipeline from an enumerable.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="source">The enumerable source.</param>
    /// <returns>A new DataPipeline.</returns>
    public static DataPipeline<T> ToPipeline<T>(this IEnumerable<T> source)
    {
        return DataPipeline<T>.From(source);
    }

    /// <summary>
    /// Creates a DataPipeline from a streaming data loader.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <typeparam name="TInput">The input type.</typeparam>
    /// <typeparam name="TOutput">The output type.</typeparam>
    /// <param name="loader">The streaming data loader.</param>
    /// <param name="shuffle">Whether to shuffle the data.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>A new DataPipeline of input/output tuples.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a pipeline from a streaming data loader,
    /// allowing you to chain operations like Map, Filter, Batch, etc.
    ///
    /// Example:
    /// <code>
    /// var pipeline = streamingLoader.ToPipeline()
    ///     .Map(batch => NormalizeBatch(batch))
    ///     .Shuffle(1000)
    ///     .Prefetch(2);
    /// </code>
    /// </para>
    /// </remarks>
    public static DataPipeline<(TInput[] Inputs, TOutput[] Outputs)> ToPipeline<T, TInput, TOutput>(
        this IStreamingDataLoader<T, TInput, TOutput> loader,
        bool shuffle = true,
        int? seed = null)
    {
        return new DataPipeline<(TInput[] Inputs, TOutput[] Outputs)>(
            () => loader.GetBatches(shuffle, dropLast: false, seed));
    }

    /// <summary>
    /// Creates an async DataPipeline from a streaming data loader.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <typeparam name="TInput">The input type.</typeparam>
    /// <typeparam name="TOutput">The output type.</typeparam>
    /// <param name="loader">The streaming data loader.</param>
    /// <param name="shuffle">Whether to shuffle the data.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>A new async DataPipeline of input/output tuples.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates an async pipeline from a streaming data loader,
    /// enabling efficient prefetching and parallel processing.
    ///
    /// Example:
    /// <code>
    /// await foreach (var batch in streamingLoader.ToAsyncPipeline().Prefetch(2))
    /// {
    ///     await model.TrainOnBatchAsync(batch);
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public static AsyncDataPipeline<(TInput[] Inputs, TOutput[] Outputs)> ToAsyncPipeline<T, TInput, TOutput>(
        this IStreamingDataLoader<T, TInput, TOutput> loader,
        bool shuffle = true,
        int? seed = null)
    {
        return new AsyncDataPipeline<(TInput[] Inputs, TOutput[] Outputs)>(
            new StreamingLoaderAsyncProvider<T, TInput, TOutput>(loader, shuffle, seed));
    }

    /// <summary>
    /// Creates a DataPipeline of individual samples from a streaming data loader.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <typeparam name="TInput">The input type.</typeparam>
    /// <typeparam name="TOutput">The output type.</typeparam>
    /// <param name="loader">The streaming data loader.</param>
    /// <param name="shuffle">Whether to shuffle the data.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>A new DataPipeline of individual input/output samples.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Unlike ToPipeline which yields batches, this yields
    /// individual samples. Useful when you want to apply per-sample operations before
    /// re-batching.
    ///
    /// Example:
    /// <code>
    /// var pipeline = streamingLoader.ToSamplePipeline()
    ///     .Map(sample => AugmentSample(sample))
    ///     .Shuffle(5000)
    ///     .Batch(64);
    /// </code>
    /// </para>
    /// </remarks>
    public static DataPipeline<(TInput Input, TOutput Output)> ToSamplePipeline<T, TInput, TOutput>(
        this IStreamingDataLoader<T, TInput, TOutput> loader,
        bool shuffle = true,
        int? seed = null)
    {
        return new DataPipeline<(TInput Input, TOutput Output)>(
            () => FlattenBatches(loader.GetBatches(shuffle, dropLast: false, seed)));
    }

    private static IEnumerable<(TInput Input, TOutput Output)> FlattenBatches<TInput, TOutput>(
        IEnumerable<(TInput[] Inputs, TOutput[] Outputs)> batches)
    {
        foreach (var (inputs, outputs) in batches)
        {
            for (int i = 0; i < inputs.Length && i < outputs.Length; i++)
            {
                yield return (inputs[i], outputs[i]);
            }
        }
    }

    /// <summary>
    /// Creates batches with padding to ensure uniform batch sizes.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="source">The source pipeline.</param>
    /// <param name="batchSize">Number of elements per batch.</param>
    /// <param name="padValue">Value to use for padding.</param>
    /// <returns>A new DataPipeline with padded batches.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> PaddedBatch ensures all batches have the same
    /// size by adding padding values to the last batch. This is useful when
    /// your model requires fixed batch sizes.
    /// </para>
    /// </remarks>
    public static DataPipeline<T[]> PaddedBatch<T>(
        this DataPipeline<T> source,
        int batchSize,
        T padValue)
    {
        return new DataPipeline<T[]>(() => PaddedBatchIterator(source, batchSize, padValue));
    }

    private static IEnumerable<T[]> PaddedBatchIterator<T>(
        IEnumerable<T> source,
        int batchSize,
        T padValue)
    {
        var batch = new List<T>(batchSize);

        foreach (var item in source)
        {
            batch.Add(item);
            if (batch.Count == batchSize)
            {
                yield return batch.ToArray();
                batch.Clear();
            }
        }

        if (batch.Count > 0)
        {
            // Pad to batchSize
            while (batch.Count < batchSize)
            {
                batch.Add(padValue);
            }
            yield return batch.ToArray();
        }
    }

    /// <summary>
    /// Applies window-based operations to the pipeline.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="source">The source pipeline.</param>
    /// <param name="windowSize">Size of each window.</param>
    /// <param name="shift">Number of elements to shift between windows. Default is windowSize (non-overlapping).</param>
    /// <returns>A new DataPipeline with windowed elements.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Window groups consecutive elements together.
    /// With windowSize=5 and shift=1, you get overlapping windows useful for
    /// sequence models.
    /// </para>
    /// </remarks>
    public static DataPipeline<T[]> Window<T>(
        this DataPipeline<T> source,
        int windowSize,
        int? shift = null)
    {
        int actualShift = shift ?? windowSize;
        return new DataPipeline<T[]>(() => WindowIterator(source, windowSize, actualShift));
    }

    private static IEnumerable<T[]> WindowIterator<T>(
        IEnumerable<T> source,
        int windowSize,
        int shift)
    {
        var buffer = new Queue<T>(windowSize);

        foreach (var item in source)
        {
            buffer.Enqueue(item);

            if (buffer.Count == windowSize)
            {
                yield return buffer.ToArray();

                // Remove 'shift' elements
                for (int i = 0; i < shift && buffer.Count > 0; i++)
                {
                    buffer.Dequeue();
                }
            }
        }

        // Yield final window only if it's complete (partial windows are discarded)
        if (buffer.Count == windowSize)
        {
            yield return buffer.ToArray();
        }
    }

    /// <summary>
    /// Samples elements with replacement using the given weights.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="source">The source pipeline.</param>
    /// <param name="weights">Weight for each element.</param>
    /// <param name="numSamples">Number of samples to draw.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>A new DataPipeline with sampled elements.</returns>
    public static DataPipeline<T> Sample<T>(
        this DataPipeline<T> source,
        IReadOnlyList<double> weights,
        int numSamples,
        int? seed = null)
    {
        return new DataPipeline<T>(() => SampleIterator(source.ToList(), weights, numSamples, seed));
    }

    private static IEnumerable<T> SampleIterator<T>(
        IReadOnlyList<T> source,
        IReadOnlyList<double> weights,
        int numSamples,
        int? seed)
    {
        if (source.Count != weights.Count)
        {
            throw new ArgumentException("Weights must have same length as source.");
        }

        Random random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        // Build cumulative distribution
        double sum = 0;
        for (int i = 0; i < weights.Count; i++)
        {
            sum += weights[i];
        }

        double[] cumulative = new double[weights.Count];
        double cumSum = 0;
        for (int i = 0; i < weights.Count; i++)
        {
            cumSum += weights[i] / sum;
            cumulative[i] = cumSum;
        }

        // Sample
        for (int s = 0; s < numSamples; s++)
        {
            double u = random.NextDouble();
            int idx = Array.BinarySearch(cumulative, u);
            if (idx < 0) idx = ~idx;
            if (idx >= source.Count) idx = source.Count - 1;
            yield return source[idx];
        }
    }
}

/// <summary>
/// Internal provider for async streaming data loader iteration.
/// </summary>
internal class StreamingLoaderAsyncProvider<T, TInput, TOutput> :
    IAsyncEnumerableProvider<(TInput[] Inputs, TOutput[] Outputs)>
{
    private readonly IStreamingDataLoader<T, TInput, TOutput> _loader;
    private readonly bool _shuffle;
    private readonly int? _seed;

    public StreamingLoaderAsyncProvider(
        IStreamingDataLoader<T, TInput, TOutput> loader,
        bool shuffle,
        int? seed)
    {
        _loader = loader;
        _shuffle = shuffle;
        _seed = seed;
    }

    public async IAsyncEnumerable<(TInput[] Inputs, TOutput[] Outputs)> GetAsyncEnumerable(
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct)
    {
        await foreach (var batch in _loader.GetBatchesAsync(_shuffle, dropLast: false, _seed, ct))
        {
            yield return batch;
        }
    }
}
