using System.Collections.Concurrent;

namespace AiDotNet.Data.Loaders;

/// <summary>
/// Wraps a batch-producing function with asynchronous prefetching for pipelined data loading.
/// </summary>
/// <remarks>
/// <para>
/// Prefetching overlaps data loading with model computation by preparing the next N batches
/// in advance on a background thread. This hides I/O latency and keeps the GPU busy.
/// </para>
/// </remarks>
/// <typeparam name="TBatch">The batch type produced by the data source.</typeparam>
public class PrefetchDataLoader<TBatch> : IDisposable
{
    private readonly PrefetchDataLoaderOptions _options;
    private readonly BlockingCollection<TBatch> _buffer;
    private CancellationTokenSource? _cts;
    private Task? _prefetchTask;
    private bool _disposed;

    /// <summary>
    /// Gets the number of batches currently buffered.
    /// </summary>
    public int BufferedCount => _buffer.Count;

    /// <summary>
    /// Creates a new prefetch data loader.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    public PrefetchDataLoader(PrefetchDataLoaderOptions? options = null)
    {
        _options = options ?? new PrefetchDataLoaderOptions();
        _buffer = new BlockingCollection<TBatch>(_options.PrefetchCount);
    }

    /// <summary>
    /// Starts prefetching batches from the provided source.
    /// </summary>
    /// <param name="batchProducer">Function that yields batches.</param>
    /// <returns>An enumerable of prefetched batches.</returns>
    public IEnumerable<TBatch> Prefetch(IEnumerable<TBatch> batchProducer)
    {
        _cts = new CancellationTokenSource();
        var token = _cts.Token;

        // Clear any leftover items
        while (_buffer.TryTake(out _)) { }

        if (_options.UseBackgroundThread)
        {
            _prefetchTask = Task.Run(() =>
            {
                try
                {
                    foreach (var batch in batchProducer)
                    {
                        if (token.IsCancellationRequested) break;
                        _buffer.Add(batch, token);
                    }
                }
                catch (OperationCanceledException) { }
                finally
                {
                    _buffer.CompleteAdding();
                }
            }, token);
        }
        else
        {
            // Synchronous mode: just pass through
            foreach (var batch in batchProducer)
            {
                yield return batch;
            }
            yield break;
        }

        // Consume prefetched batches
        foreach (var batch in _buffer.GetConsumingEnumerable(token))
        {
            yield return batch;
        }
    }

    /// <summary>
    /// Stops prefetching and cleans up resources.
    /// </summary>
    public void Stop()
    {
        _cts?.Cancel();
        try { _prefetchTask?.Wait(TimeSpan.FromMilliseconds(_options.TimeoutMs)); }
        catch (AggregateException) { /* Expected on cancellation */ }
        _cts?.Dispose();
        _cts = null;
        _prefetchTask = null;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            Stop();
            _buffer.Dispose();
            _disposed = true;
        }
    }
}
