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
    private BlockingCollection<TBatch> _buffer;
    private CancellationTokenSource? _cts;
    private Task? _prefetchTask;
    private bool _disposed;

    /// <summary>
    /// Gets the number of batches currently buffered.
    /// </summary>
    public int BufferedCount => _disposed ? 0 : _buffer.Count;

    /// <summary>
    /// Creates a new prefetch data loader.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    public PrefetchDataLoader(PrefetchDataLoaderOptions? options = null)
    {
        _options = options ?? new PrefetchDataLoaderOptions();
        _options.Validate();
        _buffer = new BlockingCollection<TBatch>(_options.PrefetchCount);
    }

    /// <summary>
    /// Starts prefetching batches from the provided source.
    /// </summary>
    /// <param name="batchProducer">Function that yields batches.</param>
    /// <returns>An enumerable of prefetched batches.</returns>
    public IEnumerable<TBatch> Prefetch(IEnumerable<TBatch> batchProducer)
    {
        if (batchProducer == null) throw new ArgumentNullException(nameof(batchProducer));
        if (_disposed) throw new ObjectDisposedException(nameof(PrefetchDataLoader<TBatch>));
        // Stop any previous prefetch and create a fresh buffer
        // (BlockingCollection cannot be reused after CompleteAdding)
        Stop();
        _buffer = new BlockingCollection<TBatch>(_options.PrefetchCount);
        _cts = new CancellationTokenSource();
        var token = _cts.Token;

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
        if (_disposed) return;
        _cts?.Cancel();
        try
        {
            if (_prefetchTask != null && !_prefetchTask.Wait(TimeSpan.FromMilliseconds(_options.TimeoutMs)))
            {
                // Task didn't complete in time — drain the buffer to unblock the producer
                while (_buffer.TryTake(out _)) { }
                _prefetchTask.Wait(TimeSpan.FromMilliseconds(1000));
            }
        }
        catch (AggregateException) { /* Expected on cancellation */ }
        _cts?.Dispose();
        _cts = null;
        _prefetchTask = null;
    }

    private void DisposeBuffer()
    {
        // Drain any remaining items before disposing
        while (_buffer.TryTake(out _)) { }
        _buffer.Dispose();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            Stop();
            DisposeBuffer();
            _disposed = true;
        }
    }
}
