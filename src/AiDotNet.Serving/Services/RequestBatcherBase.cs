using System.Collections.Concurrent;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Scheduling;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.Extensions.Logging;

namespace AiDotNet.Serving.Services;

/// <summary>
/// Base class for request batchers that provides common functionality for batching inference requests.
/// </summary>
/// <remarks>
/// <para>
/// This base class provides shared implementation details for request batchers including:
/// - Statistics tracking (total requests, batches, latency)
/// - Thread-safe queue management
/// - Common configuration handling
/// </para>
/// <para><b>For Beginners:</b> A request batcher collects multiple inference requests and processes them together.
///
/// Benefits of batching:
/// - Higher throughput (more predictions per second)
/// - Better GPU/hardware utilization
/// - Lower per-request cost
///
/// This base class provides common functionality that all batchers need,
/// while allowing different batching strategies (timeout-based, size-based, continuous, etc.)
/// </para>
/// </remarks>
public abstract class RequestBatcherBase : IRequestBatcher, IDisposable
{
    /// <summary>
    /// Model repository for accessing loaded models.
    /// </summary>
    protected readonly IModelRepository ModelRepository;

    /// <summary>
    /// Logger for diagnostics.
    /// </summary>
    protected readonly ILogger Logger;

    /// <summary>
    /// Serving configuration options.
    /// </summary>
    protected readonly ServingOptions Options;

    /// <summary>
    /// Lock object for thread-safe statistics updates.
    /// </summary>
    protected readonly object StatsLock = new();

    /// <summary>
    /// Total number of requests received.
    /// </summary>
    protected long TotalRequests;

    /// <summary>
    /// Total number of batches processed.
    /// </summary>
    protected long TotalBatches;

    /// <summary>
    /// Sum of all batch sizes for averaging.
    /// </summary>
    protected long TotalBatchSize;

    /// <summary>
    /// Sum of all processing times in milliseconds.
    /// </summary>
    protected double TotalLatencyMs;

    /// <summary>
    /// Track whether the object has been disposed.
    /// </summary>
    protected bool Disposed;

    /// <summary>
    /// Initializes a new instance of the RequestBatcherBase.
    /// </summary>
    /// <param name="modelRepository">The model repository for accessing loaded models.</param>
    /// <param name="logger">Logger for diagnostics.</param>
    /// <param name="options">Serving options configuration.</param>
    protected RequestBatcherBase(
        IModelRepository modelRepository,
        ILogger logger,
        ServingOptions options)
    {
        ModelRepository = modelRepository ?? throw new ArgumentNullException(nameof(modelRepository));
        Logger = logger ?? throw new ArgumentNullException(nameof(logger));
        Options = options ?? throw new ArgumentNullException(nameof(options));
    }

    /// <summary>
    /// Queues a prediction request to be processed in the next batch.
    /// </summary>
    /// <typeparam name="T">The numeric type used by the model.</typeparam>
    /// <param name="modelName">The name of the model to use for prediction.</param>
    /// <param name="input">The input features.</param>
    /// <param name="priority">The priority level for this request.</param>
    /// <returns>A task that completes with the prediction result.</returns>
    public abstract Task<Vector<T>> QueueRequest<T>(string modelName, Vector<T> input, RequestPriority priority = RequestPriority.Normal);

    /// <summary>
    /// Gets statistics about the batcher's performance.
    /// </summary>
    /// <returns>A dictionary containing batcher statistics.</returns>
    public virtual Dictionary<string, object> GetStatistics()
    {
        lock (StatsLock)
        {
            return new Dictionary<string, object>
            {
                ["totalRequests"] = TotalRequests,
                ["totalBatches"] = TotalBatches,
                ["averageBatchSize"] = TotalBatches > 0 ? (double)TotalBatchSize / TotalBatches : 0.0,
                ["averageLatencyMs"] = TotalBatches > 0 ? TotalLatencyMs / TotalBatches : 0.0
            };
        }
    }

    /// <summary>
    /// Gets detailed performance metrics including latency percentiles.
    /// </summary>
    /// <returns>A dictionary containing detailed performance metrics.</returns>
    public abstract Dictionary<string, object> GetPerformanceMetrics();

    /// <summary>
    /// Records a batch processing event for statistics.
    /// </summary>
    /// <param name="batchSize">The size of the batch that was processed.</param>
    /// <param name="latencyMs">The latency in milliseconds.</param>
    protected void RecordBatch(int batchSize, double latencyMs)
    {
        lock (StatsLock)
        {
            TotalBatches++;
            TotalBatchSize += batchSize;
            TotalLatencyMs += latencyMs;
        }
    }

    /// <summary>
    /// Increments the total request count.
    /// </summary>
    protected void IncrementRequestCount()
    {
        Interlocked.Increment(ref TotalRequests);
    }

    /// <summary>
    /// Creates a result vector from model output.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="tcs">The task completion source to set the result on.</param>
    /// <param name="result">The result vector.</param>
    protected static void SetResult<T>(TaskCompletionSource<Vector<T>> tcs, Vector<T> result)
    {
        tcs.TrySetResult(result);
    }

    /// <summary>
    /// Sets an exception on a task completion source.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="tcs">The task completion source to set the exception on.</param>
    /// <param name="exception">The exception to set.</param>
    protected static void SetException<T>(TaskCompletionSource<Vector<T>> tcs, Exception exception)
    {
        tcs.TrySetException(exception);
    }

    /// <summary>
    /// Disposes the request batcher.
    /// </summary>
    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes managed and unmanaged resources.
    /// </summary>
    /// <param name="disposing">True if called from Dispose(), false if from finalizer.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (!Disposed)
        {
            if (disposing)
            {
                // Dispose managed resources
                DisposeManagedResources();
            }

            Disposed = true;
        }
    }

    /// <summary>
    /// Disposes managed resources. Override in derived classes to clean up specific resources.
    /// </summary>
    protected virtual void DisposeManagedResources()
    {
        // Base implementation does nothing - override in derived classes
    }
}
