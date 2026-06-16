using System.Collections.Concurrent;
using System.Diagnostics;
using AiDotNet.Serving.Batching;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Monitoring;
using AiDotNet.Serving.Scheduling;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace AiDotNet.Serving.Services;

/// <summary>
/// Request batcher that uses continuous batching techniques for maximum throughput.
/// </summary>
/// <remarks>
/// <para>
/// Unlike traditional batching which processes fixed-size batches at fixed intervals,
/// continuous batching dynamically adds and removes requests from the batch at each
/// iteration. This maximizes throughput by always running at full capacity.
/// </para>
/// <para><b>For Beginners:</b> Continuous batching is like a conveyor belt vs. batch processing.
///
/// Traditional batching:
/// - Wait for N requests (or timeout)
/// - Process all N together
/// - Wait for all N to complete
/// - Start over
///
/// Continuous batching:
/// - Process requests as they arrive
/// - When one request completes, immediately start another
/// - Always keep the "pipeline" full
///
/// Benefits:
/// - Higher throughput (no waiting for full batches)
/// - Lower latency (no waiting for slow requests)
/// - Better resource utilization
///
/// This is especially useful for:
/// - Variable-length processing times
/// - High-throughput serving scenarios
/// - Mixed workloads (short and long requests together)
/// </para>
/// </remarks>
public class ContinuousBatchingRequestBatcher : RequestBatcherBase
{
    private readonly ConcurrentQueue<ContinuousRequest> _requestQueue = new();
    private readonly Task _processingLoop;
    private readonly CancellationTokenSource _cts = new();
    private readonly PerformanceMetrics _performanceMetrics;

    // Configuration
    private readonly int _maxBatchSize;
    private readonly int _iterationIntervalMs;
    private readonly bool _enableAdaptiveBatchSize;
    private readonly double _targetLatencyMs;

    // Adaptive batch-size tracking
    private int _currentBatchSize;
    private readonly Queue<double> _latencyHistory = new();
    private const int MaxLatencyHistorySize = 50;
    private readonly object _batchSizeLock = new();

    // Statistics
    private long _requestIdCounter;
    private int _inFlightCount;

    /// <summary>
    /// Gets the number of requests in the batch currently being processed.
    /// </summary>
    public int RunningRequestCount => Volatile.Read(ref _inFlightCount);

    /// <summary>
    /// Gets the current queue depth.
    /// </summary>
    public int QueuedRequestCount => _requestQueue.Count;

    /// <summary>
    /// Initializes a new instance of the ContinuousBatchingRequestBatcher.
    /// </summary>
    /// <param name="modelRepository">The model repository for accessing loaded models.</param>
    /// <param name="logger">Logger for diagnostics.</param>
    /// <param name="options">Serving options configuration.</param>
    public ContinuousBatchingRequestBatcher(
        IModelRepository modelRepository,
        ILogger<ContinuousBatchingRequestBatcher> logger,
        IOptions<ServingOptions> options)
        : base(modelRepository, logger, options.Value)
    {
        // Configure from options or use defaults optimized for continuous batching
        _maxBatchSize = Options.MaxBatchSize > 0 ? Options.MaxBatchSize : 32;

        // Validate BatchingWindowMs - must be positive for meaningful iteration intervals
        // If invalid or zero, use sensible default (100ms window / 10 = 10ms iteration)
        var batchingWindowMs = Options.BatchingWindowMs > 0 ? Options.BatchingWindowMs : 100;
        _iterationIntervalMs = Math.Max(1, batchingWindowMs / 10); // Run loop faster than traditional batching

        _enableAdaptiveBatchSize = Options.AdaptiveBatchSize;
        _targetLatencyMs = Options.TargetLatencyMs > 0 ? Options.TargetLatencyMs : 50;

        _currentBatchSize = Math.Max(1, _maxBatchSize / 2); // Start at half capacity

        _performanceMetrics = Options.EnablePerformanceMetrics
            ? new PerformanceMetrics(Options.MaxLatencySamples)
            : new PerformanceMetrics(0);

        // Start the continuous processing loop
        _processingLoop = Task.Run(() => ProcessingLoop(_cts.Token));

        Logger.LogInformation(
            "ContinuousBatchingRequestBatcher initialized: maxBatchSize={MaxBatchSize}, iterationMs={IterationMs}, adaptiveBatchSize={Adaptive}",
            _maxBatchSize, _iterationIntervalMs, _enableAdaptiveBatchSize);
    }

    /// <summary>
    /// Queues a prediction request for continuous batching.
    /// </summary>
    /// <typeparam name="T">The numeric type used by the model.</typeparam>
    /// <param name="modelName">The name of the model to use for prediction.</param>
    /// <param name="input">The input features.</param>
    /// <param name="priority">The priority level for this request.
    /// Note: In continuous batching mode, priority is stored for metadata purposes but requests
    /// are processed in strict FIFO order. This design choice optimizes for throughput and
    /// fairness in high-load scenarios where continuous batching provides the most benefit.
    /// For priority-aware scheduling, consider using the standard RequestBatcher instead.</param>
    /// <returns>A task that completes with the prediction result.</returns>
    public override Task<Vector<T>> QueueRequest<T>(string modelName, Vector<T> input, RequestPriority priority = RequestPriority.Normal)
    {
        var tcs = new TaskCompletionSource<Vector<T>>(TaskCreationOptions.RunContinuationsAsynchronously);

        var request = new ContinuousRequest
        {
            RequestId = Interlocked.Increment(ref _requestIdCounter),
            ModelName = modelName,
            NumericType = typeof(T).Name,
            Input = input,
            InputLength = input.Length,
            CompletionSource = tcs,
            Priority = priority,
            EnqueueTime = DateTime.UtcNow
        };

        // Check backpressure
        if (Options.MaxQueueSize > 0 && _requestQueue.Count >= Options.MaxQueueSize)
        {
            tcs.SetException(new InvalidOperationException("Request queue is full. Please try again later."));
            Logger.LogWarning("Request rejected due to backpressure. Queue size: {QueueSize}", _requestQueue.Count);
            return tcs.Task;
        }

        _requestQueue.Enqueue(request);
        IncrementRequestCount();

        return tcs.Task;
    }

    /// <summary>
    /// Gets detailed performance metrics.
    /// </summary>
    /// <returns>Dictionary of performance metrics.</returns>
    public override Dictionary<string, object> GetPerformanceMetrics()
    {
        if (!Options.EnablePerformanceMetrics)
        {
            return new Dictionary<string, object>
            {
                ["metricsEnabled"] = false
            };
        }

        var metrics = _performanceMetrics.GetAllMetrics();
        metrics["metricsEnabled"] = true;
        metrics["batchingStrategy"] = "Continuous";
        metrics["currentBatchSize"] = Volatile.Read(ref _currentBatchSize);
        metrics["maxBatchSize"] = _maxBatchSize;
        metrics["queuedRequests"] = _requestQueue.Count;
        metrics["inFlightRequests"] = Volatile.Read(ref _inFlightCount);

        return metrics;
    }

    /// <summary>
    /// Gets statistics about the batcher.
    /// </summary>
    /// <returns>Dictionary of statistics.</returns>
    public override Dictionary<string, object> GetStatistics()
    {
        var stats = base.GetStatistics();
        stats["batchingStrategy"] = "Continuous";
        stats["queuedRequests"] = _requestQueue.Count;
        stats["inFlightRequests"] = Volatile.Read(ref _inFlightCount);
        stats["currentBatchSize"] = Volatile.Read(ref _currentBatchSize);
        stats["maxBatchSize"] = _maxBatchSize;
        stats["adaptiveBatchSize"] = _enableAdaptiveBatchSize;
        return stats;
    }

    /// <summary>
    /// Main processing loop that continuously admits queued requests and runs them as batched
    /// model forward passes.
    /// </summary>
    /// <remarks>
    /// Each iteration admits up to the current (adaptive) batch size from the queue, groups the
    /// admitted requests by model + numeric type + input shape, and runs a single
    /// <c>PredictBatch</c> forward per group — sharing one forward pass across all requests in the
    /// group. New requests join the very next iteration (continuous admission), and the batch size
    /// adapts to observed latency.
    /// </remarks>
    private async Task ProcessingLoop(CancellationToken cancellationToken)
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            try
            {
                int processed = ProcessAvailableRequests();

                // Only sleep when idle, so queued requests are admitted promptly.
                if (processed == 0)
                {
                    await Task.Delay(_iterationIntervalMs, cancellationToken).ConfigureAwait(false);
                }
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                Logger.LogError(ex, "Error in continuous batching processing loop");
                await Task.Delay(100, cancellationToken).ConfigureAwait(false); // Back off on errors
            }
        }
    }

    /// <summary>
    /// Admits up to the current batch size from the queue and processes the admitted requests as
    /// batched forward passes, grouped by model + numeric type + input length.
    /// </summary>
    /// <returns>The number of requests admitted (0 when the queue was empty).</returns>
    private int ProcessAvailableRequests()
    {
        int budget = Math.Max(1, Volatile.Read(ref _currentBatchSize));

        var admitted = new List<ContinuousRequest>(budget);
        while (admitted.Count < budget && _requestQueue.TryDequeue(out var request))
        {
            admitted.Add(request);
        }

        if (admitted.Count == 0)
        {
            return 0;
        }

        Volatile.Write(ref _inFlightCount, admitted.Count);
        try
        {
            // Group so every batched forward pass is well-formed: same model, same numeric type,
            // and same input width (rows of a matrix must have equal length).
            foreach (var group in admitted.GroupBy(r => (r.ModelName, r.NumericType, r.InputLength)))
            {
                var requests = group.ToList();
                switch (group.Key.NumericType)
                {
                    case "Double":
                        ProcessBatch<double>(group.Key.ModelName, requests);
                        break;
                    case "Single":
                        ProcessBatch<float>(group.Key.ModelName, requests);
                        break;
                    case "Decimal":
                        ProcessBatch<decimal>(group.Key.ModelName, requests);
                        break;
                    default:
                        foreach (var request in requests)
                        {
                            SetRequestException(request, new NotSupportedException(
                                $"Numeric type '{group.Key.NumericType}' is not supported"));
                        }
                        break;
                }
            }
        }
        finally
        {
            Volatile.Write(ref _inFlightCount, 0);
        }

        return admitted.Count;
    }

    /// <summary>
    /// Runs a single batched forward pass for a group of same-shape requests against one model and
    /// scatters the per-row results back to the individual requests.
    /// </summary>
    private void ProcessBatch<T>(string modelName, List<ContinuousRequest> requests)
    {
        var model = ModelRepository.GetModel<T>(modelName);
        if (model == null)
        {
            var notFound = new InvalidOperationException($"Model '{modelName}' not found or wrong numeric type");
            foreach (var request in requests)
            {
                FailTyped<T>(request, notFound);
            }
            return;
        }

        var stopwatch = Stopwatch.StartNew();
        try
        {
            // Stack the inputs into a single batch matrix (one row per request).
            int batchSize = requests.Count;
            int inputDim = requests[0].InputLength;
            var batchMatrix = new Matrix<T>(batchSize, inputDim);
            for (int i = 0; i < batchSize; i++)
            {
                if (requests[i].Input is not Vector<T> inputVector)
                {
                    FailTyped<T>(requests[i], new InvalidOperationException(
                        $"Request input for model '{modelName}' was not a Vector<{typeof(T).Name}>."));
                    continue;
                }

                for (int j = 0; j < inputDim; j++)
                {
                    batchMatrix[i, j] = inputVector[j];
                }
            }

            // Single shared model forward pass for the whole group.
            var predictions = model.PredictBatch(batchMatrix);

            for (int i = 0; i < batchSize; i++)
            {
                if (requests[i].CompletionSource is TaskCompletionSource<Vector<T>> tcs)
                {
                    tcs.TrySetResult(predictions.GetRow(i));
                }
            }

            stopwatch.Stop();
            var latencyMs = stopwatch.Elapsed.TotalMilliseconds;
            RecordBatch(batchSize, latencyMs);
            if (Options.EnablePerformanceMetrics)
            {
                _performanceMetrics.RecordBatch(batchSize, latencyMs);
            }
            if (_enableAdaptiveBatchSize)
            {
                UpdateAdaptiveBatchSize(latencyMs);
            }

            Logger.LogDebug(
                "Continuous batch for model '{ModelName}': size={BatchSize}, latency={LatencyMs}ms",
                modelName, batchSize, latencyMs);
        }
        catch (Exception ex)
        {
            Logger.LogError(ex, "Error processing continuous batch for model '{ModelName}'", modelName);
            foreach (var request in requests)
            {
                FailTyped<T>(request, ex);
            }
        }
    }

    /// <summary>
    /// Fails a single request's typed completion source.
    /// </summary>
    private static void FailTyped<T>(ContinuousRequest request, Exception exception)
    {
        if (request.CompletionSource is TaskCompletionSource<Vector<T>> tcs)
        {
            tcs.TrySetException(exception);
        }
    }

    /// <summary>
    /// Sets an exception on a request using type-safe pattern matching.
    /// Avoids reflection by checking the NumericType and casting appropriately.
    /// </summary>
    private static void SetRequestException(ContinuousRequest request, Exception exception)
    {
        // Use type-safe pattern matching based on the stored NumericType
        // This avoids reflection overhead and provides compile-time safety
        switch (request.NumericType)
        {
            case "Double":
                if (request.CompletionSource is TaskCompletionSource<Vector<double>> doubleTcs)
                {
                    doubleTcs.TrySetException(exception);
                }
                break;
            case "Single":
                if (request.CompletionSource is TaskCompletionSource<Vector<float>> floatTcs)
                {
                    floatTcs.TrySetException(exception);
                }
                break;
            case "Decimal":
                if (request.CompletionSource is TaskCompletionSource<Vector<decimal>> decimalTcs)
                {
                    decimalTcs.TrySetException(exception);
                }
                break;
        }
    }

    /// <summary>
    /// Updates the adaptive batch size based on observed latency: grow the batch when latency is
    /// comfortably under target (throughput headroom), shrink it when latency exceeds target.
    /// </summary>
    private void UpdateAdaptiveBatchSize(double latencyMs)
    {
        lock (_batchSizeLock)
        {
            // Track latency history
            _latencyHistory.Enqueue(latencyMs);
            if (_latencyHistory.Count > MaxLatencyHistorySize)
            {
                _latencyHistory.Dequeue();
            }

            // Calculate average latency
            var avgLatency = _latencyHistory.Average();

            // Adjust batch size based on latency
            if (avgLatency < _targetLatencyMs * 0.8 && _currentBatchSize < _maxBatchSize)
            {
                // Latency is good, grow the batch for more throughput.
                _currentBatchSize = Math.Min(_currentBatchSize + 1, _maxBatchSize);
                Logger.LogDebug("Increased batch size to {BatchSize} (avgLatency={AvgLatency}ms)",
                    _currentBatchSize, avgLatency);
            }
            else if (avgLatency > _targetLatencyMs * 1.5 && _currentBatchSize > 1)
            {
                // Latency is too high, shrink the batch.
                _currentBatchSize = Math.Max(_currentBatchSize - 1, 1);
                Logger.LogDebug("Decreased batch size to {BatchSize} (avgLatency={AvgLatency}ms)",
                    _currentBatchSize, avgLatency);
            }
        }
    }

    /// <summary>
    /// Disposes managed resources.
    /// </summary>
    protected override void DisposeManagedResources()
    {
        _cts.Cancel();

        // Use Task.WhenAny with a timeout task to avoid synchronous blocking
        // which could deadlock if called from a synchronization context
        try
        {
            var timeoutTask = Task.Delay(TimeSpan.FromSeconds(5));
            var completedTask = Task.WhenAny(_processingLoop, timeoutTask).GetAwaiter().GetResult();

            if (completedTask == timeoutTask)
            {
                Logger.LogWarning("Processing loop did not complete within timeout during disposal");
            }
        }
        catch (AggregateException)
        {
            // Expected on cancellation
        }
        catch (OperationCanceledException)
        {
            // Expected on cancellation
        }

        _cts.Dispose();

        // Fail any requests still queued at shutdown (in-flight requests are processed
        // synchronously inside the loop, which has already stopped at this point).
        while (_requestQueue.TryDequeue(out var request))
        {
            SetRequestException(request, new OperationCanceledException("Batcher is shutting down"));
        }

        base.DisposeManagedResources();
    }

}
