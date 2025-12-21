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
    private readonly ConcurrentDictionary<long, ContinuousRequest> _runningRequests = new();
    private readonly Task _processingLoop;
    private readonly CancellationTokenSource _cts = new();
    private readonly PerformanceMetrics _performanceMetrics;

    // Configuration
    private readonly int _maxConcurrentRequests;
    private readonly int _iterationIntervalMs;
    private readonly bool _enableAdaptiveConcurrency;
    private readonly double _targetLatencyMs;

    // Adaptive concurrency tracking
    private int _currentConcurrency;
    private readonly Queue<double> _latencyHistory = new();
    private const int MaxLatencyHistorySize = 50;
    private readonly object _concurrencyLock = new();

    // Statistics
    private long _requestIdCounter;

    /// <summary>
    /// Gets the current number of requests being processed.
    /// </summary>
    public int RunningRequestCount => _runningRequests.Count;

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
        _maxConcurrentRequests = Options.MaxBatchSize > 0 ? Options.MaxBatchSize : 32;

        // Validate BatchingWindowMs - must be positive for meaningful iteration intervals
        // If invalid or zero, use sensible default (100ms window / 10 = 10ms iteration)
        var batchingWindowMs = Options.BatchingWindowMs > 0 ? Options.BatchingWindowMs : 100;
        _iterationIntervalMs = Math.Max(1, batchingWindowMs / 10); // Run loop faster than traditional batching

        _enableAdaptiveConcurrency = Options.AdaptiveBatchSize;
        _targetLatencyMs = Options.TargetLatencyMs > 0 ? Options.TargetLatencyMs : 50;

        _currentConcurrency = Math.Max(1, _maxConcurrentRequests / 2); // Start at half capacity

        _performanceMetrics = Options.EnablePerformanceMetrics
            ? new PerformanceMetrics(Options.MaxLatencySamples)
            : new PerformanceMetrics(0);

        // Start the continuous processing loop
        _processingLoop = Task.Run(() => ProcessingLoop(_cts.Token));

        Logger.LogInformation(
            "ContinuousBatchingRequestBatcher initialized: maxConcurrency={MaxConcurrency}, iterationMs={IterationMs}, adaptiveConcurrency={Adaptive}",
            _maxConcurrentRequests, _iterationIntervalMs, _enableAdaptiveConcurrency);
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
        metrics["currentConcurrency"] = _currentConcurrency;
        metrics["maxConcurrency"] = _maxConcurrentRequests;
        metrics["queuedRequests"] = _requestQueue.Count;
        metrics["runningRequests"] = _runningRequests.Count;

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
        stats["runningRequests"] = _runningRequests.Count;
        stats["currentConcurrency"] = _currentConcurrency;
        stats["maxConcurrency"] = _maxConcurrentRequests;
        stats["adaptiveConcurrency"] = _enableAdaptiveConcurrency;
        return stats;
    }

    /// <summary>
    /// Main processing loop that continuously schedules and processes requests.
    /// </summary>
    private async Task ProcessingLoop(CancellationToken cancellationToken)
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            try
            {
                // Try to fill available slots with new requests
                await ScheduleNewRequests(cancellationToken).ConfigureAwait(false);

                // Small delay to prevent tight loop
                await Task.Delay(_iterationIntervalMs, cancellationToken).ConfigureAwait(false);
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
    /// Schedules new requests up to the current concurrency limit.
    /// </summary>
    private async Task ScheduleNewRequests(CancellationToken cancellationToken)
    {
        int availableSlots;
        lock (_concurrencyLock)
        {
            availableSlots = _currentConcurrency - _runningRequests.Count;
        }

        // Schedule new requests to fill available slots
        var tasks = new List<Task>();
        while (availableSlots > 0 && _requestQueue.TryDequeue(out var request))
        {
            _runningRequests[request.RequestId] = request;
            tasks.Add(ProcessRequestAsync(request, cancellationToken));
            availableSlots--;
        }

        // Wait for all newly scheduled requests to at least start
        if (tasks.Count > 0)
        {
            // Don't await completion - let them run continuously
            _ = Task.WhenAll(tasks);
        }
    }

    /// <summary>
    /// Processes a single request asynchronously.
    /// </summary>
    private async Task ProcessRequestAsync(ContinuousRequest request, CancellationToken cancellationToken)
    {
        var stopwatch = Stopwatch.StartNew();

        try
        {
            // Process based on numeric type
            if (request.NumericType == "Double")
            {
                await ProcessTypedRequest<double>(request, cancellationToken);
            }
            else if (request.NumericType == "Single")
            {
                await ProcessTypedRequest<float>(request, cancellationToken);
            }
            else if (request.NumericType == "Decimal")
            {
                await ProcessTypedRequest<decimal>(request, cancellationToken);
            }
            else
            {
                SetRequestException(request, new NotSupportedException($"Numeric type '{request.NumericType}' is not supported"));
            }
        }
        catch (Exception ex)
        {
            Logger.LogError(ex, "Error processing request {RequestId} for model '{ModelName}'",
                request.RequestId, request.ModelName);
            SetRequestException(request, ex);
        }
        finally
        {
            stopwatch.Stop();
            var latencyMs = stopwatch.Elapsed.TotalMilliseconds;

            // Remove from running requests
            _runningRequests.TryRemove(request.RequestId, out _);

            // Record metrics
            RecordBatch(1, latencyMs);
            if (Options.EnablePerformanceMetrics)
            {
                _performanceMetrics.RecordBatch(1, latencyMs);
            }

            // Update adaptive concurrency
            if (_enableAdaptiveConcurrency)
            {
                UpdateAdaptiveConcurrency(latencyMs);
            }
        }
    }

    /// <summary>
    /// Processes a typed request.
    /// </summary>
    private Task ProcessTypedRequest<T>(ContinuousRequest request, CancellationToken cancellationToken)
    {
        var model = ModelRepository.GetModel<T>(request.ModelName);
        if (model == null)
        {
            if (request.CompletionSource is TaskCompletionSource<Vector<T>> errorTcs)
            {
                errorTcs.TrySetException(new InvalidOperationException(
                    $"Model '{request.ModelName}' not found or wrong numeric type"));
            }
            return Task.CompletedTask;
        }

        try
        {
            var input = (Vector<T>)request.Input;
            var result = model.Predict(input);

            if (request.CompletionSource is TaskCompletionSource<Vector<T>> tcs)
            {
                tcs.TrySetResult(result);
            }
        }
        catch (Exception ex)
        {
            if (request.CompletionSource is TaskCompletionSource<Vector<T>> exTcs)
            {
                exTcs.TrySetException(ex);
            }
        }

        return Task.CompletedTask;
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
    /// Updates the adaptive concurrency level based on observed latency.
    /// </summary>
    private void UpdateAdaptiveConcurrency(double latencyMs)
    {
        lock (_concurrencyLock)
        {
            // Track latency history
            _latencyHistory.Enqueue(latencyMs);
            if (_latencyHistory.Count > MaxLatencyHistorySize)
            {
                _latencyHistory.Dequeue();
            }

            // Calculate average latency
            var avgLatency = _latencyHistory.Average();

            // Adjust concurrency based on latency
            if (avgLatency < _targetLatencyMs * 0.8 && _currentConcurrency < _maxConcurrentRequests)
            {
                // Latency is good, increase concurrency
                _currentConcurrency = Math.Min(_currentConcurrency + 1, _maxConcurrentRequests);
                Logger.LogDebug("Increased concurrency to {Concurrency} (avgLatency={AvgLatency}ms)",
                    _currentConcurrency, avgLatency);
            }
            else if (avgLatency > _targetLatencyMs * 1.5 && _currentConcurrency > 1)
            {
                // Latency is too high, decrease concurrency
                _currentConcurrency = Math.Max(_currentConcurrency - 1, 1);
                Logger.LogDebug("Decreased concurrency to {Concurrency} (avgLatency={AvgLatency}ms)",
                    _currentConcurrency, avgLatency);
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

        // Fail any remaining requests
        while (_requestQueue.TryDequeue(out var request))
        {
            SetRequestException(request, new OperationCanceledException("Batcher is shutting down"));
        }

        foreach (var request in _runningRequests.Values)
        {
            SetRequestException(request, new OperationCanceledException("Batcher is shutting down"));
        }

        base.DisposeManagedResources();
    }

    /// <summary>
    /// Internal class representing a request in the continuous batching queue.
    /// </summary>
    private class ContinuousRequest
    {
        public long RequestId { get; set; }
        public string ModelName { get; set; } = string.Empty;
        public string NumericType { get; set; } = string.Empty;
        public object Input { get; set; } = null!;
        public object CompletionSource { get; set; } = null!;
        public RequestPriority Priority { get; set; } = RequestPriority.Normal;
        public DateTime EnqueueTime { get; set; }
    }
}
