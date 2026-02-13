using System.Diagnostics;
using System.Threading.Channels;
using AiDotNet.Serving.Batching;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Monitoring;
using AiDotNet.Serving.Padding;
using AiDotNet.Serving.Scheduling;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using AiDotNet.Validation;

namespace AiDotNet.Serving.Services;

/// <summary>
/// High-performance request batcher that collects multiple inference requests
/// and processes them as a single batch to maximize throughput.
///
/// Enhanced features (Issue #410):
/// - Dynamic batching with multiple strategies (Timeout, Size, Adaptive, Bucket)
/// - Priority-based request scheduling with fair scheduling and backpressure handling
/// - Padding strategies for variable-length sequences (Minimal, Bucket, Fixed)
/// - Performance monitoring with latency percentile tracking (p50, p95, p99)
/// - Adaptive batch sizing based on latency and throughput metrics
///
/// Reliability improvements:
/// - Uses Channel-based queue and background Task for robust batch processing
/// - Eliminates Timer-based scheduling which can be unreliable in CI environments
/// - Uses SemaphoreSlim with WaitAsync for non-blocking synchronization
/// </summary>
public class RequestBatcher : IRequestBatcher, IDisposable
{
    private readonly IModelRepository _modelRepository;
    private readonly ILogger<RequestBatcher> _logger;
    private readonly ServingOptions _options;
    private readonly PriorityRequestQueue<BatchRequest>? _priorityQueue;
    private readonly Channel<BatchRequest> _requestChannel;
    private readonly CancellationTokenSource _cts = new();
    private readonly Task _processingTask;
    private readonly SemaphoreSlim _batchSignal = new(0);

    // Enhanced components
    private readonly IBatchingStrategy _batchingStrategy;
    private readonly IPaddingStrategy _paddingStrategy;
    private readonly PerformanceMetrics _performanceMetrics;

    // Statistics tracking
    private long _totalRequests = 0;
    private long _totalBatches = 0;
    private long _totalBatchSize = 0;
    private DateTime _oldestRequestTime = DateTime.MaxValue;
    private readonly object _timeLock = new();

    /// <summary>
    /// Initializes a new instance of the RequestBatcher.
    /// </summary>
    /// <param name="modelRepository">The model repository for accessing loaded models</param>
    /// <param name="logger">Logger for diagnostics</param>
    /// <param name="options">Configuration options for batching behavior</param>
    public RequestBatcher(
        IModelRepository modelRepository,
        ILogger<RequestBatcher> logger,
        IOptions<ServingOptions> options)
    {
        Guard.NotNull(modelRepository);
        _modelRepository = modelRepository;
        Guard.NotNull(logger);
        _logger = logger;
        _options = options?.Value ?? throw new ArgumentNullException(nameof(options));

        // Initialize request channel with bounded capacity for backpressure
        var channelOptions = new BoundedChannelOptions(_options.MaxQueueSize > 0 ? _options.MaxQueueSize : 1000)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = true,
            SingleWriter = false
        };
        _requestChannel = Channel.CreateBounded<BatchRequest>(channelOptions);

        // Initialize priority queue only if enabled
        if (_options.EnablePriorityScheduling)
        {
            _priorityQueue = new PriorityRequestQueue<BatchRequest>(_options.MaxQueueSize);
        }

        // Initialize batching strategy
        _batchingStrategy = CreateBatchingStrategy();
        _logger.LogInformation("Using batching strategy: {Strategy}", _batchingStrategy.Name);

        // Initialize padding strategy
        _paddingStrategy = CreatePaddingStrategy();
        _logger.LogInformation("Using padding strategy: {Strategy}", _paddingStrategy.Name);

        // Initialize performance metrics
        _performanceMetrics = _options.EnablePerformanceMetrics
            ? new PerformanceMetrics(_options.MaxLatencySamples)
            : new PerformanceMetrics(0);

        // Start the batch processing background task
        _processingTask = Task.Run(() => ProcessingLoopAsync(_cts.Token));
        _logger.LogInformation("RequestBatcher started with Channel-based processing");
    }

    /// <summary>
    /// Creates the batching strategy based on configuration.
    /// </summary>
    private IBatchingStrategy CreateBatchingStrategy()
    {
        return _options.BatchingStrategy switch
        {
            BatchingStrategyType.Timeout => new TimeoutBatchingStrategy(_options.BatchingWindowMs, _options.MaxBatchSize),
            BatchingStrategyType.Size => new SizeBatchingStrategy(_options.MaxBatchSize, _options.BatchingWindowMs),
            BatchingStrategyType.Bucket => new BucketBatchingStrategy(_options.BucketSizes, _options.MaxBatchSize, _options.BatchingWindowMs),
            BatchingStrategyType.Continuous => new ContinuousBatchingStrategy(
                _options.MaxBatchSize,
                Math.Max(1, _options.BatchingWindowMs / 10),
                _options.TargetLatencyMs,
                _options.AdaptiveBatchSize),
            BatchingStrategyType.Adaptive => new AdaptiveBatchingStrategy(
                _options.MinBatchSize,
                _options.MaxBatchSize,
                _options.BatchingWindowMs,
                _options.TargetLatencyMs,
                _options.LatencyToleranceFactor),
            _ => new AdaptiveBatchingStrategy(
                _options.MinBatchSize,
                _options.MaxBatchSize,
                _options.BatchingWindowMs,
                _options.TargetLatencyMs,
                _options.LatencyToleranceFactor)
        };
    }

    /// <summary>
    /// Creates the padding strategy based on configuration.
    /// </summary>
    private IPaddingStrategy CreatePaddingStrategy()
    {
        return _options.PaddingStrategy switch
        {
            PaddingStrategyType.Bucket => new BucketPaddingStrategy(_options.BucketSizes),
            PaddingStrategyType.Fixed => new FixedSizePaddingStrategy(_options.FixedPaddingSize),
            PaddingStrategyType.Minimal => new MinimalPaddingStrategy(),
            _ => new MinimalPaddingStrategy()
        };
    }

    /// <summary>
    /// Queues a prediction request to be processed in the next batch.
    /// </summary>
    /// <typeparam name="T">The numeric type used by the model</typeparam>
    /// <param name="modelName">The name of the model to use for prediction</param>
    /// <param name="input">The input features</param>
    /// <param name="priority">The priority level for this request</param>
    /// <returns>A task that completes with the prediction result</returns>
    public Task<Vector<T>> QueueRequest<T>(string modelName, Vector<T> input, RequestPriority priority = RequestPriority.Normal)
    {
        var tcs = new TaskCompletionSource<Vector<T>>(TaskCreationOptions.RunContinuationsAsynchronously);

        var request = new BatchRequest
        {
            ModelName = modelName,
            NumericType = GetNumericType<T>(),
            Input = input,
            CompletionSource = tcs,
            Priority = priority,
            EnqueueTime = DateTime.UtcNow
        };

        // Use priority queue if enabled
        if (_options.EnablePriorityScheduling && _priorityQueue != null)
        {
            lock (_timeLock)
            {
                if (!_priorityQueue.TryEnqueue(request, priority))
                {
                    // Queue is full - backpressure handling
                    tcs.SetException(new InvalidOperationException("Request queue is full. Please try again later."));
                    _logger.LogWarning("Request rejected due to backpressure. Queue size: {QueueSize}", _priorityQueue.Count);
                    return tcs.Task;
                }

                // Track oldest request time
                if (request.EnqueueTime < _oldestRequestTime)
                    _oldestRequestTime = request.EnqueueTime;
            }

            // Signal that there are requests to process
            try { _batchSignal.Release(); } catch (SemaphoreFullException) { }
        }
        else
        {
            // Try to write to the channel (non-blocking)
            if (!_requestChannel.Writer.TryWrite(request))
            {
                tcs.SetException(new InvalidOperationException("Request queue is full. Please try again later."));
                _logger.LogWarning("Request rejected due to backpressure. Channel is full.");
                return tcs.Task;
            }

            lock (_timeLock)
            {
                // Track oldest request time
                if (request.EnqueueTime < _oldestRequestTime)
                    _oldestRequestTime = request.EnqueueTime;
            }
        }

        Interlocked.Increment(ref _totalRequests);

        return tcs.Task;
    }

    /// <summary>
    /// Background processing loop that reads from the channel and processes batches.
    /// This replaces the Timer-based approach for more reliable batch processing.
    /// </summary>
    private async Task ProcessingLoopAsync(CancellationToken cancellationToken)
    {
        var pendingRequests = new List<BatchRequest>();
        var batchingWindowMs = Math.Max(1, _options.BatchingWindowMs);

        _logger.LogDebug("Processing loop started with batching window: {WindowMs}ms", batchingWindowMs);

        try
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                // Read requests from the channel with a timeout
                using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
                timeoutCts.CancelAfter(batchingWindowMs);

                try
                {
                    // For priority queue mode, wait on the semaphore signal
                    if (_options.EnablePriorityScheduling && _priorityQueue != null)
                    {
                        await _batchSignal.WaitAsync(timeoutCts.Token).ConfigureAwait(false);

                        // Drain priority queue
                        while (_priorityQueue.TryDequeue(out var request))
                        {
                            if (request != null)
                            {
                                pendingRequests.Add(request);
                            }
                        }
                    }
                    else
                    {
                        // Read from channel - this is the main batching entry point
                        while (pendingRequests.Count < _options.MaxBatchSize)
                        {
                            if (_requestChannel.Reader.TryRead(out var request))
                            {
                                pendingRequests.Add(request);
                            }
                            else
                            {
                                // No more items immediately available
                                // Wait for more items or timeout
                                try
                                {
                                    var waitTask = _requestChannel.Reader.WaitToReadAsync(timeoutCts.Token);
                                    if (!await waitTask.ConfigureAwait(false))
                                    {
                                        // Channel was completed (no more items will be written)
                                        break;
                                    }
                                }
                                catch (OperationCanceledException)
                                {
                                    // Timeout reached, process what we have
                                    break;
                                }
                            }
                        }
                    }
                }
                catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
                {
                    // Timeout - process what we have
                }

                // Process the batch if we have any requests
                if (pendingRequests.Count > 0)
                {
                    try
                    {
                        ProcessBatchedRequests(pendingRequests);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error processing batch of {Count} requests", pendingRequests.Count);
                        // Fail all pending requests
                        foreach (var req in pendingRequests)
                        {
                            SetException(req, ex);
                        }
                    }
                    finally
                    {
                        pendingRequests.Clear();
                    }
                }
                else
                {
                    // Small delay to prevent tight loop when idle
                    await Task.Delay(1, cancellationToken).ConfigureAwait(false);
                }
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            _logger.LogInformation("Processing loop cancelled");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Unexpected error in processing loop");
        }
        finally
        {
            // Fail any remaining pending requests
            foreach (var req in pendingRequests)
            {
                SetException(req, new OperationCanceledException("Batcher is shutting down"));
            }

            _logger.LogDebug("Processing loop stopped");
        }
    }

    /// <summary>
    /// Processes a list of pending batch requests.
    /// </summary>
    private void ProcessBatchedRequests(List<BatchRequest> requests)
    {
        if (requests.Count == 0)
        {
            return;
        }

        // Reset oldest request time
        lock (_timeLock)
        {
            _oldestRequestTime = DateTime.MaxValue;
        }

        // Group requests by model name and numeric type
        var groupedRequests = requests.GroupBy(r => (r.ModelName, r.NumericType));

        foreach (var group in groupedRequests)
        {
            var modelName = group.Key.ModelName;
            var numericType = group.Key.NumericType;
            var batchRequests = group.ToList();

            try
            {
                // Process based on numeric type
                switch (numericType)
                {
                    case NumericType.Double:
                        ProcessBatch<double>(modelName, batchRequests);
                        break;
                    case NumericType.Float:
                        ProcessBatch<float>(modelName, batchRequests);
                        break;
                    case NumericType.Decimal:
                        ProcessBatch<decimal>(modelName, batchRequests);
                        break;
                    default:
                        // Unsupported type - fail all requests in this group
                        foreach (var req in batchRequests)
                        {
                            SetException(req, new NotSupportedException($"Numeric type '{numericType}' is not supported"));
                        }

                        break;
                }
            }
            catch (InvalidOperationException ex)
            {
                // Model not found or invalid operation
                _logger.LogError(ex, "Invalid operation processing batch for model '{ModelName}'", modelName);
                foreach (var req in batchRequests)
                {
                    SetException(req, ex);
                }
            }
            catch (ArgumentException ex)
            {
                // Dimension mismatch or invalid argument
                _logger.LogError(ex, "Invalid argument processing batch for model '{ModelName}'", modelName);
                foreach (var req in batchRequests)
                {
                    SetException(req, ex);
                }
            }
            catch (InvalidCastException ex)
            {
                // Type casting error
                _logger.LogError(ex, "Type cast error processing batch for model '{ModelName}'", modelName);
                foreach (var req in batchRequests)
                {
                    SetException(req, ex);
                }
            }
            catch (Exception ex)
            {
                // Unexpected error - fail all requests in this batch
                _logger.LogError(ex, "Unexpected error processing batch for model '{ModelName}'", modelName);
                foreach (var req in batchRequests)
                {
                    SetException(req, ex);
                }
            }
        }
    }

    /// <summary>
    /// Processes a batch of requests for a specific model and numeric type.
    /// </summary>
    private void ProcessBatch<T>(string modelName, List<BatchRequest> requests)
    {
        if (requests.Count == 0)
        {
            return;
        }

        var stopwatch = Stopwatch.StartNew();

        // Get the model
        var model = _modelRepository.GetModel<T>(modelName);
        if (model == null)
        {
            foreach (var req in requests)
            {
                SetException(req, new InvalidOperationException($"Model '{modelName}' not found or wrong numeric type"));
            }
            return;
        }

        try
        {
            // Create a batch matrix from all inputs
            var inputVectors = requests.Select(r => (Vector<T>)r.Input).ToArray();
            var batchSize = inputVectors.Length;
            var inputDim = inputVectors[0].Length;

            // Build the batch matrix (each row is one input)
            var batchMatrix = new Matrix<T>(batchSize, inputDim);
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < inputDim; j++)
                {
                    batchMatrix[i, j] = inputVectors[i][j];
                }
            }

            // Perform batch prediction - single model forward pass
            var predictions = model.PredictBatch(batchMatrix);

            // Distribute results back to individual requests
            for (int i = 0; i < batchSize; i++)
            {
                var result = predictions.GetRow(i);
                SetResult(requests[i], result);
            }

            // Update statistics and metrics
            stopwatch.Stop();
            var latencyMs = stopwatch.Elapsed.TotalMilliseconds;

            Interlocked.Increment(ref _totalBatches);
            Interlocked.Add(ref _totalBatchSize, batchSize);

            if (_options.EnablePerformanceMetrics)
            {
                _performanceMetrics.RecordBatch(batchSize, latencyMs);

                // Record padding metrics: calculate actual vs total elements
                // For now, assuming minimal padding (no padding overhead in current implementation)
                var actualElements = batchSize * inputDim;
                var totalElements = batchSize * inputDim;
                var paddingElements = totalElements - actualElements;
                _performanceMetrics.RecordBatchUtilization(actualElements, paddingElements);
            }

            // Update batching strategy with performance feedback
            _batchingStrategy.UpdatePerformanceFeedback(batchSize, latencyMs);

            _logger.LogDebug(
                "Processed batch for model '{ModelName}': size={BatchSize}, latency={LatencyMs}ms",
                modelName, batchSize, latencyMs);
        }
        catch (ArgumentException ex)
        {
            // Dimension mismatch or invalid argument in batch prediction
            _logger.LogError(ex, "Invalid argument in batch prediction for model '{ModelName}'", modelName);
            foreach (var req in requests)
            {
                SetException(req, ex);
            }
        }
        catch (InvalidOperationException ex)
        {
            // Model operation error during batch prediction
            _logger.LogError(ex, "Model operation error in batch prediction for '{ModelName}'", modelName);
            foreach (var req in requests)
            {
                SetException(req, ex);
            }
        }
        catch (IndexOutOfRangeException ex)
        {
            // Matrix indexing error
            _logger.LogError(ex, "Index out of range in batch prediction for '{ModelName}'", modelName);
            foreach (var req in requests)
            {
                SetException(req, ex);
            }
        }
        catch (Exception ex)
        {
            // Unexpected error - fail all requests in batch
            _logger.LogError(ex, "Unexpected error in batch prediction for '{ModelName}'", modelName);
            foreach (var req in requests)
            {
                SetException(req, ex);
            }
        }
    }

    /// <summary>
    /// Sets the result for a batch request.
    /// </summary>
    private static void SetResult<T>(BatchRequest request, Vector<T> result)
    {
        if (request.CompletionSource is TaskCompletionSource<Vector<T>> tcs)
        {
            tcs.SetResult(result);
        }
    }

    /// <summary>
    /// Sets an exception for a batch request.
    /// </summary>
    private static void SetException(BatchRequest request, Exception exception)
    {
        var tcsType = request.CompletionSource.GetType();
        var setExceptionMethod = tcsType.GetMethod("SetException", new[] { typeof(Exception) });
        setExceptionMethod?.Invoke(request.CompletionSource, new object[] { exception });
    }

    /// <summary>
    /// Gets statistics about the batcher's performance.
    /// </summary>
    public Dictionary<string, object> GetStatistics()
    {
        var queueDepth = _options.EnablePriorityScheduling && _priorityQueue != null
            ? _priorityQueue.Count
            : _requestChannel.Reader.Count;

        var stats = new Dictionary<string, object>
        {
            ["totalRequests"] = _totalRequests,
            ["totalBatches"] = _totalBatches,
            ["queuedRequests"] = queueDepth,
            ["averageBatchSize"] = _totalBatches > 0 ? (double)_totalBatchSize / _totalBatches : 0.0,
            ["batchingStrategy"] = _batchingStrategy.Name,
            ["paddingStrategy"] = _paddingStrategy.Name
        };

        // Add priority queue stats if enabled
        if (_options.EnablePriorityScheduling && _priorityQueue != null)
        {
            var priorityCounts = _priorityQueue.GetPriorityCounts();
            stats["priorityQueues"] = priorityCounts;
        }

        return stats;
    }

    /// <summary>
    /// Gets detailed performance metrics including latency percentiles.
    /// </summary>
    public Dictionary<string, object> GetPerformanceMetrics()
    {
        if (!_options.EnablePerformanceMetrics)
        {
            return new Dictionary<string, object>
            {
                ["metricsEnabled"] = false
            };
        }

        var metrics = _performanceMetrics.GetAllMetrics();
        metrics["metricsEnabled"] = true;
        metrics["batchingStrategy"] = _batchingStrategy.Name;
        metrics["paddingStrategy"] = _paddingStrategy.Name;

        return metrics;
    }

    /// <summary>
    /// Disposes the request batcher and stops the background processing task.
    /// </summary>
    public void Dispose()
    {
        // Signal cancellation and complete the channel
        _cts.Cancel();
        _requestChannel.Writer.TryComplete();

        // Wait for the processing task to complete with timeout
        try
        {
            if (!_processingTask.Wait(TimeSpan.FromSeconds(5)))
            {
                _logger.LogWarning("Processing task did not complete within timeout");
            }
        }
        catch (AggregateException ex) when (ex.InnerExceptions.All(e => e is TaskCanceledException or OperationCanceledException))
        {
            // Expected during cancellation
        }

        // Dispose resources
        _cts.Dispose();
        _batchSignal.Dispose();

        GC.SuppressFinalize(this);
    }

    private static NumericType GetNumericType<T>()
    {
        if (typeof(T) == typeof(float))
        {
            return NumericType.Float;
        }

        if (typeof(T) == typeof(decimal))
        {
            return NumericType.Decimal;
        }

        if (typeof(T) == typeof(double))
        {
            return NumericType.Double;
        }

        throw new NotSupportedException($"Numeric type '{typeof(T)}' is not supported.");
    }
}
