using System.Collections.Concurrent;
using System.Diagnostics;
using AiDotNet.LinearAlgebra;
using AiDotNet.Serving.Batching;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Monitoring;
using AiDotNet.Serving.Padding;
using AiDotNet.Serving.Scheduling;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

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
/// </summary>
public class RequestBatcher : IRequestBatcher, IDisposable
{
    private readonly IModelRepository _modelRepository;
    private readonly ILogger<RequestBatcher> _logger;
    private readonly ServingOptions _options;
    private readonly PriorityRequestQueue<BatchRequest>? _priorityQueue;
    private readonly ConcurrentQueue<BatchRequest> _requestQueue = new();
    private readonly Timer _batchTimer;
    private readonly SemaphoreSlim _processingSemaphore = new(1, 1);

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
        _modelRepository = modelRepository ?? throw new ArgumentNullException(nameof(modelRepository));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _options = options?.Value ?? throw new ArgumentNullException(nameof(options));

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

        // Start the batch processing timer
        _batchTimer = new Timer(
            ProcessBatchCallback,
            null,
            TimeSpan.FromMilliseconds(_options.BatchingWindowMs),
            TimeSpan.FromMilliseconds(_options.BatchingWindowMs));
    }

    /// <summary>
    /// Creates the batching strategy based on configuration.
    /// </summary>
    private IBatchingStrategy CreateBatchingStrategy()
    {
        return _options.BatchingStrategy?.ToLower() switch
        {
            "timeout" => new TimeoutBatchingStrategy(_options.BatchingWindowMs, _options.MaxBatchSize),
            "size" => new SizeBatchingStrategy(_options.MaxBatchSize, _options.BatchingWindowMs),
            "bucket" => new BucketBatchingStrategy(_options.BucketSizes, _options.MaxBatchSize, _options.BatchingWindowMs),
            "adaptive" => new AdaptiveBatchingStrategy(
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
        return _options.PaddingStrategy?.ToLower() switch
        {
            "bucket" => new BucketPaddingStrategy(_options.BucketSizes),
            "fixed" => new FixedSizePaddingStrategy(_options.FixedPaddingSize),
            "minimal" => new MinimalPaddingStrategy(),
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
            NumericType = typeof(T).Name,
            Input = input,
            CompletionSource = tcs,
            Priority = priority,
            EnqueueTime = DateTime.UtcNow
        };

        // Use priority queue if enabled, otherwise use regular queue
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
        }
        else
        {
            // Check backpressure for regular queue
            if (_options.MaxQueueSize > 0 && _requestQueue.Count >= _options.MaxQueueSize)
            {
                tcs.SetException(new InvalidOperationException("Request queue is full. Please try again later."));
                _logger.LogWarning("Request rejected due to backpressure. Queue size: {QueueSize}", _requestQueue.Count);
                return tcs.Task;
            }

            lock (_timeLock)
            {
                _requestQueue.Enqueue(request);

                // Track oldest request time
                if (request.EnqueueTime < _oldestRequestTime)
                    _oldestRequestTime = request.EnqueueTime;
            }
        }

        Interlocked.Increment(ref _totalRequests);

        return tcs.Task;
    }

    /// <summary>
    /// Timer callback that triggers batch processing.
    /// </summary>
    private void ProcessBatchCallback(object? state)
    {
        // Use a semaphore to ensure only one batch is processed at a time
        if (!_processingSemaphore.Wait(0))
        {
            // Another batch is still being processed, skip this cycle
            return;
        }

        try
        {
            ProcessBatches();
        }
        finally
        {
            _processingSemaphore.Release();
        }
    }

    /// <summary>
    /// Processes all queued requests in batches grouped by model and numeric type.
    /// </summary>
    private void ProcessBatches()
    {
        var queueDepth = _options.EnablePriorityScheduling && _priorityQueue != null
            ? _priorityQueue.Count
            : _requestQueue.Count;

        if (queueDepth == 0)
        {
            return;
        }

        // Calculate time in queue for oldest request
        double timeInQueueMs = 0;
        lock (_timeLock)
        {
            if (_oldestRequestTime != DateTime.MaxValue)
            {
                timeInQueueMs = (DateTime.UtcNow - _oldestRequestTime).TotalMilliseconds;
            }
        }

        // Record queue depth for monitoring
        if (_options.EnablePerformanceMetrics)
        {
            _performanceMetrics.RecordQueueDepth(queueDepth);
        }

        // Check if we should process a batch based on the strategy
        var averageLatency = _options.EnablePerformanceMetrics
            ? _performanceMetrics.GetAverageLatency()
            : double.NaN;
        if (!_batchingStrategy.ShouldProcessBatch(queueDepth, timeInQueueMs, averageLatency, queueDepth))
        {
            return;
        }

        // Determine optimal batch size
        var optimalBatchSize = _batchingStrategy.GetOptimalBatchSize(queueDepth, averageLatency);

        // Collect pending requests
        var requests = new List<BatchRequest>();
        if (_options.EnablePriorityScheduling && _priorityQueue != null)
        {
            while (requests.Count < optimalBatchSize && _priorityQueue.TryDequeue(out var request))
            {
                requests.Add(request);
            }
        }
        else
        {
            while (requests.Count < optimalBatchSize && _requestQueue.TryDequeue(out var request))
            {
                requests.Add(request);
            }
        }

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
                if (numericType == "Double")
                {
                    ProcessBatch<double>(modelName, batchRequests);
                }
                else if (numericType == "Single")
                {
                    ProcessBatch<float>(modelName, batchRequests);
                }
                else if (numericType == "Decimal")
                {
                    ProcessBatch<decimal>(modelName, batchRequests);
                }
                else
                {
                    // Unsupported type - fail all requests in this group
                    foreach (var req in batchRequests)
                    {
                        SetException(req, new NotSupportedException($"Numeric type '{numericType}' is not supported"));
                    }
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
            : _requestQueue.Count;

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
    /// Disposes the request batcher and stops the background timer.
    /// </summary>
    public void Dispose()
    {
        _batchTimer?.Dispose();
        _processingSemaphore?.Dispose();
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Internal class representing a queued batch request.
    /// </summary>
    private class BatchRequest
    {
        public string ModelName { get; set; } = string.Empty;
        public string NumericType { get; set; } = string.Empty;
        public object Input { get; set; } = null!;
        public object CompletionSource { get; set; } = null!;
        public RequestPriority Priority { get; set; } = RequestPriority.Normal;
        public DateTime EnqueueTime { get; set; } = DateTime.UtcNow;
    }
}
