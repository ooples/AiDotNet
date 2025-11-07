using System.Collections.Concurrent;
using System.Diagnostics;
using AiDotNet.LinearAlgebra;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Models;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace AiDotNet.Serving.Services;

/// <summary>
/// High-performance request batcher that collects multiple inference requests
/// and processes them as a single batch to maximize throughput.
///
/// This class implements dynamic request batching:
/// 1. Incoming requests are queued in a thread-safe queue
/// 2. A background task periodically collects queued requests
/// 3. Requests are batched together and sent to the model as a single forward pass
/// 4. Individual results are returned to each request via TaskCompletionSource
/// </summary>
public class RequestBatcher : IRequestBatcher, IDisposable
{
    private readonly IModelRepository _modelRepository;
    private readonly ILogger<RequestBatcher> _logger;
    private readonly ServingOptions _options;
    private readonly ConcurrentQueue<BatchRequest> _requestQueue = new();
    private readonly Timer _batchTimer;
    private readonly SemaphoreSlim _processingSemaphore = new(1, 1);

    // Statistics tracking
    private long _totalRequests = 0;
    private long _totalBatches = 0;
    private long _totalBatchSize = 0;

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

        // Start the batch processing timer
        _batchTimer = new Timer(
            ProcessBatchCallback,
            null,
            TimeSpan.FromMilliseconds(_options.BatchingWindowMs),
            TimeSpan.FromMilliseconds(_options.BatchingWindowMs));
    }

    /// <summary>
    /// Queues a prediction request to be processed in the next batch.
    /// </summary>
    /// <typeparam name="T">The numeric type used by the model</typeparam>
    /// <param name="modelName">The name of the model to use for prediction</param>
    /// <param name="input">The input features</param>
    /// <returns>A task that completes with the prediction result</returns>
    public Task<Vector<T>> QueueRequest<T>(string modelName, Vector<T> input)
    {
        var tcs = new TaskCompletionSource<Vector<T>>(TaskCreationOptions.RunContinuationsAsynchronously);

        var request = new BatchRequest
        {
            ModelName = modelName,
            NumericType = typeof(T).Name,
            Input = input,
            CompletionSource = tcs
        };

        _requestQueue.Enqueue(request);
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
        if (_requestQueue.IsEmpty)
        {
            return;
        }

        // Collect all pending requests
        var requests = new List<BatchRequest>();
        while (_requestQueue.TryDequeue(out var request) &&
               (_options.MaxBatchSize <= 0 || requests.Count < _options.MaxBatchSize))
        {
            requests.Add(request);
        }

        if (requests.Count == 0)
        {
            return;
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

            // Update statistics
            Interlocked.Increment(ref _totalBatches);
            Interlocked.Add(ref _totalBatchSize, batchSize);
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
        var stats = new Dictionary<string, object>
        {
            ["totalRequests"] = _totalRequests,
            ["totalBatches"] = _totalBatches,
            ["queuedRequests"] = _requestQueue.Count,
            ["averageBatchSize"] = _totalBatches > 0 ? (double)_totalBatchSize / _totalBatches : 0.0
        };

        return stats;
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
    }
}
