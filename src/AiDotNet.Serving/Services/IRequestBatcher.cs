using AiDotNet.Serving.Scheduling;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Serving.Services;

/// <summary>
/// Interface for the request batcher that collects and processes inference requests in batches.
/// </summary>
public interface IRequestBatcher
{
    /// <summary>
    /// Queues a prediction request to be processed in the next batch.
    /// </summary>
    /// <typeparam name="T">The numeric type used by the model</typeparam>
    /// <param name="modelName">The name of the model to use for prediction</param>
    /// <param name="input">The input features</param>
    /// <param name="priority">The priority level for this request</param>
    /// <returns>A task that completes with the prediction result</returns>
    Task<Vector<T>> QueueRequest<T>(string modelName, Vector<T> input, RequestPriority priority = RequestPriority.Normal);

    /// <summary>
    /// Gets statistics about the batcher's performance.
    /// </summary>
    /// <returns>A dictionary containing batcher statistics</returns>
    Dictionary<string, object> GetStatistics();

    /// <summary>
    /// Gets detailed performance metrics including latency percentiles.
    /// </summary>
    /// <returns>A dictionary containing detailed performance metrics</returns>
    Dictionary<string, object> GetPerformanceMetrics();
}
