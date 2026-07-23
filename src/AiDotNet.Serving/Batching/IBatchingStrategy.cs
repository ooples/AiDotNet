namespace AiDotNet.Serving.Batching;

/// <summary>
/// Interface for batching strategies that determine when to process accumulated requests.
/// </summary>
public interface IBatchingStrategy
{
    /// <summary>
    /// Gets the name of the batching strategy.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Determines whether a batch should be processed based on the current state.
    /// </summary>
    /// <param name="queuedRequests">Number of requests currently queued</param>
    /// <param name="timeInQueueMs">Time in milliseconds since the oldest request was queued</param>
    /// <param name="averageLatencyMs">Average latency of recent batches in milliseconds</param>
    /// <param name="queueDepth">Current queue depth</param>
    /// <returns>True if the batch should be processed; otherwise, false</returns>
    bool ShouldProcessBatch(int queuedRequests, double timeInQueueMs, double averageLatencyMs, int queueDepth);

    /// <summary>
    /// Determines the optimal batch size for the current state.
    /// </summary>
    /// <param name="queuedRequests">Number of requests currently queued</param>
    /// <param name="averageLatencyMs">Average latency of recent batches in milliseconds</param>
    /// <returns>The optimal batch size</returns>
    int GetOptimalBatchSize(int queuedRequests, double averageLatencyMs);

    /// <summary>
    /// Updates the strategy with performance feedback.
    /// </summary>
    /// <param name="batchSize">Size of the batch that was processed</param>
    /// <param name="latencyMs">Latency in milliseconds for processing the batch</param>
    void UpdatePerformanceFeedback(int batchSize, double latencyMs);
}
