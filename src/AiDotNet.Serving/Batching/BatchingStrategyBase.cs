namespace AiDotNet.Serving.Batching;

/// <summary>
/// Base class for batching strategies that provides common functionality.
/// </summary>
/// <remarks>
/// <para>
/// This base class provides shared implementation details for batching strategies
/// including statistics tracking and common helper methods. Concrete strategies
/// should inherit from this class and implement the abstract methods.
/// </para>
/// <para><b>For Beginners:</b> A batching strategy decides when to process queued requests.
///
/// This base class provides:
/// - Common statistics tracking (batches processed, latency history)
/// - Helper methods for calculating averages and thresholds
/// - Default implementations that can be overridden
///
/// Strategies differ in how they decide when to batch:
/// - Timeout: Process after X milliseconds
/// - Size: Process when batch reaches X requests
/// - Adaptive: Dynamically adjust based on latency
/// - Continuous: LLM-style batching with dynamic sequences
/// </para>
/// </remarks>
public abstract class BatchingStrategyBase : IBatchingStrategy
{
    /// <summary>
    /// Maximum number of latency samples to keep for averaging.
    /// </summary>
    protected const int MaxLatencySamples = 100;

    /// <summary>
    /// Lock object for thread-safe access to shared state.
    /// </summary>
    protected readonly object SyncLock = new();

    /// <summary>
    /// Circular buffer of recent latency measurements.
    /// </summary>
    protected readonly Queue<double> LatencyHistory = new();

    /// <summary>
    /// Total number of batches processed.
    /// </summary>
    protected long TotalBatchesProcessed;

    /// <summary>
    /// Sum of all latencies for averaging.
    /// </summary>
    protected double TotalLatencyMs;

    /// <summary>
    /// Gets the name of the batching strategy.
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Determines whether a batch should be processed based on the current state.
    /// </summary>
    /// <param name="queuedRequests">Number of requests currently queued.</param>
    /// <param name="timeInQueueMs">Time in milliseconds since the oldest request was queued.</param>
    /// <param name="averageLatencyMs">Average latency of recent batches in milliseconds.</param>
    /// <param name="queueDepth">Current queue depth.</param>
    /// <returns>True if the batch should be processed; otherwise, false.</returns>
    public abstract bool ShouldProcessBatch(int queuedRequests, double timeInQueueMs, double averageLatencyMs, int queueDepth);

    /// <summary>
    /// Determines the optimal batch size for the current state.
    /// </summary>
    /// <param name="queuedRequests">Number of requests currently queued.</param>
    /// <param name="averageLatencyMs">Average latency of recent batches in milliseconds.</param>
    /// <returns>The optimal batch size.</returns>
    public abstract int GetOptimalBatchSize(int queuedRequests, double averageLatencyMs);

    /// <summary>
    /// Updates the strategy with performance feedback.
    /// </summary>
    /// <param name="batchSize">Size of the batch that was processed.</param>
    /// <param name="latencyMs">Latency in milliseconds for processing the batch.</param>
    public virtual void UpdatePerformanceFeedback(int batchSize, double latencyMs)
    {
        lock (SyncLock)
        {
            // Track latency history
            LatencyHistory.Enqueue(latencyMs);
            if (LatencyHistory.Count > MaxLatencySamples)
            {
                var removed = LatencyHistory.Dequeue();
                TotalLatencyMs -= removed;
            }
            TotalLatencyMs += latencyMs;

            TotalBatchesProcessed++;
        }
    }

    /// <summary>
    /// Gets the current average latency from the history.
    /// </summary>
    /// <returns>Average latency in milliseconds, or 0 if no samples.</returns>
    protected double GetAverageLatency()
    {
        lock (SyncLock)
        {
            if (LatencyHistory.Count == 0)
                return 0;
            return TotalLatencyMs / LatencyHistory.Count;
        }
    }

    /// <summary>
    /// Gets the specified percentile from the latency history.
    /// </summary>
    /// <param name="percentile">The percentile to calculate (0-100).</param>
    /// <returns>The latency at the specified percentile, or 0 if no samples.</returns>
    protected double GetLatencyPercentile(double percentile)
    {
        lock (SyncLock)
        {
            if (LatencyHistory.Count == 0)
                return 0;

            var sorted = LatencyHistory.OrderBy(x => x).ToList();
            int index = (int)Math.Ceiling(percentile / 100.0 * sorted.Count) - 1;
            index = Math.Max(0, Math.Min(index, sorted.Count - 1));
            return sorted[index];
        }
    }

    /// <summary>
    /// Gets statistics about the batching strategy's performance.
    /// </summary>
    /// <returns>Dictionary of statistics.</returns>
    public virtual Dictionary<string, object> GetStatistics()
    {
        lock (SyncLock)
        {
            return new Dictionary<string, object>
            {
                ["name"] = Name,
                ["totalBatchesProcessed"] = TotalBatchesProcessed,
                ["averageLatencyMs"] = GetAverageLatency(),
                ["p50LatencyMs"] = GetLatencyPercentile(50),
                ["p95LatencyMs"] = GetLatencyPercentile(95),
                ["p99LatencyMs"] = GetLatencyPercentile(99),
                ["sampleCount"] = LatencyHistory.Count
            };
        }
    }

    /// <summary>
    /// Clamps a value between a minimum and maximum.
    /// </summary>
    protected static int Clamp(int value, int min, int max)
    {
        return Math.Max(min, Math.Min(max, value));
    }
}
