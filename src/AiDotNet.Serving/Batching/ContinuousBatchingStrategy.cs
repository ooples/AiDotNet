namespace AiDotNet.Serving.Batching;

/// <summary>
/// Continuous batching strategy that processes requests as soon as capacity is available.
/// </summary>
/// <remarks>
/// <para>
/// Unlike traditional batching which waits for a batch to fill or a timeout to expire,
/// continuous batching processes requests immediately when resources are available.
/// This maximizes throughput and minimizes latency for variable-duration workloads.
/// </para>
/// <para><b>For Beginners:</b> Continuous batching is like a conveyor belt in a factory.
///
/// Traditional batching: Wait until you have 10 items, then process them all together.
/// Continuous batching: Start processing each item as soon as there's capacity.
///
/// Benefits:
/// - Lowest possible latency (no waiting for batch to fill)
/// - Maximum throughput (always using full capacity)
/// - Better for variable-length requests (fast ones don't wait for slow ones)
///
/// When to use:
/// - High-throughput serving scenarios
/// - Variable processing times (like LLM inference)
/// - When latency is critical
/// </para>
/// </remarks>
public class ContinuousBatchingStrategy : BatchingStrategyBase
{
    private readonly int _maxConcurrency;
    private readonly int _minWaitMs;
    private readonly double _targetLatencyMs;
    private readonly bool _adaptiveConcurrency;

    private int _currentOptimalConcurrency;
    private DateTime _lastProcessTime = DateTime.MinValue;

    /// <summary>
    /// Initializes a new instance of the ContinuousBatchingStrategy.
    /// </summary>
    /// <param name="maxConcurrency">Maximum number of concurrent requests to process.</param>
    /// <param name="minWaitMs">Minimum wait time between processing attempts (prevents busy loop).</param>
    /// <param name="targetLatencyMs">Target latency for adaptive concurrency.</param>
    /// <param name="adaptiveConcurrency">Whether to adapt concurrency based on latency.</param>
    public ContinuousBatchingStrategy(
        int maxConcurrency = 32,
        int minWaitMs = 1,
        double targetLatencyMs = 50,
        bool adaptiveConcurrency = true)
    {
        if (maxConcurrency < 1)
            throw new ArgumentOutOfRangeException(nameof(maxConcurrency), "Max concurrency must be at least 1.");
        if (minWaitMs < 0)
            throw new ArgumentOutOfRangeException(nameof(minWaitMs), "Min wait time cannot be negative.");
        if (targetLatencyMs <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetLatencyMs), "Target latency must be positive.");

        _maxConcurrency = maxConcurrency;
        _minWaitMs = minWaitMs;
        _targetLatencyMs = targetLatencyMs;
        _adaptiveConcurrency = adaptiveConcurrency;
        _currentOptimalConcurrency = Math.Max(1, maxConcurrency / 2); // Start at half capacity
    }

    /// <summary>
    /// Gets the name of this batching strategy.
    /// </summary>
    public override string Name => "Continuous";

    /// <summary>
    /// Determines whether to process a batch. For continuous batching, this is true
    /// whenever there are requests and capacity is available.
    /// </summary>
    /// <param name="queuedRequests">Number of requests currently queued.</param>
    /// <param name="timeInQueueMs">Time since the oldest request was queued.</param>
    /// <param name="averageLatencyMs">Average latency of recent batches.</param>
    /// <param name="queueDepth">Current queue depth (may differ from queuedRequests if using priority queues).</param>
    /// <returns>True if a batch should be processed.</returns>
    public override bool ShouldProcessBatch(int queuedRequests, double timeInQueueMs, double averageLatencyMs, int queueDepth)
    {
        // No requests - nothing to process
        if (queuedRequests == 0)
            return false;

        // Enforce minimum wait to prevent busy loop
        var timeSinceLastProcess = (DateTime.UtcNow - _lastProcessTime).TotalMilliseconds;
        if (timeSinceLastProcess < _minWaitMs)
            return false;

        // Process immediately if we have requests
        // The caller should check capacity separately
        _lastProcessTime = DateTime.UtcNow;
        return true;
    }

    /// <summary>
    /// Gets the optimal batch size. For continuous batching, this returns the current
    /// optimal concurrency level, adjusted for available requests.
    /// </summary>
    /// <param name="queuedRequests">Number of requests currently queued.</param>
    /// <param name="averageLatencyMs">Average latency of recent batches.</param>
    /// <returns>The optimal batch size.</returns>
    public override int GetOptimalBatchSize(int queuedRequests, double averageLatencyMs)
    {
        lock (SyncLock)
        {
            // Return the lesser of queued requests and optimal concurrency
            return Math.Min(queuedRequests, _currentOptimalConcurrency);
        }
    }

    /// <summary>
    /// Updates the strategy with performance feedback, adjusting concurrency if adaptive mode is enabled.
    /// </summary>
    /// <param name="batchSize">Size of the batch that was processed.</param>
    /// <param name="latencyMs">Latency in milliseconds for processing the batch.</param>
    public override void UpdatePerformanceFeedback(int batchSize, double latencyMs)
    {
        base.UpdatePerformanceFeedback(batchSize, latencyMs);

        if (!_adaptiveConcurrency)
            return;

        lock (SyncLock)
        {
            // Calculate per-request latency
            double perRequestLatency = batchSize > 0 ? latencyMs / batchSize : latencyMs;

            // Adjust concurrency based on latency
            if (perRequestLatency < _targetLatencyMs * 0.8)
            {
                // Latency is good, try increasing concurrency
                _currentOptimalConcurrency = Math.Min(_currentOptimalConcurrency + 1, _maxConcurrency);
            }
            else if (perRequestLatency > _targetLatencyMs * 1.5)
            {
                // Latency is too high, decrease concurrency
                _currentOptimalConcurrency = Math.Max(_currentOptimalConcurrency - 1, 1);
            }
        }
    }

    /// <summary>
    /// Gets the current optimal concurrency level.
    /// </summary>
    public int CurrentConcurrency
    {
        get
        {
            lock (SyncLock)
            {
                return _currentOptimalConcurrency;
            }
        }
    }

    /// <summary>
    /// Gets statistics about the continuous batching strategy.
    /// </summary>
    /// <returns>Dictionary of statistics.</returns>
    public override Dictionary<string, object> GetStatistics()
    {
        var stats = base.GetStatistics();
        lock (SyncLock)
        {
            stats["currentConcurrency"] = _currentOptimalConcurrency;
            stats["maxConcurrency"] = _maxConcurrency;
            stats["targetLatencyMs"] = _targetLatencyMs;
            stats["adaptiveConcurrency"] = _adaptiveConcurrency;
        }
        return stats;
    }
}
