namespace AiDotNet.Serving.Batching;

/// <summary>
/// Adaptive batching strategy that dynamically adjusts batch size based on latency and throughput.
/// This strategy aims to maximize throughput while maintaining latency SLAs.
/// </summary>
public class AdaptiveBatchingStrategy : IBatchingStrategy
{
    private readonly int _minBatchSize;
    private readonly int _maxBatchSize;
    private readonly int _maxWaitMs;
    private readonly double _targetLatencyMs;
    private readonly double _latencyToleranceFactor;

    private int _currentOptimalBatchSize;
    private double _recentAverageLatency;
    private readonly object _lock = new();

    /// <summary>
    /// Initializes a new instance of the AdaptiveBatchingStrategy.
    /// </summary>
    /// <param name="minBatchSize">Minimum batch size</param>
    /// <param name="maxBatchSize">Maximum batch size</param>
    /// <param name="maxWaitMs">Maximum wait time before processing</param>
    /// <param name="targetLatencyMs">Target latency in milliseconds</param>
    /// <param name="latencyToleranceFactor">Tolerance factor for latency (e.g., 2.0 means p99 should be &lt; 2x p50)</param>
    public AdaptiveBatchingStrategy(
        int minBatchSize,
        int maxBatchSize,
        int maxWaitMs,
        double targetLatencyMs,
        double latencyToleranceFactor = 2.0)
    {
        _minBatchSize = minBatchSize;
        _maxBatchSize = maxBatchSize;
        _maxWaitMs = maxWaitMs;
        _targetLatencyMs = targetLatencyMs;
        _latencyToleranceFactor = latencyToleranceFactor;
        _currentOptimalBatchSize = minBatchSize;
        _recentAverageLatency = targetLatencyMs;
    }

    public string Name => "Adaptive";

    public bool ShouldProcessBatch(int queuedRequests, double timeInQueueMs, double averageLatencyMs, int queueDepth)
    {
        if (queuedRequests == 0)
            return false;

        // Process if we have enough requests for optimal batch size
        if (queuedRequests >= _currentOptimalBatchSize)
            return true;

        // Process if we're approaching max wait time
        if (timeInQueueMs >= _maxWaitMs)
            return true;

        // Process if queue is building up (backpressure detection)
        if (queueDepth > _currentOptimalBatchSize * 2)
            return true;

        return false;
    }

    public int GetOptimalBatchSize(int queuedRequests, double averageLatencyMs)
    {
        lock (_lock)
        {
            return Math.Min(Math.Min(queuedRequests, _currentOptimalBatchSize), _maxBatchSize);
        }
    }

    public void UpdatePerformanceFeedback(int batchSize, double latencyMs)
    {
        lock (_lock)
        {
            // Exponential moving average of latency
            const double alpha = 0.3; // Smoothing factor
            _recentAverageLatency = alpha * latencyMs + (1 - alpha) * _recentAverageLatency;

            // Adapt batch size based on latency
            if (_recentAverageLatency < _targetLatencyMs)
            {
                // Latency is good, try increasing batch size
                _currentOptimalBatchSize = Math.Min(_currentOptimalBatchSize + 5, _maxBatchSize);
            }
            else if (_recentAverageLatency > _targetLatencyMs * _latencyToleranceFactor)
            {
                // Latency is too high, decrease batch size
                _currentOptimalBatchSize = Math.Max(_currentOptimalBatchSize - 5, _minBatchSize);
            }
            // Otherwise, keep current batch size
        }
    }
}
