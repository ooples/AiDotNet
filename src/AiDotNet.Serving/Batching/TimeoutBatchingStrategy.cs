namespace AiDotNet.Serving.Batching;

/// <summary>
/// Timeout-based batching strategy that processes batches when a time threshold is reached.
/// </summary>
public class TimeoutBatchingStrategy : IBatchingStrategy
{
    private readonly int _timeoutMs;
    private readonly int _maxBatchSize;

    /// <summary>
    /// Initializes a new instance of the TimeoutBatchingStrategy.
    /// </summary>
    /// <param name="timeoutMs">Maximum time to wait before processing a batch</param>
    /// <param name="maxBatchSize">Maximum batch size</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown if timeoutMs is negative or maxBatchSize is less than 1.</exception>
    public TimeoutBatchingStrategy(int timeoutMs, int maxBatchSize)
    {
        if (timeoutMs < 0)
            throw new ArgumentOutOfRangeException(nameof(timeoutMs), "Timeout cannot be negative.");
        if (maxBatchSize < 1)
            throw new ArgumentOutOfRangeException(nameof(maxBatchSize), "Max batch size must be at least 1.");

        _timeoutMs = timeoutMs;
        _maxBatchSize = maxBatchSize;
    }

    public string Name => "Timeout";

    public bool ShouldProcessBatch(int queuedRequests, double timeInQueueMs, double averageLatencyMs, int queueDepth)
    {
        return queuedRequests > 0 && timeInQueueMs >= _timeoutMs;
    }

    public int GetOptimalBatchSize(int queuedRequests, double averageLatencyMs)
    {
        return Math.Min(queuedRequests, _maxBatchSize);
    }

    public void UpdatePerformanceFeedback(int batchSize, double latencyMs)
    {
        // No adaptation for timeout strategy
    }
}
