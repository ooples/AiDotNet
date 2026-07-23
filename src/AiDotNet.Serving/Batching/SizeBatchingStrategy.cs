namespace AiDotNet.Serving.Batching;

/// <summary>
/// Size-based batching strategy that processes batches when a size threshold is reached.
/// </summary>
public class SizeBatchingStrategy : IBatchingStrategy
{
    private readonly int _batchSize;
    private readonly int _maxWaitMs;

    /// <summary>
    /// Initializes a new instance of the SizeBatchingStrategy.
    /// </summary>
    /// <param name="batchSize">Target batch size to trigger processing</param>
    /// <param name="maxWaitMs">Maximum wait time before processing smaller batches</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown if batchSize is less than 1 or maxWaitMs is negative.</exception>
    public SizeBatchingStrategy(int batchSize, int maxWaitMs)
    {
        if (batchSize < 1)
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be at least 1.");
        if (maxWaitMs < 0)
            throw new ArgumentOutOfRangeException(nameof(maxWaitMs), "Max wait time cannot be negative.");

        _batchSize = batchSize;
        _maxWaitMs = maxWaitMs;
    }

    public string Name => "Size";

    public bool ShouldProcessBatch(int queuedRequests, double timeInQueueMs, double averageLatencyMs, int queueDepth)
    {
        // Process if we have enough requests or waited too long
        return queuedRequests >= _batchSize || (queuedRequests > 0 && timeInQueueMs >= _maxWaitMs);
    }

    public int GetOptimalBatchSize(int queuedRequests, double averageLatencyMs)
    {
        return Math.Min(queuedRequests, _batchSize);
    }

    public void UpdatePerformanceFeedback(int batchSize, double latencyMs)
    {
        // No adaptation for size strategy
    }
}
