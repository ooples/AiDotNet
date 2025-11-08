namespace AiDotNet.Serving.Batching;

/// <summary>
/// Bucket-based batching strategy that groups requests by input size into buckets.
/// This minimizes padding overhead for variable-length sequences.
/// </summary>
public class BucketBatchingStrategy : IBatchingStrategy
{
    private readonly int[] _bucketBoundaries;
    private readonly int _maxBatchSize;
    private readonly int _maxWaitMs;

    /// <summary>
    /// Initializes a new instance of the BucketBatchingStrategy.
    /// </summary>
    /// <param name="bucketBoundaries">Array of bucket boundaries (e.g., [32, 64, 128, 256, 512])</param>
    /// <param name="maxBatchSize">Maximum batch size per bucket</param>
    /// <param name="maxWaitMs">Maximum wait time before processing</param>
    public BucketBatchingStrategy(int[] bucketBoundaries, int maxBatchSize, int maxWaitMs)
    {
        _bucketBoundaries = bucketBoundaries ?? new[] { 32, 64, 128, 256, 512 };
        Array.Sort(_bucketBoundaries);
        _maxBatchSize = maxBatchSize;
        _maxWaitMs = maxWaitMs;
    }

    public string Name => "Bucket";

    /// <summary>
    /// Gets the bucket index for a given input size.
    /// </summary>
    /// <param name="inputSize">The size of the input</param>
    /// <returns>The bucket index</returns>
    public int GetBucketIndex(int inputSize)
    {
        for (int i = 0; i < _bucketBoundaries.Length; i++)
        {
            if (inputSize <= _bucketBoundaries[i])
                return i;
        }
        return _bucketBoundaries.Length; // Largest bucket
    }

    /// <summary>
    /// Gets the padded size for a bucket.
    /// </summary>
    /// <param name="bucketIndex">The bucket index</param>
    /// <returns>The padded size for the bucket</returns>
    public int GetBucketSize(int bucketIndex)
    {
        if (bucketIndex < _bucketBoundaries.Length)
            return _bucketBoundaries[bucketIndex];
        return _bucketBoundaries[^1] * 2; // Double the largest bucket for overflow
    }

    public bool ShouldProcessBatch(int queuedRequests, double timeInQueueMs, double averageLatencyMs, int queueDepth)
    {
        return queuedRequests >= _maxBatchSize || (queuedRequests > 0 && timeInQueueMs >= _maxWaitMs);
    }

    public int GetOptimalBatchSize(int queuedRequests, double averageLatencyMs)
    {
        return Math.Min(queuedRequests, _maxBatchSize);
    }

    public void UpdatePerformanceFeedback(int batchSize, double latencyMs)
    {
        // Could be enhanced to adapt bucket boundaries based on request distribution
    }
}
