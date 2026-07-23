using System.Collections.Concurrent;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Serving.Monitoring;

/// <summary>
/// Tracks performance metrics for the serving system including latency percentiles,
/// throughput, batch utilization, and queue depth.
/// </summary>
public class PerformanceMetrics
{
    private readonly ConcurrentQueue<double> _latencySamples;
    private readonly int _maxSamples;
    private readonly object _lock = new();

    // Throughput tracking
    private long _totalRequests;
    private long _totalBatches;
    private long _totalBatchSize;
    private DateTime _startTime;

    // Queue depth tracking
    private readonly ConcurrentQueue<int> _queueDepthSamples;
    private readonly int _maxQueueDepthSamples;

    // Batch utilization tracking
    private long _totalPaddingElements;
    private long _totalActualElements;

    /// <summary>
    /// Initializes a new instance of the PerformanceMetrics class.
    /// </summary>
    /// <param name="maxSamples">Maximum number of latency samples to keep for percentile calculation</param>
    /// <param name="maxQueueDepthSamples">Maximum number of queue depth samples to keep</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown if maxSamples or maxQueueDepthSamples is less than 1.</exception>
    public PerformanceMetrics(int maxSamples = 10000, int maxQueueDepthSamples = 1000)
    {
        if (maxSamples < 1)
            throw new ArgumentOutOfRangeException(nameof(maxSamples), "Max samples must be at least 1.");
        if (maxQueueDepthSamples < 1)
            throw new ArgumentOutOfRangeException(nameof(maxQueueDepthSamples), "Max queue depth samples must be at least 1.");

        _maxSamples = maxSamples;
        _maxQueueDepthSamples = maxQueueDepthSamples;
        _latencySamples = new ConcurrentQueue<double>();
        _queueDepthSamples = new ConcurrentQueue<int>();
        _startTime = DateTime.UtcNow;
    }

    /// <summary>
    /// Records a latency measurement.
    /// </summary>
    /// <param name="latencyMs">Latency in milliseconds</param>
    public void RecordLatency(double latencyMs)
    {
        _latencySamples.Enqueue(latencyMs);

        // Trim old samples if we exceed the maximum
        while (_latencySamples.Count > _maxSamples)
        {
            _latencySamples.TryDequeue(out _);
        }
    }

    /// <summary>
    /// Records a batch processing event.
    /// </summary>
    /// <param name="batchSize">Size of the batch</param>
    /// <param name="latencyMs">Latency in milliseconds</param>
    public void RecordBatch(int batchSize, double latencyMs)
    {
        Interlocked.Add(ref _totalRequests, batchSize);
        Interlocked.Increment(ref _totalBatches);
        Interlocked.Add(ref _totalBatchSize, batchSize);
        RecordLatency(latencyMs);
    }

    /// <summary>
    /// Records queue depth at a point in time.
    /// </summary>
    /// <param name="queueDepth">Current queue depth</param>
    public void RecordQueueDepth(int queueDepth)
    {
        _queueDepthSamples.Enqueue(queueDepth);

        // Trim old samples
        while (_queueDepthSamples.Count > _maxQueueDepthSamples)
        {
            _queueDepthSamples.TryDequeue(out _);
        }
    }

    /// <summary>
    /// Records batch utilization metrics (for padding efficiency).
    /// </summary>
    /// <param name="actualElements">Number of actual data elements</param>
    /// <param name="paddingElements">Number of padding elements</param>
    public void RecordBatchUtilization(int actualElements, int paddingElements)
    {
        Interlocked.Add(ref _totalActualElements, actualElements);
        Interlocked.Add(ref _totalPaddingElements, paddingElements);
    }

    /// <summary>
    /// Calculates the specified percentile from latency samples.
    /// Uses reservoir sampling for large sample sets to improve performance.
    /// </summary>
    /// <param name="percentile">Percentile to calculate (0-100)</param>
    /// <returns>The latency at the specified percentile</returns>
    public double GetLatencyPercentile(double percentile)
    {
        if (percentile < 0 || percentile > 100)
            throw new ArgumentException("Percentile must be between 0 and 100", nameof(percentile));

        var allSamples = _latencySamples.ToArray();
        if (allSamples.Length == 0)
            return 0.0;

        // For large sample sizes, use reservoir sampling to reduce sorting cost
        const int maxSortSize = 1000;
        double[] samples;

        if (allSamples.Length <= maxSortSize)
        {
            samples = allSamples;
        }
        else
        {
            // Reservoir sampling: randomly select maxSortSize samples
            samples = new double[maxSortSize];
            var random = RandomHelper.CreateSecureRandom();

            for (int i = 0; i < maxSortSize; i++)
            {
                samples[i] = allSamples[i];
            }

            for (int i = maxSortSize; i < allSamples.Length; i++)
            {
                int j = random.Next(i + 1);
                if (j < maxSortSize)
                {
                    samples[j] = allSamples[i];
                }
            }
        }

        Array.Sort(samples);
        int index = (int)Math.Ceiling(percentile / 100.0 * samples.Length) - 1;
        index = Math.Max(0, Math.Min(index, samples.Length - 1));

        return samples[index];
    }

    /// <summary>
    /// Gets the average latency across all samples.
    /// </summary>
    /// <returns>Average latency in milliseconds</returns>
    public double GetAverageLatency()
    {
        var samples = _latencySamples.ToArray();
        return samples.Length > 0 ? samples.Average() : 0.0;
    }

    /// <summary>
    /// Gets the current throughput in requests per second.
    /// </summary>
    /// <returns>Throughput in requests/second</returns>
    public double GetThroughput()
    {
        var elapsed = DateTime.UtcNow - _startTime;
        return elapsed.TotalSeconds > 0 ? _totalRequests / elapsed.TotalSeconds : 0.0;
    }

    /// <summary>
    /// Gets the average batch size.
    /// </summary>
    /// <returns>Average batch size</returns>
    public double GetAverageBatchSize()
    {
        return _totalBatches > 0 ? (double)_totalBatchSize / _totalBatches : 0.0;
    }

    /// <summary>
    /// Gets the average queue depth.
    /// </summary>
    /// <returns>Average queue depth</returns>
    public double GetAverageQueueDepth()
    {
        var samples = _queueDepthSamples.ToArray();
        return samples.Length > 0 ? samples.Average() : 0.0;
    }

    /// <summary>
    /// Gets the batch utilization percentage (actual elements / total elements including padding).
    /// </summary>
    /// <returns>Batch utilization as a percentage (0-100)</returns>
    public double GetBatchUtilization()
    {
        var total = _totalActualElements + _totalPaddingElements;
        return total > 0 ? (_totalActualElements * 100.0) / total : 100.0;
    }

    /// <summary>
    /// Gets all metrics as a dictionary.
    /// </summary>
    /// <returns>Dictionary containing all metrics</returns>
    public Dictionary<string, object> GetAllMetrics()
    {
        var metrics = new Dictionary<string, object>
        {
            ["totalRequests"] = _totalRequests,
            ["totalBatches"] = _totalBatches,
            ["throughputRequestsPerSecond"] = GetThroughput(),
            ["averageBatchSize"] = GetAverageBatchSize(),
            ["latencyP50Ms"] = GetLatencyPercentile(50),
            ["latencyP95Ms"] = GetLatencyPercentile(95),
            ["latencyP99Ms"] = GetLatencyPercentile(99),
            ["averageLatencyMs"] = GetAverageLatency(),
            ["averageQueueDepth"] = GetAverageQueueDepth(),
            ["batchUtilizationPercent"] = GetBatchUtilization(),
            ["uptimeSeconds"] = (DateTime.UtcNow - _startTime).TotalSeconds
        };

        return metrics;
    }

    /// <summary>
    /// Resets all metrics.
    /// </summary>
    public void Reset()
    {
        lock (_lock)
        {
            while (_latencySamples.TryDequeue(out _)) { }
            while (_queueDepthSamples.TryDequeue(out _)) { }

            _totalRequests = 0;
            _totalBatches = 0;
            _totalBatchSize = 0;
            _totalPaddingElements = 0;
            _totalActualElements = 0;
            _startTime = DateTime.UtcNow;
        }
    }
}
