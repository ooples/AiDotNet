namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Configuration options for the model serving framework.
/// This class defines settings for server behavior, request batching, and startup model loading.
/// </summary>
public class ServingOptions
{
    /// <summary>
    /// Gets or sets the port number on which the server will listen.
    /// Default is 5000.
    /// </summary>
    public int Port { get; set; } = 5000;

    /// <summary>
    /// Gets or sets the batching window in milliseconds.
    /// This is the maximum time the batcher will wait before processing accumulated requests.
    /// Default is 10 milliseconds.
    /// </summary>
    public int BatchingWindowMs { get; set; } = 10;

    /// <summary>
    /// Gets or sets the maximum batch size for inference requests.
    /// If set to 0 or less, there is no limit on batch size.
    /// Default is 100.
    /// </summary>
    public int MaxBatchSize { get; set; } = 100;

    /// <summary>
    /// Gets or sets the minimum batch size for adaptive batching.
    /// Default is 1.
    /// </summary>
    public int MinBatchSize { get; set; } = 1;

    /// <summary>
    /// Gets or sets whether to enable adaptive batch sizing based on latency feedback.
    /// When enabled, the batch size is dynamically adjusted to meet latency targets.
    /// Default is true.
    /// </summary>
    public bool AdaptiveBatchSize { get; set; } = true;

    /// <summary>
    /// Gets or sets the batching strategy to use.
    /// Default is Adaptive.
    /// </summary>
    public BatchingStrategyType BatchingStrategy { get; set; } = BatchingStrategyType.Adaptive;

    /// <summary>
    /// Gets or sets the target latency in milliseconds for adaptive batching.
    /// The adaptive strategy will try to maintain this latency while maximizing throughput.
    /// Default is 20 milliseconds.
    /// </summary>
    public double TargetLatencyMs { get; set; } = 20.0;

    /// <summary>
    /// Gets or sets the latency tolerance factor for adaptive batching.
    /// This defines the acceptable ratio between p99 and p50 latency.
    /// Default is 2.0 (p99 should be less than 2x p50).
    /// </summary>
    public double LatencyToleranceFactor { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the maximum queue size for backpressure handling.
    /// When the queue is full, new requests will be rejected.
    /// Set to 0 for unlimited queue size.
    /// Default is 1000.
    /// </summary>
    public int MaxQueueSize { get; set; } = 1000;

    /// <summary>
    /// Gets or sets whether to enable priority-based request scheduling.
    /// Default is false.
    /// </summary>
    public bool EnablePriorityScheduling { get; set; } = false;

    /// <summary>
    /// Gets or sets the padding strategy to use for variable-length sequences.
    /// Default is Minimal.
    /// </summary>
    public PaddingStrategyType PaddingStrategy { get; set; } = PaddingStrategyType.Minimal;

    /// <summary>
    /// Gets or sets the bucket sizes for bucket-based batching and padding.
    /// Default is [32, 64, 128, 256, 512].
    /// </summary>
    public int[] BucketSizes { get; set; } = new[] { 32, 64, 128, 256, 512 };

    /// <summary>
    /// Gets or sets the fixed size for fixed-size padding strategy.
    /// Only used when PaddingStrategy is "Fixed".
    /// Default is 512.
    /// </summary>
    public int FixedPaddingSize { get; set; } = 512;

    /// <summary>
    /// Gets or sets whether to enable detailed performance metrics collection.
    /// This includes latency percentiles, throughput, and batch utilization.
    /// Default is true.
    /// </summary>
    public bool EnablePerformanceMetrics { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum number of latency samples to keep for percentile calculation.
    /// Default is 10000.
    /// </summary>
    public int MaxLatencySamples { get; set; } = 10000;

    /// <summary>
    /// Gets or sets the root directory where model files are stored.
    /// Model paths are restricted to this directory for security.
    /// Default is "models" relative to the application directory.
    /// </summary>
    public string ModelDirectory { get; set; } = "models";

    /// <summary>
    /// Gets or sets the list of models to load at startup.
    /// </summary>
    public List<StartupModel> StartupModels { get; set; } = new();
}

