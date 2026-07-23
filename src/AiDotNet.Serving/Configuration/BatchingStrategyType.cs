namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Batching strategies for request processing.
/// </summary>
public enum BatchingStrategyType
{
    /// <summary>Process batch after timeout expires.</summary>
    Timeout,

    /// <summary>Process batch when it reaches a certain size.</summary>
    Size,

    /// <summary>Group requests by sequence length into buckets.</summary>
    Bucket,

    /// <summary>Dynamically adjust batch size based on latency and throughput.</summary>
    Adaptive,

    /// <summary>Continuous batching - process requests as capacity becomes available.</summary>
    Continuous
}

