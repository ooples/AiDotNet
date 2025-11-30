namespace AiDotNet.Serving.ContinuousBatching;

/// <summary>
/// Statistics about the batcher's operation.
/// </summary>
public class BatcherStatistics
{
    /// <summary>Total tokens generated since start.</summary>
    public long TotalTokensGenerated { get; set; }

    /// <summary>Total requests completed since start.</summary>
    public long TotalRequestsProcessed { get; set; }

    /// <summary>Total batching iterations.</summary>
    public long TotalIterations { get; set; }

    /// <summary>Tokens generated per second.</summary>
    public double TokensPerSecond { get; set; }

    /// <summary>Requests completed per second.</summary>
    public double RequestsPerSecond { get; set; }

    /// <summary>Average batch size per iteration.</summary>
    public double AverageBatchSize { get; set; }

    /// <summary>Requests currently waiting.</summary>
    public int WaitingRequests { get; set; }

    /// <summary>Requests currently being processed.</summary>
    public int RunningRequests { get; set; }

    /// <summary>Memory utilization (0-1).</summary>
    public double MemoryUtilization { get; set; }

    /// <summary>Total runtime in seconds.</summary>
    public double RuntimeSeconds { get; set; }
}
