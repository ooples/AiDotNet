namespace AiDotNet.Serving.Engine;

/// <summary>
/// A point-in-time snapshot of engine load and KV-cache utilization, used for admission control, autoscaling,
/// and the <c>/metrics</c> endpoint. These are the same signals vLLM/TGI expose (running vs waiting counts,
/// KV-cache usage, preemptions) so operators can reason about saturation and tune batch/memory limits.
/// </summary>
public sealed class EngineStatistics
{
    /// <summary>Number of sequences currently in the running batch.</summary>
    public int RunningSequences { get; init; }

    /// <summary>Number of sequences admitted but waiting for a batch slot / KV blocks.</summary>
    public int WaitingSequences { get; init; }

    /// <summary>Number of sequences preempted (swapped or recompute) and awaiting reschedule.</summary>
    public int SwappedSequences { get; init; }

    /// <summary>Fraction (0..1) of KV-cache blocks currently allocated.</summary>
    public double KvCacheUsage { get; init; }

    /// <summary>Total KV-cache blocks in the pool.</summary>
    public int TotalKvBlocks { get; init; }

    /// <summary>KV-cache blocks currently free.</summary>
    public int FreeKvBlocks { get; init; }

    /// <summary>Cumulative number of preemptions since start (a rising count signals under-provisioned KV memory).</summary>
    public long TotalPreemptions { get; init; }

    /// <summary>Cumulative number of requests finished since start.</summary>
    public long TotalFinishedRequests { get; init; }
}
