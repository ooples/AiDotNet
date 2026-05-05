namespace AiDotNet.Deployment.Configuration;

/// <summary>
/// Telemetry summary for a model's weight-streaming activity. Issue #1222
/// task #186. Returned on <c>AiModelResult.WeightStreamingReport</c> after
/// a Build that ran with streaming enabled (whether explicitly via
/// <c>ConfigureWeightStreaming</c> or auto-detected from parameter
/// count).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When weight streaming pages model weights
/// to disk to fit in RAM, this report tells you how much actually
/// happened during your training/inference run: how many cold disk
/// reads were needed, how many evictions the LRU pool performed, how
/// often the prefetcher had the right weights ready vs. caught short.
/// High eviction counts with low prefetch hits suggest you should bump
/// the pool capacity; high disk-read counts with stable evictions
/// suggest you're at steady-state and the working-set fits the budget.</para>
///
/// <para>The numbers come from the underlying
/// <c>AiDotNet.Tensors.WeightRegistry</c>'s streaming pool. AiDotNet
/// wraps them in this DTO so callers can rely on stable property names
/// across Tensors-side rewrites.</para>
/// </remarks>
public sealed class WeightStreamingReport
{
    /// <summary>
    /// True when this report was produced by an actual streaming
    /// configuration (vs. default-eager). Always <c>true</c> when this
    /// instance is non-null on the result; included for symmetry with
    /// callers that want a single is-streaming check.
    /// </summary>
    public bool StreamingEnabled { get; init; } = true;

    /// <summary>
    /// True when streaming engaged because <see cref="WeightStreamingConfig.Enabled"/>
    /// was left at <c>null</c> and the parameter-count threshold was
    /// crossed (vs. user explicitly forcing it on / off). Distinguishes
    /// "the framework decided we needed this" from "the user asked for
    /// it" in operator dashboards.
    /// </summary>
    public bool AutoDetected { get; init; }

    /// <summary>
    /// Total parameter count the model declared at the time streaming
    /// was configured. Useful when comparing across runs to verify the
    /// model size is what the user expected (catches ctor-arg drift
    /// where a config bumps the layer count without anyone noticing).
    /// </summary>
    public long ModelParameterCount { get; init; }

    /// <summary>
    /// The threshold the auto-detect compared against. Always the
    /// effective threshold actually used (env-var override applied if
    /// present), so debugging "why did this engage / not engage?" is a
    /// single-property check.
    /// </summary>
    public long EffectiveThresholdParameters { get; init; }

    /// <summary>
    /// Number of cold-disk reads the streaming pool performed during the
    /// run (cache misses where the weight had to be fetched from the
    /// backing store). Higher than expected? Either the pool is too
    /// small or the prefetcher isn't keeping up.
    /// </summary>
    public long DiskReadCount { get; init; }

    /// <summary>
    /// Number of LRU evictions. A non-trivial eviction count is normal
    /// and expected — that's the whole point of streaming. A very high
    /// count relative to the layer count (e.g. 10× more evictions than
    /// the model has layers) points at thrashing.
    /// </summary>
    public long EvictionCount { get; init; }

    /// <summary>
    /// Number of prefetch issues — calls to
    /// <c>WeightRegistry.PrefetchAsync</c> the forward path made.
    /// Should match (layer-count × forward-pass-count) approximately.
    /// </summary>
    public long PrefetchIssueCount { get; init; }

    /// <summary>
    /// Number of prefetches that completed before the requesting layer
    /// needed the weights (the win condition — async overlap between
    /// disk read and forward compute). High ratio of hits-to-issues
    /// means the prefetcher is doing its job.
    /// </summary>
    public long PrefetchHitCount { get; init; }

    /// <summary>
    /// Number of prefetches that hadn't completed by the time the
    /// requesting layer needed the weights — MaterializeScope blocked
    /// briefly waiting for the read to finish. Equivalent to a synchronous
    /// disk read for that layer; not catastrophic but signals the
    /// prefetch window is too small or the disk is under load.
    /// </summary>
    public long PrefetchMissCount { get; init; }

    /// <summary>
    /// Total bytes the streaming pool wrote to its disk-backing store
    /// during the run (for compressed pools, this is the
    /// post-compression size). Useful for IO-budget planning on
    /// constrained devices.
    /// </summary>
    public long BytesWrittenToDisk { get; init; }

    /// <summary>
    /// Total bytes the streaming pool read from its disk-backing store
    /// during the run (post-decompression for compressed pools, so
    /// represents the materialized weight bytes).
    /// </summary>
    public long BytesReadFromDisk { get; init; }
}
