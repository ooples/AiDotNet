using AiDotNet.Enums;

namespace AiDotNet.Deployment.Configuration;

/// <summary>
/// Configuration for model caching - storing loaded models in memory to avoid repeated loading.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Loading an AI model from disk takes time. Caching keeps recently-used
/// models in memory so they can be used again instantly, like keeping your frequently-used apps
/// open on your phone instead of closing and reopening them.
///
/// Benefits:
/// - Much faster inference (no model loading time)
/// - Better throughput when serving multiple requests
/// - Reduces disk I/O
///
/// Considerations:
/// - Uses memory (RAM) to store models
/// - Limited cache size - old models get evicted when full
///
/// Eviction Policies (what to remove when cache is full):
/// - LRU (Least Recently Used): Removes models you haven't used in a while (recommended)
/// - LFU (Least Frequently Used): Removes models used least often
/// - FIFO: Removes oldest models first
/// - Random: Removes random models (simple but unpredictable)
///
/// For most applications, LRU with a moderate max size works well.
/// </para>
/// </remarks>
public class CacheConfig
{
    /// <summary>
    /// Gets or sets whether caching is enabled (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Set to true to enable caching, false to disable it entirely.
    /// Caching is recommended for production systems to improve performance.
    /// </para>
    /// </remarks>
    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum number of models to cache (default: 10).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many models to keep in memory simultaneously.
    /// Higher values use more memory but reduce cache misses. 10 is a good default for most cases.
    /// </para>
    /// </remarks>
    public int MaxCacheSize { get; set; } = 10;

    /// <summary>
    /// Gets or sets the cache eviction policy (default: LRU).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Determines which model to remove when cache is full.
    /// LRU (Least Recently Used) is recommended - it removes models you haven't used recently.
    /// </para>
    /// </remarks>
    public Enums.CacheEvictionPolicy EvictionPolicy { get; set; } = Enums.CacheEvictionPolicy.LRU;

    /// <summary>
    /// Gets or sets the cache entry time-to-live in seconds (default: 3600 = 1 hour).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How long unused models stay in cache before removal.
    /// Default is 1 hour. Set to 0 to disable TTL (models only removed when cache is full).
    /// </para>
    /// </remarks>
    public int TimeToLiveSeconds { get; set; } = 3600;

    /// <summary>
    /// Gets or sets the cache entry time-to-live as a TimeSpan (default: 1 hour).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A more convenient way to specify the cache duration.
    /// This is equivalent to TimeToLiveSeconds but uses the TimeSpan type for clarity.
    /// </para>
    /// </remarks>
    public TimeSpan DefaultTTL
    {
        get => TimeSpan.FromSeconds(TimeToLiveSeconds);
        set => TimeToLiveSeconds = (int)value.TotalSeconds;
    }

    /// <summary>
    /// Gets or sets the maximum cache size in megabytes (default: 100.0 MB).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Memory-based limit for the cache. When total cached model
    /// size exceeds this, older models are evicted. Use in addition to MaxCacheSize for
    /// more precise memory control.
    /// </para>
    /// </remarks>
    public double MaxSizeMB { get; set; } = 100.0;

    /// <summary>
    /// Gets or sets whether to preload models on startup (default: false).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If true, frequently-used models are loaded into cache at startup.
    /// This eliminates first-request latency but increases startup time. Use for production servers.
    /// </para>
    /// </remarks>
    public bool PreloadModels { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to track cache hit/miss statistics (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Tracks how often models are found in cache (hits) vs loaded from disk (misses).
    /// Useful for monitoring and optimization but has tiny performance overhead. Recommended.
    /// </para>
    /// </remarks>
    public bool TrackStatistics { get; set; } = true;

    // ---------------------------------------------------------------------------------------------
    // Optimizer (training-time) caches
    //
    // The fields above govern the model-serving cache (loaded models kept in RAM). The fields below
    // govern the two IN-TRAINING optimizer caches, which are a separate concern and independently
    // tunable:
    //   * the gradient cache (per-step gradients, DefaultGradientCache), and
    //   * the model-evaluation cache (per-epoch fitness/step data, DefaultModelCache).
    // These caches are numerically transparent — a miss simply recomputes the same deterministic
    // value — so they exist purely to bound memory/time. Their defaults are the recommended values
    // (small FIFO windows) that fix the unbounded-growth training-loop leak; raise them only if a
    // workload legitimately re-queries older keys. When Enabled is false, both are disabled (every
    // lookup misses and recomputes).
    // ---------------------------------------------------------------------------------------------

    /// <summary>
    /// Maximum number of per-step gradients retained by the optimizer's gradient cache
    /// (<c>DefaultGradientCache</c>). Default 8 — enough to cover every legitimate short-window reuse
    /// (gradient-check double-eval, line-search / trust-region re-eval, DDP AllReduce read-back) while
    /// keeping the footprint independent of the number of training steps. Values &lt;= 0 fall back to the default.
    /// </summary>
    public int GradientCacheCapacity { get; set; } = 8;

    /// <summary>
    /// Maximum number of per-evaluation step-data entries retained by the optimizer's model-evaluation
    /// cache (<c>DefaultModelCache</c>). Default 8. Each entry retains a deep-copied model plus O(N)
    /// predictions, so a small bound removes the per-epoch leak; a gradient optimizer never re-queries a
    /// prior epoch's key anyway. Values &lt;= 0 fall back to the default.
    /// </summary>
    public int ModelCacheCapacity { get; set; } = 8;

    /// <summary>
    /// Eviction policy for BOTH optimizer caches (default: <see cref="Enums.CacheEvictionPolicy.FIFO"/>).
    /// FIFO is recommended for training-time caches because reuse is a short consecutive window; LRU/LFU
    /// are available for workloads that re-hit older keys. This is intentionally separate from
    /// <see cref="EvictionPolicy"/> (which governs the serving cache and defaults to LRU).
    /// </summary>
    public Enums.CacheEvictionPolicy OptimizerCacheEvictionPolicy { get; set; } = Enums.CacheEvictionPolicy.FIFO;
}
