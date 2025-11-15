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
    public CacheEvictionPolicy EvictionPolicy { get; set; } = CacheEvictionPolicy.LRU;

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

    /// <summary>
    /// Creates a disabled cache configuration (no caching).
    /// </summary>
    /// <returns>A configuration with caching disabled.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this to disable caching entirely.
    /// Models will be loaded from disk every time. Only use for debugging or very memory-constrained environments.
    /// </para>
    /// </remarks>
    public static CacheConfig Disabled()
    {
        return new CacheConfig
        {
            Enabled = false
        };
    }

    /// <summary>
    /// Creates a small cache configuration (3 models, LRU).
    /// </summary>
    /// <returns>A small cache configuration.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this for memory-constrained environments like mobile devices or edge hardware.
    /// Caches only 3 models, minimizing memory usage while still providing some performance benefit.
    /// </para>
    /// </remarks>
    public static CacheConfig Small()
    {
        return new CacheConfig
        {
            Enabled = true,
            MaxCacheSize = 3,
            EvictionPolicy = CacheEvictionPolicy.LRU
        };
    }

    /// <summary>
    /// Creates a large cache configuration (50 models, LRU).
    /// </summary>
    /// <returns>A large cache configuration.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this for servers with plenty of RAM.
    /// Caches up to 50 models and preloads frequently-used ones at startup for maximum performance.
    /// </para>
    /// </remarks>
    public static CacheConfig Large()
    {
        return new CacheConfig
        {
            Enabled = true,
            MaxCacheSize = 50,
            EvictionPolicy = CacheEvictionPolicy.LRU,
            PreloadModels = true
        };
    }
}
