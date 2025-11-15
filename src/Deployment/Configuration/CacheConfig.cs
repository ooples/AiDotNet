using AiDotNet.Enums;

namespace AiDotNet.Deployment.Configuration;

/// <summary>
/// Configuration for model caching - storing loaded models in memory to avoid repeated loading.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Loading an AI model from disk takes time. Caching keeps recently-used
/// models in memory so they can be used again instantly, like keeping your frequently-used apps
/// open on your phone instead of closing and reopening them.
///
/// **Benefits:**
/// - Much faster inference (no model loading time)
/// - Better throughput when serving multiple requests
/// - Reduces disk I/O
///
/// **Considerations:**
/// - Uses memory (RAM) to store models
/// - Limited cache size - old models get evicted when full
///
/// **Eviction Policies:** (what to remove when cache is full)
/// - **LRU (Least Recently Used)**: Removes models you haven't used in a while (recommended)
/// - **LFU (Least Frequently Used)**: Removes models used least often
/// - **FIFO**: Removes oldest models first
/// - **Random**: Removes random models (simple but unpredictable)
///
/// For most applications, **LRU** with a moderate max size works well.
/// </remarks>
public class CacheConfig
{
    /// <summary>
    /// Gets or sets whether caching is enabled (default: true).
    /// </summary>
    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum number of models to cache (default: 10).
    /// Higher = more memory usage but fewer cache misses.
    /// </summary>
    public int MaxCacheSize { get; set; } = 10;

    /// <summary>
    /// Gets or sets the cache eviction policy (default: LRU).
    /// </summary>
    public CacheEvictionPolicy EvictionPolicy { get; set; } = CacheEvictionPolicy.LRU;

    /// <summary>
    /// Gets or sets the cache entry time-to-live in seconds (default: 3600 = 1 hour).
    /// Models unused for this duration are removed even if cache isn't full.
    /// Set to 0 to disable TTL.
    /// </summary>
    public int TimeToLiveSeconds { get; set; } = 3600;

    /// <summary>
    /// Gets or sets whether to preload models on startup (default: false).
    /// If true, frequently-used models are loaded into cache at initialization.
    /// </summary>
    public bool PreloadModels { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to track cache hit/miss statistics (default: true).
    /// Useful for monitoring but has small performance overhead.
    /// </summary>
    public bool TrackStatistics { get; set; } = true;

    /// <summary>
    /// Creates a disabled cache configuration (no caching).
    /// </summary>
    public static CacheConfig Disabled()
    {
        return new CacheConfig
        {
            Enabled = false
        };
    }

    /// <summary>
    /// Creates a small cache configuration (3 models, LRU).
    /// Good for memory-constrained environments.
    /// </summary>
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
    /// Good for servers with plenty of RAM.
    /// </summary>
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
