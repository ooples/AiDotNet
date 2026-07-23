namespace AiDotNet.Data.Pipeline;

/// <summary>
/// Configuration options for disk-based pipeline caching.
/// </summary>
/// <remarks>
/// <para>
/// Controls how pipeline snapshots are stored on disk, including cache location,
/// size limits, and eviction policies.
/// </para>
/// <para><b>For Beginners:</b> These settings control where processed data is saved
/// and how much disk space it can use. Once data is cached, subsequent training epochs
/// can skip expensive preprocessing.
/// </para>
/// </remarks>
public sealed class DiskCacheOptions
{
    /// <summary>
    /// Gets or sets the directory where cache files are stored.
    /// Default is a "pipeline_cache" subdirectory of the current working directory.
    /// </summary>
    public string CacheDirectory { get; set; } = Path.Combine(
        Environment.CurrentDirectory, "pipeline_cache");

    /// <summary>
    /// Gets or sets the maximum total cache size in bytes.
    /// Default is 10 GB. Set to 0 for unlimited.
    /// </summary>
    public long MaxCacheSizeBytes { get; set; } = 10L * 1024 * 1024 * 1024;

    /// <summary>
    /// Gets or sets the eviction policy when cache is full.
    /// Default is LRU (Least Recently Used).
    /// </summary>
    public CacheEvictionPolicy EvictionPolicy { get; set; } = CacheEvictionPolicy.LeastRecentlyUsed;

    /// <summary>
    /// Gets or sets whether to verify cache integrity on load using checksums.
    /// Default is true. Disable for faster loads when data corruption is not a concern.
    /// </summary>
    public bool VerifyIntegrity { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to automatically invalidate the cache when source data changes.
    /// Default is true. When enabled, the cache stores a hash of the source configuration
    /// and invalidates if it changes.
    /// </summary>
    public bool AutoInvalidateOnSourceChange { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum age of cache entries before they are considered stale.
    /// Default is null (no time-based expiration).
    /// </summary>
    public TimeSpan? MaxAge { get; set; }

    /// <summary>
    /// Gets or sets whether to compress cached data. Reduces disk usage but increases CPU time.
    /// Default is false.
    /// </summary>
    public bool CompressData { get; set; }
}

/// <summary>
/// Policy for evicting cache entries when the cache is full.
/// </summary>
public enum CacheEvictionPolicy
{
    /// <summary>Remove the least recently used (accessed) entries first.</summary>
    LeastRecentlyUsed = 0,

    /// <summary>Remove the oldest entries first (by creation time).</summary>
    OldestFirst = 1,

    /// <summary>Remove the largest entries first.</summary>
    LargestFirst = 2
}
