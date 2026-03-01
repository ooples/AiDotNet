namespace AiDotNet.Data.Loaders;

/// <summary>
/// Configuration options for caching data loader.
/// </summary>
public sealed class CachingDataLoaderOptions
{
    /// <summary>Maximum number of batches to cache in memory. Default is 100.</summary>
    public int MaxCacheSize { get; set; } = 100;
    /// <summary>Whether to cache on disk as well (two-level cache). Default is false.</summary>
    public bool EnableDiskCache { get; set; } = false;
    /// <summary>
    /// Directory for disk cache files.
    /// Default is "{TEMP}/aidotnet/dataloader_cache". Relative paths are resolved against the current directory.
    /// </summary>
    public string DiskCacheDirectory { get; set; } = Path.Combine(Path.GetTempPath(), "aidotnet", "dataloader_cache");
    /// <summary>Cache eviction policy. Default is LRU.</summary>
    public MemoryCacheEvictionPolicy EvictionPolicy { get; set; } = MemoryCacheEvictionPolicy.LRU;

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (MaxCacheSize <= 0) throw new ArgumentOutOfRangeException(nameof(MaxCacheSize), "MaxCacheSize must be positive.");
        if (!Enum.IsDefined(typeof(MemoryCacheEvictionPolicy), EvictionPolicy))
            throw new ArgumentOutOfRangeException(nameof(EvictionPolicy), "EvictionPolicy must be a valid enum value.");
        if (EnableDiskCache && string.IsNullOrWhiteSpace(DiskCacheDirectory))
            throw new ArgumentException("DiskCacheDirectory must not be empty when disk cache is enabled.", nameof(DiskCacheDirectory));
    }
}

/// <summary>
/// Policy for evicting items from the in-memory cache when it's full.
/// </summary>
public enum MemoryCacheEvictionPolicy
{
    /// <summary>Least Recently Used: evict the item accessed longest ago.</summary>
    LRU,
    /// <summary>First In First Out: evict the oldest cached item.</summary>
    FIFO,
    /// <summary>Least Frequently Used: evict the item accessed fewest times.</summary>
    LFU
}
