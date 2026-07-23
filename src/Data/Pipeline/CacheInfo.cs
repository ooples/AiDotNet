namespace AiDotNet.Data.Pipeline;

/// <summary>
/// Information about the current state of a pipeline cache.
/// </summary>
public class CacheInfo
{
    /// <summary>Whether the cache is valid and usable.</summary>
    public bool IsValid { get; set; }

    /// <summary>Number of cached entries (tensors).</summary>
    public int EntryCount { get; set; }

    /// <summary>Total cache size in bytes.</summary>
    public long TotalSizeBytes { get; set; }

    /// <summary>Path to the cache directory.</summary>
    public string CacheDirectory { get; set; } = string.Empty;

    /// <summary>Human-readable cache size.</summary>
    public string FormattedSize
    {
        get
        {
            if (TotalSizeBytes < 1024) return $"{TotalSizeBytes} B";
            if (TotalSizeBytes < 1024 * 1024) return $"{TotalSizeBytes / 1024.0:F1} KB";
            if (TotalSizeBytes < 1024 * 1024 * 1024) return $"{TotalSizeBytes / (1024.0 * 1024):F1} MB";
            return $"{TotalSizeBytes / (1024.0 * 1024 * 1024):F2} GB";
        }
    }
}
