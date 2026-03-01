namespace AiDotNet.Data.Loaders;

/// <summary>
/// Configuration options for prefetch-enabled data loading.
/// </summary>
public sealed class PrefetchDataLoaderOptions
{
    /// <summary>Number of batches to prefetch ahead. Default is 2.</summary>
    public int PrefetchCount { get; set; } = 2;
    /// <summary>Whether to use a background thread for prefetching. Default is true.</summary>
    public bool UseBackgroundThread { get; set; } = true;
    /// <summary>Timeout in milliseconds for waiting on a prefetched batch. Default is 30000 (30s).</summary>
    public int TimeoutMs { get; set; } = 30000;

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (PrefetchCount <= 0) throw new ArgumentOutOfRangeException(nameof(PrefetchCount), "PrefetchCount must be positive.");
        if (TimeoutMs <= 0) throw new ArgumentOutOfRangeException(nameof(TimeoutMs), "TimeoutMs must be positive.");
    }
}
