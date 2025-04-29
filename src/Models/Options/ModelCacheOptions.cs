namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the model cache.
/// </summary>
public class ModelCacheOptions
{
    /// <summary>
    /// Gets or sets the default expiration time for cached items.
    /// </summary>
    public TimeSpan DefaultExpiration { get; set; } = TimeSpan.FromMinutes(30);

    /// <summary>
    /// Gets or sets the interval at which the cache is automatically cleaned up.
    /// </summary>
    public TimeSpan CleanupInterval { get; set; } = TimeSpan.FromMinutes(5);

    /// <summary>
    /// Gets or sets the maximum number of items before a cleanup is triggered.
    /// </summary>
    public int CleanupThreshold { get; set; } = 100;

    /// <summary>
    /// Gets or sets whether to enable periodic background cleanup.
    /// </summary>
    public bool EnableBackgroundCleanup { get; set; } = true;
}