namespace AiDotNet.JitCompiler;

using System.Globalization;

/// <summary>
/// Statistics about the compilation cache.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Information about cached compiled graphs.
///
/// Tells you:
/// - How many graphs are cached
/// - Approximate memory usage
/// </para>
/// </remarks>
public class CacheStats
{
    /// <summary>
    /// Gets or sets the number of cached compiled graphs.
    /// </summary>
    public int CachedGraphCount { get; set; }

    /// <summary>
    /// Gets or sets the estimated memory used by cached graphs.
    /// </summary>
    public long EstimatedMemoryBytes { get; set; }

    /// <summary>
    /// Gets a string representation of the cache statistics.
    /// </summary>
    public override string ToString()
    {
        var estimatedKb = (EstimatedMemoryBytes / 1024.0).ToString("F2", CultureInfo.InvariantCulture);

        return $"Cache Stats:\n" +
               $"  Cached graphs: {CachedGraphCount}\n" +
               $"  Estimated memory: {estimatedKb} KB";
    }
}
