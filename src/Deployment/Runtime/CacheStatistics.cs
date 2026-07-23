namespace AiDotNet.Deployment.Runtime;

/// <summary>
/// Statistics for the model cache.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> CacheStatistics provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class CacheStatistics
{
    public int TotalEntries { get; set; }
    public long TotalAccessCount { get; set; }
    public double AverageAccessCount { get; set; }
    public TimeSpan OldestEntryAge { get; set; }
    public TimeSpan NewestEntryAge { get; set; }
}
