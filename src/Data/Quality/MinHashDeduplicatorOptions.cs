namespace AiDotNet.Data.Quality;

/// <summary>
/// Configuration options for MinHash-based near-duplicate detection.
/// </summary>
public sealed class MinHashDeduplicatorOptions
{
    /// <summary>Number of hash functions for the MinHash signature. Default is 128.</summary>
    public int NumHashFunctions { get; set; } = 128;
    /// <summary>Jaccard similarity threshold for duplicate detection. Default is 0.8.</summary>
    public double SimilarityThreshold { get; set; } = 0.8;
    /// <summary>Number of bands for LSH banding. Default is 16.</summary>
    public int NumBands { get; set; } = 16;
    /// <summary>N-gram size for shingling. Default is 5.</summary>
    public int ShingleSize { get; set; } = 5;
    /// <summary>Random seed for reproducibility. When null, uses non-deterministic seed.</summary>
    public int? Seed { get; set; }
}
