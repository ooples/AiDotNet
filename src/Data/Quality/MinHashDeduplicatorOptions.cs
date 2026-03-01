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

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (NumHashFunctions <= 0) throw new ArgumentOutOfRangeException(nameof(NumHashFunctions), "NumHashFunctions must be positive.");
        if (SimilarityThreshold < 0 || SimilarityThreshold > 1) throw new ArgumentOutOfRangeException(nameof(SimilarityThreshold), "SimilarityThreshold must be between 0 and 1.");
        if (NumBands <= 0) throw new ArgumentOutOfRangeException(nameof(NumBands), "NumBands must be positive.");
        if (NumBands > NumHashFunctions) throw new ArgumentOutOfRangeException(nameof(NumBands), "NumBands must not exceed NumHashFunctions.");
        if (ShingleSize <= 0) throw new ArgumentOutOfRangeException(nameof(ShingleSize), "ShingleSize must be positive.");
    }
}
