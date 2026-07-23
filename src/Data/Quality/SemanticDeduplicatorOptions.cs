namespace AiDotNet.Data.Quality;

/// <summary>
/// Configuration options for semantic-level deduplication using embeddings.
/// </summary>
public sealed class SemanticDeduplicatorOptions
{
    /// <summary>Cosine similarity threshold for duplicate detection. Default is 0.95.</summary>
    public double SimilarityThreshold { get; set; } = 0.95;
    /// <summary>Embedding dimension. Default is 768.</summary>
    public int EmbeddingDimension { get; set; } = 768;
    /// <summary>Batch size for processing. Default is 64.</summary>
    public int BatchSize { get; set; } = 64;

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (SimilarityThreshold < 0 || SimilarityThreshold > 1) throw new ArgumentOutOfRangeException(nameof(SimilarityThreshold), "SimilarityThreshold must be between 0 and 1.");
        if (EmbeddingDimension <= 0) throw new ArgumentOutOfRangeException(nameof(EmbeddingDimension), "EmbeddingDimension must be positive.");
        if (BatchSize <= 0) throw new ArgumentOutOfRangeException(nameof(BatchSize), "BatchSize must be positive.");
    }
}
