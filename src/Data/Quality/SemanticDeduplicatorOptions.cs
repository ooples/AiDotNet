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
}
