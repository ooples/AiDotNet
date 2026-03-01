namespace AiDotNet.Data.Quality;

/// <summary>
/// Detects semantic duplicates using embedding cosine similarity.
/// </summary>
/// <remarks>
/// <para>
/// Semantic deduplication finds documents with the same meaning even if worded differently.
/// Requires pre-computed embeddings (e.g., from a sentence transformer).
/// More expensive than MinHash but catches paraphrased duplicates.
/// </para>
/// </remarks>
public class SemanticDeduplicator
{
    private readonly SemanticDeduplicatorOptions _options;

    public SemanticDeduplicator(SemanticDeduplicatorOptions? options = null)
    {
        _options = options ?? new SemanticDeduplicatorOptions();
    }

    /// <summary>
    /// Computes cosine similarity between two embedding vectors.
    /// </summary>
    public double CosineSimilarity(double[] a, double[] b)
    {
        double dotProduct = 0, normA = 0, normB = 0;
        int len = Math.Min(a.Length, b.Length);
        for (int i = 0; i < len; i++)
        {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        double denom = Math.Sqrt(normA) * Math.Sqrt(normB);
        return denom > 0 ? dotProduct / denom : 0;
    }

    /// <summary>
    /// Finds duplicate indices from pre-computed embeddings.
    /// </summary>
    /// <param name="embeddings">Array of embedding vectors, one per document.</param>
    /// <returns>Set of indices that are duplicates (should be removed).</returns>
    public HashSet<int> FindDuplicates(double[][] embeddings)
    {
        if (embeddings.Length == 0)
            return new HashSet<int>();

        // Validate all embeddings have the same dimension
        int dim = embeddings[0].Length;
        for (int i = 1; i < embeddings.Length; i++)
        {
            if (embeddings[i].Length != dim)
                throw new ArgumentException(
                    $"Embedding at index {i} has dimension {embeddings[i].Length}, expected {dim}.",
                    nameof(embeddings));
        }

        var duplicates = new HashSet<int>();

        for (int i = 0; i < embeddings.Length; i++)
        {
            if (duplicates.Contains(i)) continue;
            for (int j = i + 1; j < embeddings.Length; j++)
            {
                if (duplicates.Contains(j)) continue;
                double sim = CosineSimilarity(embeddings[i], embeddings[j]);
                if (sim >= _options.SimilarityThreshold)
                {
                    duplicates.Add(j);
                }
            }
        }

        return duplicates;
    }
}
