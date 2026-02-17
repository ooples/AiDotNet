using System.Collections.Generic;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings;

/// <summary>
/// Contains evaluation metrics for a link prediction model.
/// </summary>
/// <remarks>
/// <para>
/// Standard knowledge graph link prediction metrics in the filtered setting:
/// existing true triples are removed from the ranking before computing metrics.
/// </para>
/// <para><b>For Beginners:</b> These metrics measure how well the model predicts missing facts:
/// - MRR (Mean Reciprocal Rank): Average of 1/rank for each test triple. Higher = better. Perfect = 1.0.
/// - Hits@K: Fraction of test triples ranked in the top K. Higher = better. Perfect = 1.0.
/// - MeanRank: Average rank of correct answers. Lower = better. Perfect = 1.0.
/// </para>
/// </remarks>
public class LinkPredictionEvaluation
{
    /// <summary>
    /// Mean Reciprocal Rank: average of 1/rank for correct entities. Range: (0, 1]. Higher is better.
    /// </summary>
    public double MeanReciprocalRank { get; set; }

    /// <summary>
    /// Hits@K metrics: fraction of test triples where the correct entity is in the top K predictions.
    /// Keys are K values (e.g., 1, 3, 10).
    /// </summary>
    public Dictionary<int, double> HitsAtK { get; set; } = [];

    /// <summary>
    /// Mean rank of the correct entity across all test triples. Lower is better.
    /// </summary>
    public double MeanRank { get; set; }

    /// <summary>
    /// Number of test triples evaluated.
    /// </summary>
    public int TestTripleCount { get; set; }
}
