namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Specifies the distance metric for UMAP.
/// </summary>
public enum UMAPMetric
{
    /// <summary>
    /// Euclidean (L2) distance.
    /// </summary>
    Euclidean,

    /// <summary>
    /// Manhattan (L1) distance.
    /// </summary>
    Manhattan,

    /// <summary>
    /// Cosine distance (1 - cosine similarity).
    /// </summary>
    Cosine,

    /// <summary>
    /// Correlation distance (1 - Pearson correlation).
    /// </summary>
    Correlation
}
