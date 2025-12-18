namespace AiDotNet.Data.Structures;

/// <summary>
/// Distance functions for computing similarity between feature vectors.
/// </summary>
/// <remarks>
/// <para>
/// Distance functions are used in prototype-based meta-learning algorithms
/// to compute similarity between query examples and class prototypes.
/// Different distance functions are suitable for different types of data
/// and feature representations.
/// </para>
/// </remarks>
public enum DistanceFunction
{
    /// <summary>
    /// Euclidean distance (L2 norm).
    /// </summary>
    Euclidean,

    /// <summary>
    /// Manhattan distance (L1 norm).
    /// </summary>
    Manhattan,

    /// <summary>
    /// Cosine distance (1 - cosine similarity).
    /// </summary>
    Cosine,

    /// <summary>
    /// Mahalanobis distance.
    /// </summary>
    Mahalanobis
}