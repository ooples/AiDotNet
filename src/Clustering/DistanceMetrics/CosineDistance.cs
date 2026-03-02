using AiDotNet.Helpers;

namespace AiDotNet.Clustering.DistanceMetrics;

/// <summary>
/// Computes Cosine distance between vectors.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Cosine distance measures the angle between two vectors, ignoring their magnitudes.
/// It's derived from cosine similarity: distance = 1 - similarity.
/// </para>
/// <para>
/// Formula: d(a, b) = 1 - (a · b) / (||a|| × ||b||)
/// where a · b is the dot product and ||x|| is the L2 norm.
/// </para>
/// <para><b>For Beginners:</b> Cosine distance measures how different the "direction"
/// of two vectors is, ignoring how long they are.
///
/// Example: Vectors [1, 0] and [0, 1] are perpendicular (90°), so their cosine
/// similarity is 0 and distance is 1. Vectors [1, 2] and [2, 4] point in the
/// same direction, so their distance is 0.
///
/// Best for:
/// - Text documents (TF-IDF vectors)
/// - Any data where magnitude doesn't matter, only direction
/// - Sparse high-dimensional data
/// </para>
/// </remarks>
public class CosineDistance<T> : DistanceMetricBase<T>
{
    /// <inheritdoc />
    public override string Name => "Cosine";

    /// <inheritdoc />
    public override T Compute(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException(
                $"Vectors must have the same length. Got {a.Length} and {b.Length}.");
        }

        // Distance = 1 - similarity
        double similarity = VectorHelper.CosineSimilarity(a, b);
        return NumOps.FromDouble(1.0 - similarity);
    }

    /// <summary>
    /// Computes cosine similarity (1 - distance).
    /// </summary>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>The cosine similarity in range [-1, 1].</returns>
    public T ComputeSimilarity(Vector<T> a, Vector<T> b)
    {
        T distance = Compute(a, b);
        return NumOps.Subtract(NumOps.One, distance);
    }
}
