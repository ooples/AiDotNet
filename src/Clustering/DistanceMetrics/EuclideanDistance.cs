namespace AiDotNet.Clustering.DistanceMetrics;

/// <summary>
/// Computes Euclidean (L2) distance between vectors.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Euclidean distance is the straight-line distance between two points in Euclidean space.
/// It is the most commonly used distance metric for clustering.
/// </para>
/// <para>
/// Formula: d(a, b) = sqrt(sum((a[i] - b[i])^2))
/// </para>
/// <para><b>For Beginners:</b> Euclidean distance is what you'd measure with a ruler
/// in a straight line between two points. It's the "as the crow flies" distance.
///
/// Example: The distance between points (0, 0) and (3, 4) is 5
/// (using the Pythagorean theorem: sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5)
/// </para>
/// </remarks>
public class EuclideanDistance<T> : DistanceMetricBase<T>
{
    /// <inheritdoc />
    public override string Name => "Euclidean";

    /// <inheritdoc />
    public override T Compute(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException(
                $"Vectors must have the same length. Got {a.Length} and {b.Length}.");
        }

        T sumSquared = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            T diff = NumOps.Subtract(a[i], b[i]);
            sumSquared = NumOps.Add(sumSquared, NumOps.Multiply(diff, diff));
        }

        return Sqrt(sumSquared);
    }

    /// <summary>
    /// Computes the squared Euclidean distance (avoids square root for efficiency).
    /// </summary>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>The squared Euclidean distance.</returns>
    /// <remarks>
    /// <para>
    /// Use this when you only need to compare distances, as it avoids the expensive
    /// square root operation. The relative ordering is preserved.
    /// </para>
    /// </remarks>
    public T ComputeSquared(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException(
                $"Vectors must have the same length. Got {a.Length} and {b.Length}.");
        }

        T sumSquared = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            T diff = NumOps.Subtract(a[i], b[i]);
            sumSquared = NumOps.Add(sumSquared, NumOps.Multiply(diff, diff));
        }

        return sumSquared;
    }
}
