namespace AiDotNet.Clustering.DistanceMetrics;

/// <summary>
/// Computes Chebyshev (L∞) distance between vectors.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Chebyshev distance, also called maximum metric or L-infinity distance, is the
/// maximum absolute difference along any dimension. It's the limit of Minkowski
/// distance as p approaches infinity.
/// </para>
/// <para>
/// Formula: d(a, b) = max(|a[i] - b[i]|)
/// </para>
/// <para><b>For Beginners:</b> Chebyshev distance is the largest difference between
/// any pair of corresponding elements. It's like a "worst case" distance.
///
/// Example: The distance between points (0, 0) and (3, 4) is 4
/// (max(|3-0|, |4-0|) = max(3, 4) = 4)
///
/// Named after Pafnuty Chebyshev, it's useful when:
/// - You care about the maximum deviation in any single feature
/// - Movement is possible in all directions at once (like a king in chess)
/// - You need a robust metric that isn't sensitive to many small differences
/// </para>
/// </remarks>
public class ChebyshevDistance<T> : DistanceMetricBase<T>
{
    /// <inheritdoc />
    public override string Name => "Chebyshev";

    /// <inheritdoc />
    public override T Compute(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException(
                $"Vectors must have the same length. Got {a.Length} and {b.Length}.");
        }

        if (a.Length == 0)
        {
            return NumOps.Zero;
        }

        T maxDiff = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            T diff = NumOps.Subtract(a[i], b[i]);
            T absDiff = Abs(diff);
            maxDiff = Max(maxDiff, absDiff);
        }

        return maxDiff;
    }
}
