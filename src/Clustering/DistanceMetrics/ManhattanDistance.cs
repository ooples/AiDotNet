namespace AiDotNet.Clustering.DistanceMetrics;

/// <summary>
/// Computes Manhattan (L1) distance between vectors.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Manhattan distance, also called taxicab or city-block distance, is the sum of
/// absolute differences along each dimension. It's named after the grid-like
/// street layout of Manhattan.
/// </para>
/// <para>
/// Formula: d(a, b) = sum(|a[i] - b[i]|)
/// </para>
/// <para><b>For Beginners:</b> Manhattan distance is like walking along city blocks.
/// You can only move horizontally or vertically, not diagonally.
///
/// Example: The distance between points (0, 0) and (3, 4) is 7
/// (|3-0| + |4-0| = 3 + 4 = 7)
///
/// Good for:
/// - High-dimensional data where Euclidean distance loses meaning
/// - When features have different scales
/// - Sparse data
/// </para>
/// </remarks>
public class ManhattanDistance<T> : DistanceMetricBase<T>
{
    /// <inheritdoc />
    public override string Name => "Manhattan";

    /// <inheritdoc />
    public override T Compute(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException(
                $"Vectors must have the same length. Got {a.Length} and {b.Length}.");
        }

        T sum = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            T diff = NumOps.Subtract(a[i], b[i]);
            sum = NumOps.Add(sum, Abs(diff));
        }

        return sum;
    }
}
