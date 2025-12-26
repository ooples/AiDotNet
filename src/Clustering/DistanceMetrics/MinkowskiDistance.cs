namespace AiDotNet.Clustering.DistanceMetrics;

/// <summary>
/// Computes Minkowski (Lp) distance between vectors.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Minkowski distance is a generalization of both Euclidean (p=2) and Manhattan (p=1)
/// distances. Different values of p produce different distance behaviors.
/// </para>
/// <para>
/// Formula: d(a, b) = (sum(|a[i] - b[i]|^p))^(1/p)
/// </para>
/// <para><b>For Beginners:</b> Minkowski distance is a "tunable" distance metric.
/// By changing the parameter p, you get different behaviors:
///
/// - p = 1: Manhattan distance (city block)
/// - p = 2: Euclidean distance (straight line)
/// - p → ∞: Chebyshev distance (maximum difference)
///
/// Higher p values emphasize larger differences more. Lower p values treat
/// all differences more equally.
/// </para>
/// </remarks>
public class MinkowskiDistance<T> : DistanceMetricBase<T>
{
    private readonly double _p;

    /// <summary>
    /// Initializes a new instance with the specified p value.
    /// </summary>
    /// <param name="p">The order of the Minkowski distance (must be >= 1).</param>
    /// <exception cref="ArgumentException">Thrown if p is less than 1.</exception>
    public MinkowskiDistance(double p = 2.0)
    {
        if (p < 1.0)
        {
            throw new ArgumentException("Minkowski p parameter must be >= 1.", nameof(p));
        }
        _p = p;
    }

    /// <summary>
    /// Gets the p parameter (order) of this Minkowski distance.
    /// </summary>
    public double P => _p;

    /// <inheritdoc />
    public override string Name => $"Minkowski (p={_p:F2})";

    /// <inheritdoc />
    public override T Compute(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException(
                $"Vectors must have the same length. Got {a.Length} and {b.Length}.");
        }

        // Special case: p = 1 is Manhattan
        if (Math.Abs(_p - 1.0) < 1e-10)
        {
            return ComputeManhattan(a, b);
        }

        // Special case: p = 2 is Euclidean
        if (Math.Abs(_p - 2.0) < 1e-10)
        {
            return ComputeEuclidean(a, b);
        }

        // General case
        T sum = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            T diff = NumOps.Subtract(a[i], b[i]);
            T absDiff = Abs(diff);
            sum = NumOps.Add(sum, Pow(absDiff, _p));
        }

        return Pow(sum, 1.0 / _p);
    }

    private T ComputeManhattan(Vector<T> a, Vector<T> b)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            T diff = NumOps.Subtract(a[i], b[i]);
            sum = NumOps.Add(sum, Abs(diff));
        }
        return sum;
    }

    private T ComputeEuclidean(Vector<T> a, Vector<T> b)
    {
        T sumSquared = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            T diff = NumOps.Subtract(a[i], b[i]);
            sumSquared = NumOps.Add(sumSquared, NumOps.Multiply(diff, diff));
        }
        return Sqrt(sumSquared);
    }
}
