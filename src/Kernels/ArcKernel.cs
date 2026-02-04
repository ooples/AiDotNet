namespace AiDotNet.Kernels;

/// <summary>
/// Arc (Angular) Kernel based on the angle between vectors.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Arc Kernel (also called Angular Kernel) is based on the
/// angle θ between two vectors rather than their Euclidean distance.
///
/// In mathematical terms:
/// k(x, x') = 1 - (2/π) × arccos(cosine_similarity(x, x'))
///          = 1 - (2/π) × θ
///
/// Where θ is the angle between x and x' in radians.
///
/// Properties:
/// - Returns values in [0, 1]
/// - k(x, x) = 1 (same vector)
/// - k(x, x') = 0 when vectors are opposite (θ = π)
/// - k(x, x') = 0.5 when vectors are orthogonal (θ = π/2)
///
/// Unlike the Cosine Kernel, the Arc Kernel is positive semi-definite,
/// making it suitable for use as a GP kernel.
///
/// Applications:
/// - Directional data (wind direction, compass headings)
/// - Spherical data (locations on Earth, molecular orientations)
/// - Any data where only direction matters
/// </para>
/// </remarks>
public class ArcKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Output scale factor.
    /// </summary>
    private readonly double _outputScale;

    /// <summary>
    /// Order of the arc kernel.
    /// </summary>
    private readonly int _order;

    /// <summary>
    /// Initializes a new Arc Kernel.
    /// </summary>
    /// <param name="order">Order of the kernel (0, 1, or 2). Default is 1.</param>
    /// <param name="outputScale">Output scale factor. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The order parameter controls the kernel's behavior:
    /// - Order 0: Step function kernel (like infinite-width NN with step activation)
    /// - Order 1: Linear angle kernel (like infinite-width NN with ReLU)
    /// - Order 2: Quadratic angle kernel (like infinite-width NN with ReQU)
    ///
    /// Order 1 is the most commonly used.
    /// </para>
    /// </remarks>
    public ArcKernel(int order = 1, double outputScale = 1.0)
    {
        if (order < 0 || order > 2)
            throw new ArgumentException("Order must be 0, 1, or 2.", nameof(order));
        if (outputScale <= 0)
            throw new ArgumentException("Output scale must be positive.", nameof(outputScale));

        _order = order;
        _outputScale = outputScale;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the kernel order.
    /// </summary>
    public int Order => _order;

    /// <summary>
    /// Gets the output scale.
    /// </summary>
    public double OutputScale => _outputScale;

    /// <summary>
    /// Calculates the arc kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The arc kernel value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Computes the angular similarity between vectors.
    /// The kernel value depends on the angle θ between x1 and x2.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        if (x1.Length != x2.Length)
            throw new ArgumentException("Vectors must have the same length.");

        // Compute norms and dot product
        double dotProduct = 0;
        double norm1Sq = 0;
        double norm2Sq = 0;

        for (int i = 0; i < x1.Length; i++)
        {
            double v1 = _numOps.ToDouble(x1[i]);
            double v2 = _numOps.ToDouble(x2[i]);

            dotProduct += v1 * v2;
            norm1Sq += v1 * v1;
            norm2Sq += v2 * v2;
        }

        double norm1 = Math.Sqrt(norm1Sq);
        double norm2 = Math.Sqrt(norm2Sq);

        if (norm1 < 1e-10 || norm2 < 1e-10)
        {
            return _numOps.FromDouble(0);
        }

        // Cosine similarity (clamp to [-1, 1] for numerical stability)
        double cosSim = Math.Max(-1, Math.Min(1, dotProduct / (norm1 * norm2)));

        // Compute angle
        double theta = Math.Acos(cosSim);

        // Compute kernel based on order
        double result = _order switch
        {
            0 => ComputeOrder0(theta, norm1, norm2),
            1 => ComputeOrder1(theta, norm1, norm2, cosSim),
            2 => ComputeOrder2(theta, norm1, norm2, cosSim),
            _ => ComputeOrder1(theta, norm1, norm2, cosSim)
        };

        return _numOps.FromDouble(_outputScale * result);
    }

    /// <summary>
    /// Order 0: J_0(θ) = 1 - θ/π
    /// </summary>
    private static double ComputeOrder0(double theta, double norm1, double norm2)
    {
        return 1.0 - theta / Math.PI;
    }

    /// <summary>
    /// Order 1: J_1(θ) = (1/π) × (sin(θ) + (π - θ) × cos(θ))
    /// </summary>
    private static double ComputeOrder1(double theta, double norm1, double norm2, double cosSim)
    {
        double sinTheta = Math.Sqrt(1 - cosSim * cosSim);
        return (1.0 / Math.PI) * (sinTheta + (Math.PI - theta) * cosSim) * norm1 * norm2;
    }

    /// <summary>
    /// Order 2: J_2(θ) = (1/π) × (3 × sin(θ) × cos(θ) + (π - θ) × (1 + 2cos²(θ)))
    /// </summary>
    private static double ComputeOrder2(double theta, double norm1, double norm2, double cosSim)
    {
        double sinTheta = Math.Sqrt(1 - cosSim * cosSim);
        double result = (1.0 / Math.PI) * (3 * sinTheta * cosSim + (Math.PI - theta) * (1 + 2 * cosSim * cosSim));
        return result * norm1 * norm1 * norm2 * norm2;
    }
}
