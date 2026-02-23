namespace AiDotNet.Kernels;

/// <summary>
/// Cosine Similarity Kernel that measures angular distance between vectors.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Cosine Kernel computes the cosine of the angle between two vectors.
/// It measures similarity based on direction rather than magnitude.
///
/// In mathematical terms:
/// k(x, x') = (x · x') / (||x|| × ||x'||)
///
/// Where:
/// - x · x' is the dot product
/// - ||x|| is the Euclidean norm (length) of x
///
/// Properties:
/// - Returns values in [-1, 1]
/// - k(x, x') = 1 when vectors point in the same direction
/// - k(x, x') = 0 when vectors are orthogonal
/// - k(x, x') = -1 when vectors point in opposite directions
///
/// This kernel is particularly useful for:
/// - Text classification (TF-IDF vectors)
/// - Document similarity
/// - Any application where magnitude doesn't matter, only direction
/// </para>
/// </remarks>
public class CosineKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Optional output scale factor.
    /// </summary>
    private readonly double _outputScale;

    /// <summary>
    /// Initializes a new Cosine Kernel.
    /// </summary>
    /// <param name="outputScale">Optional scaling factor for the kernel output. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a cosine similarity kernel.
    /// The outputScale parameter can be used to scale the kernel values,
    /// which affects the variance of the GP.
    /// </para>
    /// </remarks>
    public CosineKernel(double outputScale = 1.0)
    {
        if (outputScale <= 0)
            throw new ArgumentException("Output scale must be positive.", nameof(outputScale));

        _outputScale = outputScale;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the output scale.
    /// </summary>
    public double OutputScale => _outputScale;

    /// <summary>
    /// Calculates the cosine similarity between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The cosine similarity scaled by outputScale.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Computes how similar the directions of two vectors are.
    /// Zero-length vectors return 0 similarity (undefined case handled gracefully).
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        if (x1.Length != x2.Length)
            throw new ArgumentException("Vectors must have the same length.");

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

        double cosineSim = dotProduct / (norm1 * norm2);
        return _numOps.FromDouble(_outputScale * cosineSim);
    }
}
