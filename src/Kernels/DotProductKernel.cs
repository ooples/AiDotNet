namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Dot Product (Linear) kernel with optional inhomogeneity.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Dot Product kernel measures similarity by computing the inner product
/// (dot product) between two vectors, optionally with a constant offset.
///
/// In mathematical terms: k(x, x') = σ₀² + x · x'
///
/// Where:
/// - σ₀² is the variance of the constant offset (inhomogeneity parameter)
/// - x · x' is the dot product (sum of element-wise products)
/// </para>
/// <para>
/// Why use the Dot Product kernel?
///
/// 1. **Linear relationships**: When your data has a linear relationship, this kernel works well
///    - Equivalent to Bayesian linear regression when used in a GP
///
/// 2. **Interpretability**: Each feature's contribution is directly proportional to its value
///
/// 3. **Computational efficiency**: Very fast to compute, O(d) for d-dimensional vectors
///
/// 4. **Feature importance**: In a GP with this kernel, the predictive variance grows
///    indefinitely away from the origin, which can be useful for extrapolation
///
/// The inhomogeneity parameter σ₀² adds a constant "bias" to all similarities, which allows
/// the model to capture a non-zero mean function.
/// </para>
/// </remarks>
public class DotProductKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The variance of the constant offset (σ₀²).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This parameter adds a constant baseline to all similarities.
    ///
    /// - If σ₀² = 0: Pure dot product, similarity can be negative
    /// - If σ₀² > 0: All similarities are shifted up by this amount
    ///
    /// A non-zero σ₀² allows the GP to model functions with non-zero mean at the origin.
    /// </para>
    /// </remarks>
    private readonly double _sigma0Squared;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the Dot Product kernel.
    /// </summary>
    /// <param name="sigma0Squared">The variance of the constant offset. Default is 0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a Dot Product kernel with optional inhomogeneity.
    ///
    /// - sigma0Squared = 0: Standard linear kernel
    /// - sigma0Squared > 0: Linear kernel with bias (polynomial degree 1 with bias)
    ///
    /// For most linear regression tasks, the default (0) works well.
    /// Add inhomogeneity if you want the model to have a non-zero intercept even
    /// when all features are zero.
    /// </para>
    /// </remarks>
    public DotProductKernel(double sigma0Squared = 0.0)
    {
        if (sigma0Squared < 0)
            throw new ArgumentException("Sigma0 squared must be non-negative.", nameof(sigma0Squared));

        _sigma0Squared = sigma0Squared;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Calculates the Dot Product kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value: σ₀² + x1 · x2.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This computes the similarity between two vectors using their dot product.
    ///
    /// The dot product is: x1[0]*x2[0] + x1[1]*x2[1] + ... + x1[n]*x2[n]
    ///
    /// Interpretation:
    /// - Large positive value: Vectors point in similar directions
    /// - Near zero: Vectors are orthogonal (perpendicular)
    /// - Large negative value: Vectors point in opposite directions
    ///
    /// Unlike RBF kernel, this kernel can return negative values (unless σ₀² is large enough).
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        if (x1.Length != x2.Length)
            throw new ArgumentException("Vectors must have the same length.");

        T dotProduct = _numOps.Zero;
        for (int i = 0; i < x1.Length; i++)
        {
            dotProduct = _numOps.Add(dotProduct, _numOps.Multiply(x1[i], x2[i]));
        }

        return _numOps.Add(_numOps.FromDouble(_sigma0Squared), dotProduct);
    }
}
