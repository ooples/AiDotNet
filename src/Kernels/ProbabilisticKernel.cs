namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Probabilistic kernel for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Probabilistic kernel combines aspects of both the cosine similarity and the Gaussian kernel,
/// making it useful for capturing both directional and magnitude-based relationships between data points.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The Probabilistic kernel is special because it considers both:
/// 1. The angle between data points (like cosine similarity)
/// 2. The difference in their magnitudes (like Gaussian kernel)
/// </para>
/// <para>
/// Think of the Probabilistic kernel as a "smart similarity detector" that can tell if two data points
/// are pointing in similar directions AND have similar sizes. This is particularly useful when both
/// the direction and magnitude of your data are important.
/// </para>
/// <para>
/// The formula for the Probabilistic kernel is:
/// k(x, y) = (x·y / v(||x||²·||y||²)) · exp(-||x||² - ||y||²)²/(2s²))
/// where:
/// - x and y are the two data points being compared
/// - x·y is the dot product (a measure of how aligned the vectors are)
/// - ||x|| and ||y|| are the magnitudes (lengths) of the vectors
/// - s (sigma) is a parameter that controls sensitivity to magnitude differences
/// </para>
/// <para>
/// Common uses include:
/// - Text classification where both word presence and frequency matter
/// - Image recognition where both pattern and intensity are important
/// - Any application where both direction and magnitude of data provide useful information
/// </para>
/// </remarks>
public class ProbabilisticKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The sigma parameter that controls sensitivity to differences in vector magnitudes.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Sigma (s) determines how much the kernel cares about differences in the "size" of your data points.
    /// 
    /// Think of it like this:
    /// - Small sigma values (e.g., 0.1): Very sensitive to differences in magnitude
    /// - Large sigma values (e.g., 10.0): Less sensitive to differences in magnitude
    /// 
    /// The default value is 1.0, which provides a balanced sensitivity for many applications.
    /// If your data points vary greatly in magnitude and you want the kernel to be less affected by this,
    /// you might want to use a larger sigma value.
    /// </remarks>
    private readonly T _sigma;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is a helper object that allows the kernel to perform mathematical
    /// operations regardless of what numeric type (like double, float, decimal) you're using.
    /// You don't need to interact with this directly.
    /// </remarks>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the Probabilistic kernel with an optional sigma parameter.
    /// </summary>
    /// <param name="sigma">The parameter that controls sensitivity to differences in vector magnitudes. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Probabilistic kernel for use. You can optionally
    /// provide a value for the sigma parameter:
    /// </para>
    /// <para>
    /// Sigma controls how much the kernel cares about differences in the "size" of your data points:
    /// - Smaller values make the kernel more sensitive to magnitude differences
    /// - Larger values make the kernel less sensitive to magnitude differences
    /// - The default value is 1.0, which works well for many applications
    /// </para>
    /// <para>
    /// When might you want to change this parameter?
    /// - Use a smaller sigma when the magnitude differences in your data are meaningful and should strongly affect similarity
    /// - Use a larger sigma when you want the kernel to focus more on directional similarity and be less affected by magnitude
    /// </para>
    /// <para>
    /// The best value often depends on your specific dataset and problem, so you might
    /// need to experiment with different values to find what works best.
    /// </para>
    /// </remarks>
    public ProbabilisticKernel(T? sigma = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _sigma = sigma ?? _numOps.One;
    }

    /// <summary>
    /// Calculates the Probabilistic kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Probabilistic kernel formula.
    /// </para>
    /// <para>
    /// The calculation works in two main parts:
    /// 1. Cosine similarity part: Measures if the vectors point in similar directions
    /// 2. Gaussian part: Measures if the vectors have similar magnitudes (sizes)
    /// </para>
    /// <para>
    /// The result is a similarity measure where:
    /// - Values closer to 1 mean the points are very similar (both in direction and magnitude)
    /// - Values closer to 0 mean the points are very different
    /// - Negative values are possible when vectors point in opposite directions
    /// </para>
    /// <para>
    /// This kernel is particularly useful when both the direction and the magnitude of your data
    /// contain important information. For example, in text analysis, it can capture both which
    /// words appear (direction) and how frequently they appear (magnitude).
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T dotProduct = x1.DotProduct(x2);
        T x1Norm = x1.DotProduct(x1);
        T x2Norm = x2.DotProduct(x2);
        T exponent = _numOps.Divide(_numOps.Negate(_numOps.Square(_numOps.Subtract(x1Norm, x2Norm))),
                                    _numOps.Multiply(_numOps.FromDouble(2), _numOps.Square(_sigma)));

        return _numOps.Multiply(_numOps.Exp(exponent),
                                _numOps.Divide(dotProduct, _numOps.Sqrt(_numOps.Multiply(x1Norm, x2Norm))));
    }
}
