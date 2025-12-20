namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Rational Quadratic kernel for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Rational Quadratic kernel is a similarity measure that behaves like a mixture of
/// Gaussian kernels with different length scales. It's particularly useful for data with
/// patterns occurring at multiple scales.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The Rational Quadratic kernel is like a "smooth similarity detector" that works well for data
/// with both small and large-scale patterns.
/// </para>
/// <para>
/// Think of it like this: If you're looking at a landscape, some features are small (like rocks)
/// and some are large (like mountains). The Rational Quadratic kernel can detect similarities
/// at both these scales simultaneously, unlike some other kernels that might focus on just one scale.
/// </para>
/// <para>
/// The formula for the Rational Quadratic kernel is:
/// k(x, y) = 1 - ||x - y||²/(||x - y||² + c)
/// where:
/// - x and y are the two data points being compared
/// - ||x - y||² is the squared Euclidean distance between them
/// - c is a parameter that controls the kernel's behavior
/// </para>
/// <para>
/// Common uses include:
/// - Time series analysis where patterns occur at different time scales
/// - Image processing where features exist at different resolutions
/// - Any data where both local and global patterns are important
/// </para>
/// </remarks>
public class RationalQuadraticKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The scale parameter that controls the behavior of the kernel.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The parameter c controls how quickly the similarity decreases as points get farther apart.
    /// 
    /// Think of it like this:
    /// - Small c values (e.g., 0.1): Similarity drops quickly with distance (focuses on local patterns)
    /// - Large c values (e.g., 10.0): Similarity drops slowly with distance (considers broader patterns)
    /// 
    /// The default value is 1.0, which provides a balanced sensitivity for many applications.
    /// </remarks>
    private readonly T _c;

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
    /// Initializes a new instance of the Rational Quadratic kernel with an optional scale parameter.
    /// </summary>
    /// <param name="c">The scale parameter that controls how quickly similarity decreases with distance. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Rational Quadratic kernel for use. You can optionally
    /// provide a value for the c parameter:
    /// </para>
    /// <para>
    /// The c parameter controls how the kernel responds to distance between points:
    /// - Smaller values make the kernel more sensitive to small distances (local patterns)
    /// - Larger values make the kernel more sensitive to large distances (global patterns)
    /// - The default value is 1.0, which works well for many applications
    /// </para>
    /// <para>
    /// When might you want to change this parameter?
    /// - Use a smaller c when you want to focus on very local, fine-grained patterns in your data
    /// - Use a larger c when you want to capture broader, more global patterns
    /// </para>
    /// <para>
    /// The best value often depends on your specific dataset and problem, so you might
    /// need to experiment with different values to find what works best.
    /// </para>
    /// </remarks>
    public RationalQuadraticKernel(T? c = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _c = c ?? _numOps.FromDouble(1.0);
    }

    /// <summary>
    /// Calculates the Rational Quadratic kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Rational Quadratic kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. Finding the difference between the two vectors (x1 - x2)
    /// 2. Calculating the squared distance between them
    /// 3. Applying the formula: 1 - distance²/(distance² + c)
    /// </para>
    /// <para>
    /// The result is a similarity measure where:
    /// - Values closer to 1 mean the points are very similar
    /// - Values closer to 0 mean the points are very different
    /// - The value is always between 0 and 1
    /// </para>
    /// <para>
    /// What makes this kernel special is that it can detect similarities at multiple scales simultaneously.
    /// This means it works well for complex data where patterns exist at different levels of detail.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var diff = x1.Subtract(x2);
        var squaredDistance = diff.DotProduct(diff);

        return _numOps.Subtract(_numOps.One, _numOps.Divide(squaredDistance, _numOps.Add(squaredDistance, _c)));
    }
}
