namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Spherical kernel for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Spherical kernel is a type of compactly supported kernel, which means it becomes
/// exactly zero beyond a certain distance. This property makes it computationally efficient
/// for large datasets as it creates sparse matrices.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The Spherical kernel is special because it has a "limited range" - if two points are too far apart
/// (farther than the sigma parameter), the kernel says they have zero similarity. This is different from
/// many other kernels that might say points have a very small similarity even when they're very far apart.
/// </para>
/// <para>
/// Think of it like this: The Spherical kernel creates a bubble of influence around each data point.
/// Points inside this bubble are considered similar (with similarity decreasing as distance increases),
/// while points outside the bubble are considered completely dissimilar.
/// </para>
/// <para>
/// The formula for the Spherical kernel is:
/// k(x, y) = 1.5 * (1 - ||x - y||/s) if ||x - y|| = s
/// k(x, y) = 0 if ||x - y|| > s
/// where:
/// - x and y are the two data points being compared
/// - ||x - y|| is the Euclidean distance between them
/// - s (sigma) is the radius parameter that determines the kernel's range
/// </para>
/// <para>
/// Common uses include:
/// - Spatial data analysis
/// - Geostatistics
/// - Large datasets where computational efficiency is important
/// </para>
/// </remarks>
public class SphericalKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The radius parameter that determines the range of influence for the kernel.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The sigma parameter controls how far the "bubble of influence" extends around each data point:
    /// 
    /// Think of it like this:
    /// - Larger sigma values (e.g., 2.0): Create a larger bubble, allowing points that are farther apart to still have some similarity
    /// - Smaller sigma values (e.g., 0.5): Create a smaller bubble, only considering points that are very close to be similar
    /// 
    /// The default value is 1.0, which provides a balanced range for many applications.
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
    /// Initializes a new instance of the Spherical kernel with an optional radius parameter.
    /// </summary>
    /// <param name="sigma">The radius parameter that determines the range of influence for the kernel. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Spherical kernel for use. You can optionally
    /// provide a value for the sigma parameter:
    /// </para>
    /// <para>
    /// The sigma parameter controls the size of the "bubble of influence" around each data point:
    /// - Larger values create a bigger bubble, allowing more distant points to have similarity
    /// - Smaller values create a smaller bubble, restricting similarity to only very close points
    /// - The default value is 1.0, which works well for many applications
    /// </para>
    /// <para>
    /// When might you want to change this parameter?
    /// - Use a smaller sigma when you want to focus on very local, fine-grained patterns in your data
    /// - Use a larger sigma when you want to capture broader, more global patterns
    /// </para>
    /// <para>
    /// The best value often depends on your specific dataset and problem, so you might
    /// need to experiment with different values to find what works best.
    /// </para>
    /// </remarks>
    public SphericalKernel(T? sigma = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _sigma = sigma ?? _numOps.FromDouble(1.0);
    }

    /// <summary>
    /// Calculates the Spherical kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Spherical kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. Finding the difference between the two vectors (x1 - x2)
    /// 2. Calculating the Euclidean distance between them (the straight-line distance)
    /// 3. Normalizing this distance by dividing by sigma
    /// 4. If the normalized distance is greater than 1 (outside the bubble), returning 0
    /// 5. Otherwise, calculating 1.5 * (1 - normalized distance)
    /// </para>
    /// <para>
    /// The result is a similarity measure where:
    /// - Values closer to 1.5 mean the points are very similar (close together)
    /// - Values closer to 0 mean the points are less similar (farther apart)
    /// - Exactly 0 means the points are too far apart to be considered similar at all
    /// </para>
    /// <para>
    /// What makes this kernel special is its "compact support" property - it becomes exactly zero
    /// beyond a certain distance. This makes it computationally efficient for large datasets
    /// because it creates sparse matrices (matrices with lots of zeros).
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var diff = x1.Subtract(x2);
        var distance = _numOps.Sqrt(diff.DotProduct(diff));
        var normalizedDistance = _numOps.Divide(distance, _sigma);

        if (_numOps.GreaterThan(normalizedDistance, _numOps.One))
        {
            return _numOps.Zero;
        }

        var oneMinusR = _numOps.Subtract(_numOps.One, normalizedDistance);
        return _numOps.Multiply(_numOps.FromDouble(1.5), oneMinusR);
    }
}
