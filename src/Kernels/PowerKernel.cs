namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Power kernel for measuring dissimilarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Power kernel is a negative distance-based kernel that measures how different two data points are,
/// rather than how similar they are. It's sometimes called the "negative distance kernel."
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures relationships between data points.
/// While most kernels measure similarity (higher values mean more similar points), the Power kernel is special
/// because it measures dissimilarity - higher values (less negative) mean the points are more different from each other.
/// </para>
/// <para>
/// Think of the Power kernel as a "difference detector" that emphasizes how far apart points are.
/// This can be useful when you want your algorithm to focus on the differences between data points
/// rather than their similarities.
/// </para>
/// <para>
/// The formula for the Power kernel is:
/// k(x, y) = -||x - y||^d
/// where:
/// - x and y are the two data points being compared
/// - ||x - y|| is the Euclidean distance between them
/// - d is the degree parameter
/// - The negative sign makes this a dissimilarity measure
/// </para>
/// <para>
/// Common uses include:
/// - Clustering algorithms where distance is more important than similarity
/// - Outlier detection
/// - Applications where the magnitude of difference matters more than similarity patterns
/// </para>
/// </remarks>
public class PowerKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The degree parameter that controls how the distance affects the kernel value.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The degree determines how strongly the distance between points affects the result.
    /// 
    /// Think of it like this:
    /// - degree = 1.0: The effect is directly proportional to distance
    /// - degree = 2.0: The effect grows quadratically with distance (default)
    /// - degree = 0.5: The effect grows with the square root of distance
    /// 
    /// Higher values amplify the differences between points that are far apart,
    /// while lower values reduce this effect.
    /// </remarks>
    private readonly T _degree;

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
    /// Initializes a new instance of the Power kernel with an optional degree parameter.
    /// </summary>
    /// <param name="degree">The degree parameter that controls how distance affects the kernel value. Default is 2.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Power kernel for use. You can optionally
    /// provide a value for the degree parameter:
    /// </para>
    /// <para>
    /// The degree controls how strongly the distance between points affects the result:
    /// - Lower values (like 0.5): Make the kernel less sensitive to distance
    /// - Higher values (like 3.0): Make the kernel more sensitive to distance
    /// - The default value is 2.0, which is a good starting point for many applications
    /// </para>
    /// <para>
    /// When might you want to change this parameter?
    /// - Use a higher degree when you want to emphasize points that are very different
    /// - Use a lower degree when you want a more gradual effect of distance
    /// </para>
    /// <para>
    /// The best value often depends on your specific dataset and problem, so you might
    /// need to experiment with different values to find what works best.
    /// </para>
    /// </remarks>
    public PowerKernel(T? degree = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _degree = degree ?? _numOps.FromDouble(2.0);
    }

    /// <summary>
    /// Calculates the Power kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the dissimilarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how different they are from each other using the Power kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. Finding the difference between the two vectors (x1 - x2)
    /// 2. Calculating the Euclidean distance (straight-line distance) between them
    /// 3. Raising this distance to the power of the degree parameter
    /// 4. Making the result negative (to indicate dissimilarity)
    /// </para>
    /// <para>
    /// The result is a dissimilarity measure where:
    /// - Values closer to zero (less negative) mean the points are more similar
    /// - More negative values mean the points are more different
    /// </para>
    /// <para>
    /// Unlike many other kernels, the Power kernel doesn't have an upper bound - the values
    /// can become increasingly negative as points get farther apart. This can be useful when
    /// you want to emphasize the magnitude of differences between points.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var diff = x1.Subtract(x2);
        var distance = _numOps.Sqrt(diff.DotProduct(diff));

        return _numOps.Negate(_numOps.Power(distance, _degree));
    }
}
