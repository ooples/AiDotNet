namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Multiquadric kernel for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Multiquadric kernel is a popular kernel function used in machine learning and spatial analysis
/// to measure the similarity between data points.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The Multiquadric kernel is particularly useful for spatial data and regression problems.
/// </para>
/// <para>
/// Think of the Multiquadric kernel as a "similarity detector" that works well for many types of data.
/// Unlike some other kernels, the Multiquadric kernel increases with distance, which gives it some
/// unique properties that can be useful for certain types of problems.
/// </para>
/// <para>
/// The formula for the Multiquadric kernel is:
/// k(x, y) = v(||x - y||² + c²)
/// where:
/// - x and y are the two data points being compared
/// - ||x - y|| is the Euclidean distance between them
/// - c is a parameter that controls the kernel's behavior
/// </para>
/// <para>
/// This kernel is often used in radial basis function networks, spatial interpolation,
/// and various machine learning algorithms where you need to measure similarity between points.
/// </para>
/// </remarks>
public class MultiquadricKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The shape parameter that controls the behavior of the kernel function.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The '_c' parameter (sometimes called the "shape parameter") controls
    /// how the kernel responds to distance between points.
    /// 
    /// Think of it as a "sensitivity adjustment":
    /// - A smaller value makes the kernel more sensitive to small distances between points
    /// - A larger value makes the kernel less sensitive to small distances between points
    /// 
    /// The default value is 1.0, which works well as a starting point for many applications.
    /// You might want to adjust this parameter based on the scale of your data and the specific
    /// problem you're trying to solve.
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
    /// Initializes a new instance of the Multiquadric kernel with an optional shape parameter.
    /// </summary>
    /// <param name="c">The shape parameter that controls the kernel's behavior. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Multiquadric kernel for use. You can optionally
    /// provide a value for the shape parameter that controls how the kernel behaves.
    /// </para>
    /// <para>
    /// If you don't specify a value, the default is 1.0, which works well as a starting point
    /// for many applications.
    /// </para>
    /// <para>
    /// When might you want to change this parameter?
    /// - If your data points are very close together, you might want a smaller value
    /// - If your data points are far apart, you might want a larger value
    /// </para>
    /// <para>
    /// The best value often depends on your specific dataset and problem, so you might
    /// need to experiment with different values to find what works best.
    /// </para>
    /// </remarks>
    public MultiquadricKernel(T? c = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _c = c ?? _numOps.FromDouble(1.0);
    }

    /// <summary>
    /// Calculates the Multiquadric kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Multiquadric kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. Finding the difference between the two vectors (diff)
    /// 2. Calculating the squared Euclidean distance between them (squaredDistance)
    /// 3. Adding the square of the shape parameter (c²)
    /// 4. Taking the square root of the result
    /// </para>
    /// <para>
    /// Unlike many other kernels, the Multiquadric kernel value increases as the distance between
    /// points increases. This gives it some unique properties that can be useful for certain types
    /// of problems, especially in spatial interpolation.
    /// </para>
    /// <para>
    /// In practical terms, this means that points that are far apart will have a higher kernel value
    /// than points that are close together, which is the opposite of what many other kernels do.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var diff = x1.Subtract(x2);
        var squaredDistance = diff.DotProduct(diff);
        return _numOps.Sqrt(_numOps.Add(squaredDistance, _numOps.Multiply(_c, _c)));
    }
}
