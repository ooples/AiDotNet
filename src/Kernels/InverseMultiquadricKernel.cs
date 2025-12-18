namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Inverse Multiquadric kernel for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Inverse Multiquadric kernel is a radial basis function kernel that measures similarity
/// based on the distance between points. Unlike the Gaussian kernel, it decreases more slowly
/// as points get farther apart, making it useful for data with long-range dependencies.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The Inverse Multiquadric kernel is like a "similarity detector" that gives higher values when points
/// are close together and lower values when they're far apart.
/// </para>
/// <para>
/// Think of this kernel as a "distance translator" - it takes the distance between two points and
/// converts it into a similarity score. Points that are close together get a similarity score close to 1,
/// while points that are far apart get a score closer to 0, but the decrease is more gradual than with
/// some other kernels.
/// </para>
/// <para>
/// This kernel has a parameter 'c' that controls how quickly the similarity decreases with distance.
/// A larger value of 'c' makes the kernel more "tolerant" of distance, meaning points can be farther
/// apart and still be considered somewhat similar.
/// </para>
/// <para>
/// The Inverse Multiquadric kernel is often used in machine learning tasks like regression, classification,
/// and interpolation, especially when you want to capture both local and more distant relationships in your data.
/// </para>
/// </remarks>
public class InverseMultiquadricKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The shape parameter that controls how quickly similarity decreases with distance.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Think of the '_c' parameter as a "tolerance knob" for distance.
    /// 
    /// A larger value of '_c' makes the kernel more tolerant of distance, meaning even points
    /// that are somewhat far apart will still have a meaningful similarity score.
    /// 
    /// A smaller value of '_c' makes the kernel more sensitive to distance, meaning the
    /// similarity score drops more quickly as points get farther apart.
    /// 
    /// The default value of 1.0 provides a good balance for many applications, but you can
    /// adjust it based on your specific needs.
    /// </remarks>
    private readonly T _c;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the Inverse Multiquadric kernel with an optional shape parameter.
    /// </summary>
    /// <param name="c">The shape parameter that controls how quickly similarity decreases with distance. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Inverse Multiquadric kernel for use. You can optionally
    /// provide a value for the 'c' parameter, which controls how the kernel behaves.
    /// </para>
    /// <para>
    /// If you don't specify a value for 'c', it will default to 1.0, which works well for many applications.
    /// </para>
    /// <para>
    /// When might you want to change the 'c' parameter?
    /// - If your data points tend to be far apart and you still want to capture relationships between them,
    ///   use a larger value of 'c' (like 2.0 or 5.0).
    /// - If your data points are close together and you want to be more selective about what's considered
    ///   "similar," use a smaller value of 'c' (like 0.1 or 0.5).
    /// </para>
    /// <para>
    /// The best value for 'c' often depends on your specific dataset and problem, so you might need to
    /// experiment with different values to find what works best.
    /// </para>
    /// </remarks>
    public InverseMultiquadricKernel(T? c = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _c = c ?? _numOps.FromDouble(1.0);
    }

    /// <summary>
    /// Calculates the Inverse Multiquadric kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Inverse Multiquadric kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. Finding the difference between the two vectors
    /// 2. Calculating the squared distance between them (similar to the Euclidean distance squared)
    /// 3. Adding the square of the 'c' parameter to this distance
    /// 4. Taking the square root of this sum
    /// 5. Calculating the reciprocal (1 divided by the result)
    /// </para>
    /// <para>
    /// The result is a similarity measure where:
    /// - A value close to 1 means the vectors are very similar (close together)
    /// - A value close to 0 means the vectors are very different (far apart)
    /// </para>
    /// <para>
    /// Unlike the Gaussian kernel, which drops to nearly zero very quickly as distance increases,
    /// the Inverse Multiquadric kernel decreases more slowly. This means it can still detect some
    /// similarity between points that are moderately far apart, which can be useful for capturing
    /// long-range dependencies in your data.
    /// </para>
    /// <para>
    /// This kernel is particularly useful in regression problems, interpolation tasks, and when
    /// working with data where distant points might still influence each other.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var diff = x1.Subtract(x2);
        var squaredDistance = diff.DotProduct(diff);
        return _numOps.Divide(_numOps.One, _numOps.Sqrt(_numOps.Add(squaredDistance, _numOps.Multiply(_c, _c))));
    }
}
