namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Log kernel for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Log kernel is a kernel function that uses the negative logarithm of the distance between points
/// to measure similarity. It can be useful for certain types of data where similarity decreases
/// logarithmically with distance.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The Log kernel is special because it uses logarithms to determine similarity.
/// </para>
/// <para>
/// Think of the Log kernel as a "similarity detector" that is very sensitive to small distances but less
/// sensitive to large distances. When two points are very close together, a small change in distance makes
/// a big difference in similarity. But when two points are already far apart, even a large additional
/// distance doesn't change the similarity much.
/// </para>
/// <para>
/// This is similar to how we perceive sound volume: the difference between a whisper and normal speech
/// seems much greater than the difference between a shout and a very loud shout, even though the actual
/// increase in sound energy might be the same.
/// </para>
/// <para>
/// The Log kernel can be useful for data where small differences are very important when points are similar,
/// but less important when points are already quite different.
/// </para>
/// </remarks>
public class LogKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// Controls the power to which the logarithm of the distance is raised.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The '_degree' parameter controls how strongly the logarithm affects the similarity calculation.
    /// 
    /// A higher degree value makes the kernel more sensitive to differences in distance.
    /// A lower degree value makes the kernel less sensitive to differences in distance.
    /// 
    /// The default value is 1.0, which applies the logarithm directly without raising it to any power.
    /// </remarks>
    private readonly T _degree;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the Log kernel with an optional degree parameter.
    /// </summary>
    /// <param name="degree">The power to which the logarithm of the distance is raised. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Log kernel for use. You can optionally
    /// provide a value for the degree parameter that controls how the kernel behaves.
    /// </para>
    /// <para>
    /// If you don't specify a value, the default is 1.0, which means the logarithm is used directly.
    /// </para>
    /// <para>
    /// When might you want to change this parameter?
    /// - Increase the degree if you want the kernel to be more sensitive to differences in distance
    /// - Decrease the degree if you want the kernel to be less sensitive to differences in distance
    /// </para>
    /// <para>
    /// The best value for this parameter often depends on your specific dataset and problem,
    /// so you might need to experiment with different values to find what works best.
    /// </para>
    /// </remarks>
    public LogKernel(T? degree = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _degree = degree ?? _numOps.FromDouble(1.0);
    }

    /// <summary>
    /// Calculates the Log kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Log kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. Finding the difference between the two vectors
    /// 2. Calculating the Euclidean distance (straight-line distance) between them
    /// 3. Taking the logarithm of this distance
    /// 4. Raising the logarithm to the power specified by the degree parameter
    /// 5. Making the result negative (so closer points have higher similarity)
    /// </para>
    /// <para>
    /// The result is a similarity measure where:
    /// - A higher (less negative) value means the points are more similar
    /// - A lower (more negative) value means the points are less similar
    /// </para>
    /// <para>
    /// One important note: This kernel will produce an error if the distance between points is exactly zero
    /// (because the logarithm of zero is undefined). In practice, this usually only happens when comparing
    /// a point to itself, and you might need to add a small value to the distance in such cases.
    /// </para>
    /// <para>
    /// The Log kernel is particularly useful for data where you want to be very sensitive to small
    /// differences between similar points, but less sensitive to differences between points that are
    /// already quite different.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var diff = x1.Subtract(x2);
        var distance = _numOps.Sqrt(diff.DotProduct(diff));

        return _numOps.Negate(_numOps.Power(_numOps.Log(distance), _degree));
    }
}
