namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Exponential kernel function for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Exponential kernel is a radial basis function kernel that decreases exponentially with the
/// distance between data points. It is similar to the Gaussian kernel but uses the L1 norm (Manhattan distance)
/// instead of the L2 norm (Euclidean distance squared).
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The Exponential kernel is like a "similarity detector" that gives higher values when points are close
/// together and lower values when they're far apart.
/// </para>
/// <para>
/// Think of the Exponential kernel as measuring similarity that fades quickly as points get farther apart.
/// It's like the brightness of a flashlight - very bright up close, but quickly gets dimmer as you move away.
/// Unlike some other kernels, the Exponential kernel never completely reaches zero, though it gets very close
/// for distant points.
/// </para>
/// <para>
/// This kernel is useful when you want a similarity measure that's sensitive to small distances but still
/// gives some weight to moderately distant points.
/// </para>
/// </remarks>
public class ExponentialKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The scaling parameter that controls how quickly similarity decreases with distance.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Think of sigma as a "sensitivity knob" for your similarity measurement.
    /// A smaller sigma makes the kernel more sensitive to differences between data points,
    /// causing similarity to drop off more quickly with distance.
    /// 
    /// In practical terms:
    /// - With a small sigma, only very close points will be considered similar
    /// - With a large sigma, even somewhat distant points will still have meaningful similarity values
    /// 
    /// Adjusting sigma lets you control how "picky" your similarity measure is about distance.
    /// </remarks>
    private readonly T _sigma;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the Exponential kernel with an optional scaling parameter.
    /// </summary>
    /// <param name="sigma">The scaling parameter that controls how quickly similarity decreases with distance. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Exponential kernel with your chosen settings.
    /// If you don't specify any settings, it will use a default value of 1.0 for sigma,
    /// which works well for many problems.
    /// </para>
    /// <para>
    /// The sigma parameter controls how quickly the similarity decreases as points get farther apart.
    /// A smaller sigma means the similarity drops off more quickly with distance.
    /// </para>
    /// <para>
    /// If you're just starting out, you can use the default value. As you become more experienced,
    /// you might want to experiment with different values to see what works best for your specific data.
    /// </para>
    /// </remarks>
    public ExponentialKernel(T? sigma = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _sigma = sigma ?? _numOps.FromDouble(1.0);
    }

    /// <summary>
    /// Calculates the Exponential kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Exponential kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. Finding the difference between the two vectors
    /// 2. Calculating the Euclidean distance between them (how far apart they are)
    /// 3. Applying the exponential function to the negative distance divided by sigma
    /// </para>
    /// <para>
    /// The result is a number between 0 and 1, where:
    /// - 1 means the vectors are identical (zero distance)
    /// - Values close to 1 mean the vectors are very similar (small distance)
    /// - Values close to 0 mean the vectors are very different (large distance)
    /// </para>
    /// <para>
    /// Unlike some other kernels, the Exponential kernel never actually reaches 0, though it gets
    /// very close for distant points. This means it always considers points to have at least some
    /// tiny amount of similarity, no matter how far apart they are.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var diff = x1.Subtract(x2);
        var distance = _numOps.Sqrt(diff.DotProduct(diff));

        return _numOps.Exp(_numOps.Divide(_numOps.Negate(distance), _sigma));
    }
}
