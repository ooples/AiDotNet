namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Cauchy kernel function for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Cauchy kernel is based on the Cauchy distribution from probability theory. It is a
/// long-tailed kernel that decreases more slowly than the Gaussian kernel, making it more
/// robust to outliers in the data.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The Cauchy kernel is a specialized similarity measure that works well when your data might contain
/// unusual or extreme values (outliers).
/// </para>
/// <para>
/// Think of the Cauchy kernel as a "forgiving" similarity measure. When comparing two data points,
/// it doesn't penalize large differences as severely as other kernels might. This makes it useful
/// when you want your AI model to be less sensitive to occasional extreme values in your data,
/// similar to how a good teacher might not let one bad test score heavily impact a student's overall grade.
/// </para>
/// </remarks>
public class CauchyKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The scaling parameter that controls the width of the kernel.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Think of sigma as a "sensitivity knob" for your similarity measurement.
    /// A smaller sigma makes the kernel more sensitive to differences between data points,
    /// while a larger sigma makes the kernel more tolerant of differences.
    /// 
    /// In practical terms, if you set a small sigma, only very similar data points will be
    /// considered close to each other. With a large sigma, even somewhat different data points
    /// will be considered relatively similar.
    /// </remarks>
    private readonly T _sigma;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the Cauchy kernel with an optional scaling parameter.
    /// </summary>
    /// <param name="sigma">The scaling parameter that controls the width of the kernel. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Cauchy kernel with your chosen settings.
    /// If you don't specify any settings, it will use a default value of 1.0 for sigma,
    /// which works well for many problems.
    /// </para>
    /// <para>
    /// The sigma parameter controls how quickly the similarity decreases as points get farther apart.
    /// A larger sigma means that points can be farther apart and still be considered similar.
    /// </para>
    /// <para>
    /// If you're just starting out, you can use the default value. As you become more experienced,
    /// you might want to experiment with different values to see what works best for your specific data.
    /// </para>
    /// </remarks>
    public CauchyKernel(T? sigma = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _sigma = sigma ?? _numOps.FromDouble(1.0);
    }

    /// <summary>
    /// Calculates the Cauchy kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Cauchy kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. Finding the difference between the two vectors (subtracting one from the other)
    /// 2. Calculating the squared distance between them (dot product of the difference with itself)
    /// 3. Applying the Cauchy formula: 1 / (1 + squared_distance / sigma²)
    /// </para>
    /// <para>
    /// The result is a number between 0 and 1, where:
    /// - Values closer to 1 mean the vectors are very similar
    /// - Values closer to 0 mean the vectors are very different
    /// </para>
    /// <para>
    /// Unlike some other kernels, the Cauchy kernel decreases more slowly as the distance increases,
    /// making it less sensitive to outliers or extreme differences between data points.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var diff = x1.Subtract(x2);
        var squaredDistance = diff.DotProduct(diff);

        return _numOps.Divide(_numOps.One, _numOps.Add(_numOps.One, _numOps.Divide(squaredDistance, _numOps.Square(_sigma))));
    }
}
