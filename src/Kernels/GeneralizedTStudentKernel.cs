namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Generalized T-Student kernel for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Generalized T-Student kernel is a kernel function based on the Student's t-distribution.
/// It is particularly useful for handling data with outliers because it decreases more slowly
/// than the Gaussian kernel as points get farther apart.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The T-Student kernel is like a "similarity detector" that gives higher values when points are close
/// together and lower values when they're far apart, but it's more tolerant of occasional large distances
/// than some other kernels.
/// </para>
/// <para>
/// Think of this kernel as a "forgiving" similarity measure. While many kernels quickly decide that distant
/// points have almost zero similarity, the T-Student kernel decreases more gradually, still giving some
/// weight to points that are moderately far apart. This makes it useful when your data might contain
/// outliers or when distant points might still have meaningful relationships.
/// </para>
/// <para>
/// This kernel is particularly valuable in machine learning tasks where robustness to outliers is important,
/// such as in financial data analysis, anomaly detection, or noisy real-world datasets.
/// </para>
/// </remarks>
public class GeneralizedTStudentKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The degree parameter that controls how quickly similarity decreases with distance.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Think of the degree parameter as controlling how "forgiving" your similarity
    /// measure is about distance.
    /// 
    /// A smaller degree value makes the kernel more tolerant of large distances, meaning even points
    /// that are quite far apart will still have some meaningful similarity score.
    /// 
    /// A larger degree value makes the kernel decrease similarity more quickly as distance increases,
    /// becoming more strict about what it considers "similar."
    /// 
    /// The default value of 1.0 provides a good balance for many applications, but you can adjust it
    /// based on how much you want to emphasize or de-emphasize distant points in your analysis.
    /// </remarks>
    private readonly T _degree;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the Generalized T-Student kernel with an optional degree parameter.
    /// </summary>
    /// <param name="degree">The parameter that controls how quickly similarity decreases with distance. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Generalized T-Student kernel with your chosen settings.
    /// If you don't specify any settings, it will use a default value of 1.0 for the degree parameter,
    /// which works well for many problems.
    /// </para>
    /// <para>
    /// The degree parameter controls how the kernel responds to the distance between points:
    /// </para>
    /// <para>
    /// - With a smaller degree (e.g., 0.5), the kernel decreases more slowly as distance increases,
    ///   making it more tolerant of outliers or distant points
    /// - With a larger degree (e.g., 2.0), the kernel decreases more rapidly with distance,
    ///   making it more sensitive to distance differences
    /// </para>
    /// <para>
    /// If you're just starting out, you can use the default value. As you become more experienced,
    /// you might want to try different values to see what works best for your specific data,
    /// especially if your data contains outliers or has an unusual distribution.
    /// </para>
    /// </remarks>
    public GeneralizedTStudentKernel(T? degree = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _degree = degree ?? _numOps.FromDouble(1.0);
    }

    /// <summary>
    /// Calculates the Generalized T-Student kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Generalized T-Student kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. Finding the difference between the two vectors
    /// 2. Computing the squared Euclidean distance between them (the dot product of the difference with itself)
    /// 3. Applying the formula: 1 / (1 + distance^degree)
    /// </para>
    /// <para>
    /// The result is a number between 0 and 1, where:
    /// - 1 means the vectors are identical (zero distance)
    /// - Values close to 1 mean the vectors are very similar (small distance)
    /// - Values close to 0 mean the vectors are very different (large distance)
    /// </para>
    /// <para>
    /// Unlike the Gaussian kernel which drops off exponentially with distance, the T-Student kernel
    /// follows a power law decay, making it decrease more slowly for large distances. This property
    /// makes it more robust when dealing with outliers in your data.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var diff = x1.Subtract(x2);
        var squaredDistance = diff.DotProduct(diff);

        return _numOps.Divide(_numOps.One, _numOps.Add(_numOps.One, _numOps.Power(squaredDistance, _degree)));
    }
}
