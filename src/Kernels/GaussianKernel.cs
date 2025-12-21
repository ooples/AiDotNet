namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Gaussian (Radial Basis Function) kernel for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Gaussian kernel, also known as the Radial Basis Function (RBF) kernel, is one of the most widely used
/// kernel functions in machine learning. It measures similarity based on the Euclidean distance between points
/// and transforms this distance using an exponential function.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The Gaussian kernel is like a "similarity detector" that gives higher values when points are close
/// together and lower values when they're far apart.
/// </para>
/// <para>
/// Think of the Gaussian kernel as a bell-shaped curve centered on each data point. When you compare two points,
/// the kernel value tells you how much their "bells" overlap. Points that are close together have a lot of
/// overlap (high similarity), while distant points have little overlap (low similarity).
/// </para>
/// <para>
/// This kernel is particularly popular because it works well for many different types of data and problems.
/// It's often a good first choice when you're not sure which kernel to use.
/// </para>
/// </remarks>
public class GaussianKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The bandwidth parameter that controls how quickly similarity decreases with distance.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Think of sigma as a "zoom level" for your similarity measurement.
    /// A smaller sigma makes the kernel more focused on local patterns (only very close points are considered similar),
    /// while a larger sigma makes it consider broader patterns (even somewhat distant points can be similar).
    /// 
    /// In practical terms:
    /// - With a small sigma, the similarity drops off quickly as points get farther apart
    /// - With a large sigma, the similarity decreases more gradually with distance
    /// 
    /// Adjusting sigma lets you control how "picky" your similarity measure is about distance.
    /// </remarks>
    private readonly double _sigma;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the Gaussian kernel with an optional bandwidth parameter.
    /// </summary>
    /// <param name="sigma">The bandwidth parameter that controls how quickly similarity decreases with distance. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Gaussian kernel with your chosen settings.
    /// If you don't specify any settings, it will use a default value of 1.0 for sigma,
    /// which works well for many problems.
    /// </para>
    /// <para>
    /// The sigma parameter (sometimes called the bandwidth or length scale) controls the width of the
    /// Gaussian "bell curve" used for measuring similarity. It determines how quickly the similarity
    /// decreases as points get farther apart.
    /// </para>
    /// <para>
    /// If you're just starting out, you can use the default value. As you become more experienced,
    /// you might want to try different values of sigma to see what works best for your specific data.
    /// A common approach is to try values on different scales (like 0.1, 1.0, 10.0) to see which
    /// gives the best results.
    /// </para>
    /// </remarks>
    public GaussianKernel(double sigma = 1.0)
    {
        _sigma = sigma;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Calculates the Gaussian kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <exception cref="ArgumentException">Thrown when the vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Gaussian kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. Checking that both vectors have the same length (number of features)
    /// 2. Computing the squared Euclidean distance between the vectors (the sum of squared differences for each feature)
    /// 3. Applying the Gaussian formula: exp(-distance²/(2*sigma²))
    /// </para>
    /// <para>
    /// The result is a number between 0 and 1, where:
    /// - 1 means the vectors are identical (zero distance)
    /// - Values close to 1 mean the vectors are very similar (small distance)
    /// - Values close to 0 mean the vectors are very different (large distance)
    /// </para>
    /// <para>
    /// This similarity measure has a "bell curve" shape - similarity drops off gradually at first
    /// for small distances, then more rapidly as distance increases, and finally very slowly as
    /// distance becomes very large.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        if (x1.Length != x2.Length)
            throw new ArgumentException("Vectors must have the same length.");

        T squaredDistance = _numOps.Zero;
        for (int i = 0; i < x1.Length; i++)
        {
            T diff = _numOps.Subtract(x1[i], x2[i]);
            squaredDistance = _numOps.Add(squaredDistance, _numOps.Multiply(diff, diff));
        }

        T exponent = _numOps.Divide(squaredDistance, _numOps.FromDouble(2 * _sigma * _sigma));
        return _numOps.Exp(_numOps.Negate(exponent));
    }
}
