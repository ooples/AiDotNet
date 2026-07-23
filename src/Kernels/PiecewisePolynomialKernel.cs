namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Piecewise Polynomial kernel for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Piecewise Polynomial kernel is a compact kernel function that produces a similarity measure
/// that becomes exactly zero when points are far enough apart.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The Piecewise Polynomial kernel is special because it has a "cutoff distance" - if two points are
/// farther apart than this distance, the kernel says they have zero similarity.
/// </para>
/// <para>
/// This property makes the Piecewise Polynomial kernel useful for:
/// - Speeding up calculations in large datasets (since many calculations become zero)
/// - Problems where you only want nearby points to influence each other
/// - Creating sparse matrices in kernel methods
/// </para>
/// <para>
/// The formula for this kernel is:
/// k(x, y) = (1 - ||x-y||/c)^(j+1) if ||x-y|| = c, and 0 otherwise
/// where:
/// - x and y are the two data points being compared
/// - ||x-y|| is the Euclidean distance between them
/// - c is the cutoff distance parameter
/// - j is the degree parameter
/// </para>
/// </remarks>
public class PiecewisePolynomialKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The degree of the polynomial used in the kernel function.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The degree controls the "smoothness" of the kernel function.
    /// Higher values create a smoother function that decreases more gradually as points get farther apart.
    /// The default value is 3, which works well for many applications.
    /// </remarks>
    private readonly int _degree;

    /// <summary>
    /// The cutoff distance parameter that determines when the kernel value becomes zero.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This parameter sets the maximum distance between points where they're still
    /// considered to have some similarity. If points are farther apart than this value, the kernel
    /// will return zero (meaning "no similarity").
    /// 
    /// Think of it as a "neighborhood radius" - points outside this radius don't influence each other.
    /// The default value is 1.0, but you might want to adjust it based on the scale of your data.
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
    /// Initializes a new instance of the Piecewise Polynomial kernel with optional degree and cutoff parameters.
    /// </summary>
    /// <param name="degree">The degree of the polynomial. Default is 3.</param>
    /// <param name="c">The cutoff distance parameter. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Piecewise Polynomial kernel for use. You can optionally
    /// provide values for:
    /// </para>
    /// <para>
    /// 1. degree - Controls how smoothly the similarity decreases with distance (default: 3)
    ///    - Higher values create a smoother decrease
    ///    - Lower values create a more abrupt decrease
    /// </para>
    /// <para>
    /// 2. c - The cutoff distance beyond which points have zero similarity (default: 1.0)
    ///    - Smaller values mean fewer points influence each other
    ///    - Larger values mean more points influence each other
    /// </para>
    /// <para>
    /// When might you want to change these parameters?
    /// - Change the degree if you want to control how smoothly the similarity decreases
    /// - Change the cutoff distance to match the scale of your data or to control sparsity
    /// </para>
    /// <para>
    /// The best values often depend on your specific dataset and problem, so you might
    /// need to experiment with different values to find what works best.
    /// </para>
    /// </remarks>
    public PiecewisePolynomialKernel(int degree = 3, T? c = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _degree = degree;
        _c = c ?? _numOps.One;
    }

    /// <summary>
    /// Calculates the Piecewise Polynomial kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Piecewise Polynomial kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. Finding the Euclidean distance (straight-line distance) between the two points
    /// 2. Checking if this distance is greater than the cutoff parameter (c)
    ///    - If yes, it returns 0 (no similarity)
    ///    - If no, it continues with the calculation
    /// 3. Computing (1 - distance/c)^(degree+1) to get the similarity value
    /// </para>
    /// <para>
    /// The result is a similarity measure where:
    /// - A value close to 1 means the points are very similar (close together)
    /// - A value close to 0 means the points are less similar (farther apart)
    /// - Exactly 0 means the points are beyond the cutoff distance
    /// </para>
    /// <para>
    /// This "cutoff" property makes calculations more efficient for large datasets,
    /// as many point pairs will have zero similarity and can be ignored in further calculations.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T distance = x1.EuclideanDistance(x2);

        if (_numOps.GreaterThan(distance, _c))
        {
            return _numOps.Zero;
        }

        T j = _numOps.FromDouble(_degree);
        T term = _numOps.Subtract(_numOps.One, _numOps.Divide(distance, _c));

        return _numOps.Power(term, _numOps.Add(j, _numOps.One));
    }
}
