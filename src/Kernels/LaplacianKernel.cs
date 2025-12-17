namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Laplacian kernel for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Laplacian kernel is a radial basis function kernel that uses the Manhattan distance (L1 norm)
/// instead of the Euclidean distance (L2 norm) used by the Gaussian kernel. It's particularly useful
/// for data that has sparse features or when you want to be less sensitive to outliers.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The Laplacian kernel is like a "similarity detector" that gives higher values when points are close
/// together and lower values when they're far apart.
/// </para>
/// <para>
/// Think of this kernel as a "distance translator" - it takes the distance between two points and
/// converts it into a similarity score between 0 and 1. Points that are identical get a score of 1,
/// while points that are very different get a score closer to 0.
/// </para>
/// <para>
/// What makes the Laplacian kernel special is how it measures distance - it uses what's called the
/// "Manhattan distance" or "city block distance." Imagine you're in a city with a grid of streets:
/// you can only travel along the streets (not diagonally through buildings). The Manhattan distance
/// is the total number of blocks you'd need to walk. This makes the Laplacian kernel less sensitive
/// to outliers compared to kernels that use the straight-line (Euclidean) distance.
/// </para>
/// <para>
/// The Laplacian kernel is often used in machine learning tasks like classification, regression,
/// and anomaly detection, especially when dealing with high-dimensional or sparse data.
/// </para>
/// </remarks>
public class LaplacianKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The bandwidth parameter that controls how quickly similarity decreases with distance.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Think of the '_sigma' parameter as a "sensitivity knob" for distance.
    /// 
    /// A larger value of '_sigma' makes the kernel less sensitive to distance, meaning even points
    /// that are somewhat far apart will still have a meaningful similarity score.
    /// 
    /// A smaller value of '_sigma' makes the kernel more sensitive to distance, meaning the
    /// similarity score drops more quickly as points get farther apart.
    /// 
    /// The default value of 1.0 provides a good balance for many applications, but you can
    /// adjust it based on your specific needs.
    /// </remarks>
    private readonly T _sigma;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the Laplacian kernel with an optional bandwidth parameter.
    /// </summary>
    /// <param name="sigma">The bandwidth parameter that controls how quickly similarity decreases with distance. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Laplacian kernel for use. You can optionally
    /// provide a value for the 'sigma' parameter, which controls how the kernel behaves.
    /// </para>
    /// <para>
    /// If you don't specify a value for 'sigma', it will default to 1.0, which works well for many applications.
    /// </para>
    /// <para>
    /// When might you want to change the 'sigma' parameter?
    /// - If your data points tend to be far apart and you still want to capture relationships between them,
    ///   use a larger value of 'sigma' (like 2.0 or 5.0).
    /// - If your data points are close together and you want to be more selective about what's considered
    ///   "similar," use a smaller value of 'sigma' (like 0.1 or 0.5).
    /// </para>
    /// <para>
    /// The best value for 'sigma' often depends on your specific dataset and problem, so you might need to
    /// experiment with different values to find what works best.
    /// </para>
    /// </remarks>
    public LaplacianKernel(T? sigma = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _sigma = sigma ?? _numOps.FromDouble(1.0);
    }

    /// <summary>
    /// Calculates the Laplacian kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Laplacian kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. Finding the difference between the two vectors
    /// 2. Calculating the Manhattan distance between them (the sum of the absolute differences)
    /// 3. Dividing this distance by the 'sigma' parameter
    /// 4. Taking the negative of this value
    /// 5. Calculating e (approximately 2.718) raised to this power
    /// </para>
    /// <para>
    /// The result is a similarity measure where:
    /// - A value of 1 means the vectors are identical
    /// - A value close to 0 means the vectors are very different (far apart)
    /// - Values in between represent varying degrees of similarity
    /// </para>
    /// <para>
    /// The Laplacian kernel has a more "peaked" shape compared to the Gaussian kernel, which means
    /// it's less influenced by points that are far away. This can be beneficial when you want your
    /// model to focus more on local patterns in your data.
    /// </para>
    /// <para>
    /// This kernel is particularly useful when working with sparse data (where many values are zero)
    /// or when you want your model to be robust against outliers.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var diff = x1.Subtract(x2);
        var distance = diff.Transform(_numOps.Abs).Sum();

        return _numOps.Exp(_numOps.Negate(_numOps.Divide(distance, _sigma)));
    }
}
