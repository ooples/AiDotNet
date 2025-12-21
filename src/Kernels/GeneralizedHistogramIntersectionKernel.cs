namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Generalized Histogram Intersection kernel for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Generalized Histogram Intersection kernel is an extension of the standard Histogram Intersection kernel,
/// which is commonly used in image recognition and classification tasks. It measures similarity by finding
/// the "overlap" between two vectors, with an additional parameter (beta) to control the influence of this overlap.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The Histogram Intersection kernel is especially useful when your data represents frequencies, counts,
/// or distributions (like histograms in image processing).
/// </para>
/// <para>
/// Think of this kernel as comparing two histograms (or any two sets of non-negative values) by looking at
/// their overlap. For each pair of corresponding values, it takes the smaller one (the overlap) and adds it
/// to the total similarity. The beta parameter lets you adjust how this overlap is weighted.
/// </para>
/// <para>
/// This kernel works best when your data contains non-negative values, such as pixel intensities, word frequencies,
/// or other count-based features. It's particularly popular in computer vision and document classification.
/// </para>
/// </remarks>
public class GeneralizedHistogramIntersectionKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The parameter that controls the influence of the intersection values.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Think of beta as a "weighting knob" for your similarity measurement.
    /// 
    /// When beta equals 1.0 (the default), this is the standard Histogram Intersection kernel
    /// that simply sums the minimum values between corresponding elements.
    /// 
    /// When beta is greater than 1.0, it gives more weight to larger intersection values,
    /// making the kernel more sensitive to features with high overlap.
    /// 
    /// When beta is less than 1.0 but greater than 0, it reduces the influence of larger
    /// intersection values, making the kernel more balanced across all features.
    /// 
    /// Adjusting beta lets you control how different overlapping values contribute to the
    /// overall similarity measure.
    /// </remarks>
    private readonly T _beta;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the Generalized Histogram Intersection kernel with an optional beta parameter.
    /// </summary>
    /// <param name="beta">The parameter that controls the influence of intersection values. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Generalized Histogram Intersection kernel with your chosen settings.
    /// If you don't specify any settings, it will use a default value of 1.0 for beta,
    /// which gives you the standard Histogram Intersection kernel.
    /// </para>
    /// <para>
    /// The beta parameter controls how the intersection values (the minimum of each pair of corresponding elements)
    /// are weighted in the final similarity calculation:
    /// </para>
    /// <para>
    /// - When beta = 1.0: This is the standard Histogram Intersection kernel
    /// - When beta > 1.0: Larger intersection values have more influence
    /// - When beta < 1.0: The influence of larger intersection values is reduced
    /// </para>
    /// <para>
    /// If you're just starting out, you can use the default value. As you become more experienced,
    /// you might want to experiment with different values to see what works best for your specific data.
    /// </para>
    /// </remarks>
    public GeneralizedHistogramIntersectionKernel(T? beta = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _beta = beta ?? _numOps.FromDouble(1.0);
    }

    /// <summary>
    /// Calculates the Generalized Histogram Intersection kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Generalized Histogram Intersection kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. For each pair of corresponding elements in the two vectors, finding the smaller value (the "overlap")
    /// 2. Raising this minimum value to the power of beta
    /// 3. Summing up all these values to get the total similarity
    /// </para>
    /// <para>
    /// The result is a measure of similarity where higher values indicate greater similarity between the vectors.
    /// Unlike some other kernels, this value is not normalized to be between 0 and 1 - the maximum possible
    /// value depends on the data.
    /// </para>
    /// <para>
    /// This kernel works best when your data contains non-negative values. If your data includes negative values,
    /// you might want to preprocess it or choose a different kernel function.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T sum = _numOps.Zero;
        for (int i = 0; i < x1.Length; i++)
        {
            sum = _numOps.Add(sum, _numOps.Power(MathHelper.Min(x1[i], x2[i]), _beta));
        }

        return sum;
    }
}
