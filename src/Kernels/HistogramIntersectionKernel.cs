namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Histogram Intersection kernel for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Histogram Intersection kernel is a similarity measure commonly used in image recognition and
/// classification tasks. It measures similarity by finding the "overlap" between two vectors.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The Histogram Intersection kernel is especially useful when your data represents frequencies, counts,
/// or distributions (like histograms in image processing).
/// </para>
/// <para>
/// Think of this kernel as comparing two histograms (or any two sets of non-negative values) by looking at
/// their overlap. For each pair of corresponding values, it takes the smaller one (the overlap) and adds it
/// to the total similarity. The more overlap there is, the higher the similarity score.
/// </para>
/// <para>
/// This kernel works best when your data contains non-negative values, such as pixel intensities, word frequencies,
/// or other count-based features. It's particularly popular in computer vision and document classification.
/// </para>
/// <para>
/// Note: For proper use of the Histogram Intersection kernel, input vectors should contain non-negative values.
/// If your data has negative values, you might need to preprocess it before using this kernel.
/// </para>
/// </remarks>
public class HistogramIntersectionKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the Histogram Intersection kernel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Histogram Intersection kernel for use. Unlike some other kernels,
    /// this kernel doesn't have any parameters to adjust - it works the same way for all data.
    /// </para>
    /// <para>
    /// This makes it simpler to use, as you don't need to worry about tuning parameters. Just create
    /// an instance of this class, and you're ready to calculate similarities between your data points.
    /// </para>
    /// <para>
    /// Remember that this kernel works best with non-negative data that represents counts, frequencies,
    /// or distributions. If your data has negative values, you might want to consider a different kernel
    /// or preprocess your data to make all values non-negative.
    /// </para>
    /// </remarks>
    public HistogramIntersectionKernel()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Calculates the Histogram Intersection kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Histogram Intersection kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. For each pair of corresponding elements in the two vectors, finding the minimum (smaller) value
    /// 2. Summing up all these minimum values to get the total similarity
    /// </para>
    /// <para>
    /// The result is a measure of similarity where higher values indicate greater similarity between the vectors.
    /// For non-negative vectors, the result will be:
    /// - Highest when the vectors are identical
    /// - Lower when the vectors have less overlap
    /// - Zero when the vectors have no overlap at all (e.g., when one vector has all zeros where the other has positive values)
    /// </para>
    /// <para>
    /// This kernel is particularly intuitive: it directly measures how much the two vectors "overlap" with each other.
    /// It's often used in image recognition (comparing histograms of image features), document classification
    /// (comparing word frequency distributions), and other applications where data can be represented as counts or frequencies.
    /// </para>
    /// <para>
    /// Important: For this kernel to work properly and give meaningful results, your input vectors should contain
    /// non-negative values. If your data has negative values, consider preprocessing it first.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T sum = _numOps.Zero;
        for (int i = 0; i < x1.Length; i++)
        {
            sum = _numOps.Add(sum, MathHelper.Min(x1[i], x2[i]));
        }

        return sum;
    }
}
