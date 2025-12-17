namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Hellinger kernel for measuring similarity between probability distributions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Hellinger kernel is based on the Hellinger distance, which is a metric used to quantify the similarity
/// between two probability distributions. This kernel is particularly useful when working with data that
/// represents distributions, such as histograms, word frequencies, or any normalized non-negative data.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The Hellinger kernel is specifically designed for comparing data that represents probabilities or
/// frequencies (like how often words appear in documents, or the distribution of pixel intensities in images).
/// </para>
/// <para>
/// Think of this kernel as a "distribution comparator" that works well when your data points represent
/// how things are distributed or how frequently they occur. It's particularly good at handling sparse data
/// (where many values are zero) and is less sensitive to outliers than some other kernels.
/// </para>
/// <para>
/// This kernel works best when your data contains non-negative values that sum to 1 (or can be normalized
/// to sum to 1), such as probability distributions, normalized histograms, or frequency counts.
/// </para>
/// <para>
/// Note: For proper use of the Hellinger kernel, input vectors should contain non-negative values and
/// ideally should be normalized (sum to 1). If your data doesn't meet these criteria, you might need
/// to preprocess it before using this kernel.
/// </para>
/// </remarks>
public class HellingerKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the Hellinger kernel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Hellinger kernel for use. Unlike some other kernels,
    /// the Hellinger kernel doesn't have any parameters to adjust - it works the same way for all data.
    /// </para>
    /// <para>
    /// This makes it simpler to use, as you don't need to worry about tuning parameters. Just create
    /// an instance of this class, and you're ready to calculate similarities between your data points.
    /// </para>
    /// <para>
    /// Remember that this kernel works best with non-negative data that represents distributions or
    /// frequencies. If your data has negative values or doesn't represent a distribution, you might
    /// want to consider a different kernel or preprocess your data.
    /// </para>
    /// </remarks>
    public HellingerKernel()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Calculates the Hellinger kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector, representing a probability distribution.</param>
    /// <param name="x2">The second vector, representing a probability distribution.</param>
    /// <returns>The kernel value representing the similarity between the two distributions.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Hellinger kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. For each pair of corresponding elements in the two vectors, multiplying them together
    /// 2. Taking the square root of this product
    /// 3. Summing up all these square roots to get the total similarity
    /// </para>
    /// <para>
    /// The result is a measure of similarity where higher values indicate greater similarity between the vectors.
    /// For normalized probability distributions (vectors whose elements sum to 1), the result will be between 0 and 1:
    /// - A value close to 1 means the distributions are very similar
    /// - A value close to 0 means the distributions are very different
    /// </para>
    /// <para>
    /// This kernel is related to the Bhattacharyya coefficient and is particularly useful for comparing
    /// probability distributions or normalized histograms. It's often used in text classification,
    /// image recognition, and other applications where data can be represented as distributions.
    /// </para>
    /// <para>
    /// Important: For this kernel to work properly, your input vectors should contain non-negative values.
    /// For the most meaningful results, they should also be normalized (sum to 1) to represent proper
    /// probability distributions.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T sum = _numOps.Zero;
        for (int i = 0; i < x1.Length; i++)
        {
            var sqrtProduct = _numOps.Sqrt(_numOps.Multiply(x1[i], x2[i]));
            sum = _numOps.Add(sum, sqrtProduct);
        }

        return sum;
    }
}
