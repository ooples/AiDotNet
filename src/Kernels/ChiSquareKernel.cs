namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Chi-Square kernel function for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Chi-Square kernel is based on the Chi-Square distance, which is particularly effective
/// for histogram data (such as image histograms, text frequency counts, etc.). It is commonly
/// used in computer vision and document classification tasks.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The Chi-Square kernel is especially good at comparing data that represents counts or frequencies,
/// like how often words appear in documents or how many pixels of each color are in an image.
/// </para>
/// <para>
/// Think of the Chi-Square kernel as a specialized tool for comparing "how much" of something exists
/// in two different samples. It's particularly sensitive to differences in smaller values, which makes
/// it good at detecting subtle patterns in your data. For example, it might notice that two documents
/// use rare words in similar ways, even if their common words are different.
/// </para>
/// <para>
/// This kernel works best when your data contains only non-negative values, such as counts,
/// frequencies, or proportions.
/// </para>
/// </remarks>
public class ChiSquareKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the Chi-Square kernel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Chi-Square kernel with the necessary
    /// mathematical operations for your chosen numeric type.
    /// </para>
    /// <para>
    /// Unlike some other kernels, the Chi-Square kernel doesn't have any parameters to adjust.
    /// This makes it simpler to use - you don't need to worry about tuning parameters to get
    /// good results.
    /// </para>
    /// </remarks>
    public ChiSquareKernel()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Calculates the Chi-Square kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Chi-Square kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. For each pair of corresponding values in the vectors:
    ///    a. Finding the difference between the values
    ///    b. Squaring this difference
    ///    c. Dividing by the sum of the values
    /// 2. Adding up all these individual results
    /// 3. Subtracting the sum from 1 to get a similarity measure (where higher values mean more similar)
    /// </para>
    /// <para>
    /// The result is a number between 0 and 1, where:
    /// - Values closer to 1 mean the vectors are very similar
    /// - Values closer to 0 mean the vectors are very different
    /// </para>
    /// <para>
    /// The method includes a safety check to avoid division by zero when the sum of corresponding
    /// values is zero.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T sum = _numOps.Zero;
        for (int i = 0; i < x1.Length; i++)
        {
            var diff = _numOps.Subtract(x1[i], x2[i]);
            var numerator = _numOps.Square(diff);
            var denominator = _numOps.Add(x1[i], x2[i]);

            if (!_numOps.Equals(denominator, _numOps.Zero))
            {
                sum = _numOps.Add(sum, _numOps.Divide(numerator, denominator));
            }
        }

        return _numOps.Subtract(_numOps.One, sum);
    }
}
