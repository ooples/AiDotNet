namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Additive Chi-Squared kernel function for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Additive Chi-Squared kernel is a variation of the Chi-Squared distance measure that is
/// particularly useful for histogram comparison in image recognition, document classification,
/// and other applications where data is represented as frequency distributions.
/// </para>
/// <para>
/// The kernel is defined as K(x,y) = -log(1 + S[(x_i - y_i)²/(x_i + y_i)]) for all dimensions i.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a way to measure how similar two data points are to each other.
/// Think of it like a special ruler that can measure distance in complex data spaces. The Additive 
/// Chi-Squared kernel is particularly good at comparing data that represents counts or frequencies 
/// (like how many times words appear in documents, or how many pixels of each color appear in images).
/// </para>
/// <para>
/// Unlike regular distance measures where smaller values mean "closer" (more similar), kernel functions
/// typically return larger values for more similar items. This kernel transforms the Chi-Squared distance
/// so that more similar items have higher values.
/// </para>
/// </remarks>
public class AdditiveChiSquaredKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the Additive Chi-Squared kernel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor prepares the kernel function for use. It doesn't need any
    /// parameters because the Additive Chi-Squared kernel doesn't have any adjustable settings.
    /// It simply sets up the mathematical operations needed for the calculations.
    /// </para>
    /// </remarks>
    public AdditiveChiSquaredKernel()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Calculates the Additive Chi-Squared kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Additive Chi-Squared formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. Finding the difference between corresponding elements in both vectors
    /// 2. Squaring those differences
    /// 3. Dividing by the sum of the corresponding elements
    /// 4. Adding up all these values
    /// 5. Applying a logarithmic transformation to the result
    /// </para>
    /// <para>
    /// Higher output values indicate greater similarity between the vectors. This is useful in
    /// machine learning algorithms that need to compare data points, such as Support Vector Machines
    /// or clustering algorithms.
    /// </para>
    /// <para>
    /// Note: The method handles the case where the denominator (x_i + y_i) is zero to avoid
    /// division by zero errors.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T sum = _numOps.Zero;
        for (int i = 0; i < x1.Length; i++)
        {
            T numerator = _numOps.Square(_numOps.Subtract(x1[i], x2[i]));
            T denominator = _numOps.Add(x1[i], x2[i]);
            if (!_numOps.Equals(denominator, _numOps.Zero))
            {
                sum = _numOps.Add(sum, _numOps.Divide(numerator, denominator));
            }
        }

        return _numOps.Negate(_numOps.Log(_numOps.Add(_numOps.One, sum)));
    }
}
