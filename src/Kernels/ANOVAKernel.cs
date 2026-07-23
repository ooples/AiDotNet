namespace AiDotNet.Kernels;

/// <summary>
/// Implements the ANOVA (Analysis of Variance) kernel function for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The ANOVA kernel is a specialized kernel function that is particularly useful for problems involving
/// analysis of variance. It combines aspects of the RBF (Radial Basis Function) kernel with a polynomial approach.
/// </para>
/// <para>
/// The kernel is defined as the sum of exponential terms raised to a specified power (degree), where each term
/// represents the similarity between corresponding dimensions of the input vectors.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are to each other.
/// Think of it like a special measuring tape that works in complex data spaces. The ANOVA kernel is particularly
/// useful when you want to analyze how different factors (variables) contribute to the overall variation in your data.
/// </para>
/// <para>
/// The name "ANOVA" comes from "Analysis of Variance," which is a statistical technique used to determine if there
/// are significant differences between the means of different groups. This kernel helps machine learning algorithms
/// capture these kinds of relationships in the data.
/// </para>
/// </remarks>
public class ANOVAKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The width parameter that controls the influence of distance between points.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Think of sigma as a "zoom level" for the similarity measurement. A smaller sigma means
    /// the kernel is more sensitive to small differences between data points, while a larger sigma makes
    /// the kernel more tolerant of differences.
    /// </remarks>
    private readonly T _sigma;

    /// <summary>
    /// The polynomial degree parameter that controls the complexity of the kernel.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The degree determines how "complex" the patterns the kernel can capture are.
    /// Higher degrees can capture more complex relationships but might also lead to overfitting
    /// (where the model learns noise in the data rather than the true pattern).
    /// </remarks>
    private readonly int _degree;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the ANOVA kernel with optional parameters.
    /// </summary>
    /// <param name="sigma">The width parameter that controls the influence of distance between points. Default is 1.0.</param>
    /// <param name="degree">The polynomial degree parameter that controls the complexity of the kernel. Default is 2.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the ANOVA kernel with your chosen settings. If you don't
    /// specify any settings, it will use default values that work reasonably well for many problems.
    /// </para>
    /// <para>
    /// The sigma parameter controls how quickly the similarity decreases as points get farther apart.
    /// A smaller sigma means only very close points are considered similar.
    /// </para>
    /// <para>
    /// The degree parameter controls the complexity of the patterns the kernel can detect. Higher values
    /// can capture more complex relationships but might make your model more prone to overfitting
    /// (learning the noise in your data rather than the true pattern).
    /// </para>
    /// </remarks>
    public ANOVAKernel(T? sigma = default, int degree = 2)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _sigma = sigma ?? _numOps.FromDouble(1.0);
        _degree = degree;
    }

    /// <summary>
    /// Calculates the ANOVA kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the ANOVA kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. For each dimension (feature) in your data:
    ///    a. Finding the squared difference between the corresponding values
    ///    b. Applying a Gaussian (bell curve) function to this difference
    ///    c. Raising the result to the power specified by the degree parameter
    /// 2. Adding up all these values to get the final similarity score
    /// </para>
    /// <para>
    /// Higher output values indicate greater similarity between the vectors. This is useful in
    /// machine learning algorithms like Support Vector Machines (SVMs) that need to compare data points.
    /// </para>
    /// <para>
    /// The ANOVA kernel is particularly good at identifying which dimensions (features) in your data
    /// contribute most to the differences between data points, similar to how ANOVA statistical tests
    /// identify which factors contribute most to variation in experimental results.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T result = _numOps.Zero;
        for (int i = 0; i < x1.Length; i++)
        {
            T term = _numOps.Exp(_numOps.Negate(_numOps.Divide(_numOps.Square(_numOps.Subtract(x1[i], x2[i])), _numOps.Multiply(_numOps.FromDouble(2), _numOps.Square(_sigma)))));
            result = _numOps.Add(result, _numOps.Power(term, _numOps.FromDouble(_degree)));
        }

        return result;
    }
}
