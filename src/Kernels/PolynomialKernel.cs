namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Polynomial kernel for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Polynomial kernel is a popular kernel function used in machine learning algorithms
/// like Support Vector Machines (SVMs) to find patterns in higher-dimensional spaces.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The Polynomial kernel is special because it can find complex patterns in your data by implicitly
/// mapping your data to a higher-dimensional space without actually performing the expensive calculations
/// that would normally be required.
/// </para>
/// <para>
/// Think of the Polynomial kernel as a "pattern detector" that can identify more complex relationships
/// than simple linear patterns. For example, if you're trying to classify data that can't be separated
/// by a straight line, the Polynomial kernel can help find a curved boundary instead.
/// </para>
/// <para>
/// The formula for the Polynomial kernel is:
/// k(x, y) = (x·y + c)^d
/// where:
/// - x and y are the two data points being compared
/// - x·y is the dot product (a measure of how aligned the vectors are)
/// - c is a constant term (coef0) that influences the kernel's behavior
/// - d is the degree of the polynomial
/// </para>
/// <para>
/// Common uses include:
/// - Classification problems where data isn't linearly separable
/// - Natural language processing tasks
/// - Image recognition
/// </para>
/// </remarks>
public class PolynomialKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The degree of the polynomial used in the kernel function.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The degree determines the "complexity" of patterns the kernel can detect.
    /// 
    /// Think of it like this:
    /// - degree = 1: Can only find straight-line patterns (linear)
    /// - degree = 2: Can find curved patterns (quadratic)
    /// - degree = 3: Can find more complex curved patterns (cubic)
    /// - Higher degrees: Can find increasingly complex patterns
    /// 
    /// The default value is 3.0, which works well for many applications.
    /// Higher values can find more complex patterns but might lead to overfitting
    /// (where the model learns the training data too specifically and performs poorly on new data).
    /// </remarks>
    private readonly T _degree;

    /// <summary>
    /// The constant coefficient added to the dot product before raising to the power of the degree.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This parameter (sometimes called the "bias" term) controls how much influence
    /// the degree has on the result.
    /// 
    /// Think of it as a "flexibility adjustment":
    /// - When coef0 = 0: Only interactions between features matter
    /// - When coef0 > 0: The kernel can better handle data that doesn't pass through the origin
    /// 
    /// The default value is 1.0, which works well as a starting point for many applications.
    /// </remarks>
    private readonly T _coef0;

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
    /// Initializes a new instance of the Polynomial kernel with optional degree and coefficient parameters.
    /// </summary>
    /// <param name="degree">The degree of the polynomial. Default is 3.0.</param>
    /// <param name="coef0">The constant coefficient added to the dot product. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Polynomial kernel for use. You can optionally
    /// provide values for:
    /// </para>
    /// <para>
    /// 1. degree - Controls the complexity of patterns the kernel can detect (default: 3.0)
    ///    - Higher values can find more complex patterns
    ///    - Lower values create simpler models
    /// </para>
    /// <para>
    /// 2. coef0 - Controls how much influence the degree has on the result (default: 1.0)
    ///    - Higher values give more weight to higher-order terms
    ///    - A value of 0 means only interactions between features matter
    /// </para>
    /// <para>
    /// When might you want to change these parameters?
    /// - Change the degree if your data has complex patterns (higher degree) or simple patterns (lower degree)
    /// - Change coef0 if your model isn't performing well with the default value
    /// </para>
    /// <para>
    /// The best values often depend on your specific dataset and problem, so you might
    /// need to experiment with different values to find what works best.
    /// </para>
    /// </remarks>
    public PolynomialKernel(T? degree = default, T? coef0 = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _degree = degree ?? _numOps.FromDouble(3.0);
        _coef0 = coef0 ?? _numOps.FromDouble(1.0);
    }

    /// <summary>
    /// Calculates the Polynomial kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Polynomial kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. Computing the dot product between the two vectors (a measure of their alignment)
    /// 2. Adding the constant coefficient (coef0) to this dot product
    /// 3. Raising the result to the power of the degree
    /// </para>
    /// <para>
    /// The result is a similarity measure where higher values indicate greater similarity.
    /// This similarity measure has been transformed to capture more complex relationships
    /// than a simple linear comparison would.
    /// </para>
    /// <para>
    /// What makes this kernel powerful is that it implicitly maps your data to a higher-dimensional
    /// space (where more complex patterns can be found) without actually having to calculate
    /// all the new coordinates, which would be computationally expensive.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var dotProduct = x1.DotProduct(x2);
        return _numOps.Power(_numOps.Add(dotProduct, _coef0), _degree);
    }
}
