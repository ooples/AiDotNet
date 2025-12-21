namespace AiDotNet.Kernels;

/// <summary>
/// Implements the B-Spline kernel function for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The B-Spline kernel is based on B-spline functions, which are piecewise polynomial functions
/// with compact support. This kernel is particularly useful for problems requiring smooth
/// interpolation and approximation.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The B-Spline kernel is a specialized similarity measure that works well for data that needs smooth
/// transitions between points. Think of B-splines like drawing a smooth curve through a set of points,
/// where the curve doesn't necessarily pass through all points but creates a smooth approximation.
/// </para>
/// <para>
/// This kernel is particularly useful when you want your AI model to produce smooth outputs
/// and avoid abrupt changes in predictions, similar to how a skilled artist might draw a smooth
/// curve rather than connecting dots with straight lines.
/// </para>
/// </remarks>
public class BSplineKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The degree of the B-spline function used in the kernel calculation.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The degree determines how smooth the B-spline function is.
    /// Higher degrees create smoother functions but require more computation.
    /// A degree of 3 (the default) creates cubic B-splines, which provide a good
    /// balance between smoothness and computational efficiency for most applications.
    /// </remarks>
    private readonly int _degree;

    /// <summary>
    /// The spacing between knots in the B-spline function.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Knots are the points where pieces of the B-spline function connect.
    /// The knot spacing controls how the kernel responds to differences between data points.
    /// A smaller spacing makes the kernel more sensitive to small differences,
    /// while a larger spacing makes it more tolerant of differences.
    /// </remarks>
    private readonly T _knotSpacing;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the B-Spline kernel with optional parameters.
    /// </summary>
    /// <param name="degree">The degree of the B-spline function. Default is 3 (cubic B-spline).</param>
    /// <param name="knotSpacing">The spacing between knots in the B-spline function. Default is 1.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the B-Spline kernel with your chosen settings.
    /// If you don't specify any settings, it will use default values that work well for many problems.
    /// </para>
    /// <para>
    /// The degree parameter (default 3) controls how smooth the kernel function is. Higher values create
    /// smoother functions but require more computation. The default value of 3 creates a cubic B-spline,
    /// which is commonly used because it provides a good balance of smoothness and efficiency.
    /// </para>
    /// <para>
    /// The knotSpacing parameter (default 1) controls how the kernel responds to differences between data points.
    /// </para>
    /// </remarks>
    public BSplineKernel(int degree = 3, T? knotSpacing = default)
    {
        _degree = degree;
        _numOps = MathHelper.GetNumericOperations<T>();
        _knotSpacing = knotSpacing ?? _numOps.One;
    }

    /// <summary>
    /// Calculates the B-Spline kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the B-Spline kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. For each dimension (feature) in the vectors, finding the difference between the corresponding values
    /// 2. Scaling this difference using the knot spacing parameter
    /// 3. Applying the B-spline basis function to this scaled difference
    /// 4. Multiplying all these individual dimension results together
    /// </para>
    /// <para>
    /// Higher output values indicate greater similarity between the vectors.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T result = _numOps.One;
        for (int i = 0; i < x1.Length; i++)
        {
            T diff = _numOps.Divide(_numOps.Subtract(x1[i], x2[i]), _knotSpacing);
            result = _numOps.Multiply(result, BSplineBasis(_degree, diff));
        }

        return result;
    }

    /// <summary>
    /// Calculates the value of the B-spline basis function of a given degree.
    /// </summary>
    /// <param name="degree">The degree of the B-spline basis function.</param>
    /// <param name="x">The input value.</param>
    /// <returns>The value of the B-spline basis function.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates a special mathematical function called the "B-spline basis function"
    /// which is used in the kernel calculation. B-spline basis functions are building blocks for creating
    /// smooth curves and surfaces.
    /// </para>
    /// <para>
    /// The method uses a recursive approach, meaning it calls itself with simpler versions of the problem:
    /// - For degree 0 (the simplest case), it returns 1 if x is between 0 and 1, otherwise 0
    /// - For higher degrees, it combines two simpler B-splines of one degree lower
    /// </para>
    /// <para>
    /// This recursive definition creates increasingly smooth functions as the degree increases,
    /// similar to how combining simple shapes can create more complex and smoother shapes.
    /// </para>
    /// </remarks>
    private T BSplineBasis(int degree, T x)
    {
        if (degree == 0)
        {
            return (_numOps.GreaterThanOrEquals(x, _numOps.Zero) && _numOps.LessThan(x, _numOps.One)) ? _numOps.One : _numOps.Zero;
        }

        T left = _numOps.Multiply(x, BSplineBasis(degree - 1, x));
        T right = _numOps.Multiply(_numOps.Subtract(_numOps.FromDouble(degree + 1), x), BSplineBasis(degree - 1, _numOps.Subtract(x, _numOps.One)));

        return _numOps.Divide(_numOps.Add(left, right), _numOps.FromDouble(degree));
    }
}
