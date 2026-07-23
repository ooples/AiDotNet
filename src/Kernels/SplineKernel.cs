namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Spline kernel for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Spline kernel is a specialized kernel function that is particularly useful for
/// smoothing and interpolation tasks. It's based on the mathematical concept of splines,
/// which are piecewise polynomial functions used to create smooth curves.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The Spline kernel is special because it's designed to create smooth transitions between data points,
/// similar to how a flexible ruler (a physical spline) creates a smooth curve when bent to pass through
/// a set of points.
/// </para>
/// <para>
/// Think of it like this: The Spline kernel looks at each dimension (feature) of your data separately,
/// calculates a similarity score for each dimension, and then combines these scores by multiplying them
/// together. This approach helps it capture complex relationships in your data while maintaining smoothness.
/// </para>
/// <para>
/// The formula for the Spline kernel for each dimension is:
/// k(x, y) = 1 + x*y + x*y*min(x,y) - (x+y)/2*min(x,y)² + min(x,y)³/3
/// 
/// For simplicity, this implementation uses the form:
/// k(x, y) = 1 + x*y*min(x,y) + 0.5*min(x,y)³
/// 
/// The overall kernel value is the product of these values across all dimensions.
/// </para>
/// <para>
/// Common uses include:
/// - Smoothing noisy data
/// - Interpolation problems
/// - Function approximation
/// - Regression tasks where smoothness is important
/// </para>
/// </remarks>
public class SplineKernel<T> : IKernelFunction<T>
{
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
    /// Initializes a new instance of the Spline kernel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Spline kernel for use. Unlike some other
    /// kernels, the Spline kernel doesn't have any parameters that you need to set - it works
    /// right out of the box.
    /// </para>
    /// <para>
    /// The Spline kernel is particularly good at creating smooth transitions between data points,
    /// which makes it useful for:
    /// - Smoothing noisy data
    /// - Interpolation (estimating values between known data points)
    /// - Function approximation (finding a smooth function that fits your data)
    /// </para>
    /// <para>
    /// When might you want to use this kernel?
    /// - When you're working with data that should have smooth transitions
    /// - When you're doing regression and want a smooth prediction function
    /// - When you're working with time series data and want to capture smooth trends
    /// </para>
    /// </remarks>
    public SplineKernel()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Calculates the Spline kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Spline kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. Starting with a result of 1
    /// 2. For each dimension (feature) in the vectors:
    ///    a. Finding the minimum value between the two vectors at that dimension
    ///    b. Calculating the product of the two values at that dimension
    ///    c. Multiplying the minimum by the product
    ///    d. Calculating the cube of the minimum value
    ///    e. Multiplying the cube by 0.5
    ///    f. Adding 1, the minimum-product, and the half-cube together
    ///    g. Multiplying the running result by this sum
    /// 3. Returning the final product
    /// </para>
    /// <para>
    /// The result is a similarity measure where higher values indicate greater similarity.
    /// </para>
    /// <para>
    /// What makes this kernel special is its ability to create smooth transitions between data points,
    /// which is particularly useful for regression tasks and function approximation.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T result = _numOps.One;
        for (int i = 0; i < x1.Length; i++)
        {
            T min = MathHelper.Min(x1[i], x2[i]);
            T product = _numOps.Multiply(x1[i], x2[i]);
            T minProduct = _numOps.Multiply(min, product);
            T minCubed = _numOps.Power(min, _numOps.FromDouble(3));
            T halfMinCubed = _numOps.Multiply(_numOps.FromDouble(0.5), minCubed);

            T innerSum = _numOps.Add(_numOps.One, minProduct);
            innerSum = _numOps.Add(innerSum, halfMinCubed);

            result = _numOps.Multiply(result, innerSum);
        }

        return result;
    }
}
