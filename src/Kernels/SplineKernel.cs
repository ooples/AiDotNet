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
    /// 1. For each dimension (feature), with m = min(x, y):
    ///    1 + x*y + x*y*m - (x + y)/2 * m² + m³/3
    /// 2. Multiplying these per-dimension values together
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
        // Vapnik's spline kernel (Vapnik 1998; Souza's kernel survey):
        // prod_i [1 + x_i y_i + x_i y_i m_i - (x_i + y_i) / 2 * m_i^2 + m_i^3 / 3], m_i = min(x_i, y_i).
        var engine = AiDotNetEngine.Current;
        var min = engine.Min(x1, x2);
        var minSquared = engine.Multiply(min, min);
        var product = engine.Multiply(x1, x2);
        var productTimesMin = engine.Multiply(product, min);
        var halfSumTimesMinSquared = engine.Multiply(engine.Multiply(engine.Add(x1, x2), minSquared), _numOps.FromDouble(0.5));
        var minCubedOverThree = engine.Divide(engine.Multiply(minSquared, min), _numOps.FromDouble(3));

        var inner = engine.Add(engine.Add(product, productTimesMin), engine.Subtract(minCubedOverThree, halfSumTimesMinSquared));
        return engine.Product(engine.Add(inner, Vector<T>.CreateDefault(x1.Length, _numOps.One)));
    }
}
