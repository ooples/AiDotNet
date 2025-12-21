namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Matérn kernel for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Matérn kernel is a flexible kernel function that generalizes the Radial Basis Function (RBF) kernel
/// by introducing a parameter that controls the smoothness of the resulting function.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The Matérn kernel is particularly useful because it lets you control exactly how "smooth" your model's
/// predictions will be.
/// </para>
/// <para>
/// Think of the Matérn kernel as a "similarity detector" with an adjustable sensitivity. When two points
/// are close together, the kernel gives a high similarity score. As the points get farther apart, the
/// similarity decreases. The special thing about the Matérn kernel is that you can control exactly how
/// quickly this similarity drops off and how smooth the transition is.
/// </para>
/// <para>
/// This kernel has two important parameters:
/// - The nu (?) parameter controls the smoothness of the function
/// - The length parameter controls how quickly the similarity decreases with distance
/// </para>
/// <para>
/// Common values for nu include 0.5 (which gives an exponential kernel), 1.5 (which is once differentiable),
/// and 2.5 (which is twice differentiable). The default value is 1.5, which works well for many applications.
/// </para>
/// <para>
/// The Matérn kernel is particularly useful for modeling physical processes, spatial data, and any application
/// where you need precise control over the smoothness assumptions in your model.
/// </para>
/// </remarks>
public class MaternKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// Controls the smoothness of the kernel function.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The '_nu' parameter determines how smooth your model's predictions will be.
    /// 
    /// Think of it as a "smoothness knob":
    /// - Lower values (like 0.5) create rougher, less smooth predictions that can change quickly
    /// - Higher values (like 2.5 or higher) create very smooth predictions that change gradually
    /// 
    /// Common values and what they mean:
    /// - 0.5: Creates an exponential kernel (predictions can change direction suddenly)
    /// - 1.5: Creates a kernel that's once differentiable (predictions change direction smoothly)
    /// - 2.5: Creates a kernel that's twice differentiable (predictions change direction very smoothly)
    /// 
    /// The default value is 1.5, which provides a good balance between flexibility and smoothness
    /// for many applications.
    /// </remarks>
    private readonly T _nu;

    /// <summary>
    /// Controls how quickly the similarity decreases with distance between points.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The '_length' parameter (also called the length scale) controls how far apart
    /// two points can be while still being considered similar.
    /// 
    /// Think of it as a "distance vision" parameter:
    /// - A smaller value means only very close points are considered similar (near-sighted)
    /// - A larger value means even distant points can be considered similar (far-sighted)
    /// 
    /// The default value is 1.0, which works well as a starting point for many applications.
    /// You might want to increase this if your data points are spread far apart, or decrease it
    /// if your data points are densely packed and local patterns are important.
    /// </remarks>
    private readonly T _length;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the Matérn kernel with optional parameters.
    /// </summary>
    /// <param name="nu">Controls the smoothness of the kernel function. Default is 1.5.</param>
    /// <param name="length">Controls how quickly similarity decreases with distance. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Matérn kernel for use. You can optionally
    /// provide values for the two parameters that control how the kernel behaves.
    /// </para>
    /// <para>
    /// If you don't specify values, the defaults are:
    /// - nu = 1.5: Creates a kernel that produces smooth predictions (once differentiable)
    /// - length = 1.0: A standard value for how quickly similarity decreases with distance
    /// </para>
    /// <para>
    /// When might you want to change these parameters?
    /// - Change nu if you want to control how smooth your model's predictions will be
    /// - Change length if you want to control how far the influence of each data point extends
    /// </para>
    /// <para>
    /// The best values for these parameters often depend on your specific dataset and problem,
    /// so you might need to experiment with different values to find what works best.
    /// </para>
    /// </remarks>
    public MaternKernel(T? nu = default, T? length = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _nu = nu ?? _numOps.FromDouble(1.5);
        _length = length ?? _numOps.One;
    }

    /// <summary>
    /// Calculates the Matérn kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Matérn kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. Finding the Euclidean distance (straight-line distance) between the two points
    /// 2. Scaling this distance based on the length parameter and nu parameter
    /// 3. Using special mathematical functions (Bessel functions and Gamma functions) to calculate
    ///    the final similarity value
    /// </para>
    /// <para>
    /// The result is a similarity measure where:
    /// - A value close to 1 means the points are very similar
    /// - A value close to 0 means the points are very different
    /// </para>
    /// <para>
    /// Don't worry about the complex math happening inside this method - the important thing to understand
    /// is that it measures similarity in a way that gives you control over the smoothness of your model's
    /// predictions through the nu parameter.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T distance = _numOps.Sqrt(x1.Subtract(x2).DotProduct(x1.Subtract(x2)));
        T scaledDistance = _numOps.Multiply(_numOps.Sqrt(_numOps.Multiply(_numOps.FromDouble(2), _nu)),
                                            _numOps.Divide(distance, _length));

        T besselTerm = ModifiedBesselFunction(_nu, scaledDistance);

        T powerTerm = _numOps.Power(_numOps.FromDouble(2), _numOps.Subtract(_numOps.One, _nu));
        T gammaTerm = StatisticsHelper<T>.Gamma(_nu);

        return _numOps.Multiply(_numOps.Multiply(powerTerm, _numOps.Divide(_numOps.One, gammaTerm)),
                                _numOps.Multiply(besselTerm, _numOps.Power(scaledDistance, _nu)));
    }

    /// <summary>
    /// Calculates the modified Bessel function of the second kind.
    /// </summary>
    /// <param name="order">The order of the Bessel function.</param>
    /// <param name="x">The input value.</param>
    /// <returns>The value of the modified Bessel function.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This is a helper method that implements a special mathematical function
    /// needed for the Matérn kernel calculation. You don't need to understand the details of this
    /// function to use the Matérn kernel effectively.
    /// </remarks>
    private T ModifiedBesselFunction(T order, T x)
    {
        double nu = Convert.ToDouble(order);
        double xDouble = Convert.ToDouble(x);

        if (xDouble < 0)
        {
            throw new ArgumentException("x must be non-negative for modified Bessel function");
        }

        if (xDouble == 0)
        {
            return nu == 0 ? _numOps.FromDouble(double.PositiveInfinity) : _numOps.FromDouble(double.PositiveInfinity);
        }

        if (xDouble <= 2)
        {
            // Use series expansion for small x
            return ModifiedBesselFunctionSeries(order, x);
        }
        else
        {
            // Use asymptotic expansion for large x
            return ModifiedBesselFunctionAsymptotic(order, x);
        }
    }

    /// <summary>
    /// Calculates the modified Bessel function using a series expansion for small input values.
    /// </summary>
    /// <param name="order">The order of the Bessel function.</param>
    /// <param name="x">The input value.</param>
    /// <returns>The value of the modified Bessel function.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This is a specialized mathematical helper method used internally by the kernel.
    /// It uses a mathematical technique called "series expansion" to calculate the Bessel function
    /// when the input value is small.
    /// </remarks>
    private T ModifiedBesselFunctionSeries(T order, T x)
    {
        T sum = _numOps.Zero;
        T term = _numOps.One;
        int k = 0;
        T xOver2 = _numOps.Divide(x, _numOps.FromDouble(2));

        while (true)
        {
            T a = _numOps.Power(xOver2, _numOps.Add(order, _numOps.FromDouble(2 * k)));
            T b = StatisticsHelper<T>.Gamma(_numOps.Add(order, _numOps.FromDouble(k + 1)));
            T c = StatisticsHelper<T>.Gamma(_numOps.FromDouble(k + 1));
            term = _numOps.Divide(a, _numOps.Multiply(b, c));

            if (_numOps.LessThan(_numOps.Abs(term), _numOps.FromDouble(1e-15)))
            {
                break;
            }

            sum = _numOps.Add(sum, term);
            k++;
        }

        return sum;
    }

    /// <summary>
    /// Calculates the modified Bessel function using an asymptotic expansion for large input values.
    /// </summary>
    /// <param name="order">The order of the Bessel function.</param>
    /// <param name="x">The input value.</param>
    /// <returns>The value of the modified Bessel function using an asymptotic approximation.</returns>
    /// <remarks>
    /// <para>
    /// This method implements an asymptotic expansion of the modified Bessel function of the second kind,
    /// which is more efficient and numerically stable for large input values.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is a specialized mathematical helper method used internally by the kernel.
    /// It uses a mathematical technique called "asymptotic expansion" to calculate the Bessel function
    /// when the input value is large.
    /// </para>
    /// <para>
    /// Think of this method as a mathematical shortcut that gives a good approximation when the
    /// input number is large. Instead of doing many calculations in a series (which could be slow
    /// or inaccurate for large numbers), this method uses a formula that works well specifically
    /// for large values.
    /// </para>
    /// <para>
    /// The formula uses:
    /// - A square root term (sqrtPiOver2x) that scales the result based on the input value
    /// - An exponential term (exp_x) that decreases rapidly as x increases
    /// - Correction terms (p and q) that improve the accuracy of the approximation
    /// </para>
    /// <para>
    /// You don't need to understand the mathematical details to use the Matérn kernel effectively.
    /// This method is just part of the internal machinery that makes the kernel work correctly.
    /// </para>
    /// </remarks>
    private T ModifiedBesselFunctionAsymptotic(T order, T x)
    {
        /// <summary>
        /// Calculates the square root of p/(2x) term used in the asymptotic expansion.
        /// </summary>
        T sqrtPiOver2x = _numOps.Divide(_numOps.Sqrt(_numOps.FromDouble(Math.PI / 2)), _numOps.Sqrt(x));

        /// <summary>
        /// Calculates the exponential decay term e^(-x) used in the asymptotic expansion.
        /// </summary>
        T exp_x = _numOps.Exp(_numOps.Negate(x));

        /// <summary>
        /// The first term in the asymptotic series expansion.
        /// </summary>
        T p = _numOps.One;

        /// <summary>
        /// The second term in the asymptotic series expansion, which improves accuracy.
        /// </summary>
        T q = _numOps.Divide(_numOps.Multiply(_numOps.Multiply(_numOps.FromDouble(4), order), _numOps.Subtract(order, _numOps.One)), _numOps.Multiply(_numOps.FromDouble(8), x));

        return _numOps.Multiply(sqrtPiOver2x, _numOps.Multiply(exp_x, _numOps.Add(p, q)));
    }
}
