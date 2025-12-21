namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Bessel kernel function for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Bessel kernel is based on the Bessel functions of the first kind, which are solutions to 
/// Bessel's differential equation. This kernel is particularly useful for problems involving 
/// circular or cylindrical data patterns.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The Bessel kernel is a specialized similarity measure that works well for certain types of data,
/// especially data with circular or radial patterns (like sound waves, vibrations, or heat distribution
/// in circular objects). It's named after Friedrich Bessel, a mathematician who studied these special
/// mathematical functions in the 19th century.
/// </para>
/// <para>
/// Think of the Bessel kernel as a way to compare data points while taking into account their
/// "wave-like" or "oscillating" properties, similar to how ripples spread in water.
/// </para>
/// </remarks>
public class BesselKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The order of the Bessel function to use in the kernel calculation.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The order parameter determines which specific Bessel function to use.
    /// Different orders capture different types of oscillatory patterns in your data.
    /// Order 0 (the default) is the most commonly used and works well for many applications.
    /// </remarks>
    private readonly T _order;

    /// <summary>
    /// The scaling parameter that controls the width of the kernel.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Think of sigma as a "zoom level" for your similarity measurement.
    /// A smaller sigma makes the kernel more sensitive to small differences between data points,
    /// while a larger sigma makes the kernel more tolerant of differences.
    /// </remarks>
    private readonly T _sigma;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the Bessel kernel with optional parameters.
    /// </summary>
    /// <param name="order">The order of the Bessel function to use. Default is 0.</param>
    /// <param name="sigma">The scaling parameter that controls the width of the kernel. Default is 1.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Bessel kernel with your chosen settings.
    /// If you don't specify any settings, it will use default values that work well for many problems.
    /// </para>
    /// <para>
    /// The order parameter (default 0) determines which specific Bessel function to use in the calculations.
    /// The sigma parameter (default 1) controls how quickly the similarity decreases as points get farther apart.
    /// </para>
    /// </remarks>
    public BesselKernel(T? order = default, T? sigma = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _order = order ?? _numOps.Zero; // Default order is 0 (Bessel function of the first kind, order 0)
        _sigma = sigma ?? _numOps.One;
    }

    /// <summary>
    /// Calculates the Bessel kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Bessel kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. Finding the distance between the two points
    /// 2. Scaling this distance using the sigma parameter
    /// 3. Applying the Bessel function to this scaled distance
    /// 4. Dividing by the scaled distance raised to the power of the order
    /// </para>
    /// <para>
    /// Higher output values indicate greater similarity between the vectors.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T norm = x1.EuclideanDistance(x2);
        T scaledNorm = _numOps.Divide(norm, _sigma);

        return _numOps.Divide(BesselFunction(_order, scaledNorm), _numOps.Power(scaledNorm, _order));
    }

    /// <summary>
    /// Calculates the value of the Bessel function of the first kind.
    /// </summary>
    /// <param name="order">The order of the Bessel function.</param>
    /// <param name="x">The input value.</param>
    /// <returns>The value of the Bessel function J_order(x).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates a special mathematical function called the "Bessel function"
    /// which is used in the kernel calculation. Bessel functions are solutions to a specific type of
    /// differential equation that appears in many physics problems involving waves and vibrations.
    /// </para>
    /// <para>
    /// The method chooses the most efficient calculation approach based on the input values:
    /// - For negative x, it throws an exception (Bessel functions are defined for non-negative inputs)
    /// - For negative orders, it uses a mathematical relationship to convert to positive orders
    /// - For x = 0, it returns 1 if order = 0, otherwise 0
    /// - For small x values, it uses a series expansion
    /// - For large x values, it uses an asymptotic expansion
    /// </para>
    /// </remarks>
    private T BesselFunction(T order, T x)
    {
        // Convert order and x to double for easier comparisons
        double orderDouble = Convert.ToDouble(order);
        double xDouble = Convert.ToDouble(x);

        if (xDouble < 0)
        {
            throw new ArgumentException("x must be non-negative for Bessel function");
        }

        if (orderDouble < 0)
        {
            // Use the relation J_(-n)(x) = (-1)^n * J_n(x)
            T result = BesselFunction(_numOps.Abs(order), x);
            return orderDouble % 2 == 0 ? result : _numOps.Negate(result);
        }

        if (xDouble == 0)
        {
            return orderDouble == 0 ? _numOps.One : _numOps.Zero;
        }

        if (xDouble <= 12 || xDouble < Math.Abs(orderDouble))
        {
            // Use series expansion for small x or when x < |order|
            return BesselFunctionSeries(order, x);
        }
        else
        {
            // Use asymptotic expansion for large x
            return BesselFunctionAsymptotic(order, x);
        }
    }

    /// <summary>
    /// Calculates the Bessel function using a series expansion approach.
    /// </summary>
    /// <param name="order">The order of the Bessel function.</param>
    /// <param name="x">The input value.</param>
    /// <returns>The value of the Bessel function calculated using series expansion.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates the Bessel function using a mathematical technique called
    /// "series expansion," which is like breaking down a complex calculation into a sum of simpler terms.
    /// </para>
    /// <para>
    /// The series expansion is particularly accurate for smaller input values. The method adds up terms
    /// until the result is precise enough (when additional terms become extremely small).
    /// </para>
    /// <para>
    /// This approach is similar to how you might approximate p by adding more and more decimal places.
    /// </para>
    /// </remarks>
    private T BesselFunctionSeries(T order, T x)
    {
        int maxIterations = 100;
        T sum = _numOps.Zero;
        T term = _numOps.One;
        T factorial = _numOps.One;
        T xSquaredOver4 = _numOps.Divide(_numOps.Square(x), _numOps.FromDouble(4));

        for (int k = 0; k < maxIterations; k++)
        {
            sum = _numOps.Add(sum, term);

            term = _numOps.Divide(
                _numOps.Multiply(_numOps.Negate(term), xSquaredOver4),
                _numOps.Multiply(factorial, _numOps.Add(order, _numOps.FromDouble(k + 1)))
            );

            if (_numOps.LessThan(_numOps.Abs(term), _numOps.FromDouble(1e-15)))
            {
                break;
            }

            factorial = _numOps.Multiply(factorial, _numOps.FromDouble(k + 1));
        }

        return _numOps.Multiply(sum, _numOps.Power(_numOps.Divide(x, _numOps.FromDouble(2)), order));
    }

    /// <summary>
    /// Calculates the Bessel function using an asymptotic expansion approach.
    /// </summary>
    /// <param name="order">The order of the Bessel function.</param>
    /// <param name="x">The input value.</param>
    /// <returns>The value of the Bessel function calculated using asymptotic expansion.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates the Bessel function using a mathematical technique called
    /// "asymptotic expansion," which is a special approximation that works well for large input values.
    /// </para>
    /// <para>
    /// Think of it like using a shortcut formula that becomes more accurate as the input gets larger.
    /// For large values, this approach is both faster and more numerically stable than the series expansion.
    /// </para>
    /// <para>
    /// The asymptotic expansion approximates the Bessel function as a combination of sine and cosine waves
    /// with specific amplitudes, which matches the oscillatory behavior of Bessel functions for large inputs.
    /// </para>
    /// </remarks>
    private T BesselFunctionAsymptotic(T order, T x)
    {
        T mu = _numOps.Subtract(_numOps.Multiply(order, order), _numOps.FromDouble(0.25));
        T theta = _numOps.Subtract(x, _numOps.Multiply(_numOps.FromDouble(0.25 * Math.PI), _numOps.Add(_numOps.Multiply(_numOps.FromDouble(2), order), _numOps.One)));

        T p = _numOps.One;
        T q = _numOps.Divide(mu, _numOps.Multiply(_numOps.FromDouble(8), x));

        T cosTheta = MathHelper.Cos(theta);
        T sinTheta = MathHelper.Sin(theta);

        T sqrtX = _numOps.Sqrt(x);
        T sqrtPi = _numOps.Sqrt(_numOps.FromDouble(Math.PI));
        T factor = _numOps.Divide(_numOps.Sqrt(_numOps.FromDouble(2)), _numOps.Multiply(sqrtPi, sqrtX));

        return _numOps.Multiply(factor, _numOps.Add(_numOps.Multiply(p, cosTheta), _numOps.Multiply(q, sinTheta)));
    }
}
