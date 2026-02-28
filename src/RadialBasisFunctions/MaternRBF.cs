namespace AiDotNet.RadialBasisFunctions;

/// <summary>
/// Implements a Matérn Radial Basis Function (RBF) that provides a flexible family of kernels.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This class implements the Matérn family of radial basis functions, which are defined using modified Bessel functions
/// and provide a flexible set of kernels with varying degrees of smoothness. The Matérn RBF is defined as:
/// f(r) = [2^(1-?)/G(?)] × (v(2?)r/l)^? × K_?(v(2?)r/l)
/// where r is the radial distance, ? (nu) is a smoothness parameter, l is the length scale parameter,
/// G is the Gamma function, and K_? is the modified Bessel function of the second kind of order ?.
/// </para>
/// <para>
/// The Matérn function is commonly used in spatial statistics, machine learning, and geostatistics.
/// It generalizes many other RBFs; for example, when ? ? 8, it becomes the Gaussian RBF, and
/// when ? = 0.5, it becomes the exponential RBF. Special half-integer values of ? (0.5, 1.5, 2.5)
/// result in simpler forms that can be computed without Bessel functions.
/// </para>
/// <para><b>For Beginners:</b> A Radial Basis Function (RBF) is a special type of mathematical function
/// that depends only on the distance from a center point.
/// 
/// The Matérn RBF is like a "Swiss Army knife" of radial basis functions - it provides a whole family
/// of different shapes by adjusting a parameter called nu (?). This makes it very flexible for
/// different types of data.
/// 
/// This RBF has two main parameters:
/// - nu (?): Controls the smoothness of the function. Common values are 0.5, 1.5, and 2.5
/// - lengthScale (l): Controls how quickly the function decreases with distance
/// 
/// When nu = 0.5, the function decreases rapidly (exponentially) with distance.
/// When nu = 1.5, the function is smoother and decreases more gradually.
/// As nu increases, the function becomes even smoother, approaching a bell curve shape.
/// 
/// The Matérn RBF is popular in machine learning and statistics because you can adjust its
/// smoothness to match the characteristics of your data.
/// </para>
/// </remarks>
public class MaternRBF<T> : IRadialBasisFunction<T>
{
    /// <summary>
    /// The numeric operations provider for type T, used for mathematical calculations.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// The smoothness parameter (nu) controlling the differentiability of the function.
    /// </summary>
    private readonly double _nu;

    /// <summary>
    /// The length scale parameter controlling the width of the function.
    /// </summary>
    private readonly T _lengthScale;

    /// <summary>
    /// Initializes a new instance of the <see cref="MaternRBF{T}"/> class with specified parameters.
    /// </summary>
    /// <param name="nu">The smoothness parameter, defaults to 1.5.</param>
    /// <param name="lengthScale">The length scale parameter, defaults to 1.0.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the Matérn Radial Basis Function with specified nu and length scale parameters.
    /// The nu parameter controls the smoothness of the function, with higher values giving smoother functions.
    /// Common values for nu are 0.5, 1.5, and 2.5. The length scale parameter controls how quickly the 
    /// function decreases with distance.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Matérn RBF with specific smoothness and width settings.
    /// 
    /// The two parameters you can adjust are:
    /// - nu: Controls how smooth the function is. Smaller values (like 0.5) make it less smooth, while
    ///   larger values make it more smooth. Common values are:
    ///   - 0.5: Makes an exponential function (sharper)
    ///   - 1.5: Makes a function that decreases more smoothly (default)
    ///   - 2.5: Makes an even smoother function
    /// 
    /// - lengthScale: Controls the "width" of the function:
    ///   - Smaller values make the function decrease rapidly with distance
    ///   - Larger values make the function decrease more gradually
    /// 
    /// If you're not sure what values to use, the defaults (nu = 1.5, lengthScale = 1.0) are a good starting point
    /// that provide moderate smoothness and width.
    /// </para>
    /// </remarks>
    public MaternRBF(double nu = 1.5, double lengthScale = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _nu = nu;
        _lengthScale = _numOps.FromDouble(lengthScale);
    }

    /// <summary>
    /// Computes the value of the Matérn Radial Basis Function for a given radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The computed function value.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the value of the Matérn RBF for a given radius r. The formula used is
    /// f(r) = [2^(1-?)/G(?)] × (v(2?)r/l)^? × K_?(v(2?)r/l), where G is the Gamma function
    /// and K_? is the modified Bessel function of the second kind of order ?.
    /// For r = 0, the function returns 1 to avoid numerical issues.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes the "height" or "value" of the Matérn function
    /// at a specific distance (r) from the center.
    /// 
    /// The calculation involves several steps including specialized mathematical functions like
    /// the Bessel function and the Gamma function. These complex calculations are handled automatically
    /// for you.
    /// 
    /// The result is a single number representing the function's value at the given distance:
    /// - At the center (r = 0), the value is exactly 1
    /// - As you move away from the center, the value decreases in a way that depends on the nu parameter:
    ///   - For nu = 0.5, it decreases quickly (exponentially)
    ///   - For nu = 1.5, it decreases more gradually
    ///   - For higher nu values, it decreases even more smoothly
    /// </para>
    /// </remarks>
    public T Compute(T r)
    {
        T scaledR = _numOps.Divide(r, _lengthScale);

        if (_numOps.Equals(scaledR, _numOps.Zero))
        {
            return _numOps.One;
        }

        // Use closed-form expressions for half-integer nu values (more numerically stable)
        double sqrt2nu = Math.Sqrt(2 * _nu);
        double scaledRDouble = Convert.ToDouble(scaledR);
        double x = sqrt2nu * scaledRDouble;

        // For nu = 0.5: k(r) = exp(-x)
        if (Math.Abs(_nu - 0.5) < 1e-10)
        {
            return _numOps.Exp(_numOps.Negate(_numOps.FromDouble(x)));
        }

        // For nu = 1.5: k(r) = (1 + x) * exp(-x)
        if (Math.Abs(_nu - 1.5) < 1e-10)
        {
            T expTerm = _numOps.Exp(_numOps.Negate(_numOps.FromDouble(x)));
            return _numOps.Multiply(_numOps.Add(_numOps.One, _numOps.FromDouble(x)), expTerm);
        }

        // For nu = 2.5: k(r) = (1 + x + x^2/3) * exp(-x)
        if (Math.Abs(_nu - 2.5) < 1e-10)
        {
            T expTerm = _numOps.Exp(_numOps.Negate(_numOps.FromDouble(x)));
            T xSquaredOver3 = _numOps.FromDouble(x * x / 3.0);
            return _numOps.Multiply(
                _numOps.Add(_numOps.Add(_numOps.One, _numOps.FromDouble(x)), xSquaredOver3),
                expTerm);
        }

        // General case using Bessel functions
        T term1 = _numOps.Power(_numOps.FromDouble(2), _numOps.FromDouble(1 - _nu));
        T term2 = _numOps.FromDouble(1 / MathHelper.Gamma(_nu));
        T term3 = _numOps.Power(_numOps.FromDouble(x), _numOps.FromDouble(_nu));
        T term4 = _numOps.FromDouble(MathHelper.BesselK(_nu, x));

        return _numOps.Multiply(_numOps.Multiply(_numOps.Multiply(term1, term2), term3), term4);
    }

    /// <summary>
    /// Computes the derivative of the Matérn RBF with respect to the radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to r.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Matérn RBF with respect to the radius r.
    /// The derivative has different formulations depending on the value of ?. Special cases are
    /// implemented for ? = 0.5 and ? = 1.5, which have simpler forms. For other values of ?,
    /// the derivative is computed using the general formula involving Bessel functions.
    /// At r = 0, the derivative is 0 due to symmetry.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes how fast the function's value changes
    /// as you move away from the center point.
    /// 
    /// The derivative tells you the "slope" or "rate of change" of the function at a specific distance.
    /// For the Matérn RBF:
    /// - At the center point (r = 0), the derivative is zero (flat)
    /// - As you move away from the center, the function starts to decrease, so the derivative becomes negative
    /// - The exact behavior of the derivative depends on the nu parameter
    /// 
    /// This method handles different cases based on the nu value:
    /// - For nu = 0.5, there's a simpler formula that can be calculated more efficiently
    /// - For nu = 1.5, there's another simplified formula
    /// - For other values of nu, a more general (and complex) calculation is used
    /// 
    /// This derivative is useful in machine learning applications when optimizing parameters
    /// using gradient-based methods.
    /// </para>
    /// </remarks>
    public T ComputeDerivative(T r)
    {
        // For r = 0, the derivative is 0 due to symmetry
        if (_numOps.Equals(r, _numOps.Zero))
        {
            return _numOps.Zero;
        }

        T scaledR = _numOps.Divide(r, _lengthScale);
        double sqrt2nu = Math.Sqrt(2 * _nu);
        T sqrtTerm = _numOps.FromDouble(sqrt2nu);
        T x = _numOps.Multiply(sqrtTerm, scaledR); // v(2?)r/l

        // Common terms from the original function
        T term1 = _numOps.Power(_numOps.FromDouble(2), _numOps.FromDouble(1 - _nu));
        T term2 = _numOps.FromDouble(1 / MathHelper.Gamma(_nu));

        // For special case ? = 0.5, the derivative has a simpler form
        if (Math.Abs(_nu - 0.5) < 1e-10)
        {
            // For ? = 0.5, K_0.5(x) = v(p/2x) * e^(-x)
            // The derivative simplifies considerably
            T expTerm = _numOps.Exp(_numOps.Negate(x));
            return _numOps.Multiply(
                _numOps.Negate(_numOps.Divide(sqrtTerm, _lengthScale)),
                expTerm
            );
        }

        // For special case ? = 1.5, the derivative also has a simpler form
        if (Math.Abs(_nu - 1.5) < 1e-10)
        {
            // For ? = 1.5: k(r) = (1+x)*exp(-x)
            // dk/dx = exp(-x) - (1+x)*exp(-x) = -x*exp(-x)
            // dk/dr = dk/dx * dx/dr = -x*exp(-x) * ?(2?)/l
            T expTerm = _numOps.Exp(_numOps.Negate(x));
            T factor = _numOps.Multiply(
                _numOps.FromDouble(sqrt2nu / Convert.ToDouble(_lengthScale)),
                x
            );
            return _numOps.Multiply(_numOps.Negate(factor), expTerm);
        }

        // For special case ? = 2.5, use closed-form derivative
        if (Math.Abs(_nu - 2.5) < 1e-10)
        {
            // For ? = 2.5: k(r) = (1 + x + x²/3)*exp(-x)
            // dk/dx = (1 + 2x/3)*exp(-x) - (1 + x + x²/3)*exp(-x)
            //       = exp(-x)*(-x/3 - x²/3) = -exp(-x)*x*(1+x)/3
            // dk/dr = dk/dx * dx/dr = -exp(-x)*x*(1+x)/3 * ?(2?)/l
            T expTerm = _numOps.Exp(_numOps.Negate(x));
            T xTimesOnePlusX = _numOps.Multiply(x, _numOps.Add(_numOps.One, x));
            T dkdx = _numOps.Negate(_numOps.Divide(xTimesOnePlusX, _numOps.FromDouble(3.0)));
            T chainFactor = _numOps.Divide(sqrtTerm, _lengthScale);
            return _numOps.Multiply(_numOps.Multiply(dkdx, chainFactor), expTerm);
        }

        // For general case, we need to use the recurrence relation for Bessel functions
        // d/dr[K_?(x)] = -K_(?-1)(x) - (?/x)K_?(x) where x = v(2?)r/l

        double xDouble = Convert.ToDouble(x);
        double besselKnu = MathHelper.BesselK(_nu, xDouble);
        double besselKnuMinus1 = MathHelper.BesselK(_nu - 1, xDouble);

        // Calculate d/dx[K_?(x)]
        T dBesselK = _numOps.Add(
            _numOps.Negate(_numOps.FromDouble(besselKnuMinus1)),
            _numOps.Multiply(
                _numOps.Negate(_numOps.FromDouble(_nu / xDouble)),
                _numOps.FromDouble(besselKnu)
            )
        );

        // Calculate d/dr[x] = v(2?)/l
        T dxdr = _numOps.Divide(sqrtTerm, _lengthScale);

        // Calculate d/dr[x^?] = ?*x^(?-1) * d/dr[x]
        T dxPowerNu = _numOps.Multiply(
            _numOps.Multiply(
                _numOps.FromDouble(_nu),
                _numOps.Power(x, _numOps.FromDouble(_nu - 1))
            ),
            dxdr
        );

        // Apply product rule: d/dr[x^? * K_?(x)] = x^? * d/dr[K_?(x)] + K_?(x) * d/dr[x^?]
        T term3 = _numOps.Power(x, _numOps.FromDouble(_nu));
        T term4 = _numOps.FromDouble(besselKnu);

        T productRule = _numOps.Add(
            _numOps.Multiply(term3, _numOps.Multiply(dBesselK, dxdr)),
            _numOps.Multiply(term4, dxPowerNu)
        );

        // Combine with the constant terms
        return _numOps.Multiply(_numOps.Multiply(term1, term2), productRule);
    }

    /// <summary>
    /// Computes the derivative of the Matérn RBF with respect to the length scale parameter.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to the length scale.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Matérn RBF with respect to the length scale parameter.
    /// The derivative has different formulations depending on the value of ?. Special cases are
    /// implemented for ? = 0.5 and ? = 1.5, which have simpler forms. For other values of ?,
    /// the derivative is computed using the general formula involving Bessel functions.
    /// At r = 0, the derivative is 0.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how the function's value would change
    /// if you were to adjust the length scale parameter.
    /// 
    /// This is particularly important in machine learning applications:
    /// - When training a model with Matérn RBFs, we often need to adjust the length scale to fit the data better
    /// - This derivative tells us exactly how changing the length scale affects the output of the function
    /// - With this information, learning algorithms can automatically find the optimal value of length scale
    /// 
    /// Like the radius derivative, this method handles different cases based on the nu value:
    /// - For nu = 0.5, there's a simpler formula
    /// - For nu = 1.5, there's another simplified formula
    /// - For other values of nu, a more general calculation is used
    /// 
    /// At the center point (r = 0), the derivative is always zero because changing the length scale
    /// doesn't affect the function value at the center (which is always 1).
    /// </para>
    /// </remarks>
    public T ComputeWidthDerivative(T r)
    {
        // For r = 0, the width derivative is 0
        if (_numOps.Equals(r, _numOps.Zero))
        {
            return _numOps.Zero;
        }

        T scaledR = _numOps.Divide(r, _lengthScale);
        double sqrt2nu = Math.Sqrt(2 * _nu);
        T sqrtTerm = _numOps.FromDouble(sqrt2nu);
        T x = _numOps.Multiply(sqrtTerm, scaledR); // v(2?)r/l

        // Common terms from the original function
        T term1 = _numOps.Power(_numOps.FromDouble(2), _numOps.FromDouble(1 - _nu));
        T term2 = _numOps.FromDouble(1 / MathHelper.Gamma(_nu));
        T term3 = _numOps.Power(x, _numOps.FromDouble(_nu));

        double xDouble = Convert.ToDouble(x);
        double besselKnu = MathHelper.BesselK(_nu, xDouble);
        T term4 = _numOps.FromDouble(besselKnu);

        // The width derivative involves d/dl[x] = -v(2?)r/l²
        T dxdl = _numOps.Negate(_numOps.Divide(x, _lengthScale));

        // For special case ? = 0.5, the width derivative has a simpler form
        if (Math.Abs(_nu - 0.5) < 1e-10)
        {
            // For ? = 0.5: k(r) = exp(-x), x = r/l
            // dk/dl = dk/dx * dx/dl = (-exp(-x)) * (-x/l) = (x/l)*exp(-x)
            // Since dxdl = -x/l, we have dk/dl = -dxdl * exp(-x)
            T expTerm = _numOps.Exp(_numOps.Negate(x));
            return _numOps.Multiply(_numOps.Negate(dxdl), expTerm);
        }

        // For special case ? = 1.5, the width derivative also has a simpler form
        if (Math.Abs(_nu - 1.5) < 1e-10)
        {
            // For ? = 1.5: k(r) = (1+x)*exp(-x)
            // dk/dx = -x*exp(-x)
            // dk/dl = dk/dx * dx/dl = (-x*exp(-x)) * (-x/l) = x²/l * exp(-x)
            // Since dxdl = -x/l, we have dk/dl = x * (-dxdl) * exp(-x)
            T expTerm = _numOps.Exp(_numOps.Negate(x));
            return _numOps.Multiply(x, _numOps.Multiply(_numOps.Negate(dxdl), expTerm));
        }

        // For general case, we need to use the recurrence relation for Bessel functions
        double besselKnuMinus1 = MathHelper.BesselK(_nu - 1, xDouble);

        // Calculate d/dx[K_?(x)]
        T dBesselK = _numOps.Add(
            _numOps.Negate(_numOps.FromDouble(besselKnuMinus1)),
            _numOps.Multiply(
                _numOps.Negate(_numOps.FromDouble(_nu / xDouble)),
                term4
            )
        );

        // Calculate d/dl[x^?] = ?*x^(?-1) * d/dl[x]
        T dxPowerNu = _numOps.Multiply(
            _numOps.Multiply(
                _numOps.FromDouble(_nu),
                _numOps.Power(x, _numOps.FromDouble(_nu - 1))
            ),
            dxdl
        );

        // Apply product rule: d/dl[x^? * K_?(x)] = x^? * d/dl[K_?(x)] + K_?(x) * d/dl[x^?]
        T productRule = _numOps.Add(
            _numOps.Multiply(term3, _numOps.Multiply(dBesselK, dxdl)),
            _numOps.Multiply(term4, dxPowerNu)
        );

        // Combine with the constant terms
        return _numOps.Multiply(_numOps.Multiply(term1, term2), productRule);
    }
}
