namespace AiDotNet.RadialBasisFunctions;

/// <summary>
/// Implements a Rational Quadratic Radial Basis Function (RBF) of the form 1 - r²/(r² + e²).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This class provides an implementation of a Radial Basis Function (RBF) that uses a rational quadratic form
/// of f(r) = 1 - r²/(r² + e²), where r is the radial distance and e (epsilon) is a shape parameter
/// controlling the width of the function. The rational quadratic RBF is infinitely differentiable and
/// decreases from 1 at r = 0 to 0 as r approaches infinity. It has a smoother and more gradual decay
/// compared to the Gaussian RBF, which can be beneficial in certain applications.
/// </para>
/// <para>
/// This function forms a proper correlation function and has applications in various fields including
/// machine learning, geostatistics, and scattered data interpolation. The rational quadratic is also
/// related to the Student-t process in statistical modeling.
/// </para>
/// <para><b>For Beginners:</b> A Radial Basis Function (RBF) is a special type of mathematical function
/// that depends only on the distance from a center point.
/// 
/// The Rational Quadratic RBF is shaped like a hill or a bell curve - it starts at its highest point
/// in the center (with a value of 1) and gradually decreases in all directions, eventually approaching
/// zero but never quite reaching it. Compared to the Gaussian RBF, it decreases more slowly as you
/// move away from the center, giving it "fatter tails."
/// 
/// This RBF has a parameter called epsilon (e) that controls the width of the hill:
/// - A larger epsilon value creates a wider hill that decreases more gradually with distance
/// - A smaller epsilon value creates a narrower hill that drops off more quickly
/// 
/// The rational quadratic function is useful when you want a smooth function that doesn't decay
/// as rapidly as the Gaussian. It can capture longer-range dependencies in your data, making it
/// valuable in many machine learning and spatial statistics applications.
/// </para>
/// </remarks>
public class RationalQuadraticRBF<T> : IRadialBasisFunction<T>
{
    /// <summary>
    /// The shape parameter (epsilon) controlling the width of the function.
    /// </summary>
    private readonly T _epsilon;

    /// <summary>
    /// The numeric operations provider for type T, used for mathematical calculations.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="RationalQuadraticRBF{T}"/> class with a specified shape parameter.
    /// </summary>
    /// <param name="epsilon">The shape parameter, defaults to 1.0.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the Rational Quadratic Radial Basis Function with a specified shape parameter epsilon.
    /// Epsilon controls the width and decay rate of the function. Larger values of epsilon result in a wider function
    /// with a more gradual decay, while smaller values result in a narrower function with a faster decay.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Rational Quadratic RBF with a specific width setting.
    /// 
    /// The epsilon parameter controls the "width" of the function:
    /// - Larger epsilon values (like 5.0) make the function very wide, decreasing slowly with distance
    /// - Smaller epsilon values (like 0.1) make the function very narrow, falling off quickly as you move away from the center
    /// 
    /// If you're not sure what value to use, the default (epsilon = 1.0) is a good starting point
    /// that provides a moderate decay with distance.
    /// </para>
    /// </remarks>
    public RationalQuadraticRBF(double epsilon = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _epsilon = _numOps.FromDouble(epsilon);
    }

    /// <summary>
    /// Computes the value of the Rational Quadratic Radial Basis Function for a given radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The computed function value 1 - r²/(r² + e²).</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the value of the Rational Quadratic RBF for a given radius r. The formula used is
    /// 1 - r²/(r² + e²), which decreases with distance. The function equals 1 at r = 0 and approaches 0
    /// as r approaches infinity.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes the "height" or "value" of the Rational Quadratic function
    /// at a specific distance (r) from the center.
    /// 
    /// The calculation involves:
    /// 1. Squaring the distance (r² = r * r)
    /// 2. Squaring the epsilon parameter (e² = e * e)
    /// 3. Adding these squared values (r² + e²)
    /// 4. Dividing r² by this sum (r²/(r² + e²))
    /// 5. Subtracting this fraction from 1 (1 - r²/(r² + e²))
    /// 
    /// The result is a single number representing the function's value at the given distance.
    /// This value is always between 0 and 1:
    /// - At the center (r = 0), the value is exactly 1
    /// - As you move away from the center, the value gets smaller, approaching 0 but never quite reaching it
    /// - The rate of decrease depends on the epsilon parameter
    /// </para>
    /// </remarks>
    public T Compute(T r)
    {
        T rSquared = _numOps.Power(r, _numOps.FromDouble(2));
        T epsilonSquared = _numOps.Power(_epsilon, _numOps.FromDouble(2));
        T denominator = _numOps.Add(rSquared, epsilonSquared);
        T fraction = _numOps.Divide(rSquared, denominator);
        return _numOps.Subtract(_numOps.One, fraction);
    }

    /// <summary>
    /// Computes the derivative of the Rational Quadratic RBF with respect to the radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to r.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Rational Quadratic RBF with respect to the radius r.
    /// The formula for the derivative is -2re²/(r² + e²)², which is always negative for positive r and e,
    /// indicating that the function always decreases with distance.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes how fast the function's value changes
    /// as you move away from the center point.
    /// 
    /// The derivative tells you the "slope" or "rate of change" of the function at a specific distance.
    /// For the Rational Quadratic RBF, the derivative is:
    /// - Zero at the center point (r = 0)
    /// - Negative for any positive distance, meaning the function always decreases as you move away from the center
    /// - At first, the magnitude of the derivative increases with distance (getting steeper)
    /// - After a certain point, the magnitude starts decreasing (the slope becomes less steep)
    /// 
    /// This derivative is useful in machine learning applications when optimizing parameters
    /// using gradient-based methods.
    /// </para>
    /// </remarks>
    public T ComputeDerivative(T r)
    {
        // Derivative with respect to r: -2re²/(r² + e²)²

        // Calculate r²
        T rSquared = _numOps.Multiply(r, r);

        // Calculate e²
        T epsilonSquared = _numOps.Multiply(_epsilon, _epsilon);

        // Calculate r² + e²
        T denominator = _numOps.Add(rSquared, epsilonSquared);

        // Calculate (r² + e²)²
        T denominatorSquared = _numOps.Multiply(denominator, denominator);

        // Calculate 2re²
        T twoREpsilonSquared = _numOps.Multiply(
            _numOps.Multiply(_numOps.FromDouble(2.0), r),
            epsilonSquared
        );

        // Calculate -2re²
        T negativeTwoREpsilonSquared = _numOps.Negate(twoREpsilonSquared);

        // Return -2re²/(r² + e²)²
        return _numOps.Divide(negativeTwoREpsilonSquared, denominatorSquared);
    }

    /// <summary>
    /// Computes the derivative of the Rational Quadratic RBF with respect to the shape parameter epsilon.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to epsilon.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Rational Quadratic RBF with respect to the shape parameter epsilon.
    /// The formula for this derivative is 2er²/(r² + e²)². The sign of this derivative is positive for positive e and r,
    /// indicating that increasing epsilon increases the function value at any non-zero radius.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how the function's value would change
    /// if you were to adjust the shape parameter (epsilon).
    /// 
    /// This is particularly important in machine learning applications:
    /// - When training a model with Rational Quadratic RBFs, we often need to adjust epsilon to fit the data better
    /// - This derivative tells us exactly how changing epsilon affects the output of the function
    /// - With this information, learning algorithms can automatically find the optimal value of epsilon
    /// 
    /// The derivative is zero at the center point (r = 0) and positive everywhere else, which means that 
    /// increasing the epsilon parameter:
    /// - Has no effect on the function value at the center (which is always 1)
    /// - Increases the function value at all other points (makes the function decay more slowly with distance)
    /// </para>
    /// </remarks>
    public T ComputeWidthDerivative(T r)
    {
        // Derivative with respect to e: 2er²/(r² + e²)²

        // Calculate r²
        T rSquared = _numOps.Multiply(r, r);

        // Calculate e²
        T epsilonSquared = _numOps.Multiply(_epsilon, _epsilon);

        // Calculate r² + e²
        T denominator = _numOps.Add(rSquared, epsilonSquared);

        // Calculate (r² + e²)²
        T denominatorSquared = _numOps.Multiply(denominator, denominator);

        // Calculate 2er²
        T twoEpsilonRSquared = _numOps.Multiply(
            _numOps.Multiply(_numOps.FromDouble(2.0), _epsilon),
            rSquared
        );

        // Return 2er²/(r² + e²)²
        return _numOps.Divide(twoEpsilonRSquared, denominatorSquared);
    }
}
