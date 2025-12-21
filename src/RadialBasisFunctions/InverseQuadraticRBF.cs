namespace AiDotNet.RadialBasisFunctions;

/// <summary>
/// Implements an Inverse Quadratic Radial Basis Function (RBF) of the form 1/(1 + (er)²).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This class provides an implementation of a Radial Basis Function (RBF) that uses an inverse quadratic form
/// of f(r) = 1/(1 + (er)²), where r is the radial distance and e (epsilon) is a shape parameter
/// controlling the width of the function. The inverse quadratic RBF is infinitely differentiable and
/// decreases more slowly than the Gaussian RBF but faster than the inverse multiquadric RBF as distance increases.
/// It has properties that make it useful for scattered data interpolation and solving differential equations.
/// </para>
/// <para><b>For Beginners:</b> A Radial Basis Function (RBF) is a special type of mathematical function
/// that depends only on the distance from a center point.
/// 
/// The Inverse Quadratic RBF looks like a smooth bell-shaped curve that flattens out at larger distances.
/// At the center point (r = 0), it has its maximum value of 1, and as you move away from the center,
/// the function value gradually decreases toward zero, but never quite reaches it.
/// 
/// This RBF has a parameter called epsilon (e) that controls the shape and width of the function:
/// - A larger epsilon value creates a narrower bell curve that drops off quickly with distance
/// - A smaller epsilon value creates a wider bell curve that extends further
/// 
/// The inverse quadratic function is similar to the Gaussian RBF in shape but has "longer tails,"
/// meaning it decreases more slowly at larger distances. This property can be useful when you want
/// data points to have influence over a broader range.
/// </para>
/// </remarks>
public class InverseQuadraticRBF<T> : IRadialBasisFunction<T>
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
    /// Initializes a new instance of the <see cref="InverseQuadraticRBF{T}"/> class with a specified shape parameter.
    /// </summary>
    /// <param name="epsilon">The shape parameter, defaults to 1.0.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the Inverse Quadratic Radial Basis Function with a specified shape parameter epsilon.
    /// Epsilon controls the shape and width of the function. Larger values of epsilon result in a narrower bell curve
    /// with a faster drop-off, while smaller values result in a broader bell curve with a more gradual drop-off.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Inverse Quadratic RBF with a specific width setting.
    /// 
    /// The epsilon parameter controls the "width" of the bell curve:
    /// - Larger epsilon values (like 5.0) make the bell curve very narrow, falling off quickly as you move away from the center
    /// - Smaller epsilon values (like 0.1) make the bell curve very wide, decreasing slowly with distance
    /// 
    /// Unlike some other RBFs, the Inverse Quadratic always has a value of exactly 1 at the center (r = 0),
    /// regardless of the epsilon value. The epsilon parameter only affects how quickly the function decreases
    /// as you move away from the center.
    /// 
    /// If you're not sure what value to use, the default (epsilon = 1.0) is a good starting point
    /// that provides a moderate decrease with distance.
    /// </para>
    /// </remarks>
    public InverseQuadraticRBF(double epsilon = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _epsilon = _numOps.FromDouble(epsilon);
    }

    /// <summary>
    /// Computes the value of the Inverse Quadratic Radial Basis Function for a given radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The computed function value 1/(1 + (er)²).</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the value of the Inverse Quadratic RBF for a given radius r. The formula used is
    /// 1/(1 + (er)²), which decreases with distance. The function equals 1 at r = 0 and approaches 0
    /// as r approaches infinity.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes the "height" or "value" of the Inverse Quadratic function
    /// at a specific distance (r) from the center.
    /// 
    /// The calculation involves:
    /// 1. Multiplying the distance (r) by the epsilon parameter (er)
    /// 2. Squaring this product ((er)²)
    /// 3. Adding 1 to this squared value (1 + (er)²)
    /// 4. Dividing 1 by this sum (1/(1 + (er)²))
    /// 
    /// The result is a single number representing the function's value at the given distance.
    /// This value is always between 0 and 1:
    /// - At the center (r = 0), the value is exactly 1
    /// - As you move away from the center, the value gets smaller, approaching 0 but never quite reaching it
    /// </para>
    /// </remarks>
    public T Compute(T r)
    {
        T epsilonR = _numOps.Multiply(_epsilon, r);
        T denominator = _numOps.Add(_numOps.One, _numOps.Power(epsilonR, _numOps.FromDouble(2)));
        return _numOps.Divide(_numOps.One, denominator);
    }

    /// <summary>
    /// Computes the derivative of the Inverse Quadratic RBF with respect to the radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to r.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Inverse Quadratic RBF with respect to the radius r.
    /// The formula for the derivative is -2e²r/(1 + (er)²)², which is always negative for positive r and e,
    /// indicating that the function always decreases with distance.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes how fast the function's value changes
    /// as you move away from the center point.
    /// 
    /// The derivative tells you the "slope" or "rate of change" of the function at a specific distance.
    /// For the Inverse Quadratic RBF, the derivative is:
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
        // Derivative with respect to r: -2e²r/(1 + (er)²)²

        // Calculate er
        T epsilonR = _numOps.Multiply(_epsilon, r);

        // Calculate (er)²
        T epsilonRSquared = _numOps.Multiply(epsilonR, epsilonR);

        // Calculate 1 + (er)²
        T denominator = _numOps.Add(_numOps.One, epsilonRSquared);

        // Calculate (1 + (er)²)²
        T denominatorSquared = _numOps.Multiply(denominator, denominator);

        // Calculate e²
        T epsilonSquared = _numOps.Multiply(_epsilon, _epsilon);

        // Calculate 2e²r
        T twoEpsilonSquaredR = _numOps.Multiply(
            _numOps.Multiply(_numOps.FromDouble(2.0), epsilonSquared),
            r
        );

        // Calculate -2e²r
        T negativeTwoEpsilonSquaredR = _numOps.Negate(twoEpsilonSquaredR);

        // Return -2e²r/(1 + (er)²)²
        return _numOps.Divide(negativeTwoEpsilonSquaredR, denominatorSquared);
    }

    /// <summary>
    /// Computes the derivative of the Inverse Quadratic RBF with respect to the shape parameter epsilon.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to epsilon.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Inverse Quadratic RBF with respect to the shape parameter epsilon.
    /// The formula for this derivative is -2er²/(1 + (er)²)². The sign of this derivative depends on e and r:
    /// it is negative for positive values, indicating that increasing epsilon decreases the function value
    /// at any non-zero radius.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how the function's value would change
    /// if you were to adjust the shape parameter (epsilon).
    /// 
    /// This is particularly important in machine learning applications:
    /// - When training a model with Inverse Quadratic RBFs, we often need to adjust epsilon to fit the data better
    /// - This derivative tells us exactly how changing epsilon affects the output of the function
    /// - With this information, learning algorithms can automatically find the optimal value of epsilon
    /// 
    /// The derivative is zero at the center point (r = 0) and negative everywhere else, which means that 
    /// increasing the epsilon parameter:
    /// - Has no effect on the function value at the center
    /// - Decreases the function value at all other points
    /// </para>
    /// </remarks>
    public T ComputeWidthDerivative(T r)
    {
        // Derivative with respect to e: -2er²/(1 + (er)²)²

        // Calculate er
        T epsilonR = _numOps.Multiply(_epsilon, r);

        // Calculate (er)²
        T epsilonRSquared = _numOps.Multiply(epsilonR, epsilonR);

        // Calculate 1 + (er)²
        T denominator = _numOps.Add(_numOps.One, epsilonRSquared);

        // Calculate (1 + (er)²)²
        T denominatorSquared = _numOps.Multiply(denominator, denominator);

        // Calculate r²
        T rSquared = _numOps.Multiply(r, r);

        // Calculate 2er²
        T twoEpsilonRSquared = _numOps.Multiply(
            _numOps.Multiply(_numOps.FromDouble(2.0), _epsilon),
            rSquared
        );

        // Calculate -2er²
        T negativeTwoEpsilonRSquared = _numOps.Negate(twoEpsilonRSquared);

        // Return -2er²/(1 + (er)²)²
        return _numOps.Divide(negativeTwoEpsilonRSquared, denominatorSquared);
    }
}
