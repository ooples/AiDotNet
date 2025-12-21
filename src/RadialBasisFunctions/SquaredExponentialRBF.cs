namespace AiDotNet.RadialBasisFunctions;

/// <summary>
/// Implements a Squared Exponential (Gaussian) Radial Basis Function (RBF) of the form exp(-(er)²).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This class provides an implementation of a Radial Basis Function (RBF) that uses a squared exponential form
/// of f(r) = exp(-(er)²), where r is the radial distance and e (epsilon) is a shape parameter
/// controlling the width of the function. The squared exponential RBF, also known as the Gaussian RBF,
/// is one of the most widely used RBFs due to its smoothness properties. It is infinitely differentiable
/// and has exponential decay, making it suitable for a wide range of applications in machine learning,
/// interpolation, and function approximation.
/// </para>
/// <para>
/// The squared exponential RBF corresponds to a Gaussian probability distribution and has the unique
/// property of being a universal approximator, meaning it can approximate any continuous function to
/// arbitrary precision given sufficient basis functions. It is also the only RBF that is both rotation
/// and translation invariant.
/// </para>
/// <para><b>For Beginners:</b> A Radial Basis Function (RBF) is a special type of mathematical function
/// that depends only on the distance from a center point.
/// 
/// The Squared Exponential RBF, also commonly known as the Gaussian RBF, looks like a bell curve or
/// a mountain peak - it's at its highest at the center point (with a value of 1) and gradually decreases
/// in all directions, eventually approaching zero but never quite reaching it.
/// 
/// This RBF has a parameter called epsilon (e) that controls the width of the bell curve:
/// - A larger epsilon value creates a narrower bell curve that drops off quickly with distance
/// - A smaller epsilon value creates a wider bell curve that extends further
/// 
/// The squared exponential is the most popular RBF for many applications because:
/// - It's very smooth (it has derivatives of all orders)
/// - It has a simple mathematical form and properties
/// - Its shape resembles many natural processes and distributions
/// 
/// If you're familiar with statistics, it has the same shape as the Gaussian (normal) distribution
/// from probability theory.
/// </para>
/// </remarks>
public class SquaredExponentialRBF<T> : IRadialBasisFunction<T>
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
    /// Initializes a new instance of the <see cref="SquaredExponentialRBF{T}"/> class with a specified shape parameter.
    /// </summary>
    /// <param name="epsilon">The shape parameter, defaults to 1.0.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the Squared Exponential Radial Basis Function with a specified shape parameter epsilon.
    /// Epsilon controls the width of the function. Larger values of epsilon result in a narrower function
    /// with a faster decay, while smaller values result in a wider function with a more gradual decay.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Squared Exponential RBF with a specific width setting.
    /// 
    /// The epsilon parameter controls the "width" of the bell curve:
    /// - Larger epsilon values (like 5.0) make the bell curve very narrow, falling off quickly as you move away from the center
    /// - Smaller epsilon values (like 0.1) make the bell curve very wide, decreasing slowly with distance
    /// 
    /// Think of it like adjusting the spread of a flashlight beam:
    /// - A high epsilon value is like a focused beam that covers a small area intensely
    /// - A low epsilon value is like a wide beam that covers a large area more diffusely
    /// 
    /// If you're not sure what value to use, the default (epsilon = 1.0) is a good starting point
    /// that provides a moderate decrease with distance.
    /// </para>
    /// </remarks>
    public SquaredExponentialRBF(double epsilon = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _epsilon = _numOps.FromDouble(epsilon);
    }

    /// <summary>
    /// Computes the value of the Squared Exponential Radial Basis Function for a given radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The computed function value exp(-(er)²).</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the value of the Squared Exponential RBF for a given radius r. The formula used is
    /// exp(-(er)²), which decreases exponentially with the square of the distance. The function equals 1
    /// at r = 0 and approaches 0 as r approaches infinity.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes the "height" or "value" of the Squared Exponential function
    /// at a specific distance (r) from the center.
    /// 
    /// The calculation involves:
    /// 1. Multiplying the distance (r) by the epsilon parameter (er)
    /// 2. Squaring this product ((er)²)
    /// 3. Negating this squared value (-(er)²)
    /// 4. Computing the exponential function (e raised to this power)
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
        T epsilonR = _numOps.Multiply(_epsilon, r);
        T squaredEpsilonR = _numOps.Power(epsilonR, _numOps.FromDouble(2));
        T negativeSquaredEpsilonR = _numOps.Negate(squaredEpsilonR);
        return _numOps.Exp(negativeSquaredEpsilonR);
    }

    /// <summary>
    /// Computes the derivative of the Squared Exponential RBF with respect to the radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to r.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Squared Exponential RBF with respect to the radius r.
    /// The formula for the derivative is -2e²r * exp(-(er)²), which is always negative for positive r and e,
    /// indicating that the function always decreases with distance.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes how fast the function's value changes
    /// as you move away from the center point.
    /// 
    /// The derivative tells you the "slope" or "rate of change" of the function at a specific distance.
    /// For the Squared Exponential RBF, the derivative is:
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
        // Derivative with respect to r: -2e²r * exp(-(er)²)

        // Calculate er
        T epsilonR = _numOps.Multiply(_epsilon, r);

        // Calculate (er)²
        T squaredEpsilonR = _numOps.Multiply(epsilonR, epsilonR);

        // Calculate -(er)²
        T negativeSquaredEpsilonR = _numOps.Negate(squaredEpsilonR);

        // Calculate exp(-(er)²)
        T expTerm = _numOps.Exp(negativeSquaredEpsilonR);

        // Calculate e²
        T epsilonSquared = _numOps.Multiply(_epsilon, _epsilon);

        // Calculate 2e²r
        T twoEpsilonSquaredR = _numOps.Multiply(
            _numOps.Multiply(_numOps.FromDouble(2.0), epsilonSquared),
            r
        );

        // Calculate -2e²r
        T negativeTwoEpsilonSquaredR = _numOps.Negate(twoEpsilonSquaredR);

        // Return -2e²r * exp(-(er)²)
        return _numOps.Multiply(negativeTwoEpsilonSquaredR, expTerm);
    }

    /// <summary>
    /// Computes the derivative of the Squared Exponential RBF with respect to the shape parameter epsilon.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to epsilon.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Squared Exponential RBF with respect to the shape parameter epsilon.
    /// The formula for this derivative is -2er² * exp(-(er)²). The sign of this derivative depends on r: it is
    /// negative for non-zero r, indicating that increasing epsilon decreases the function value at any non-zero radius.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how the function's value would change
    /// if you were to adjust the shape parameter (epsilon).
    /// 
    /// This is particularly important in machine learning applications:
    /// - When training a model with Squared Exponential RBFs, we often need to adjust epsilon to fit the data better
    /// - This derivative tells us exactly how changing epsilon affects the output of the function
    /// - With this information, learning algorithms can automatically find the optimal value of epsilon
    /// 
    /// The derivative is zero at the center point (r = 0) and negative everywhere else, which means that 
    /// increasing the epsilon parameter:
    /// - Has no effect on the function value at the center (which is always 1)
    /// - Decreases the function value at all other points (makes the bell curve narrower)
    /// </para>
    /// </remarks>
    public T ComputeWidthDerivative(T r)
    {
        // Derivative with respect to e: -2er² * exp(-(er)²)

        // Calculate er
        T epsilonR = _numOps.Multiply(_epsilon, r);

        // Calculate (er)²
        T squaredEpsilonR = _numOps.Multiply(epsilonR, epsilonR);

        // Calculate -(er)²
        T negativeSquaredEpsilonR = _numOps.Negate(squaredEpsilonR);

        // Calculate exp(-(er)²)
        T expTerm = _numOps.Exp(negativeSquaredEpsilonR);

        // Calculate r²
        T rSquared = _numOps.Multiply(r, r);

        // Calculate 2er²
        T twoEpsilonRSquared = _numOps.Multiply(
            _numOps.Multiply(_numOps.FromDouble(2.0), _epsilon),
            rSquared
        );

        // Calculate -2er²
        T negativeTwoEpsilonRSquared = _numOps.Negate(twoEpsilonRSquared);

        // Return -2er² * exp(-(er)²)
        return _numOps.Multiply(negativeTwoEpsilonRSquared, expTerm);
    }
}
