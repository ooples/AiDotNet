namespace AiDotNet.RadialBasisFunctions;

/// <summary>
/// Implements a Gaussian Radial Basis Function (RBF) of the form exp(-e*r²).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This class provides an implementation of a Radial Basis Function (RBF) that uses a Gaussian form
/// of f(r) = exp(-e*r²), where r is the radial distance and e (epsilon) is a width parameter
/// controlling how quickly the function decreases with distance. The Gaussian RBF is one of the most
/// widely used RBFs due to its smooth behavior and mathematical properties. It is infinitely differentiable
/// and has exponential decay, making it suitable for a wide range of applications in machine learning,
/// interpolation, and function approximation.
/// </para>
/// <para><b>For Beginners:</b> A Radial Basis Function (RBF) is a special type of mathematical function
/// that depends only on the distance from a center point.
/// 
/// The Gaussian RBF is shaped like a bell curve or a mountain peak - it's at its highest at the center point
/// and gradually decreases in all directions, eventually approaching zero. This particular RBF is named
/// "Gaussian" because it uses the same mathematical form as the Gaussian (normal) distribution from statistics.
/// 
/// This RBF has a parameter called epsilon (e) that controls the width of the bell curve:
/// - A larger epsilon value creates a narrower bell curve that drops off quickly with distance
/// - A smaller epsilon value creates a wider bell curve that extends further
/// 
/// The Gaussian RBF is very popular in machine learning applications like neural networks and regression
/// because it has nice mathematical properties and creates smooth interpolations between data points.
/// </para>
/// </remarks>
public class GaussianRBF<T> : IRadialBasisFunction<T>
{
    /// <summary>
    /// The width parameter (epsilon) controlling how quickly the function decreases with distance.
    /// </summary>
    private readonly T _epsilon;

    /// <summary>
    /// The numeric operations provider for type T, used for mathematical calculations.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="GaussianRBF{T}"/> class with a specified width parameter.
    /// </summary>
    /// <param name="epsilon">The width parameter, defaults to 1.0.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the Gaussian Radial Basis Function with a specified width parameter epsilon.
    /// Epsilon controls how quickly the function decreases with distance. Larger values of epsilon result in a
    /// faster decrease (narrower width), while smaller values result in a slower decrease (wider width).
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Gaussian RBF with a specific width setting.
    /// 
    /// The epsilon parameter controls the "width" of the bell curve:
    /// - Larger epsilon values (like 5.0) make the bell curve very narrow, falling off quickly as you move away from the center
    /// - Smaller epsilon values (like 0.1) make the bell curve very wide, decreasing slowly with distance
    /// 
    /// If you're not sure what value to use, the default (epsilon = 1.0) is a good starting point
    /// that provides a moderate decrease with distance.
    /// </para>
    /// </remarks>
    public GaussianRBF(double epsilon = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _epsilon = _numOps.FromDouble(epsilon);
    }

    /// <summary>
    /// Computes the value of the Gaussian Radial Basis Function for a given radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The computed function value exp(-e*r²).</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the value of the Gaussian RBF for a given radius r. The formula used is
    /// exp(-e*r²), which decreases exponentially with the square of the distance. The function equals 1
    /// at r = 0 and approaches 0 as r approaches infinity.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes the "height" or "value" of the Gaussian function
    /// at a specific distance (r) from the center.
    /// 
    /// The calculation involves:
    /// 1. Squaring the distance (r² = r * r)
    /// 2. Multiplying the squared distance by the epsilon parameter
    /// 3. Negating this product to make it negative
    /// 4. Computing the exponential function (e raised to this power)
    /// 
    /// The result is a single number representing the function's value at the given distance.
    /// This value is always between 0 and 1:
    /// - At the center (r = 0), the value is exactly 1
    /// - As you move away from the center, the value gets smaller, approaching 0 but never quite reaching it
    /// </para>
    /// </remarks>
    public T Compute(T r)
    {
        return _numOps.Exp(_numOps.Negate(_numOps.Multiply(_epsilon, _numOps.Multiply(r, r))));
    }

    /// <summary>
    /// Computes the derivative of the Gaussian RBF with respect to the radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to r.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Gaussian RBF with respect to the radius r.
    /// The formula for the derivative is -2er * exp(-e*r²), which is always negative for positive r and e,
    /// indicating that the function always decreases with distance.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes how fast the function's value changes
    /// as you move away from the center point.
    /// 
    /// The derivative tells you the "slope" or "rate of change" of the function at a specific distance.
    /// For the Gaussian RBF, the derivative is:
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
        // Derivative with respect to r: -2er * exp(-e*r²)

        // Calculate -2er
        T minusTwoEpsilonR = _numOps.Multiply(
            _numOps.Multiply(_numOps.FromDouble(-2.0), _epsilon),
            r
        );

        // Multiply by exp(-e*r²)
        return _numOps.Multiply(minusTwoEpsilonR, Compute(r));
    }

    /// <summary>
    /// Computes the derivative of the Gaussian RBF with respect to the width parameter epsilon.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to epsilon.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Gaussian RBF with respect to the width parameter epsilon.
    /// The formula for this derivative is -r² * exp(-e*r²). The sign of this derivative depends on r: it is
    /// negative for non-zero r, indicating that increasing epsilon decreases the function value at any non-zero radius.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how the function's value would change
    /// if you were to adjust the width parameter (epsilon).
    /// 
    /// This is particularly important in machine learning applications:
    /// - When training a model with Gaussian RBFs, we often need to adjust epsilon to fit the data better
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
        // Derivative with respect to e: -r² * exp(-e*r²)

        // Calculate -r²
        T rSquared = _numOps.Multiply(r, r);
        T negativeRSquared = _numOps.Negate(rSquared);

        // Multiply by exp(-e*r²)
        return _numOps.Multiply(negativeRSquared, Compute(r));
    }
}
