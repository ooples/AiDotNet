namespace AiDotNet.RadialBasisFunctions;

/// <summary>
/// Implements an Exponential Radial Basis Function (RBF) of the form exp(-e*r).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This class provides an implementation of a Radial Basis Function (RBF) that uses an exponential decay
/// of the form f(r) = exp(-e*r), where r is the radial distance and e (epsilon) is a width parameter
/// controlling how quickly the function decreases with distance. The exponential RBF is sometimes called
/// the Laplacian RBF and is related to the distribution of the same name. It decreases less rapidly than
/// the Gaussian RBF for small distances but has a more gradual asymptotic behavior for large distances.
/// </para>
/// <para><b>For Beginners:</b> A Radial Basis Function (RBF) is a special type of mathematical function
/// that depends only on the distance from a center point.
/// 
/// The Exponential RBF decreases as you move away from the center point, following an exponential decay pattern.
/// Think of it like a hill or mountain that starts at its highest point in the center and then gradually
/// slopes downward in all directions, never quite reaching zero.
/// 
/// This specific RBF has a parameter called epsilon (e) that controls how quickly the "hill" drops off:
/// - A larger epsilon value creates a steeper hill that drops off quickly with distance
/// - A smaller epsilon value creates a more gradual slope that extends further
/// 
/// The exponential RBF is useful in many machine learning applications, especially when you want a smoother
/// decrease with distance compared to a Gaussian RBF.
/// </para>
/// </remarks>
public class ExponentialRBF<T> : IRadialBasisFunction<T>
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
    /// Initializes a new instance of the <see cref="ExponentialRBF{T}"/> class with a specified width parameter.
    /// </summary>
    /// <param name="epsilon">The width parameter, defaults to 1.0.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the Exponential Radial Basis Function with a specified width parameter epsilon.
    /// Epsilon controls how quickly the function decreases with distance. Larger values of epsilon result in a
    /// faster decrease (narrower width), while smaller values result in a slower decrease (wider width).
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Exponential RBF with a specific width setting.
    /// 
    /// The epsilon parameter controls the "width" of the function:
    /// - Larger epsilon values (like 5.0) make the function very narrow, falling off quickly as you move away from the center
    /// - Smaller epsilon values (like 0.1) make the function very wide, decreasing slowly with distance
    /// 
    /// If you're not sure what value to use, the default (epsilon = 1.0) is a good starting point
    /// that provides a moderate decrease with distance.
    /// </para>
    /// </remarks>
    public ExponentialRBF(double epsilon = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _epsilon = _numOps.FromDouble(epsilon);
    }

    /// <summary>
    /// Computes the value of the Exponential Radial Basis Function for a given radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The computed function value exp(-e*r).</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the value of the Exponential RBF for a given radius r. The formula used is
    /// exp(-e*r), which decreases exponentially with distance. The function equals 1 at r = 0 and approaches 0
    /// as r approaches infinity.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes the "height" or "value" of the Exponential function
    /// at a specific distance (r) from the center.
    /// 
    /// The calculation involves:
    /// 1. Multiplying the distance (r) by the epsilon parameter
    /// 2. Negating this product to make it negative
    /// 3. Computing the exponential function (e raised to this power)
    /// 
    /// The result is a single number representing the function's value at the given distance.
    /// This value is always between 0 and 1:
    /// - At the center (r = 0), the value is exactly 1
    /// - As you move away from the center, the value gets smaller, approaching 0 but never quite reaching it
    /// </para>
    /// </remarks>
    public T Compute(T r)
    {
        T negativeEpsilonR = _numOps.Multiply(_numOps.Negate(_epsilon), r);
        return _numOps.Exp(negativeEpsilonR);
    }

    /// <summary>
    /// Computes the derivative of the Exponential RBF with respect to the radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to r.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Exponential RBF with respect to the radius r.
    /// The formula for the derivative is -e * exp(-e*r), which is always negative for positive r and e,
    /// indicating that the function always decreases with distance.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes how fast the function's value changes
    /// as you move away from the center point.
    /// 
    /// The derivative tells you the "slope" or "rate of change" of the function at a specific distance.
    /// For the Exponential RBF, the derivative is:
    /// - Always negative (for positive r and epsilon), meaning the function always decreases as you move away from the center
    /// - Has its largest magnitude near the center and gradually approaches zero as distance increases
    /// 
    /// This derivative is useful in machine learning applications when optimizing parameters
    /// using gradient-based methods.
    /// </para>
    /// </remarks>
    public T ComputeDerivative(T r)
    {
        // Derivative with respect to r: -e * exp(-e*r)
        T negativeEpsilon = _numOps.Negate(_epsilon);
        return _numOps.Multiply(negativeEpsilon, Compute(r));
    }

    /// <summary>
    /// Computes the derivative of the Exponential RBF with respect to the width parameter epsilon.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to epsilon.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Exponential RBF with respect to the width parameter epsilon.
    /// The formula for this derivative is -r * exp(-e*r). The sign of this derivative depends on r: it is
    /// negative for positive r, indicating that increasing epsilon decreases the function value at any positive radius.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how the function's value would change
    /// if you were to adjust the width parameter (epsilon).
    /// 
    /// This is particularly important in machine learning applications:
    /// - When training a model with Exponential RBFs, we often need to adjust epsilon to fit the data better
    /// - This derivative tells us exactly how changing epsilon affects the output of the function
    /// - With this information, learning algorithms can automatically find the optimal value of epsilon
    /// 
    /// The derivative is always negative for positive distances, which means that increasing the epsilon
    /// parameter always decreases the function value at any given positive distance.
    /// </para>
    /// </remarks>
    public T ComputeWidthDerivative(T r)
    {
        // Derivative with respect to e: -r * exp(-e*r)
        T negativeR = _numOps.Negate(r);
        return _numOps.Multiply(negativeR, Compute(r));
    }
}
