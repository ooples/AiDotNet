namespace AiDotNet.RadialBasisFunctions;

/// <summary>
/// Implements an Inverse Multiquadric Radial Basis Function (RBF) of the form 1/v(r² + e²).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This class provides an implementation of a Radial Basis Function (RBF) that uses an inverse multiquadric form
/// of f(r) = 1/v(r² + e²), where r is the radial distance and e (epsilon) is a shape parameter
/// controlling the width of the function. The inverse multiquadric RBF is infinitely differentiable and
/// decreases more slowly than the Gaussian RBF as distance increases. It is often used in interpolation
/// problems and has good numerical properties for solving partial differential equations.
/// </para>
/// <para><b>For Beginners:</b> A Radial Basis Function (RBF) is a special type of mathematical function
/// that depends only on the distance from a center point.
/// 
/// The Inverse Multiquadric RBF looks like an upside-down cone that flattens out at larger distances.
/// At the center point (r = 0), it has its highest value of 1/e, and as you move away from the center,
/// the function value gradually decreases toward zero, but never quite reaches it.
/// 
/// This RBF has a parameter called epsilon (e) that controls the shape and width of the function:
/// - A larger epsilon value creates a narrower peak with a faster initial drop-off
/// - A smaller epsilon value creates a broader peak with a more gradual initial drop-off
/// 
/// Unlike some other RBFs (like the Gaussian), the inverse multiquadric function has "long tails,"
/// meaning it decreases more slowly at larger distances. This property makes it useful for problems
/// where you want influence to extend further from the center points.
/// </para>
/// </remarks>
public class InverseMultiquadricRBF<T> : IRadialBasisFunction<T>
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
    /// Initializes a new instance of the <see cref="InverseMultiquadricRBF{T}"/> class with a specified shape parameter.
    /// </summary>
    /// <param name="epsilon">The shape parameter, defaults to 1.0.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the Inverse Multiquadric Radial Basis Function with a specified shape parameter epsilon.
    /// Epsilon controls the shape and width of the function. Larger values of epsilon result in a narrower peak
    /// with a faster initial drop-off, while smaller values result in a broader peak with a more gradual drop-off.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Inverse Multiquadric RBF with a specific shape setting.
    /// 
    /// The epsilon parameter controls how the function looks:
    /// - Larger epsilon values (like 5.0) create a sharp, narrow peak that quickly flattens out
    /// - Smaller epsilon values (like 0.1) create a wide, gradual curve that extends further
    /// 
    /// The epsilon parameter also determines the maximum value of the function at the center:
    /// - The value at the center (r = 0) is always 1/e
    /// - So with epsilon = 1.0, the center value is 1.0
    /// - With epsilon = 0.5, the center value would be 2.0
    /// 
    /// If you're not sure what value to use, the default (epsilon = 1.0) is a good starting point.
    /// </para>
    /// </remarks>
    public InverseMultiquadricRBF(double epsilon = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _epsilon = _numOps.FromDouble(epsilon);
    }

    /// <summary>
    /// Computes the value of the Inverse Multiquadric Radial Basis Function for a given radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The computed function value 1/v(r² + e²).</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the value of the Inverse Multiquadric RBF for a given radius r. The formula used is
    /// 1/v(r² + e²), which decreases with distance. The function reaches its maximum value of 1/e at r = 0
    /// and approaches 0 as r approaches infinity.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes the "height" or "value" of the Inverse Multiquadric function
    /// at a specific distance (r) from the center.
    /// 
    /// The calculation involves:
    /// 1. Squaring the distance (r² = r * r)
    /// 2. Squaring the epsilon parameter (e² = e * e)
    /// 3. Adding these squared values together (r² + e²)
    /// 4. Taking the square root of this sum
    /// 5. Dividing 1 by this square root
    /// 
    /// The result is a single number representing the function's value at the given distance.
    /// This value is always positive and decreases as the distance increases:
    /// - At the center (r = 0), the value is at its maximum of 1/e
    /// - As you move away from the center, the value gets smaller, approaching 0 but never quite reaching it
    /// </para>
    /// </remarks>
    public T Compute(T r)
    {
        return _numOps.Divide(
            _numOps.One,
            _numOps.Sqrt(_numOps.Add(_numOps.Multiply(r, r), _numOps.Multiply(_epsilon, _epsilon)))
        );
    }

    /// <summary>
    /// Computes the derivative of the Inverse Multiquadric RBF with respect to the radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to r.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Inverse Multiquadric RBF with respect to the radius r.
    /// The formula for the derivative is -r/(r² + e²)^(3/2), which is negative for positive r,
    /// indicating that the function always decreases with distance.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes how fast the function's value changes
    /// as you move away from the center point.
    /// 
    /// The derivative tells you the "slope" or "rate of change" of the function at a specific distance.
    /// For the Inverse Multiquadric RBF, the derivative is:
    /// - Zero at the center point (r = 0)
    /// - Negative for any positive distance, meaning the function always decreases as you move away from the center
    /// - Has its largest magnitude near the center and gradually approaches zero as distance increases
    /// 
    /// This derivative is useful in machine learning applications when optimizing parameters
    /// using gradient-based methods.
    /// </para>
    /// </remarks>
    public T ComputeDerivative(T r)
    {
        // Derivative with respect to r: -r/(r² + e²)^(3/2)

        // Calculate r²
        T rSquared = _numOps.Multiply(r, r);

        // Calculate e²
        T epsilonSquared = _numOps.Multiply(_epsilon, _epsilon);

        // Calculate r² + e²
        T sum = _numOps.Add(rSquared, epsilonSquared);

        // Calculate (r² + e²)^(3/2)
        T sumSqrt = _numOps.Sqrt(sum);
        T sumPow3_2 = _numOps.Multiply(sum, sumSqrt);

        // Calculate -r
        T negativeR = _numOps.Negate(r);

        // Return -r/(r² + e²)^(3/2)
        return _numOps.Divide(negativeR, sumPow3_2);
    }

    /// <summary>
    /// Computes the derivative of the Inverse Multiquadric RBF with respect to the shape parameter epsilon.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to epsilon.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Inverse Multiquadric RBF with respect to the shape parameter epsilon.
    /// The formula for this derivative is -e/(r² + e²)^(3/2). The sign of this derivative is always negative,
    /// indicating that increasing epsilon decreases the function value at any radius.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how the function's value would change
    /// if you were to adjust the shape parameter (epsilon).
    /// 
    /// This is particularly important in machine learning applications:
    /// - When training a model with Inverse Multiquadric RBFs, we often need to adjust epsilon to fit the data better
    /// - This derivative tells us exactly how changing epsilon affects the output of the function
    /// - With this information, learning algorithms can automatically find the optimal value of epsilon
    /// 
    /// The derivative is negative for all points, which means that increasing the epsilon
    /// parameter always decreases the function value at any distance.
    /// </para>
    /// </remarks>
    public T ComputeWidthDerivative(T r)
    {
        // Derivative with respect to e: -e/(r² + e²)^(3/2)

        // Calculate r²
        T rSquared = _numOps.Multiply(r, r);

        // Calculate e²
        T epsilonSquared = _numOps.Multiply(_epsilon, _epsilon);

        // Calculate r² + e²
        T sum = _numOps.Add(rSquared, epsilonSquared);

        // Calculate (r² + e²)^(3/2)
        T sumSqrt = _numOps.Sqrt(sum);
        T sumPow3_2 = _numOps.Multiply(sum, sumSqrt);

        // Calculate -e
        T negativeEpsilon = _numOps.Negate(_epsilon);

        // Return -e/(r² + e²)^(3/2)
        return _numOps.Divide(negativeEpsilon, sumPow3_2);
    }
}
