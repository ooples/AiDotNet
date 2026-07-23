namespace AiDotNet.RadialBasisFunctions;

/// <summary>
/// Implements a Multiquadric Radial Basis Function (RBF) of the form v(r² + e²).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This class provides an implementation of a Radial Basis Function (RBF) that uses a multiquadric form
/// of f(r) = v(r² + e²), where r is the radial distance and e (epsilon) is a shape parameter
/// controlling the width of the function. The multiquadric RBF is infinitely differentiable and
/// increases with distance, unlike many other RBFs that decrease with distance. It was introduced by
/// R.L. Hardy and is often used in scattered data interpolation and solving partial differential equations.
/// </para>
/// <para>
/// A notable property of the multiquadric RBF is that it grows with distance rather than decaying,
/// which can lead to better conditioning in certain interpolation problems. However, this growth also
/// means that the corresponding interpolation matrices can be ill-conditioned for large datasets unless
/// appropriate precautions are taken.
/// </para>
/// <para><b>For Beginners:</b> A Radial Basis Function (RBF) is a special type of mathematical function
/// that depends only on the distance from a center point.
/// 
/// The Multiquadric RBF is unusual compared to most other RBFs because its value grows larger as you move
/// away from the center, rather than smaller. It looks like a cone with a rounded bottom - starting from
/// a value of e at the center point (r = 0) and gradually increasing in all directions.
/// 
/// This RBF has a parameter called epsilon (e) that controls the shape of the function:
/// - A larger epsilon value creates a flatter, more rounded shape near the center
/// - A smaller epsilon value creates a sharper, more pointed shape near the center
/// 
/// The multiquadric function is useful in certain interpolation problems where its growth properties
/// can lead to better numerical stability. However, this same growth property means it's often used
/// in combination with other techniques when working with large datasets.
/// </para>
/// </remarks>
public class MultiquadricRBF<T> : IRadialBasisFunction<T>
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
    /// Initializes a new instance of the <see cref="MultiquadricRBF{T}"/> class with a specified shape parameter.
    /// </summary>
    /// <param name="epsilon">The shape parameter, defaults to 1.0.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the Multiquadric Radial Basis Function with a specified shape parameter epsilon.
    /// Epsilon controls the shape and behavior of the function near the origin. Larger values of epsilon result
    /// in a flatter shape near the origin, while smaller values result in a sharper, more pointed shape.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Multiquadric RBF with a specific shape setting.
    /// 
    /// The epsilon parameter controls how the function looks near the center:
    /// - Larger epsilon values (like 5.0) create a rounded, flatter shape near the center
    /// - Smaller epsilon values (like 0.1) create a sharp, pointed shape near the center
    /// 
    /// The epsilon parameter also determines the minimum value of the function at the center:
    /// - The value at the center (r = 0) is always exactly equal to epsilon
    /// - So with epsilon = 1.0, the center value is 1.0
    /// - With epsilon = 0.5, the center value would be 0.5
    /// 
    /// If you're not sure what value to use, the default (epsilon = 1.0) is a good starting point.
    /// </para>
    /// </remarks>
    public MultiquadricRBF(double epsilon = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _epsilon = _numOps.FromDouble(epsilon);
    }

    /// <summary>
    /// Computes the value of the Multiquadric Radial Basis Function for a given radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The computed function value v(r² + e²).</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the value of the Multiquadric RBF for a given radius r. The formula used is
    /// v(r² + e²), which increases with distance. The function reaches its minimum value of e at r = 0
    /// and increases without bound as r increases.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes the "height" or "value" of the Multiquadric function
    /// at a specific distance (r) from the center.
    /// 
    /// The calculation involves:
    /// 1. Squaring the distance (r² = r * r)
    /// 2. Squaring the epsilon parameter (e² = e * e)
    /// 3. Adding these squared values together (r² + e²)
    /// 4. Taking the square root of this sum
    /// 
    /// The result is a single number representing the function's value at the given distance.
    /// This value increases as the distance increases:
    /// - At the center (r = 0), the value is at its minimum of e (epsilon)
    /// - As you move away from the center, the value grows larger without bound
    /// </para>
    /// </remarks>
    public T Compute(T r)
    {
        return _numOps.Sqrt(_numOps.Add(_numOps.Multiply(r, r), _numOps.Multiply(_epsilon, _epsilon)));
    }

    /// <summary>
    /// Computes the derivative of the Multiquadric RBF with respect to the radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to r.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Multiquadric RBF with respect to the radius r.
    /// The formula for the derivative is r/v(r² + e²), which is always positive for positive r,
    /// indicating that the function always increases with distance.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes how fast the function's value changes
    /// as you move away from the center point.
    /// 
    /// The derivative tells you the "slope" or "rate of change" of the function at a specific distance.
    /// For the Multiquadric RBF, the derivative is:
    /// - Zero at the center point (r = 0)
    /// - Positive for any positive distance, meaning the function always increases as you move away from the center
    /// - Approaches 1 as distance increases to infinity
    /// 
    /// This derivative is useful in machine learning applications when optimizing parameters
    /// using gradient-based methods.
    /// </para>
    /// </remarks>
    public T ComputeDerivative(T r)
    {
        // Derivative with respect to r: r/v(r² + e²)

        // Calculate r²
        T rSquared = _numOps.Multiply(r, r);

        // Calculate e²
        T epsilonSquared = _numOps.Multiply(_epsilon, _epsilon);

        // Calculate r² + e²
        T sum = _numOps.Add(rSquared, epsilonSquared);

        // Calculate v(r² + e²)
        T sqrtSum = _numOps.Sqrt(sum);

        // Return r/v(r² + e²)
        return _numOps.Divide(r, sqrtSum);
    }

    /// <summary>
    /// Computes the derivative of the Multiquadric RBF with respect to the shape parameter epsilon.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to epsilon.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Multiquadric RBF with respect to the shape parameter epsilon.
    /// The formula for this derivative is e/v(r² + e²). The sign of this derivative is always positive for positive e,
    /// indicating that increasing epsilon always increases the function value at any radius.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how the function's value would change
    /// if you were to adjust the shape parameter (epsilon).
    /// 
    /// This is particularly important in machine learning applications:
    /// - When training a model with Multiquadric RBFs, we often need to adjust epsilon to fit the data better
    /// - This derivative tells us exactly how changing epsilon affects the output of the function
    /// - With this information, learning algorithms can automatically find the optimal value of epsilon
    /// 
    /// The derivative is positive for all points, which means that increasing the epsilon
    /// parameter always increases the function value at any distance.
    /// </para>
    /// </remarks>
    public T ComputeWidthDerivative(T r)
    {
        // Derivative with respect to e: e/v(r² + e²)

        // Calculate r²
        T rSquared = _numOps.Multiply(r, r);

        // Calculate e²
        T epsilonSquared = _numOps.Multiply(_epsilon, _epsilon);

        // Calculate r² + e²
        T sum = _numOps.Add(rSquared, epsilonSquared);

        // Calculate v(r² + e²)
        T sqrtSum = _numOps.Sqrt(sum);

        // Return e/v(r² + e²)
        return _numOps.Divide(_epsilon, sqrtSum);
    }
}
