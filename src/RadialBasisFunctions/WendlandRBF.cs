namespace AiDotNet.RadialBasisFunctions;

/// <summary>
/// Implements Wendland's compactly supported Radial Basis Functions with different smoothness orders.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This class provides an implementation of Wendland's family of compactly supported Radial Basis Functions.
/// These functions are defined by a smoothness parameter k and have the form:
/// - For k = 0: f(r) = (1 - r)²  (for r = 1, 0 otherwise)
/// - For k = 1: f(r) = (1 - r)4(1 + 4r)  (for r = 1, 0 otherwise)
/// - For k = 2: f(r) = (1 - r)6(3 + 18r + 35r²)  (for r = 1, 0 otherwise)
/// where r is the normalized radial distance (actual distance divided by the support radius).
/// </para>
/// <para>
/// Wendland functions are popular in scientific computing because they combine compact support
/// (they become exactly zero beyond a certain radius) with high order smoothness properties. The parameter k
/// controls the smoothness of the function: higher k values yield more derivatives at r = 0 and r = 1,
/// resulting in smoother interpolations. These functions are particularly useful for scattered data
/// interpolation, meshless methods for solving PDEs, and computer graphics applications.
/// </para>
/// <para><b>For Beginners:</b> A Radial Basis Function (RBF) is a special type of mathematical function
/// that depends only on the distance from a center point.
/// 
/// Wendland RBFs are a family of functions that have two important properties:
/// 1. They have "compact support" - they become exactly zero beyond a certain distance (the support radius)
/// 2. They are very smooth - they transition gradually to zero at the boundary with no sudden changes
/// 
/// Think of them like smooth hills or bumps that are exactly flat (zero) beyond a certain distance.
/// You can choose from different types of Wendland functions based on how smooth you need them to be:
/// - k = 0: The basic version, reasonably smooth but with limited continuity
/// - k = 1: A smoother version, with more continuous derivatives
/// - k = 2: The smoothest version, with even more continuous derivatives
/// 
/// The higher the k value, the smoother the function, but also the more computationally expensive.
/// These functions are particularly useful in scientific computing because they combine efficiency
/// (from the compact support) with high quality results (from the smoothness).
/// </para>
/// </remarks>
public class WendlandRBF<T> : IRadialBasisFunction<T>
{
    /// <summary>
    /// The numeric operations provider for type T, used for mathematical calculations.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// The smoothness parameter controlling the order of the Wendland function.
    /// </summary>
    private readonly int _k;

    /// <summary>
    /// The support radius beyond which the function becomes zero.
    /// </summary>
    private readonly T _supportRadius;

    /// <summary>
    /// Initializes a new instance of the <see cref="WendlandRBF{T}"/> class with specified parameters.
    /// </summary>
    /// <param name="k">The smoothness parameter, defaults to 2. Supported values are 0, 1, and 2.</param>
    /// <param name="supportRadius">The radius beyond which the function becomes zero, defaults to 1.0.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the Wendland Radial Basis Function with a specified smoothness parameter k
    /// and support radius. The parameter k controls the smoothness of the function, with higher values providing
    /// more continuous derivatives. The support radius determines the distance beyond which the function
    /// becomes exactly zero.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Wendland RBF with specific smoothness and size settings.
    /// 
    /// The two parameters you can adjust are:
    /// - k: Controls how smooth the function is. Higher values (0, 1, or 2) give progressively smoother functions
    ///   with more continuous derivatives. The default k = 2 gives the smoothest function.
    /// - supportRadius: Controls the "reach" or "influence radius" of the function. Any point beyond this
    ///   distance will receive exactly zero influence. A larger value means the function affects a wider area.
    /// 
    /// For example, with supportRadius = 2.0, the function will have non-zero values for all points closer than
    /// 2.0 units to the center, and will be exactly zero for all points 2.0 or more units away.
    /// 
    /// If you're not sure what values to use, the defaults (k = 2, supportRadius = 1.0) are good starting points
    /// that provide high smoothness with a moderate range of influence.
    /// </para>
    /// </remarks>
    public WendlandRBF(int k = 2, double supportRadius = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _k = k;
        _supportRadius = _numOps.FromDouble(supportRadius);
    }

    /// <summary>
    /// Computes the value of the Wendland Radial Basis Function for a given radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The computed function value based on the k parameter, or zero if r = supportRadius.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the value of the Wendland RBF for a given radius r. The formula used depends
    /// on the k parameter and only applies when r is less than the support radius. If r is greater than or
    /// equal to the support radius, the function returns 0. The formulas are:
    /// - For k = 0: (1 - r)²
    /// - For k = 1: (1 - r)4(1 + 4r)
    /// - For k = 2: (1 - r)6(3 + 18r + 35r²)
    /// where r is normalized by dividing by the support radius.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes the function's value at a specific distance (r) from the center.
    /// 
    /// The calculation first checks if the distance is within the support radius:
    /// - If the distance is greater than or equal to the support radius, it immediately returns 0
    /// - Otherwise, it calculates a value based on the k parameter (smoothness level)
    /// 
    /// The result is a single number representing the function's value at the given distance:
    /// - At the center (r = 0), the value depends on k but is always positive
    /// - As you move away from the center, the value decreases following a smooth curve
    /// - At exactly r = supportRadius, the value becomes 0
    /// - Beyond the support radius, the value remains 0
    /// 
    /// This compact support property makes Wendland functions computationally efficient,
    /// as points beyond the support radius can be completely ignored in calculations.
    /// </para>
    /// </remarks>
    public T Compute(T r)
    {
        T normalizedR = _numOps.Divide(r, _supportRadius);

        if (_numOps.GreaterThanOrEquals(normalizedR, _numOps.One))
        {
            return _numOps.Zero;
        }

        T oneMinusR = _numOps.Subtract(_numOps.One, normalizedR);

        switch (_k)
        {
            case 0:
                return _numOps.Power(oneMinusR, _numOps.FromDouble(2));
            case 1:
                T term1 = _numOps.Power(oneMinusR, _numOps.FromDouble(4));
                T term2 = _numOps.Multiply(_numOps.FromDouble(4), normalizedR);
                return _numOps.Multiply(term1, _numOps.Add(_numOps.One, term2));
            case 2:
                T term1_k2 = _numOps.Power(oneMinusR, _numOps.FromDouble(6));
                T term2_k2 = _numOps.Multiply(_numOps.FromDouble(35), _numOps.Power(normalizedR, _numOps.FromDouble(2)));
                T term3_k2 = _numOps.Multiply(_numOps.FromDouble(18), normalizedR);
                T term4_k2 = _numOps.FromDouble(3);
                return _numOps.Multiply(term1_k2, _numOps.Add(_numOps.Add(term2_k2, term3_k2), term4_k2));
            default:
                throw new ArgumentException("Unsupported k value. Supported values are 0, 1, and 2.");
        }
    }

    /// <summary>
    /// Computes the derivative of the Wendland RBF with respect to the radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to r, or zero if r = supportRadius or r = 0.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Wendland RBF with respect to the radius r.
    /// The formula for the derivative depends on the k parameter and only applies when r is greater than 0
    /// and less than the support radius. If r is greater than or equal to the support radius or equal to 0,
    /// the derivative is 0. The formulas for the derivatives are:
    /// - For k = 0: -2(1-r)
    /// - For k = 1: (1-r)³(-4-20r)
    /// - For k = 2: (1-r)5(-18-180r-210r²)
    /// where r is normalized by dividing by the support radius.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes how fast the function's value changes
    /// as you move away from the center point.
    /// 
    /// The derivative tells you the "slope" or "rate of change" of the function at a specific distance:
    /// - If the distance exceeds the support radius, the derivative is 0 (no change, since the function is constantly 0)
    /// - If the distance is exactly 0, the derivative is also 0 (due to symmetry at the center)
    /// - Otherwise, it calculates a formula that gives the slope at that distance
    /// 
    /// For all Wendland functions:
    /// - The derivative is negative within the support radius, meaning the function decreases as distance increases
    /// - The derivative approaches 0 as you approach the boundary, allowing a smooth transition to the zero region
    /// 
    /// This smooth behavior at the boundary is one of the key features that makes Wendland functions
    /// valuable for scientific computing applications.
    /// </para>
    /// </remarks>
    public T ComputeDerivative(T r)
    {
        T normalizedR = _numOps.Divide(r, _supportRadius);

        if (_numOps.GreaterThanOrEquals(normalizedR, _numOps.One) || _numOps.Equals(normalizedR, _numOps.Zero))
        {
            return _numOps.Zero;
        }

        T oneMinusR = _numOps.Subtract(_numOps.One, normalizedR);

        switch (_k)
        {
            case 0:
                // d/dr[(1-r)^2] = -2(1-r)
                T factor = _numOps.FromDouble(-2.0);
                return _numOps.Multiply(factor, oneMinusR);

            case 1:
                // d/dr[(1-r)^4 * (1+4r)] = (1-r)^3 * (-4-20r)
                T term1 = _numOps.Power(oneMinusR, _numOps.FromDouble(3));
                T term2 = _numOps.FromDouble(-4);
                T term3 = _numOps.Multiply(_numOps.FromDouble(-20), normalizedR);
                return _numOps.Multiply(term1, _numOps.Add(term2, term3));

            case 2:
                // d/dr[(1-r)^6 * (3+18r+35r^2)] = (1-r)^5 * (-18-180r-210r^2)
                T term1_k2 = _numOps.Power(oneMinusR, _numOps.FromDouble(5));
                T term2_k2 = _numOps.FromDouble(-18);
                T term3_k2 = _numOps.Multiply(_numOps.FromDouble(-180), normalizedR);
                T term4_k2 = _numOps.Multiply(_numOps.FromDouble(-210), _numOps.Power(normalizedR, _numOps.FromDouble(2)));
                return _numOps.Multiply(term1_k2, _numOps.Add(_numOps.Add(term2_k2, term3_k2), term4_k2));

            default:
                throw new ArgumentException("Unsupported k value. Supported values are 0, 1, and 2.");
        }
    }

    /// <summary>
    /// Computes the derivative of the Wendland RBF with respect to the support radius parameter.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to the support radius, or zero if r = supportRadius.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Wendland RBF with respect to the support radius parameter.
    /// For a function f(r/s) where s is the support radius, the derivative with respect to s is
    /// -r/s² × f'(r/s), where f' is the derivative of f with respect to its argument. This derivative
    /// is useful for optimizing the support radius parameter in applications.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how the function's value would change
    /// if you were to adjust the support radius parameter.
    /// 
    /// This is particularly important in machine learning applications:
    /// - When training a model with Wendland RBFs, we often need to adjust the support radius to fit the data better
    /// - This derivative tells us exactly how changing the support radius affects the output of the function
    /// - With this information, learning algorithms can automatically find the optimal value for the support radius
    /// 
    /// For Wendland functions:
    /// - If the distance exceeds the support radius, the width derivative is 0
    /// - Otherwise, the width derivative is calculated as a product involving the radius derivative
    /// 
    /// As the support radius increases, the "reach" of the function extends further, allowing it to
    /// influence more distant points. This derivative helps algorithms determine whether extending
    /// that reach would improve the function's ability to model the data.
    /// </para>
    /// </remarks>
    public T ComputeWidthDerivative(T r)
    {
        T normalizedR = _numOps.Divide(r, _supportRadius);

        if (_numOps.GreaterThanOrEquals(normalizedR, _numOps.One))
        {
            return _numOps.Zero;
        }

        // For width derivative, we need to compute d/ds[f(r/s)]
        // This equals -r/s^2 * f'(r/s) where f' is the derivative of f

        // First, compute r/s^2
        T rOverSigmaSquared = _numOps.Divide(r, _numOps.Power(_supportRadius, _numOps.FromDouble(2)));

        // Then compute the derivative at r/s
        T derivativeValue = ComputeDerivative(r);

        // Multiply by -1
        T negativeOne = _numOps.FromDouble(-1);

        // Return -r/s^2 * f'(r/s)
        return _numOps.Multiply(negativeOne, _numOps.Multiply(rOverSigmaSquared, derivativeValue));
    }
}
