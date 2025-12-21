namespace AiDotNet.RadialBasisFunctions;

/// <summary>
/// Implements a Spherical Radial Basis Function (RBF) with compact support.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This class provides an implementation of a Spherical Radial Basis Function, which is a compactly
/// supported RBF defined as:
/// f(r) = 1 - 1.5(r/e) + 0.5(r/e)³  for r = e
/// f(r) = 0                        for r > e
/// where r is the radial distance and e (epsilon) is a shape parameter controlling the support radius.
/// </para>
/// <para>
/// Unlike many other RBFs that have non-zero values for all distances, the Spherical RBF becomes exactly
/// zero beyond a certain radius (e), giving it "compact support." This property can be computationally
/// advantageous when working with large datasets, as it leads to sparse matrices in many applications.
/// The function is C² continuous, meaning it has continuous derivatives up to order 2.
/// </para>
/// <para><b>For Beginners:</b> A Radial Basis Function (RBF) is a special type of mathematical function
/// that depends only on the distance from a center point.
/// 
/// The Spherical RBF has a unique property compared to many other RBFs: it becomes exactly zero
/// beyond a certain distance (the support radius). Think of it like a hill that completely flattens
/// out beyond a specific distance - there's a clear boundary where the function's influence stops.
/// 
/// Inside its support radius, the function has a curved shape that smoothly transitions to zero at the boundary:
/// - At the center (distance = 0), the value is exactly 1
/// - As distance increases, the value decreases in a curved pattern
/// - At exactly the support radius (epsilon), the value becomes 0
/// - Beyond the support radius, the value stays at 0
/// 
/// This "limited reach" property makes the Spherical RBF computationally efficient for large datasets,
/// as points beyond the support radius can be completely ignored in calculations.
/// </para>
/// </remarks>
public class SphericalRBF<T> : IRadialBasisFunction<T>
{
    /// <summary>
    /// The shape parameter (epsilon) controlling the support radius of the function.
    /// </summary>
    private readonly T _epsilon;

    /// <summary>
    /// The numeric operations provider for type T, used for mathematical calculations.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="SphericalRBF{T}"/> class with a specified support radius.
    /// </summary>
    /// <param name="epsilon">The shape parameter defining the support radius, defaults to 1.0.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the Spherical Radial Basis Function with a specified epsilon parameter.
    /// Epsilon defines the radius beyond which the function becomes exactly zero (the support radius).
    /// This parameter directly controls how "local" the influence of each center point will be.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Spherical RBF with a specific radius of influence.
    /// 
    /// The epsilon parameter determines the "reach" or "influence radius" of the function:
    /// - Any point within this radius will be affected by the function
    /// - Any point beyond this radius will receive exactly zero influence
    /// - A larger epsilon value means the function affects a wider area
    /// - A smaller epsilon value means the function is more localized
    /// 
    /// For example, with epsilon = 2.0, the function will have non-zero values for all points closer than
    /// 2.0 units to the center, and will be exactly zero for all points 2.0 or more units away.
    /// 
    /// If you're not sure what value to use, the default (epsilon = 1.0) is a good starting point.
    /// </para>
    /// </remarks>
    public SphericalRBF(double epsilon = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _epsilon = _numOps.FromDouble(epsilon);
    }

    /// <summary>
    /// Computes the value of the Spherical Radial Basis Function for a given radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The computed function value, or zero if r > epsilon.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the value of the Spherical RBF for a given radius r. The formula used is
    /// 1 - 1.5(r/e) + 0.5(r/e)³ for r = e, and 0 for r > e. The function equals 1 at r = 0 and
    /// smoothly decreases to 0 at r = e, remaining 0 for all larger values of r.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes the function's value at a specific distance (r) from the center.
    /// 
    /// The calculation depends on whether the distance is within the support radius (epsilon) or not:
    /// - If the distance exceeds epsilon, the function immediately returns 0
    /// - Otherwise, it calculates a polynomial expression that gives a smoothly decreasing curve
    /// 
    /// The result is a single number representing the function's value at the given distance:
    /// - At the center (r = 0), the value is exactly 1
    /// - As you move away from the center, the value decreases following a curved pattern
    /// - At exactly r = epsilon, the value becomes 0
    /// - Beyond epsilon, the value remains 0
    /// 
    /// This means that points beyond the support radius have absolutely no influence in applications
    /// using this function, which can significantly speed up computations for large datasets.
    /// </para>
    /// </remarks>
    public T Compute(T r)
    {
        if (_numOps.GreaterThan(r, _epsilon))
        {
            return _numOps.Zero;
        }
        T rDividedByEpsilon = _numOps.Divide(r, _epsilon);
        T rCubedDividedByEpsilonCubed = _numOps.Power(rDividedByEpsilon, _numOps.FromDouble(3));
        T term1 = _numOps.Multiply(_numOps.FromDouble(1.5), rDividedByEpsilon);
        T term2 = _numOps.Multiply(_numOps.FromDouble(0.5), rCubedDividedByEpsilonCubed);
        return _numOps.Subtract(_numOps.One, _numOps.Subtract(term1, term2));
    }

    /// <summary>
    /// Computes the derivative of the Spherical RBF with respect to the radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to r, or zero if r > epsilon.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Spherical RBF with respect to the radius r.
    /// For r > e, the derivative is 0. For r = e, the formula for the derivative is (1.5/e)[(r/e)² - 1].
    /// The derivative is negative for r < e, indicating that the function decreases with distance within its support.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes how fast the function's value changes
    /// as you move away from the center point.
    /// 
    /// The derivative tells you the "slope" or "rate of change" of the function at a specific distance:
    /// - If the distance exceeds epsilon, the derivative is 0 (no change, since the function is constantly 0)
    /// - Otherwise, it calculates a formula that gives the slope at that distance
    /// 
    /// For the Spherical RBF:
    /// - The derivative is negative within the support radius, meaning the function decreases as distance increases
    /// - The derivative is steepest near the center and becomes less steep as you approach the boundary
    /// - At exactly r = epsilon, the derivative is 0, allowing a smooth transition to the zero region
    /// 
    /// This derivative is useful in machine learning applications when optimizing parameters
    /// using gradient-based methods.
    /// </para>
    /// </remarks>
    public T ComputeDerivative(T r)
    {
        // For r > e, the derivative is 0
        if (_numOps.GreaterThan(r, _epsilon))
        {
            return _numOps.Zero;
        }

        // Calculate r/e
        T rDividedByEpsilon = _numOps.Divide(r, _epsilon);

        // Calculate (r/e)²
        T rDividedByEpsilonSquared = _numOps.Multiply(rDividedByEpsilon, rDividedByEpsilon);

        // Calculate (r/e)² - 1
        T term = _numOps.Subtract(rDividedByEpsilonSquared, _numOps.One);

        // Calculate 1.5/e
        T factor = _numOps.Divide(_numOps.FromDouble(1.5), _epsilon);

        // Return (1.5/e)[(r/e)² - 1]
        return _numOps.Multiply(factor, term);
    }

    /// <summary>
    /// Computes the derivative of the Spherical RBF with respect to the shape parameter epsilon.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to epsilon, or zero if r > epsilon.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Spherical RBF with respect to the shape parameter epsilon.
    /// For r > e, the derivative is 0 for practical purposes, though theoretically it involves a Dirac delta function
    /// at the boundary. For r = e, the formula is (1.5r/e²)[1 - (r/e)²]. This derivative is useful for
    /// optimizing the support radius parameter.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how the function's value would change
    /// if you were to adjust the support radius parameter (epsilon).
    /// 
    /// This is particularly important in machine learning applications:
    /// - When training a model with Spherical RBFs, we often need to adjust epsilon to fit the data better
    /// - This derivative tells us exactly how changing epsilon affects the output of the function
    /// - With this information, learning algorithms can automatically find the optimal value of epsilon
    /// 
    /// For the Spherical RBF:
    /// - If the distance exceeds epsilon, the derivative is treated as 0 for practical purposes
    /// - Otherwise, it calculates a formula that shows how the function value would change
    /// - The derivative is positive for points within the support radius, meaning that increasing epsilon
    ///   generally increases the function value at these points
    /// 
    /// The special case at the boundary r = epsilon is complex mathematically (involving what's called
    /// a "delta function"), but is approximated as 0 for practical purposes.
    /// </para>
    /// </remarks>
    public T ComputeWidthDerivative(T r)
    {
        // For r > e, the width derivative requires special handling
        if (_numOps.GreaterThan(r, _epsilon))
        {
            // The derivative at the boundary is a delta function, which we can't represent directly
            // For practical purposes, we return 0 for r > e
            return _numOps.Zero;
        }

        // Calculate r/e
        T rDividedByEpsilon = _numOps.Divide(r, _epsilon);

        // Calculate (r/e)²
        T rDividedByEpsilonSquared = _numOps.Multiply(rDividedByEpsilon, rDividedByEpsilon);

        // Calculate 1 - (r/e)²
        T term = _numOps.Subtract(_numOps.One, rDividedByEpsilonSquared);

        // Calculate 1.5r/e²
        T factor = _numOps.Divide(
            _numOps.Multiply(_numOps.FromDouble(1.5), r),
            _numOps.Multiply(_epsilon, _epsilon)
        );

        // Return (1.5r/e²)[1 - (r/e)²]
        return _numOps.Multiply(factor, term);
    }
}
