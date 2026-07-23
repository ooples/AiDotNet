namespace AiDotNet.RadialBasisFunctions;

/// <summary>
/// Implements the Bessel Radial Basis Function based on Bessel functions of the first kind.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This class provides an implementation of a Radial Basis Function (RBF) using Bessel functions of the first kind.
/// The Bessel RBF is defined as J_?(e*r)/(e*r)^?, where J_? is the Bessel function of the first kind of order ?,
/// e is the width parameter, and r is the radial distance. This RBF is particularly useful for problems with
/// circular or spherical symmetry, and in cases where oscillatory behavior is expected.
/// </para>
/// <para><b>For Beginners:</b> A Radial Basis Function (RBF) is a special type of mathematical function
/// that depends only on the distance from a center point.
/// 
/// The Bessel RBF uses Bessel functions, which are important in physics and engineering for problems
/// with circular or cylindrical symmetry. Think of them as functions that can model wave-like behavior,
/// like the vibrations on a circular drum or the pattern of ripples on a pond.
/// 
/// This particular RBF has two main parameters:
/// - epsilon (e): Controls the width of the function (how quickly it changes with distance)
/// - nu (?): Controls the order of the Bessel function (affects the shape and oscillatory behavior)
/// 
/// Bessel RBFs are useful when your data or problem has circular patterns or oscillatory features.
/// </para>
/// </remarks>
public class BesselRBF<T> : IRadialBasisFunction<T>
{
    /// <summary>
    /// The width parameter (epsilon) controlling how quickly the function decreases with distance.
    /// </summary>
    private readonly T _epsilon;

    /// <summary>
    /// The order parameter (nu) of the Bessel function.
    /// </summary>
    private readonly T _nu;

    /// <summary>
    /// The numeric operations provider for type T, used for mathematical calculations.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="BesselRBF{T}"/> class with specified parameters.
    /// </summary>
    /// <param name="epsilon">The width parameter, defaults to 1.0.</param>
    /// <param name="nu">The order of the Bessel function, defaults to 0.0.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the Bessel Radial Basis Function with specified width and order parameters.
    /// The width parameter (epsilon) controls how quickly the function decreases with distance, while the
    /// order parameter (nu) determines the specific Bessel function used. Common values for nu include 0 and 1,
    /// corresponding to the Bessel functions J0 and J1.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Bessel RBF with specific settings.
    /// 
    /// The two parameters you can adjust are:
    /// - epsilon: Controls the "width" of the function - smaller values make the function change more gradually
    ///   with distance, while larger values make it change more rapidly
    /// - nu: Controls the "shape" of the function - different values give different patterns of peaks and valleys
    /// 
    /// If you're not sure what values to use, the defaults (epsilon = 1.0, nu = 0.0) are a good starting point.
    /// These defaults use the zero-order Bessel function (J0) with a moderate width.
    /// </para>
    /// </remarks>
    public BesselRBF(double epsilon = 1.0, double nu = 0.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _epsilon = _numOps.FromDouble(epsilon);
        _nu = _numOps.FromDouble(nu);
    }

    /// <summary>
    /// Computes the value of the Bessel Radial Basis Function for a given radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The computed function value J_?(e*r)/(e*r)^?.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the value of the Bessel RBF for a given radius r. The formula used is
    /// J_?(e*r)/(e*r)^?, where J_? is the Bessel function of the first kind of order ?.
    /// For r = 0, a special case is handled to avoid division by zero, returning 1.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes the "height" or "value" of the Bessel function
    /// at a specific distance (r) from the center.
    /// 
    /// The calculation involves:
    /// 1. Multiplying the distance (r) by the width parameter (epsilon)
    /// 2. Computing the Bessel function of the first kind for this product
    /// 3. Dividing by the same product raised to the power of nu
    /// 
    /// A special case is handled when the distance is very close to zero to avoid
    /// mathematical errors (like division by zero).
    /// 
    /// The result is a single number representing the function's value at the given distance.
    /// </para>
    /// </remarks>
    public T Compute(T r)
    {
        T epsilonR = _numOps.Multiply(_epsilon, r);

        // Handle the case when epsilonR is very close to zero
        if (MathHelper.AlmostEqual(epsilonR, _numOps.Zero))
        {
            return _numOps.One;
        }

        T besselValue = MathHelper.BesselJ(_nu, epsilonR);
        T denominator = _numOps.Power(epsilonR, _nu);

        return _numOps.Divide(besselValue, denominator);
    }

    /// <summary>
    /// Computes the derivative of the Bessel RBF with respect to the radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to r.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Bessel RBF with respect to the radius r.
    /// The formula for the derivative is complex and involves both the Bessel function of order ?
    /// and order ?-1. Special cases are handled for r = 0, depending on the value of ?.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes how fast the function's value changes
    /// as you move away from the center point.
    /// 
    /// The derivative tells you the "slope" or "rate of change" of the function at a specific distance.
    /// A positive derivative means the function is increasing as distance increases, while a negative
    /// derivative means it's decreasing.
    /// 
    /// Special cases are handled when the distance is exactly zero:
    /// - For nu = 0, the derivative at r=0 is negative and depends on epsilon
    /// - For other values of nu, the derivative at r=0 is zero
    /// 
    /// This derivative is useful in machine learning applications when optimizing parameters
    /// using gradient-based methods.
    /// </para>
    /// </remarks>
    public T ComputeDerivative(T r)
    {
        // Handle the case when r is very close to zero
        if (MathHelper.AlmostEqual(r, _numOps.Zero))
        {
            // For r=0, the derivative is 0 for nu > 1
            // For nu = 0, the derivative is -epsilon^2/2
            // For nu = 1, the derivative is 0
            if (MathHelper.AlmostEqual(_nu, _numOps.Zero))
            {
                T halfEpsilonSquared = _numOps.Multiply(_numOps.Multiply(_epsilon, _epsilon), _numOps.FromDouble(0.5));
                return _numOps.Negate(halfEpsilonSquared);
            }
            return _numOps.Zero;
        }

        T epsilonR = _numOps.Multiply(_epsilon, r);

        // The derivative of J_nu(epsilon*r)/(epsilon*r)^nu with respect to r is:
        // epsilon * [J_(nu-1)(epsilon*r) - (nu/r)*J_nu(epsilon*r)] / (epsilon*r)^nu

        T besselNu = MathHelper.BesselJ(_nu, epsilonR);
        T besselNuMinus1 = MathHelper.BesselJ(_numOps.Subtract(_nu, _numOps.One), epsilonR);

        T term1 = besselNuMinus1;
        T term2 = _numOps.Multiply(_numOps.Divide(_nu, r), besselNu);
        T numerator = _numOps.Multiply(_epsilon, _numOps.Subtract(term1, term2));

        T denominator = _numOps.Power(epsilonR, _nu);

        return _numOps.Divide(numerator, denominator);
    }

    /// <summary>
    /// Computes the derivative of the Bessel RBF with respect to the width parameter epsilon.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to epsilon.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Bessel RBF with respect to the width parameter epsilon.
    /// This derivative is useful for gradient-based optimization of the width parameter. The formula
    /// involves both the Bessel function of order ? and order ?-1, as well as special cases for r = 0.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how the function's value would change
    /// if you were to adjust the width parameter (epsilon).
    /// 
    /// This is particularly important in machine learning applications:
    /// - When training a model with Bessel RBFs, we often need to adjust epsilon to fit the data better
    /// - This derivative tells us exactly how changing epsilon affects the output of the function
    /// - With this information, learning algorithms can automatically find the optimal value of epsilon
    /// 
    /// Like the radius derivative, special cases are handled when the distance is exactly zero,
    /// depending on the value of nu:
    /// - For nu = 0, the width derivative at r=0 is 0
    /// - For nu = 1, the width derivative at r=0 is 0.5
    /// - For other values, the width derivative at r=0 is 0
    /// </para>
    /// </remarks>
    public T ComputeWidthDerivative(T r)
    {
        // Handle the case when r is very close to zero
        if (MathHelper.AlmostEqual(r, _numOps.Zero))
        {
            // For r=0, the width derivative depends on nu
            if (MathHelper.AlmostEqual(_nu, _numOps.Zero))
            {
                return _numOps.Zero; // For nu=0, the width derivative at r=0 is 0
            }
            else if (MathHelper.AlmostEqual(_nu, _numOps.One))
            {
                return _numOps.FromDouble(0.5); // For nu=1, the width derivative at r=0 is 0.5
            }
            return _numOps.Zero;
        }

        T epsilonR = _numOps.Multiply(_epsilon, r);

        // The derivative of J_nu(epsilon*r)/(epsilon*r)^nu with respect to epsilon is:
        // r * [J_(nu-1)(epsilon*r) - (nu/epsilon)*J_nu(epsilon*r)] / (epsilon*r)^nu - (nu/epsilon) * J_nu(epsilon*r)/(epsilon*r)^nu
        // = r * J_(nu-1)(epsilon*r)/(epsilon*r)^nu - (nu*r/epsilon) * J_nu(epsilon*r)/(epsilon*r)^nu - (nu/epsilon) * J_nu(epsilon*r)/(epsilon*r)^nu
        // = r * J_(nu-1)(epsilon*r)/(epsilon*r)^nu - (nu*(r+1)/epsilon) * J_nu(epsilon*r)/(epsilon*r)^nu

        T besselNu = MathHelper.BesselJ(_nu, epsilonR);
        T besselNuMinus1 = MathHelper.BesselJ(_numOps.Subtract(_nu, _numOps.One), epsilonR);

        T term1 = _numOps.Multiply(r, besselNuMinus1);

        T rPlusOne = _numOps.Add(r, _numOps.One);
        T nuRPlusOne = _numOps.Multiply(_nu, rPlusOne);
        T term2 = _numOps.Multiply(_numOps.Divide(nuRPlusOne, _epsilon), besselNu);

        T numerator = _numOps.Subtract(term1, term2);
        T denominator = _numOps.Power(epsilonR, _nu);

        return _numOps.Divide(numerator, denominator);
    }
}
