namespace AiDotNet.RadialBasisFunctions;

/// <summary>
/// Implements a Wave (Sinc) Radial Basis Function (RBF) of the form sin(er)/(er).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This class provides an implementation of a Radial Basis Function (RBF) that uses a wave form
/// of f(r) = sin(er)/(er), where r is the radial distance and e (epsilon) is a shape parameter
/// controlling the frequency of oscillations. This function is also known as the spherical Bessel function
/// of the first kind of order zero, or more commonly as the "sinc" function when scaled.
/// </para>
/// <para>
/// Unlike most other RBFs that monotonically decrease with distance, the Wave RBF oscillates, creating
/// positive and negative lobes. This oscillatory behavior can be useful for modeling wave-like phenomena
/// or for approximating functions with periodic components. The function equals 1 at r = 0 and approaches 0
/// as r approaches infinity, but with oscillations that cross the zero axis.
/// </para>
/// <para><b>For Beginners:</b> A Radial Basis Function (RBF) is a special type of mathematical function
/// that depends only on the distance from a center point.
/// 
/// The Wave RBF is unique among RBFs because instead of simply decreasing with distance, it creates
/// a wave-like pattern that alternates between positive and negative values as you move away from
/// the center. Think of it like the ripples that spread out when you drop a stone in water - the
/// height of the water rises and falls in circles moving outward from where the stone hit.
/// 
/// This RBF has a parameter called epsilon (e) that controls how tightly packed these "ripples" are:
/// - A larger epsilon value creates more tightly packed ripples (higher frequency oscillations)
/// - A smaller epsilon value creates more widely spaced ripples (lower frequency oscillations)
/// 
/// The Wave RBF is particularly useful for modeling phenomena that naturally have wave-like properties,
/// such as sound, electromagnetic fields, or certain types of physical simulations.
/// </para>
/// </remarks>
public class WaveRBF<T> : IRadialBasisFunction<T>
{
    /// <summary>
    /// The shape parameter (epsilon) controlling the frequency of oscillations.
    /// </summary>
    private readonly T _epsilon;

    /// <summary>
    /// The numeric operations provider for type T, used for mathematical calculations.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="WaveRBF{T}"/> class with a specified shape parameter.
    /// </summary>
    /// <param name="epsilon">The shape parameter, defaults to 1.0.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the Wave Radial Basis Function with a specified shape parameter epsilon.
    /// Epsilon controls the frequency of oscillations in the function. Larger values of epsilon result in
    /// higher frequency oscillations, while smaller values result in lower frequency oscillations.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Wave RBF with a specific oscillation frequency setting.
    /// 
    /// The epsilon parameter controls how quickly the function oscillates with distance:
    /// - Larger epsilon values (like 5.0) create rapid oscillations with many cycles in a small space
    /// - Smaller epsilon values (like 0.1) create gentle oscillations that complete fewer cycles over longer distances
    /// 
    /// You can think of epsilon as controlling the "pitch" of the wave pattern:
    /// - High epsilon is like a high-pitched sound with rapid vibrations
    /// - Low epsilon is like a low-pitched sound with slower vibrations
    /// 
    /// If you're not sure what value to use, the default (epsilon = 1.0) is a good starting point
    /// that provides moderate oscillation frequency.
    /// </para>
    /// </remarks>
    public WaveRBF(double epsilon = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _epsilon = _numOps.FromDouble(epsilon);
    }

    /// <summary>
    /// Computes the value of the Wave Radial Basis Function for a given radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The computed function value sin(er)/(er), or 1 if r is very close to zero.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the value of the Wave RBF for a given radius r. The formula used is
    /// sin(er)/(er), which creates an oscillating pattern that eventually decays with distance.
    /// For r = 0, the function experiences a removable singularity and returns a value of 1.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes the "height" or "value" of the Wave function
    /// at a specific distance (r) from the center.
    /// 
    /// The calculation involves:
    /// 1. Multiplying the distance (r) by the epsilon parameter (er)
    /// 2. Computing the sine of this product (sin(er))
    /// 3. Dividing this sine value by the product from step 1 (sin(er)/(er))
    /// 
    /// At the center point (r = 0), this formula would involve division by zero, so a special case
    /// returns the value 1 (which is the mathematical limit of the function as r approaches 0).
    /// 
    /// The result is a single number representing the function's value at the given distance:
    /// - At the center (r = 0), the value is exactly 1
    /// - As you move away from the center, the value oscillates between positive and negative
    /// - The oscillations diminish in amplitude over distance, eventually approaching 0
    /// - The first zero crossing occurs at r = p/e
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
        T sinEpsilonR = MathHelper.Sin(epsilonR);
        return _numOps.Divide(sinEpsilonR, epsilonR);
    }

    /// <summary>
    /// Computes the derivative of the Wave RBF with respect to the radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to r.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Wave RBF with respect to the radius r.
    /// The formula for the derivative is (e·r·cos(er) + sin(er))/(er)². For r = 0, the derivative
    /// is 0 due to the limit as r approaches 0.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes how fast the function's value changes
    /// as you move away from the center point.
    /// 
    /// The derivative tells you the "slope" or "rate of change" of the function at a specific distance.
    /// For the Wave RBF, the derivative:
    /// - Is zero at the center point (r = 0)
    /// - Oscillates between positive and negative values as distance increases
    /// - The oscillations diminish in amplitude over distance
    /// 
    /// This oscillating derivative creates the wave-like pattern of the function, with alternating
    /// upward and downward slopes creating the rises and falls of the "ripples."
    /// 
    /// This derivative is useful in machine learning applications when optimizing parameters
    /// using gradient-based methods.
    /// </para>
    /// </remarks>
    public T ComputeDerivative(T r)
    {
        T epsilonR = _numOps.Multiply(_epsilon, r);

        // Handle the case when epsilonR is very close to zero
        if (MathHelper.AlmostEqual(epsilonR, _numOps.Zero))
        {
            // For er ? 0, the derivative approaches 0
            return _numOps.Zero;
        }

        // Calculate cos(er)
        T cosEpsilonR = MathHelper.Cos(epsilonR);

        // Calculate sin(er)
        T sinEpsilonR = MathHelper.Sin(epsilonR);

        // Calculate e·r·cos(er)
        T epsilonRCosEpsilonR = _numOps.Multiply(epsilonR, cosEpsilonR);

        // Calculate e·r·cos(er) + sin(er)
        T numerator = _numOps.Add(epsilonRCosEpsilonR, sinEpsilonR);

        // Calculate (er)²
        T epsilonRSquared = _numOps.Multiply(epsilonR, epsilonR);

        // Return (e·r·cos(er) + sin(er))/(er)²
        return _numOps.Divide(numerator, epsilonRSquared);
    }

    /// <summary>
    /// Computes the derivative of the Wave RBF with respect to the shape parameter epsilon.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to epsilon.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Wave RBF with respect to the shape parameter epsilon.
    /// The formula for this derivative is (r²·cos(er) + sin(er)/e)/(er)² for non-zero r,
    /// and -r²/3 for r approaching 0. This derivative is useful for optimizing the oscillation frequency.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how the function's value would change
    /// if you were to adjust the shape parameter (epsilon) that controls oscillation frequency.
    /// 
    /// This is particularly important in machine learning applications:
    /// - When training a model with Wave RBFs, we often need to adjust epsilon to fit the data better
    /// - This derivative tells us exactly how changing epsilon affects the output of the function
    /// - With this information, learning algorithms can automatically find the optimal value of epsilon
    /// 
    /// For the Wave RBF, the width derivative:
    /// - Has a special formula for points very close to the center (approaches -r²/3)
    /// - For other points, follows a complex pattern that depends on both the distance and the current epsilon value
    /// - Like the function itself, the width derivative oscillates as distance increases
    /// 
    /// The width derivative helps machine learning algorithms determine whether to increase or decrease
    /// the oscillation frequency to better fit the data.
    /// </para>
    /// </remarks>
    public T ComputeWidthDerivative(T r)
    {
        T epsilonR = _numOps.Multiply(_epsilon, r);
        T rSquared = _numOps.Multiply(r, r);

        // Handle the case when epsilonR is very close to zero
        if (MathHelper.AlmostEqual(epsilonR, _numOps.Zero))
        {
            // For er ? 0, the width derivative approaches -r²/3
            T negativeRSquaredDivThree = _numOps.Divide(
                _numOps.Negate(rSquared),
                _numOps.FromDouble(3.0)
            );
            return negativeRSquaredDivThree;
        }

        // Calculate cos(er)
        T cosEpsilonR = MathHelper.Cos(epsilonR);

        // Calculate sin(er)
        T sinEpsilonR = MathHelper.Sin(epsilonR);

        // Calculate r²·cos(er)
        T rSquaredCosEpsilonR = _numOps.Multiply(rSquared, cosEpsilonR);

        // Calculate sin(er)/e
        T sinEpsilonRDivEpsilon = _numOps.Divide(sinEpsilonR, _epsilon);

        // Calculate r²·cos(er) + sin(er)/e
        T numerator = _numOps.Add(rSquaredCosEpsilonR, sinEpsilonRDivEpsilon);

        // Calculate (er)²
        T epsilonRSquared = _numOps.Multiply(epsilonR, epsilonR);

        // Return (r²·cos(er) + sin(er)/e)/(er)²
        return _numOps.Divide(numerator, epsilonRSquared);
    }
}
