namespace AiDotNet.RadialBasisFunctions;

/// <summary>
/// Implements a Polyharmonic Spline Radial Basis Function (RBF) with different forms based on a parameter k.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This class provides an implementation of Polyharmonic Spline Radial Basis Functions, which are defined as:
/// f(r) = r^k for odd k
/// f(r) = r^k * log(r) for even k
/// where r is the radial distance and k is an integer parameter (typically k = 1).
/// </para>
/// <para>
/// Polyharmonic splines are used in scattered data interpolation, numerical solutions of partial differential
/// equations, and image processing. They are particularly useful for problems in multiple dimensions
/// due to their theoretical properties. Unlike many other RBFs, polyharmonic splines do not have a width
/// parameter, making them scale-invariant. The parameter k controls the smoothness of the resulting
/// interpolation, with higher values of k producing smoother functions.
/// </para>
/// <para><b>For Beginners:</b> A Radial Basis Function (RBF) is a special type of mathematical function
/// that depends only on the distance from a center point.
/// 
/// The Polyharmonic Spline RBF is used to create smooth curves or surfaces that pass through a set of points.
/// Think of it like drawing a smooth line through dots on a page, but it works in any number of dimensions.
/// 
/// This particular RBF comes in different "flavors" depending on the value of parameter k:
/// - When k is odd (1, 3, 5, etc.): The function is simply the distance raised to the power of k
/// - When k is even (2, 4, 6, etc.): The function is the distance raised to the power of k, multiplied by the logarithm of the distance
/// 
/// A unique property of polyharmonic splines is that they don't have a width parameter like most other RBFs.
/// This makes them "scale-invariant" - scaling all your input distances by the same factor only changes
/// the output by a constant factor, which doesn't affect the shape of the resulting interpolation.
/// 
/// Common choices for k include:
/// - k = 1: "Linear" (r)
/// - k = 2: "Thin plate spline" (r² log r)
/// - k = 3: "Cubic" (r³)
/// </para>
/// </remarks>
public class PolyharmonicSplineRBF<T> : IRadialBasisFunction<T>
{
    /// <summary>
    /// The numeric operations provider for type T, used for mathematical calculations.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// The parameter k that determines the type and order of the polyharmonic spline.
    /// </summary>
    private readonly int _k;

    /// <summary>
    /// Initializes a new instance of the <see cref="PolyharmonicSplineRBF{T}"/> class with a specified k parameter.
    /// </summary>
    /// <param name="k">The order parameter for the polyharmonic spline, defaults to 2.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the Polyharmonic Spline RBF with a specified k parameter. The value of k
    /// determines the form and smoothness of the function. For odd k, the function is r^k, and for even k,
    /// the function is r^k * log(r). Commonly used values are k = 1 (linear), k = 2 (thin plate spline),
    /// and k = 3 (cubic).
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Polyharmonic Spline RBF with a specific order setting.
    /// 
    /// The k parameter controls the "order" or "smoothness" of the function:
    /// - k = 1: Creates a linear function (r)
    /// - k = 2: Creates a thin plate spline (r² log r), which is the default and commonly used
    /// - k = 3: Creates a cubic function (r³)
    /// - Higher values of k create even smoother functions
    /// 
    /// Higher values of k produce smoother interpolations, but can sometimes lead to numerical issues.
    /// If you're not sure what value to use, the default (k = 2) is a good starting point
    /// that works well for many applications.
    /// </para>
    /// </remarks>
    public PolyharmonicSplineRBF(int k = 2)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _k = k;
    }

    /// <summary>
    /// Computes the value of the Polyharmonic Spline Radial Basis Function for a given radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The computed function value based on the k parameter.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the value of the Polyharmonic Spline RBF for a given radius r.
    /// For odd values of k, the formula used is r^k.
    /// For even values of k, the formula used is r^k * log(r).
    /// For r = 0, the function returns 0 to avoid numerical issues with logarithms.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes the function's value at a specific distance (r) from the center.
    /// 
    /// The calculation depends on whether k is an odd or even number:
    /// - For odd k (1, 3, 5, ...): The result is simply r raised to the power of k (r^k)
    /// - For even k (2, 4, 6, ...): The result is r raised to power k, multiplied by the natural logarithm of r (r^k * log(r))
    /// 
    /// At the center point (r = 0), the function always returns 0.
    /// 
    /// For example, with the default k = 2:
    /// - At r = 0, the value is 0
    /// - At r = 1, the value is 0 (since log(1) = 0)
    /// - At r = 2, the value is 2² * log(2) ˜ 4 * 0.693 ˜ 2.77
    /// </para>
    /// </remarks>
    public T Compute(T r)
    {
        if (_numOps.Equals(r, _numOps.Zero))
        {
            return _numOps.Zero;
        }
        if (_k % 2 == 0)
        {
            // For even k: r^k * log(r)
            T rPowK = _numOps.Power(r, _numOps.FromDouble(_k));
            return _numOps.Multiply(rPowK, _numOps.Log(r));
        }
        else
        {
            // For odd k: r^k
            return _numOps.Power(r, _numOps.FromDouble(_k));
        }
    }

    /// <summary>
    /// Computes the derivative of the Polyharmonic Spline RBF with respect to the radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to r.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Polyharmonic Spline RBF with respect to the radius r.
    /// For odd values of k, the derivative is k * r^(k-1).
    /// For even values of k, the derivative is r^(k-1) * (k * log(r) + 1).
    /// Special cases are handled for r = 0 depending on the value of k.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes how fast the function's value changes
    /// as you move away from the center point.
    /// 
    /// The derivative tells you the "slope" or "rate of change" of the function at a specific distance.
    /// The calculation again depends on whether k is odd or even:
    /// - For odd k (1, 3, 5, ...): The derivative is k * r^(k-1)
    /// - For even k (2, 4, 6, ...): The derivative is r^(k-1) * (k * log(r) + 1)
    /// 
    /// At the center point (r = 0), special handling is needed:
    /// - For k = 1, the derivative is 1
    /// - For k > 1, the derivative is 0
    /// 
    /// For example, with k = 2:
    /// - At r = 1, the derivative is 1^1 * (2 * log(1) + 1) = 1 * (0 + 1) = 1
    /// - At r = 2, the derivative is 2^1 * (2 * log(2) + 1) = 2 * (2 * 0.693 + 1) ˜ 2 * 2.386 ˜ 4.77
    /// </para>
    /// </remarks>
    public T ComputeDerivative(T r)
    {
        // Handle r = 0 case
        if (_numOps.Equals(r, _numOps.Zero))
        {
            // The derivative at r = 0 depends on k
            if (_k == 1)
            {
                // For k = 1, the derivative is 1
                return _numOps.One;
            }
            else
            {
                // For k > 1, the derivative is 0
                return _numOps.Zero;
            }
        }
        if (_k % 2 == 0)
        {
            // For even k: d/dr[r^k * log(r)] = r^(k-1) * (k * log(r) + 1)

            // Calculate r^(k-1)
            T rPowKMinus1 = _numOps.Power(r, _numOps.FromDouble(_k - 1));

            // Calculate k * log(r)
            T kLogR = _numOps.Multiply(_numOps.FromDouble(_k), _numOps.Log(r));

            // Calculate k * log(r) + 1
            T term = _numOps.Add(kLogR, _numOps.One);

            // Return r^(k-1) * (k * log(r) + 1)
            return _numOps.Multiply(rPowKMinus1, term);
        }
        else
        {
            // For odd k: d/dr[r^k] = k * r^(k-1)

            // Calculate r^(k-1)
            T rPowKMinus1 = _numOps.Power(r, _numOps.FromDouble(_k - 1));

            // Return k * r^(k-1)
            return _numOps.Multiply(_numOps.FromDouble(_k), rPowKMinus1);
        }
    }

    /// <summary>
    /// Computes the derivative of the Polyharmonic Spline RBF with respect to a width parameter.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>Always zero since polyharmonic splines don't have a width parameter.</returns>
    /// <remarks>
    /// <para>
    /// This method always returns zero because polyharmonic spline RBFs do not have a width parameter.
    /// Unlike many other RBFs, polyharmonic splines are scale-invariant, so they don't need a parameter
    /// to control their width. The method is implemented only to satisfy the interface requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This method would normally tell you how the function's value
    /// would change if you adjusted the width parameter.
    /// 
    /// However, the Polyharmonic Spline RBF doesn't have a width parameter to adjust - its behavior
    /// is controlled only by the k parameter, which determines the order of the function.
    /// 
    /// Since there's no width parameter to vary, changing a non-existent parameter has no effect,
    /// so the derivative is zero.
    /// 
    /// This method only exists because all RBFs in this library are required to have it,
    /// but for the Polyharmonic Spline RBF, it will always return zero.
    /// </para>
    /// </remarks>
    public T ComputeWidthDerivative(T r)
    {
        // The polyharmonic spline doesn't have a width parameter,
        // so the derivative with respect to width is 0
        return _numOps.Zero;
    }
}
