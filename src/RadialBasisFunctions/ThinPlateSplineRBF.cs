namespace AiDotNet.RadialBasisFunctions;

/// <summary>
/// Implements a Thin Plate Spline Radial Basis Function (RBF) of the form r² log(r).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This class provides an implementation of a Thin Plate Spline Radial Basis Function, which is defined as
/// f(r) = r² log(r), where r is the radial distance. This is a special case of the polyharmonic spline with k = 2.
/// The Thin Plate Spline RBF does not have a width parameter, making it scale-invariant.
/// </para>
/// <para>
/// The name "Thin Plate Spline" comes from the physical analogy of bending a thin sheet of metal. This RBF
/// minimizes a measure of energy that approximates the bending energy of a thin metal plate. It is particularly
/// useful for interpolation problems in two dimensions and provides a smooth interpolation that avoids
/// unnecessary oscillations.
/// </para>
/// <para>
/// Unlike many other RBFs, the thin plate spline grows with distance rather than decaying. At r = 0, the
/// function and its first derivative are both 0, providing a high degree of smoothness at the origin.
/// </para>
/// <para><b>For Beginners:</b> A Radial Basis Function (RBF) is a special type of mathematical function
/// that depends only on the distance from a center point.
/// 
/// The Thin Plate Spline RBF is named after the physical behavior of a thin metal plate that bends
/// under pressure. Imagine pressing down on a thin sheet of metal at specific points - the way the
/// sheet bends to smoothly connect those points is similar to how this function behaves.
/// 
/// This RBF has some unique properties:
/// - Unlike most RBFs that decrease with distance, the thin plate spline actually grows as you move away from the center
/// - It equals exactly 0 at the center point (r = 0)
/// - It doesn't have a width parameter like most other RBFs, making it "scale-invariant"
/// - It creates very smooth interpolations with minimal unnecessary wiggles or oscillations
/// 
/// The thin plate spline is particularly useful for 2D interpolation problems, like reconstructing
/// a surface from a set of scattered points, or for image warping and morphing in computer graphics.
/// </para>
/// </remarks>
public class ThinPlateSplineRBF<T> : IRadialBasisFunction<T>
{
    /// <summary>
    /// The numeric operations provider for type T, used for mathematical calculations.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="ThinPlateSplineRBF{T}"/> class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The constructor initializes the Thin Plate Spline Radial Basis Function. Unlike most other RBFs,
    /// the Thin Plate Spline RBF does not take any parameters as it has the fixed form f(r) = r² log(r).
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Thin Plate Spline RBF.
    /// 
    /// Unlike other RBFs we've seen, the Thin Plate Spline doesn't need any configuration parameters
    /// because its behavior is fixed - it always follows the same mathematical formula: r² log(r).
    /// There's no "width" parameter or other settings to adjust.
    /// 
    /// This means that once you create a Thin Plate Spline RBF, it's ready to use without additional setup.
    /// </para>
    /// </remarks>
    public ThinPlateSplineRBF()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Computes the value of the Thin Plate Spline Radial Basis Function for a given radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The computed function value r² log(r), or zero if r = 0.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the value of the Thin Plate Spline RBF for a given radius r.
    /// The formula used is r² log(r). For r = 0, the function returns 0 to avoid numerical issues with logarithms.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes the function's value at a specific distance (r) from the center.
    /// 
    /// The calculation involves:
    /// 1. Squaring the distance (r² = r * r)
    /// 2. Computing the natural logarithm of the distance (log(r))
    /// 3. Multiplying these two values together (r² * log(r))
    /// 
    /// For the special case when distance is exactly 0, the function returns 0 (because log(0) is undefined).
    /// 
    /// Some key properties of this function:
    /// - At r = 0, the value is 0
    /// - For very small r (close to 0), the value is negative because log(r) is negative when r < 1
    /// - At r = 1, the value is 0 again (since log(1) = 0)
    /// - For r > 1, the value grows increasingly positive as r increases
    /// </para>
    /// </remarks>
    public T Compute(T r)
    {
        if (_numOps.Equals(r, _numOps.Zero))
        {
            return _numOps.Zero;
        }

        T r2 = _numOps.Multiply(r, r);
        return _numOps.Multiply(r2, _numOps.Log(r));
    }

    /// <summary>
    /// Computes the derivative of the Thin Plate Spline RBF with respect to the radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to r, or zero if r = 0.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Thin Plate Spline RBF with respect to the radius r.
    /// The formula for the derivative is r * (2 * log(r) + 1). At r = 0, the derivative is 0 for
    /// numerical stability.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes how fast the function's value changes
    /// as you move away from the center point.
    /// 
    /// The derivative tells you the "slope" or "rate of change" of the function at a specific distance.
    /// For the Thin Plate Spline RBF:
    /// - At the center (r = 0), the derivative is 0, meaning the function is flat at the origin
    /// - For small values of r (between 0 and about 0.61), the derivative is negative, meaning the function decreases
    /// - At r ˜ 0.61 (where 2*log(r)+1 = 0), the derivative is 0 again (a local minimum)
    /// - For r > 0.61, the derivative is positive and increasing, meaning the function grows faster and faster
    /// 
    /// This pattern creates the distinctive shape of the thin plate spline - a smooth dip around the origin
    /// followed by an increasing rise as you move further out.
    /// </para>
    /// </remarks>
    public T ComputeDerivative(T r)
    {
        // For r = 0, the derivative is 0
        if (_numOps.Equals(r, _numOps.Zero))
        {
            return _numOps.Zero;
        }

        // Calculate log(r)
        T logR = _numOps.Log(r);

        // Calculate 2 * log(r)
        T twoLogR = _numOps.Multiply(_numOps.FromDouble(2.0), logR);

        // Calculate 2 * log(r) + 1
        T term = _numOps.Add(twoLogR, _numOps.One);

        // Return r * (2 * log(r) + 1)
        return _numOps.Multiply(r, term);
    }

    /// <summary>
    /// Computes the derivative of the Thin Plate Spline RBF with respect to a width parameter.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>Always zero since thin plate splines don't have a width parameter.</returns>
    /// <remarks>
    /// <para>
    /// This method always returns zero because the Thin Plate Spline RBF does not have a width parameter.
    /// Unlike many other RBFs, the thin plate spline is scale-invariant, so it doesn't need a parameter
    /// to control its width. The method is implemented only to satisfy the interface requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This method would normally tell you how the function's value
    /// would change if you adjusted the width parameter.
    /// 
    /// However, the Thin Plate Spline RBF doesn't have a width parameter to adjust - its behavior
    /// is controlled by the mathematical formula r² log(r) with no additional parameters.
    /// 
    /// This property makes the function "scale-invariant," which means that if you scale all input
    /// distances by the same factor, the relative shape of the resulting interpolation doesn't change.
    /// 
    /// This method only exists because all RBFs in this library are required to have it,
    /// but for the Thin Plate Spline RBF, it will always return zero.
    /// </para>
    /// </remarks>
    public T ComputeWidthDerivative(T r)
    {
        // The thin plate spline doesn't have a width parameter,
        // so the derivative with respect to width is 0
        return _numOps.Zero;
    }
}
