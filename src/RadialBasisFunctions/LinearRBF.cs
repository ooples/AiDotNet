namespace AiDotNet.RadialBasisFunctions;

/// <summary>
/// Implements a Linear Radial Basis Function (RBF) of the form f(r) = r.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This class provides an implementation of a Radial Basis Function (RBF) that simply returns the radius itself.
/// Unlike most RBFs which reach their maximum at the center and decrease with distance, the Linear RBF
/// increases linearly with distance from the center. It has the simplest possible form of any RBF: f(r) = r.
/// Note that this function does not have a width parameter like most other RBFs.
/// </para>
/// <para>
/// The Linear RBF can be useful in specific applications where a direct proportionality to distance is desired.
/// It is also sometimes used as a component in more complex kernels or in combination with other RBFs.
/// However, it should be noted that this function does not approach zero as distance increases, which may
/// make it unsuitable for many typical RBF applications requiring localized influence.
/// </para>
/// <para><b>For Beginners:</b> A Radial Basis Function (RBF) is a special type of mathematical function
/// that depends only on the distance from a center point.
/// 
/// The Linear RBF is the simplest possible RBF - it's just the distance itself. Think of it as a straight line
/// starting from zero at the center and growing uniformly as you move away in any direction.
/// 
/// This is different from most other RBFs because:
/// - Most RBFs have their highest value at the center and decrease with distance
/// - The Linear RBF starts at zero at the center and increases with distance
/// - Most RBFs have a "width" parameter to control how quickly they change with distance
/// - The Linear RBF has no width parameter - it always increases at the same rate
/// 
/// The Linear RBF is rarely used alone in practice, but it can be useful in certain specific applications
/// or as a building block for more complex functions.
/// </para>
/// </remarks>
public class LinearRBF<T> : IRadialBasisFunction<T>
{
    /// <summary>
    /// The numeric operations provider for type T, used for mathematical calculations.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="LinearRBF{T}"/> class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The constructor initializes the Linear Radial Basis Function. Unlike most other RBFs,
    /// the Linear RBF does not take any parameters as it has the fixed form f(r) = r.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Linear RBF.
    /// 
    /// Unlike other RBFs we've seen, the Linear RBF doesn't need any configuration parameters
    /// because its behavior is fixed - it always returns exactly the distance value.
    /// There's no "width" parameter or other settings to adjust.
    /// </para>
    /// </remarks>
    public LinearRBF()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Computes the value of the Linear Radial Basis Function for a given radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The radius value itself.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the value of the Linear RBF for a given radius r.
    /// For the Linear RBF, this is simply the radius itself: f(r) = r.
    /// </para>
    /// <para><b>For Beginners:</b> This method simply returns the distance value unchanged.
    /// 
    /// For example:
    /// - If you input a distance of 2.5, you get back 2.5
    /// - If you input a distance of 10, you get back 10
    /// 
    /// There's no complex calculation needed - this function passes through the input value
    /// without modifying it.
    /// </para>
    /// </remarks>
    public T Compute(T r)
    {
        return r;
    }

    /// <summary>
    /// Computes the derivative of the Linear RBF with respect to the radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The constant value 1.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Linear RBF with respect to the radius r.
    /// Since the function is f(r) = r, its derivative is simply 1, regardless of the radius value.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you how fast the function's value changes
    /// as you move away from the center point.
    /// 
    /// For the Linear RBF, the rate of change is constant: the function value increases by exactly 1
    /// for each unit of distance you move. This is why the derivative is always 1, regardless of
    /// where you are.
    /// 
    /// Think of it like walking up a ramp with a steady incline - no matter where you are on the ramp,
    /// you're always climbing at the same rate.
    /// </para>
    /// </remarks>
    public T ComputeDerivative(T r)
    {
        // The derivative of r with respect to r is 1
        return _numOps.One;
    }

    /// <summary>
    /// Computes the derivative of the Linear RBF with respect to a width parameter.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The constant value 0.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Linear RBF with respect to a width parameter.
    /// Since the Linear RBF does not have a width parameter (it has the fixed form f(r) = r),
    /// this derivative is zero. The method is implemented only to satisfy the interface requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This method would normally tell you how the function's value
    /// would change if you adjusted the width parameter.
    /// 
    /// However, the Linear RBF doesn't have a width parameter to adjust - its behavior is fixed.
    /// Since there's no parameter to vary, changing a non-existent parameter has no effect,
    /// so the derivative is zero.
    /// 
    /// This method only exists because all RBFs in this library are required to have it,
    /// but for the Linear RBF, it will always return zero.
    /// </para>
    /// </remarks>
    public T ComputeWidthDerivative(T r)
    {
        // Since there is no width parameter, the derivative is 0
        return _numOps.Zero;
    }
}
