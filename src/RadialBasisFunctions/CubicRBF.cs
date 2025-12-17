namespace AiDotNet.RadialBasisFunctions;

/// <summary>
/// Implements a Cubic Radial Basis Function (RBF) that grows with the cube of the distance.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This class provides an implementation of a Radial Basis Function (RBF) that uses a cubic function
/// of the form f(r) = (r/width)³, where r is the radial distance and width is a scaling parameter.
/// Unlike many other RBFs that decrease with distance, the cubic RBF increases with the cube of the distance.
/// This makes it useful for certain regression and interpolation problems where larger responses are expected
/// for points farther from the centers.
/// </para>
/// <para><b>For Beginners:</b> A Radial Basis Function (RBF) is a special type of mathematical function
/// that depends only on the distance from a center point.
/// 
/// The Cubic RBF is unique compared to many other RBFs because its value grows larger as you move
/// away from the center, rather than smaller. Specifically, it grows with the cube of the distance
/// (distance × distance × distance).
/// 
/// Think of it like a bowl shape turned upside down - the further you go from the center,
/// the higher the value becomes, and it grows quite rapidly with distance.
/// 
/// This type of function is useful in certain modeling scenarios where you expect larger values
/// for points that are farther away from reference points. The width parameter lets you control
/// how quickly the function grows with distance.
/// </para>
/// </remarks>
public class CubicRBF<T> : IRadialBasisFunction<T>
{
    /// <summary>
    /// The numeric operations provider for type T, used for mathematical calculations.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// The width parameter controlling the scale of the function.
    /// </summary>
    private readonly T _width;

    /// <summary>
    /// Initializes a new instance of the <see cref="CubicRBF{T}"/> class with a specified width parameter.
    /// </summary>
    /// <param name="width">The width parameter, defaults to 1.0.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the Cubic Radial Basis Function with a specified width parameter.
    /// The width parameter acts as a scaling factor for the distance. Larger width values will cause
    /// the function to grow more slowly with distance, while smaller width values will cause it
    /// to grow more rapidly.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Cubic RBF with a specific width setting.
    /// 
    /// The width parameter controls how quickly the function grows as you move away from the center:
    /// - Smaller width values make the function grow very quickly with distance
    /// - Larger width values make the function grow more slowly with distance
    /// 
    /// If you're not sure what value to use, the default (width = 1.0) is a good starting point
    /// that provides moderate growth with distance.
    /// </para>
    /// </remarks>
    public CubicRBF(double width = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _width = _numOps.FromDouble(width);
    }

    /// <summary>
    /// Computes the value of the Cubic Radial Basis Function for a given radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The computed function value (r/width)³.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the value of the Cubic RBF for a given radius r. The formula used is
    /// (r/width)³, which grows as the cube of the normalized distance.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes the "height" or "value" of the Cubic function
    /// at a specific distance (r) from the center.
    /// 
    /// The calculation is straightforward:
    /// 1. Divide the distance (r) by the width parameter to get a scaled distance
    /// 2. Cube this value (multiply it by itself twice)
    /// 
    /// The result is a single number representing the function's value at the given distance.
    /// This value grows rapidly as you move away from the center point.
    /// </para>
    /// </remarks>
    public T Compute(T r)
    {
        T rOverWidth = _numOps.Divide(r, _width);
        return _numOps.Multiply(rOverWidth, _numOps.Multiply(rOverWidth, rOverWidth));
    }

    /// <summary>
    /// Computes the derivative of the Cubic RBF with respect to the radius.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to r.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Cubic RBF with respect to the radius r.
    /// The formula for the derivative is 3r²/width³, which is always positive for r > 0,
    /// indicating that the function always increases with distance.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes how fast the function's value changes
    /// as you move away from the center point.
    /// 
    /// The derivative tells you the "slope" or "rate of change" of the function at a specific distance.
    /// For the Cubic RBF, the derivative is:
    /// - Always positive (for r > 0), meaning the function always increases as you move away from the center
    /// - Grows with the square of the distance, so it increases more rapidly the further you go
    /// 
    /// This derivative is useful in machine learning applications when optimizing parameters
    /// using gradient-based methods.
    /// </para>
    /// </remarks>
    public T ComputeDerivative(T r)
    {
        T three = _numOps.FromDouble(3.0);
        T rSquared = _numOps.Multiply(r, r);
        T widthCubed = _numOps.Multiply(_width, _numOps.Multiply(_width, _width));
        return _numOps.Divide(_numOps.Multiply(three, rSquared), widthCubed);
    }

    /// <summary>
    /// Computes the derivative of the Cubic RBF with respect to the width parameter.
    /// </summary>
    /// <param name="r">The radius or distance from the center point.</param>
    /// <returns>The derivative value of the function with respect to width.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative of the Cubic RBF with respect to the width parameter.
    /// The formula for this derivative is -3r³/width4. The negative sign indicates that increasing
    /// the width parameter decreases the function value at any given radius.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how the function's value would change
    /// if you were to adjust the width parameter.
    /// 
    /// This is particularly important in machine learning applications:
    /// - When training a model with Cubic RBFs, we often need to adjust the width to fit the data better
    /// - This derivative tells us exactly how changing the width affects the output of the function
    /// - With this information, learning algorithms can automatically find the optimal value of width
    /// 
    /// The derivative is always negative (for r > 0), which means that increasing the width parameter
    /// always decreases the function value at any given distance.
    /// </para>
    /// </remarks>
    public T ComputeWidthDerivative(T r)
    {
        // For f(r) = (r/width)³, the width derivative is -3r³/width4
        T rCubed = _numOps.Multiply(r, _numOps.Multiply(r, r));
        T widthSquared = _numOps.Multiply(_width, _width);
        T widthFourth = _numOps.Multiply(widthSquared, widthSquared);
        T negThree = _numOps.FromDouble(-3.0);
        T numerator = _numOps.Multiply(negThree, rCubed);

        return _numOps.Divide(numerator, widthFourth);
    }
}
