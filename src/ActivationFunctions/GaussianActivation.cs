namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Gaussian activation function for neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// For Beginners: The Gaussian activation function is based on the bell-shaped curve that you might 
/// recognize from statistics (the "normal distribution" or "bell curve"). 
/// 
/// Key properties of the Gaussian activation function:
/// - It outputs values between 0 and 1
/// - The highest output (1) occurs when the input is 0
/// - As inputs move away from 0 (either positive or negative), the output approaches 0
/// - It's symmetric around the y-axis (f(-x) = f(x))
/// 
/// Unlike many other activation functions, Gaussian responds strongly to inputs near zero and 
/// weakly to inputs far from zero in either direction. This makes it useful for:
/// - Radial Basis Function (RBF) networks
/// - Pattern recognition tasks
/// - Problems where distance from a central point is important
/// 
/// The mathematical formula is: f(x) = exp(-x²)
/// </para>
/// </remarks>
public class GaussianActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Indicates that this activation function supports operations on individual scalar values.
    /// </summary>
    /// <returns>Always returns true as Gaussian can be applied to scalar values.</returns>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the Gaussian activation function to a single input value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The activated output value using the Gaussian function.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: This method transforms an input value using the formula:
    /// f(x) = exp(-x²)
    /// 
    /// In simpler terms:
    /// - When input is 0, the output is 1 (the peak of the bell curve)
    /// - As the input moves away from 0 in either direction, the output gets closer to 0
    /// - The output is always positive and never exceeds 1
    /// 
    /// This creates a bell-shaped response curve that is strongest at the center and 
    /// gradually fades as you move away from the center.
    /// </para>
    /// </remarks>
    public override T Activate(T input)
    {
        // f(x) = exp(-x^2)
        T negativeSquare = NumOps.Negate(NumOps.Multiply(input, input));
        return NumOps.Exp(negativeSquare);
    }

    /// <summary>
    /// Calculates the derivative of the Gaussian function for a single input value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The derivative value at the input point.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: The derivative measures how much the Gaussian function's output changes
    /// when its input changes slightly. This is used during neural network training to determine
    /// how to adjust weights.
    /// 
    /// The derivative of the Gaussian function has these properties:
    /// - At x = 0, the derivative is 0 (the function is at its peak and momentarily flat)
    /// - For positive inputs, the derivative is negative (the function is decreasing)
    /// - For negative inputs, the derivative is positive (the function is increasing)
    /// - The derivative approaches 0 as inputs get very large in either direction
    /// 
    /// The mathematical formula is: f'(x) = -2x * exp(-x²)
    /// </para>
    /// </remarks>
    public override T Derivative(T input)
    {
        // f'(x) = -2x * exp(-x^2)
        T activationValue = Activate(input);
        T negativeTwo = NumOps.FromDouble(-2);

        return NumOps.Multiply(NumOps.Multiply(negativeTwo, input), activationValue);
    }
}