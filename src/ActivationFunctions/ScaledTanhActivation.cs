using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Scaled Hyperbolic Tangent (tanh) activation function for neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Scaled Tanh activation function is a parameterized version of the standard
/// hyperbolic tangent function. Like the standard tanh, it outputs values between -1 and 1, making
/// it useful for neural networks where you want the output to be centered around zero.
/// 
/// The mathematical formula is: f(x) = (1 - e^(-ßx)) / (1 + e^(-ßx))
/// 
/// This is equivalent to the standard tanh function when ß = 2, and has these key properties:
/// - Outputs values between -1 and 1
/// - Is symmetric around the origin (f(-x) = -f(x))
/// - The parameter ß (beta) controls the steepness of the curve
/// - When ß = 2, this is exactly equivalent to the standard tanh function
/// 
/// When to use it:
/// - When you need outputs centered around zero
/// - For hidden layers in many types of neural networks
/// - When dealing with data that naturally has both positive and negative values
/// - When you want to control the steepness of the activation function
/// </para>
/// </remarks>
public class ScaledTanhActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// The steepness parameter that controls how quickly the function transitions between -1 and 1.
    /// </summary>
    private readonly T _beta;

    /// <summary>
    /// Initializes a new instance of the ScaledTanhActivation class with the specified steepness parameter.
    /// </summary>
    /// <param name="beta">The steepness parameter. Higher values make the curve steeper. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The beta parameter controls how "steep" the S-curve of the function is.
    /// - A higher beta value (e.g., 2.0) makes the transition from -1 to 1 happen more quickly
    /// - A lower beta value (e.g., 0.5) makes the transition more gradual
    /// - When beta = 2.0, this function is exactly equivalent to the standard tanh function
    /// 
    /// The default value of 1.0 works well for most applications, but you might adjust it if:
    /// - Your network is learning too slowly (try increasing beta)
    /// - Your network is unstable during training (try decreasing beta)
    /// </para>
    /// </remarks>
    public ScaledTanhActivation(double beta = 1.0)
    {
        _beta = NumOps.FromDouble(beta);
    }

    /// <summary>
    /// Indicates that this activation function supports operations on individual scalar values.
    /// </summary>
    /// <returns>Always returns true as Scaled Tanh can be applied to scalar values.</returns>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the Scaled Tanh activation function to a single input value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The activated output value between -1 and 1.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method transforms an input value using the formula:
    /// f(x) = (1 - e^(-ßx)) / (1 + e^(-ßx))
    /// 
    /// No matter how large or small the input is, the output will always be between -1 and 1:
    /// - Large positive inputs produce values close to 1
    /// - Large negative inputs produce values close to -1
    /// - An input of 0 produces an output of 0
    /// 
    /// This "squashing" property makes the Scaled Tanh useful for normalizing outputs.
    /// When ß = 2, this function is mathematically identical to the standard tanh function.
    /// </para>
    /// </remarks>
    public override T Activate(T input)
    {
        // f(x) = (1 - exp(-ßx)) / (1 + exp(-ßx))
        T negBetaX = NumOps.Negate(NumOps.Multiply(_beta, input));
        T expNegBetaX = NumOps.Exp(negBetaX);
        T numerator = NumOps.Subtract(NumOps.One, expNegBetaX);
        T denominator = NumOps.Add(NumOps.One, expNegBetaX);

        return NumOps.Divide(numerator, denominator);
    }

    /// <summary>
    /// Calculates the derivative of the Scaled Tanh function for a single input value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The derivative value at the input point.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The derivative measures how much the Scaled Tanh function's output changes
    /// when its input changes slightly. This is used during neural network training to determine
    /// how to adjust weights.
    /// 
    /// The derivative formula is: f'(x) = ß * (1 - f(x)²)
    /// 
    /// Key properties of this derivative:
    /// - It's highest at x = 0 (where the function is steepest)
    /// - It approaches zero for very large positive or negative inputs
    /// - This means the network learns most effectively from inputs near zero
    /// 
    /// The "vanishing gradient problem" can occur when inputs are very large in magnitude,
    /// causing very small derivatives that slow down learning.
    /// </para>
    /// </remarks>
    public override T Derivative(T input)
    {
        // f'(x) = ß * (1 - f(x)^2)
        T activationValue = Activate(input);
        T squaredActivation = NumOps.Multiply(activationValue, activationValue);
        T oneMinus = NumOps.Subtract(NumOps.One, squaredActivation);

        return NumOps.Multiply(_beta, oneMinus);
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>False because gradient computation is not yet implemented.</value>
    /// <remarks>
    /// <para>
    /// This activation does not yet support JIT compilation because the gradient
    /// computation (backward pass) has not been implemented in TensorOperations.ScaledTanh.
    /// </para>
    /// <para>
    /// To enable JIT support:
    /// 1. Implement the backward pass in TensorOperations.ScaledTanh
    /// 2. Test the gradient computation
    /// 3. Change SupportsJitCompilation to return true
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => false;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with ScaledTanh activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <exception cref="NotSupportedException">Thrown because gradient is not implemented.</exception>
    /// <remarks>
    /// <para>
    /// This method would map the activation to TensorOperations&lt;T&gt;.ScaledTanh(input)
    /// once the gradient computation is implemented.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        double scale = NumOps.ToDouble(_beta);
        return TensorOperations<T>.ScaledTanh(input, scale);
    }
}
