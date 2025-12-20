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
    /// The derivative formula is: f'(x) = (ß / 2) * (1 - f(x)²)
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
        // f'(x) = (ß / 2) * (1 - f(x)^2)
        T activationValue = Activate(input);
        T squaredActivation = NumOps.Multiply(activationValue, activationValue);
        T oneMinus = NumOps.Subtract(NumOps.One, squaredActivation);

        // (ß / 2) * (1 - f(x)^2)
        T half = NumOps.FromDouble(0.5);
        T scaledBeta = NumOps.Multiply(_beta, half);
        return NumOps.Multiply(scaledBeta, oneMinus);
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because gradient computation is fully implemented in TensorOperations.ScaledTanh.</value>
    /// <remarks>
    /// <para>
    /// ScaledTanh supports JIT compilation because:
    /// - The gradient computation (backward pass) is fully implemented in TensorOperations
    /// - The gradient is (β / 2) * (1 - f(x)²)
    /// - The steepness parameter β allows tuning network behavior
    /// - It can be represented as a static computation graph node
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with ScaledTanh activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps the ScaledTanh activation to TensorOperations&lt;T&gt;.ScaledTanh(input, beta),
    /// which handles both forward and backward passes for JIT compilation.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        double betaDouble = Convert.ToDouble(_beta);
        return TensorOperations<T>.ScaledTanh(input, betaDouble);
    }
}
