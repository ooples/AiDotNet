using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Bent Identity activation function for neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Bent Identity activation function is a smoother alternative to the ReLU function.
/// It behaves similarly to a linear function for positive inputs but has a gentle curve for negative inputs.
/// This helps prevent the "dying neuron" problem that can occur with ReLU, where neurons can get stuck
/// outputting zero.
/// 
/// The mathematical formula is: f(x) = ((v(x² + 1) - 1) / 2) + x
/// 
/// Key properties:
/// - Always produces a non-zero gradient, helping with training
/// - Approximates linear behavior for large positive values
/// - Provides a smooth transition around zero
/// - Has no upper or lower bounds (unlike sigmoid or tanh)
/// </para>
/// </remarks>
public class BentIdentityActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Indicates that this activation function supports operations on individual scalar values.
    /// </summary>
    /// <returns>Always returns true as Bent Identity can be applied to scalar values.</returns>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the Bent Identity activation function to a single input value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The activated output value using the Bent Identity function.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method transforms an input value using the formula:
    /// f(x) = ((v(x² + 1) - 1) / 2) + x
    /// 
    /// The function adds a non-linear component to the identity function (x),
    /// making it bend slightly while maintaining good gradient properties.
    /// </para>
    /// </remarks>
    public override T Activate(T input)
    {
        // f(x) = (sqrt(x^2 + 1) - 1) / 2 + x
        T squarePlusOne = NumOps.Add(NumOps.Multiply(input, input), NumOps.One);
        T sqrtTerm = NumOps.Sqrt(squarePlusOne);
        T firstTerm = NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Subtract(sqrtTerm, NumOps.One));

        return NumOps.Add(firstTerm, input);
    }

    /// <summary>
    /// Calculates the derivative of the Bent Identity function for a single input value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The derivative value at the input point.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The derivative measures how much the Bent Identity function's output changes
    /// when its input changes slightly. This is used during neural network training to determine
    /// how to adjust weights.
    /// 
    /// The derivative formula is: f'(x) = x / (2 * v(x² + 1)) + 1
    /// 
    /// An important property is that this derivative is always greater than 1, which helps prevent
    /// the vanishing gradient problem during training.
    /// </para>
    /// </remarks>
    public override T Derivative(T input)
    {
        // f'(x) = x / (2 * sqrt(x^2 + 1)) + 1
        T squarePlusOne = NumOps.Add(NumOps.Multiply(input, input), NumOps.One);
        T sqrtTerm = NumOps.Sqrt(squarePlusOne);
        T firstTerm = NumOps.Divide(input, NumOps.Multiply(NumOps.FromDouble(2), sqrtTerm));

        return NumOps.Add(firstTerm, NumOps.One);
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>False because gradient computation is not yet implemented.</value>
    /// <remarks>
    /// <para>
    /// This activation does not yet support JIT compilation because the gradient
    /// computation (backward pass) has not been implemented in TensorOperations.BentIdentity.
    /// </para>
    /// <para>
    /// To enable JIT support:
    /// 1. Implement the backward pass in TensorOperations.BentIdentity
    /// 2. Test the gradient computation
    /// 3. Change SupportsJitCompilation to return true
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with BentIdentity activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return TensorOperations<T>.BentIdentity(input);
    }
}