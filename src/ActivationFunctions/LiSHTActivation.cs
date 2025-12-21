using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Linearly Scaled Hyperbolic Tangent (LiSHT) activation function for neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The LiSHT activation function combines the input value with its hyperbolic tangent.
/// 
/// The formula is: f(x) = x * tanh(x)
/// 
/// This means:
/// - For positive inputs: The output is positive but scaled down
/// - For negative inputs: The output is negative but scaled down
/// - For zero: The output is zero
/// 
/// LiSHT has several advantages:
/// - It doesn't suffer from the "vanishing gradient problem" (where learning becomes very slow)
/// - It's smooth everywhere (unlike ReLU which has a sharp corner at zero)
/// - It naturally keeps values in a reasonable range
/// 
/// Think of it as a function that "squeezes" large values while preserving the sign and 
/// allowing small values to pass through with minimal change.
/// </para>
/// </remarks>
public class LiSHTActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Determines if the activation function supports operations on individual scalar values.
    /// </summary>
    /// <returns>True, as LiSHT can be applied to individual values.</returns>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the LiSHT activation function to a single input value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The result of x * tanh(x).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes a single number and transforms it using the LiSHT function.
    /// 
    /// It first calculates tanh(x), which is a value between -1 and 1, and then multiplies
    /// the original input by this value. This creates a smooth curve that passes through zero
    /// and grows more slowly than a straight line.
    /// </para>
    /// </remarks>
    public override T Activate(T input)
    {
        // f(x) = x * tanh(x)
        T tanhInput = MathHelper.Tanh(input);
        return NumOps.Multiply(input, tanhInput);
    }

    /// <summary>
    /// Calculates the derivative of the LiSHT function for a single input value.
    /// </summary>
    /// <param name="input">The input value at which to calculate the derivative.</param>
    /// <returns>The derivative value at the input point.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The derivative tells us how quickly the LiSHT function is changing at a specific point.
    /// 
    /// This is crucial for neural network training because it helps determine how to adjust the weights
    /// during backpropagation. The formula looks complex, but it's just the mathematical way to express
    /// how sensitive the output is to small changes in the input.
    /// 
    /// The derivative of LiSHT combines:
    /// - The tanh of the input
    /// - A term that accounts for how the tanh itself changes
    /// </para>
    /// </remarks>
    public override T Derivative(T input)
    {
        // f'(x) = tanh(x) + x * (1 - tanh^2(x))
        T tanhInput = MathHelper.Tanh(input);
        T tanhSquared = NumOps.Multiply(tanhInput, tanhInput);
        T oneMinus = NumOps.Subtract(NumOps.One, tanhSquared);
        T secondTerm = NumOps.Multiply(input, oneMinus);

        return NumOps.Add(tanhInput, secondTerm);
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because gradient computation is fully implemented in TensorOperations.LiSHT.</value>
    /// <remarks>
    /// <para>
    /// LiSHT supports JIT compilation because:
    /// - The gradient computation (backward pass) is fully implemented in TensorOperations
    /// - The gradient is tanh(x) + x * (1 - tanh²(x))
    /// - It helps prevent vanishing gradients
    /// - It can be represented as a static computation graph node
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with LiSHT activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps the LiSHT activation to TensorOperations&lt;T&gt;.LiSHT(input),
    /// which handles both forward and backward passes for JIT compilation.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return TensorOperations<T>.LiSHT(input);
    }
}
