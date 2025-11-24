using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Mish activation function for neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Mish activation function is a smooth, non-monotonic function that helps neural networks
/// learn complex patterns. It was introduced in 2019 and has shown good performance in many applications.
/// 
/// Mathematically, Mish is defined as: f(x) = x * tanh(softplus(x))
/// where softplus(x) = ln(1 + e^x)
/// 
/// Mish combines properties of several popular activation functions:
/// - It's smooth (no sharp corners like ReLU)
/// - It allows both positive and negative values (unlike ReLU which zeros out negatives)
/// - It's unbounded on the positive side (can output large positive values)
/// - It's bounded on the negative side (won't output extremely negative values)
/// 
/// These properties help neural networks learn more effectively in many situations.
/// </para>
/// </remarks>
public class MishActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Determines if the activation function supports operations on individual scalar values.
    /// </summary>
    /// <returns>True, as Mish can be applied to individual values.</returns>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the Mish activation function to a single input value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The activated output value using the Mish function.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method transforms a single number using the Mish formula:
    /// 
    /// 1. First, it calculates softplus(x) = ln(1 + e^x)
    /// 2. Then, it calculates tanh(softplus(x))
    /// 3. Finally, it multiplies the input by this tanh value
    /// 
    /// The result is a smooth curve that:
    /// - For large positive inputs: behaves almost like the identity function (returns the input)
    /// - For negative inputs: dampens the values but doesn't completely zero them out
    /// - Near zero: has a smooth transition between these behaviors
    /// </para>
    /// </remarks>
    public override T Activate(T input)
    {
        T softplus = NumOps.Log(NumOps.Add(NumOps.One, NumOps.Exp(input)));
        T tanh = MathHelper.Tanh(softplus);

        return NumOps.Multiply(input, tanh);
    }

    /// <summary>
    /// Calculates the derivative (gradient) of the Mish function for a single input value.
    /// </summary>
    /// <param name="input">The input value at which to calculate the derivative.</param>
    /// <returns>The derivative value at the input point.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The derivative tells us how much the Mish function's output changes when we
    /// slightly change the input. This is essential for training neural networks.
    /// 
    /// The formula for the Mish derivative is complex (as you can see from the code), but what's
    /// important to understand is:
    /// 
    /// - For large positive inputs: the derivative approaches 1
    /// - For large negative inputs: the derivative approaches 0
    /// - In between: it has a smooth transition with values that help the network learn effectively
    /// 
    /// During training, this derivative helps determine how much to adjust each neuron's weights
    /// based on the errors the network makes.
    /// </para>
    /// </remarks>
    public override T Derivative(T input)
    {
        T exp_x = NumOps.Exp(input);
        T exp_2x = NumOps.Exp(NumOps.Multiply(NumOps.FromDouble(2), input));
        T exp_3x = NumOps.Exp(NumOps.Multiply(NumOps.FromDouble(3), input));
        
        T omega = NumOps.Add(
            NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(4), NumOps.Add(input, NumOps.One)),
                NumOps.Multiply(NumOps.FromDouble(4), exp_2x)
            ),
            NumOps.Add(
                exp_3x,
                NumOps.Multiply(exp_x, NumOps.Add(NumOps.Multiply(NumOps.FromDouble(4), input), NumOps.FromDouble(6)))
            )
        );
        
        T delta = NumOps.Add(
            NumOps.Add(NumOps.Multiply(NumOps.FromDouble(2), exp_2x), NumOps.FromDouble(2)),
            NumOps.Multiply(exp_2x, NumOps.Square(NumOps.Add(input, NumOps.FromDouble(2))))
        );
        
        return NumOps.Divide(NumOps.Multiply(exp_x, omega), NumOps.Square(delta));
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>False because gradient computation is not yet implemented.</value>
    /// <remarks>
    /// <para>
    /// This activation does not yet support JIT compilation because the gradient
    /// computation (backward pass) has not been implemented in TensorOperations.Mish.
    /// </para>
    /// <para>
    /// To enable JIT support:
    /// 1. Implement the backward pass in TensorOperations.Mish
    /// 2. Test the gradient computation
    /// 3. Change SupportsJitCompilation to return true
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with Mish activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return TensorOperations<T>.Mish(input);
    }
}