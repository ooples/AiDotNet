using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the SoftSign activation function, which is a smooth alternative to the tanh function.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The SoftSign function is defined as: f(x) = x / (1 + |x|), where |x| is the absolute value of x.
/// It maps any input value to an output between -1 and 1, similar to tanh but with different properties.
/// </para>
/// <para>
/// <b>For Beginners:</b> SoftSign is an activation function that squeezes input values into a range between -1 and 1.
/// It's similar to the tanh function but approaches its limits more slowly.
/// 
/// Key properties of SoftSign:
/// - For input of 0, the output is 0
/// - For large positive inputs, the output approaches 1 (but never reaches it)
/// - For large negative inputs, the output approaches -1 (but never reaches it)
/// - The function is smooth everywhere, making it easier to train
/// - Unlike tanh, SoftSign has "polynomial" tails, meaning it approaches its limits more gradually
/// 
/// This gradual approach to limits can help prevent neurons from becoming "saturated" (stuck at extreme values)
/// during training, which can be an advantage in some neural network architectures.
/// </para>
/// </remarks>
public class SoftSignActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Indicates whether this activation function supports scalar operations.
    /// </summary>
    /// <returns>Always returns true as SoftSign can operate on individual values.</returns>
    /// <remarks>
    /// <para>
    /// Unlike functions like Softmax that require a vector of values, SoftSign can be applied
    /// independently to each individual value.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method returning true means that SoftSign works on one number at a time.
    /// Each input value is transformed independently without needing to know about other values.
    /// </para>
    /// </remarks>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the SoftSign activation function to a single input value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The result of applying SoftSign to the input, which will be between -1 and 1.</returns>
    /// <remarks>
    /// <para>
    /// Computes x / (1 + |x|) where x is the input value and |x| is the absolute value of x.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method transforms an input number using the SoftSign formula:
    /// 1. Calculate the absolute value of your input (remove any negative sign)
    /// 2. Add 1 to that absolute value
    /// 3. Divide the original input by this sum
    /// 
    /// For example:
    /// - If input is 2, the output is 2/(1+2) = 2/3 ˜ 0.67
    /// - If input is -2, the output is -2/(1+2) = -2/3 ˜ -0.67
    /// - If input is 10, the output is 10/(1+10) = 10/11 ˜ 0.91
    /// - If input is -10, the output is -10/(1+10) = -10/11 ˜ -0.91
    /// 
    /// Notice that even with large inputs like 10 or -10, the outputs stay between -1 and 1.
    /// </para>
    /// </remarks>
    public override T Activate(T input)
    {
        // f(x) = x / (1 + |x|)
        T absInput = NumOps.Abs(input);
        T denominator = NumOps.Add(NumOps.One, absInput);

        return NumOps.Divide(input, denominator);
    }

    /// <summary>
    /// Calculates the derivative of the SoftSign function for a given input.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The derivative of SoftSign at the input point.</returns>
    /// <remarks>
    /// <para>
    /// The derivative of SoftSign is: f'(x) = 1 / (1 + |x|)^2
    /// This derivative is always positive and approaches zero as |x| increases.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The derivative tells us how quickly the SoftSign function is changing at any point.
    /// It's calculated as 1 / (1 + |x|)^2.
    /// 
    /// This derivative has these important properties:
    /// - It's always positive (meaning the function is always increasing)
    /// - For input of 0, the derivative is 1 (its maximum value)
    /// - As inputs get further from 0 (either positive or negative), the derivative gets smaller
    /// - For large inputs, the derivative approaches 0
    /// 
    /// During neural network training, this derivative helps determine how much to adjust the weights.
    /// The fact that the derivative decreases more slowly than tanh can help with training deep networks.
    /// </para>
    /// </remarks>
    public override T Derivative(T input)
    {
        // f'(x) = 1 / (1 + |x|)^2
        T absInput = NumOps.Abs(input);
        T denominator = NumOps.Add(NumOps.One, absInput);
        T squaredDenominator = NumOps.Multiply(denominator, denominator);

        return NumOps.Divide(NumOps.One, squaredDenominator);
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because gradient computation is fully implemented in TensorOperations.SoftSign.</value>
    /// <remarks>
    /// <para>
    /// SoftSign supports JIT compilation because:
    /// - The gradient computation (backward pass) is fully implemented in TensorOperations
    /// - The gradient is 1 / (1 + |x|)², which is always positive and well-behaved
    /// - The slower saturation helps prevent vanishing gradients in deep networks
    /// - It can be represented as a static computation graph node
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with SoftSign activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps the SoftSign activation to TensorOperations&lt;T&gt;.SoftSign(input),
    /// which handles both forward and backward passes for JIT compilation.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return TensorOperations<T>.SoftSign(input);
    }
}
