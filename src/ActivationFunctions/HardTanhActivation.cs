using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Hard Tanh activation function for neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Hard Tanh is a simplified version of the standard Tanh (hyperbolic tangent) function.
/// 
/// While the regular Tanh creates an S-shaped curve that smoothly transitions from -1 to 1,
/// the Hard Tanh uses straight lines to approximate this curve, making it:
/// 
/// 1. Computationally faster (uses simple comparison operations instead of complex math)
/// 2. Less smooth but still useful for many neural network applications
/// 
/// The function works like this:
/// - If input = -1: output = -1
/// - If input = 1: output = 1
/// - If -1 &lt; input &lt; 1: output = input (unchanged)
/// 
/// This creates a function that "clips" or "saturates" any input to the range [-1, 1],
/// with a straight line in between.
/// 
/// Hard Tanh is often used when you want the benefits of Tanh (centered around zero,
/// outputs between -1 and 1) but need faster computation, such as in deep networks
/// or resource-constrained environments.
/// </para>
/// </remarks>
public class HardTanhActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Indicates whether this activation function can operate on individual scalar values.
    /// </summary>
    /// <returns>Always returns true as Hard Tanh can be applied to individual values.</returns>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the Hard Tanh activation function to a single input value.
    /// </summary>
    /// <param name="input">The input value to transform.</param>
    /// <returns>The transformed value, constrained between -1 and 1.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method transforms any input number into a value between -1 and 1:
    /// 
    /// - If the input is less than -1, the output is -1
    /// - If the input is greater than 1, the output is 1
    /// - If the input is between -1 and 1, the output equals the input
    /// 
    /// For example:
    /// - If input = -2, output = -1
    /// - If input = 0, output = 0
    /// - If input = 0.5, output = 0.5
    /// - If input = 2, output = 1
    /// 
    /// This creates a function that "clips" extreme values while preserving values in the middle range,
    /// which helps prevent issues with very large or very small values during neural network training.
    /// </para>
    /// </remarks>
    public override T Activate(T input)
    {
        // Hard Tanh: max(-1, min(1, x))
        T minBound = NumOps.FromDouble(-1);
        T maxBound = NumOps.One;

        return MathHelper.Max(minBound, MathHelper.Min(maxBound, input));
    }

    /// <summary>
    /// Calculates the derivative of the Hard Tanh function for a given input value.
    /// </summary>
    /// <param name="input">The input value at which to calculate the derivative.</param>
    /// <returns>The derivative value (1 if input is between -1 and 1, otherwise 0).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The derivative tells us how much the output changes when the input changes slightly.
    /// 
    /// For Hard Tanh:
    /// - If input is between -1 and 1: derivative = 1 (the output changes at the same rate as the input)
    /// - Otherwise: derivative = 0 (the output doesn't change at all when input changes)
    /// 
    /// This information is crucial during neural network training because:
    /// - It determines how much to adjust weights during backpropagation
    /// - It helps identify which neurons are actively learning (those with non-zero derivatives)
    /// - It prevents the "vanishing gradient problem" that can occur with some activation functions
    /// 
    /// When the derivative is 1, the neural network can learn efficiently in that region.
    /// When the derivative is 0, learning stops for that neuron (for inputs in that range).
    /// </para>
    /// </remarks>
    public override T Derivative(T input)
    {
        // Derivative of Hard Tanh:
        // 1 if -1 < x < 1
        // 0 otherwise
        T minBound = NumOps.FromDouble(-1);
        T maxBound = NumOps.One;

        if (NumOps.GreaterThan(input, minBound) && NumOps.LessThan(input, maxBound))
        {
            return NumOps.One;
        }

        return NumOps.Zero;
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because gradient computation is fully implemented in TensorOperations.HardTanh.</value>
    /// <remarks>
    /// <para>
    /// HardTanh supports JIT compilation because:
    /// - The gradient computation (backward pass) is fully implemented in TensorOperations
    /// - The gradient is 1 when -1 &lt; x &lt; 1, and 0 otherwise
    /// - It's computationally efficient and useful for bounded outputs
    /// - It can be represented as a static computation graph node
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with HardTanh activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps the HardTanh activation to TensorOperations&lt;T&gt;.HardTanh(input),
    /// which handles both forward and backward passes for JIT compilation.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return TensorOperations<T>.HardTanh(input);
    }
}
