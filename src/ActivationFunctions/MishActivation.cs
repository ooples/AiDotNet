using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;

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
        // Numerically stable softplus: ln(1 + e^x)
        // For large positive x: ≈ x; For large negative x: ≈ e^x
        T threshold = NumOps.FromDouble(20.0);
        T negThreshold = NumOps.Negate(threshold);
        T softplus;

        if (NumOps.GreaterThan(input, threshold))
        {
            softplus = input;
        }
        else if (NumOps.LessThan(input, negThreshold))
        {
            softplus = NumOps.Exp(input);
        }
        else
        {
            softplus = NumericalStabilityHelper.SafeLog(NumOps.Add(NumOps.One, NumOps.Exp(input)));
        }

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

        // delta = 2 + 2*e^x + e^(2x) = (1 + e^x)^2 + 1
        T delta = NumOps.Add(
            NumOps.Add(NumOps.FromDouble(2), NumOps.Multiply(NumOps.FromDouble(2), exp_x)),
            exp_2x
        );

        return NumOps.Divide(NumOps.Multiply(exp_x, omega), NumOps.Square(delta));
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because gradient computation is fully implemented in TensorOperations.Mish.</value>
    /// <remarks>
    /// <para>
    /// Mish supports JIT compilation because:
    /// - The gradient computation (backward pass) is fully implemented in TensorOperations
    /// - The operation uses IEngine for GPU acceleration
    /// - It can be represented as a static computation graph node
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with Mish activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps the Mish activation to TensorOperations&lt;T&gt;.Mish(input),
    /// which handles both forward and backward passes for JIT compilation.
    /// Mish is a smooth, self-regularizing activation function.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return TensorOperations<T>.Mish(input);
    }

    #region GPU Training Support

    /// <summary>
    /// Gets whether Mish supports GPU-resident training.
    /// </summary>
    /// <value>True because Mish has GPU kernels for both forward and backward passes.</value>
    public override bool SupportsGpuTraining => true;

    /// <summary>
    /// Applies the Mish activation function on GPU.
    /// </summary>
    /// <param name="backend">The GPU backend to use for execution.</param>
    /// <param name="input">The input GPU buffer.</param>
    /// <param name="output">The output GPU buffer to store the activated values.</param>
    /// <param name="size">The number of elements to process.</param>
    /// <remarks>
    /// Mish on GPU: output[i] = input[i] * tanh(softplus(input[i]))
    /// where softplus(x) = ln(1 + exp(x))
    /// </remarks>
    public override void ForwardGpu(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer output, int size)
    {
        backend.Mish(input, output, size);
    }

    /// <summary>
    /// Calculates the Mish backward pass gradient on GPU.
    /// </summary>
    /// <param name="backend">The GPU backend to use for execution.</param>
    /// <param name="gradOutput">The gradient flowing back from the next layer.</param>
    /// <param name="input">The input buffer from the forward pass.</param>
    /// <param name="output">Not used for Mish (can be null). Mish backward uses forward input.</param>
    /// <param name="gradInput">The output buffer to store the input gradient.</param>
    /// <param name="size">The number of elements to process.</param>
    /// <remarks>
    /// Mish backward on GPU computes the gradient using the chain rule with the
    /// derivative of the Mish function.
    /// </remarks>
    public override void BackwardGpu(IDirectGpuBackend backend, IGpuBuffer gradOutput, IGpuBuffer? input, IGpuBuffer? output, IGpuBuffer gradInput, int size)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input), "Mish backward requires the input from forward pass.");

        backend.MishBackward(gradOutput, input, gradInput, size);
    }

    #endregion
}
