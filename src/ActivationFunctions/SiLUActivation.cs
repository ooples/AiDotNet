using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the SiLU (Sigmoid Linear Unit) activation function, also known as Swish.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The SiLU function is defined as f(x) = x * sigmoid(x), where sigmoid(x) = 1/(1+e^(-x)).
/// It was introduced in 2017 and has shown strong performance in deep neural networks.
/// </para>
/// <para>
/// <b>For Beginners:</b> SiLU (or Swish) is a relatively new activation function that has become 
/// popular in modern neural networks. Unlike simpler functions like ReLU that either pass 
/// a value through or block it, SiLU smoothly scales inputs based on their value. It keeps 
/// most positive values, reduces small positive values, and allows some negative values to 
/// pass through (but reduced in magnitude). This smooth behavior helps neural networks learn 
/// more complex patterns. SiLU is used in many state-of-the-art models, especially in deep 
/// learning applications like computer vision and natural language processing.
/// </para>
/// </remarks>
public class SiLUActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Indicates whether this activation function supports scalar operations.
    /// </summary>
    /// <returns>Always returns true as SiLU supports scalar operations.</returns>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the SiLU activation function to a single input value.
    /// </summary>
    /// <param name="input">The input value to activate.</param>
    /// <returns>The result of x * sigmoid(x).</returns>
    /// <remarks>
    /// <para>
    /// The SiLU function multiplies the input by the sigmoid of the input: f(x) = x * sigmoid(x).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method takes a number and transforms it using the SiLU formula.
    /// First, it calculates the sigmoid of the input (a value between 0 and 1), then multiplies
    /// the original input by this sigmoid value. This creates a smooth curve that:
    /// - For large positive values: outputs approximately the same value (since sigmoid approaches 1)
    /// - For values near zero: outputs a smaller value (since sigmoid is around 0.5)
    /// - For negative values: can output small negative values (unlike ReLU which outputs 0)
    /// 
    /// This behavior helps neural networks learn more effectively in many situations.
    /// </para>
    /// </remarks>
    public override T Activate(T input)
    {
        // SiLU: x * sigmoid(x)
        T sigmoid = MathHelper.Sigmoid(input);
        return NumOps.Multiply(input, sigmoid);
    }

    /// <summary>
    /// Calculates the derivative of the SiLU function for a single input value.
    /// </summary>
    /// <param name="input">The input value to calculate the derivative for.</param>
    /// <returns>The derivative of SiLU at the given input.</returns>
    /// <remarks>
    /// <para>
    /// The derivative of SiLU is: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)).
    /// This derivative is used during the backpropagation phase of neural network training.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The derivative tells us how much the output changes when we slightly change the input.
    /// This information is crucial for training neural networks because it guides how the network's weights
    /// should be adjusted. The SiLU derivative has a nice property: it's non-zero for most input values,
    /// which helps prevent the "dying neuron" problem that can occur with simpler activation functions like ReLU.
    /// This means SiLU neurons can continue to learn even if they receive negative inputs.
    /// </para>
    /// </remarks>
    public override T Derivative(T input)
    {
        // Derivative of SiLU: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        T sigmoid = MathHelper.Sigmoid(input);
        T sigmoidDerivative = NumOps.Multiply(sigmoid, NumOps.Subtract(NumOps.One, sigmoid));
        T xSigmoidDerivative = NumOps.Multiply(input, sigmoidDerivative);

        return NumOps.Add(sigmoid, xSigmoidDerivative);
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because SiLU is mathematically equivalent to Swish, which is fully implemented in TensorOperations.</value>
    /// <remarks>
    /// <para>
    /// SiLU (Sigmoid Linear Unit) is mathematically identical to Swish: f(x) = x * sigmoid(x).
    /// TensorOperations.Swish provides full forward and backward pass support for JIT compilation.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with SiLU/Swish activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps SiLU to TensorOperations&lt;T&gt;.Swish(input) since SiLU and Swish
    /// are mathematically equivalent: f(x) = x * sigmoid(x).
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        // SiLU is mathematically equivalent to Swish: x * sigmoid(x)
        return TensorOperations<T>.Swish(input);
    }

    #region GPU Training Support

    /// <summary>
    /// Gets whether SiLU supports GPU-resident training.
    /// </summary>
    /// <value>True because SiLU has GPU kernels for both forward and backward passes.</value>
    /// <remarks>
    /// SiLU is mathematically equivalent to Swish, so it uses the same GPU kernels.
    /// </remarks>
    public override bool SupportsGpuTraining => true;

    /// <summary>
    /// Applies the SiLU activation function on GPU.
    /// </summary>
    /// <param name="backend">The GPU backend to use for execution.</param>
    /// <param name="input">The input GPU buffer.</param>
    /// <param name="output">The output GPU buffer to store the activated values.</param>
    /// <param name="size">The number of elements to process.</param>
    /// <remarks>
    /// SiLU on GPU: output[i] = input[i] * sigmoid(input[i])
    /// Uses the Swish kernel since SiLU and Swish are mathematically identical.
    /// </remarks>
    public override void ForwardGpu(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer output, int size)
    {
        backend.Swish(input, output, size);
    }

    /// <summary>
    /// Calculates the SiLU backward pass gradient on GPU.
    /// </summary>
    /// <param name="backend">The GPU backend to use for execution.</param>
    /// <param name="gradOutput">The gradient flowing back from the next layer.</param>
    /// <param name="input">The input buffer from the forward pass.</param>
    /// <param name="output">Not used for SiLU (can be null). SiLU backward uses forward input.</param>
    /// <param name="gradInput">The output buffer to store the input gradient.</param>
    /// <param name="size">The number of elements to process.</param>
    /// <remarks>
    /// SiLU backward on GPU: gradInput[i] = gradOutput[i] * (sigmoid(input[i]) + input[i] * sigmoid(input[i]) * (1 - sigmoid(input[i])))
    /// Uses the Swish backward kernel since SiLU and Swish are mathematically identical.
    /// </remarks>
    public override void BackwardGpu(IDirectGpuBackend backend, IGpuBuffer gradOutput, IGpuBuffer? input, IGpuBuffer? output, IGpuBuffer gradInput, int size)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input), "SiLU backward requires the input from forward pass.");

        backend.SwishBackward(gradOutput, input, gradInput, size);
    }

    #endregion
}
