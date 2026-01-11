using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Gaussian Error Linear Unit (GELU) activation function for neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The GELU activation function is a modern activation function used in many 
/// state-of-the-art neural networks, including transformers like BERT and GPT.
/// 
/// Think of GELU as a "smoother" version of ReLU that:
/// - Keeps most positive values (like ReLU does)
/// - Gradually reduces small positive values
/// - Gradually allows some small negative values through
/// - Blocks large negative values (like ReLU does)
/// 
/// GELU can be thought of as multiplying the input by the probability that the input is positive.
/// This creates a smooth curve that transitions naturally between allowing and blocking values,
/// rather than having a sharp cutoff like ReLU.
/// 
/// GELU is widely used in modern language models and has been shown to perform better than
/// older activation functions in many deep learning tasks.
/// </para>
/// </remarks>
public class GELUActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Indicates that this activation function supports operations on individual scalar values.
    /// </summary>
    /// <returns>Always returns true as GELU can be applied to scalar values.</returns>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the GELU activation function to a single input value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The activated output value using the GELU function.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method transforms an input value using the GELU formula.
    /// 
    /// The GELU function behaves differently based on the input:
    /// - For large positive inputs: the output is approximately equal to the input
    /// - For large negative inputs: the output is approximately zero
    /// - For inputs near zero: there's a smooth transition that allows some negative values
    ///   and slightly reduces some positive values
    /// 
    /// This smooth transition helps neural networks learn more effectively than functions
    /// with sharp transitions (like ReLU).
    /// 
    /// The mathematical formula used is an approximation:
    /// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/p) * (x + 0.044715 * x³)))
    /// </para>
    /// </remarks>
    public override T Activate(T input)
    {
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/p) * (x + 0.044715 * x^3)))
        T sqrt2OverPi = NumOps.Sqrt(NumOps.FromDouble(2.0 / Math.PI));
        T x3 = NumOps.Multiply(NumOps.Multiply(input, input), input);
        T inner = NumOps.Add(input, NumOps.Multiply(NumOps.FromDouble(0.044715), x3));
        T tanhTerm = NumOps.Add(NumOps.One, MathHelper.Tanh(NumOps.Multiply(sqrt2OverPi, inner)));

        return NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Multiply(input, tanhTerm));
    }

    /// <summary>
    /// Calculates the derivative of the GELU function for a single input value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The derivative value at the input point.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The derivative measures how much the GELU function's output changes
    /// when its input changes slightly. This is essential for neural network training as it
    /// determines how weights should be adjusted during backpropagation.
    /// 
    /// Key properties of the GELU derivative:
    /// - Always positive for all inputs (unlike ReLU's derivative which is zero for negative inputs)
    /// - Approaches 1 for large positive inputs
    /// - Approaches 0 for large negative inputs
    /// - Has a smooth transition in between
    /// 
    /// This smooth, non-zero derivative helps prevent the "dying ReLU" problem where neurons
    /// can become permanently inactive during training.
    /// 
    /// The mathematical formula is complex but has been simplified to:
    /// d/dx GELU(x) = 0.5 * tanh(0.0356774 * x³ + 0.797885 * x) + 
    ///                (0.0535161 * x³ + 0.398942 * x) * sech²(0.0356774 * x³ + 0.797885 * x) + 0.5
    /// 
    /// Where sech²(x) = 1 - tanh²(x)
    /// </para>
    /// </remarks>
    public override T Derivative(T input)
    {
        // d/dx GELU(x) = 0.5 * tanh(0.0356774 * x^3 + 0.797885 * x) + 
        //                (0.0535161 * x^3 + 0.398942 * x) * sech^2(0.0356774 * x^3 + 0.797885 * x) + 0.5
        T x2 = NumOps.Multiply(input, input);
        T x3 = NumOps.Multiply(x2, input);

        T term1 = NumOps.Add(
            NumOps.Multiply(NumOps.FromDouble(0.0356774), x3),
            NumOps.Multiply(NumOps.FromDouble(0.797885), input)
        );

        T term2 = NumOps.Add(
            NumOps.Multiply(NumOps.FromDouble(0.0535161), x3),
            NumOps.Multiply(NumOps.FromDouble(0.398942), input)
        );

        T tanhTerm = MathHelper.Tanh(term1);
        T sech2Term = NumOps.Subtract(NumOps.One, NumOps.Multiply(tanhTerm, tanhTerm));

        return NumOps.Add(
            NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(0.5), tanhTerm),
                NumOps.Multiply(term2, sech2Term)
            ),
            NumOps.FromDouble(0.5)
        );
    }

    /// <summary>
    /// Applies the GELU activation function to each element in a tensor.
    /// </summary>
    /// <param name="input">The input tensor to activate.</param>
    /// <returns>A new tensor with the GELU function applied to each element.</returns>
    public override Tensor<T> Activate(Tensor<T> input)
    {
        return Engine.GELU(input);
    }

    /// <summary>
    /// Calculates the derivative of the GELU function for each element in a tensor.
    /// </summary>
    /// <param name="input">The input tensor to calculate the derivative for.</param>
    /// <returns>A new tensor containing the derivatives of the GELU function for each input element.</returns>
    public override Tensor<T> Derivative(Tensor<T> input)
    {
        // Use the tensor-level derivative computation
        Tensor<T> output = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            output[i] = Derivative(input[i]);
        }
        return output;
    }

    /// <summary>
    /// Calculates the backward pass gradient for GELU using GPU-accelerated fused operation.
    /// </summary>
    /// <param name="input">The input tensor that was used in the forward pass.</param>
    /// <param name="outputGradient">The gradient flowing back from the next layer.</param>
    /// <returns>The gradient with respect to the input.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method uses a single GPU kernel to compute the gradient,
    /// which is faster than computing derivative and gradient multiplication separately.
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> input, Tensor<T> outputGradient)
    {
        return Engine.GeluBackward(outputGradient, input);
    }

    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because gradient computation is implemented.</value>
    /// <remarks>
    /// <para>
    /// This activation supports JIT compilation. The gradient computation (backward pass)
    /// is implemented in TensorOperations.GELU, enabling use in JIT-compiled computation graphs.
    /// </para>
    /// <para>
    /// GELU is widely used in transformers (BERT, GPT) and modern architectures,
    /// making it an important activation for JIT-compiled models.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with GELU activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps the GELU activation to TensorOperations&lt;T&gt;.GELU(input),
    /// which handles both forward and backward passes for JIT compilation.
    /// GELU is widely used in transformers (BERT, GPT) and modern architectures.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return TensorOperations<T>.GELU(input);
    }

    #region GPU Training Support

    /// <summary>
    /// Gets whether GELU supports GPU-resident training.
    /// </summary>
    /// <value>True because GELU has GPU kernels for both forward and backward passes.</value>
    public override bool SupportsGpuTraining => true;

    /// <summary>
    /// Applies the GELU activation function on GPU.
    /// </summary>
    /// <param name="backend">The GPU backend to use for execution.</param>
    /// <param name="input">The input GPU buffer.</param>
    /// <param name="output">The output GPU buffer to store the activated values.</param>
    /// <param name="size">The number of elements to process.</param>
    /// <remarks>
    /// GELU on GPU: output[i] = 0.5 * input[i] * (1 + tanh(sqrt(2/pi) * (input[i] + 0.044715 * input[i]^3)))
    /// </remarks>
    public override void ForwardGpu(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer output, int size)
    {
        backend.Gelu(input, output, size);
    }

    /// <summary>
    /// Calculates the GELU backward pass gradient on GPU.
    /// </summary>
    /// <param name="backend">The GPU backend to use for execution.</param>
    /// <param name="gradOutput">The gradient flowing back from the next layer.</param>
    /// <param name="input">The input buffer from the forward pass.</param>
    /// <param name="output">Not used for GELU (can be null). GELU backward uses forward input.</param>
    /// <param name="gradInput">The output buffer to store the input gradient.</param>
    /// <param name="size">The number of elements to process.</param>
    /// <remarks>
    /// GELU backward on GPU computes the gradient using the original input values.
    /// </remarks>
    public override void BackwardGpu(IDirectGpuBackend backend, IGpuBuffer gradOutput, IGpuBuffer? input, IGpuBuffer? output, IGpuBuffer gradInput, int size)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input), "GELU backward requires the input from forward pass.");

        backend.GeluBackward(gradOutput, input, gradInput, size);
    }

    #endregion
}
