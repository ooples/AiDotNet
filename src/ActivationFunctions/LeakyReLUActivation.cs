using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Leaky Rectified Linear Unit (Leaky ReLU) activation function for neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Leaky ReLU activation function is a variation of the standard ReLU function.
/// 
/// How it works:
/// - For positive inputs (x > 0): It returns the input unchanged (like a straight line)
/// - For negative inputs (x = 0): It returns a small fraction of the input (a * x)
/// 
/// The main advantage of Leaky ReLU over standard ReLU is that it never completely "turns off" 
/// neurons for negative inputs. Instead, it allows a small gradient to flow through, which helps
/// prevent the "dying ReLU" problem where neurons can stop learning during training.
/// 
/// Think of it like a water pipe that:
/// - Allows full flow when the input is positive
/// - Allows a small "leak" when the input is negative (controlled by the alpha parameter)
/// </para>
/// </remarks>
public class LeakyReLUActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// The slope coefficient for negative input values.
    /// </summary>
    private readonly T _alpha;

    /// <summary>
    /// Gets the slope coefficient for negative input values.
    /// </summary>
    public T Alpha => _alpha;

    /// <summary>
    /// Initializes a new instance of the Leaky ReLU activation function with the specified alpha parameter.
    /// </summary>
    /// <param name="alpha">
    /// The slope coefficient for negative input values. Default value is 0.01.
    /// </param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The alpha parameter determines how much of the negative inputs "leak through":
    ///
    /// - With alpha = 0.01 (default), negative inputs are multiplied by 0.01 (reduced to 1% of their value)
    /// - With alpha = 0.1, negative inputs are multiplied by 0.1 (reduced to 10% of their value)
    /// - With alpha = 0.001, negative inputs are multiplied by 0.001 (reduced to 0.1% of their value)
    ///
    /// A larger alpha means more information flows through for negative inputs, which can help with learning
    /// but might make the network less focused on positive features. The default value of 0.01 works well
    /// for most applications, but you can adjust it based on your specific needs.
    /// </para>
    /// </remarks>
    public LeakyReLUActivation(double alpha = 0.01)
    {
        _alpha = NumOps.FromDouble(alpha);
    }

    /// <summary>
    /// Indicates whether this activation function can operate on individual scalar values.
    /// </summary>
    /// <returns>Always returns true as the Leaky ReLU function can be applied to individual values.</returns>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the Leaky ReLU activation function to a single value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>
    /// The input value if it's positive, or the input value multiplied by alpha if it's negative or zero.
    /// </returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method transforms an input value using the formula:
    /// 
    /// f(x) = x        if x > 0
    /// f(x) = a * x    if x = 0
    /// 
    /// For example, with the default a = 0.01:
    /// - Input of 5 ? Output of 5 (unchanged)
    /// - Input of 0 ? Output of 0
    /// - Input of -5 ? Output of -0.05 (5 * 0.01)
    /// </para>
    /// </remarks>
    public override T Activate(T input)
    {
        return NumOps.GreaterThan(input, NumOps.Zero) ? input : NumOps.Multiply(_alpha, input);
    }

    /// <summary>
    /// Applies the Leaky ReLU activation function to a vector of values.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>A new vector with the Leaky ReLU function applied to each element.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method applies the Leaky ReLU function to each value in a collection (vector)
    /// of inputs. It processes each number individually using the same rules as the single-value version.
    /// </remarks>
    public override Vector<T> Activate(Vector<T> input)
    {
        return input.Transform(x => Activate(x));
    }

    /// <summary>
    /// Calculates the derivative (gradient) of the Leaky ReLU function for a single value.
    /// </summary>
    /// <param name="input">The input value at which to calculate the derivative.</param>
    /// <returns>1 if the input is positive, or alpha if the input is negative or zero.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The derivative tells us how much the output changes when we slightly change the input.
    /// This information is crucial during neural network training.
    /// 
    /// For the Leaky ReLU function, the derivative is very simple:
    /// - For positive inputs (x > 0): The derivative is 1 (output changes at the same rate as input)
    /// - For negative inputs (x = 0): The derivative is alpha (output changes at alpha times the rate of input)
    /// 
    /// Unlike some other activation functions, Leaky ReLU's derivative never becomes zero,
    /// which helps prevent neurons from "dying" during training.
    /// </para>
    /// </remarks>
    public override T Derivative(T input)
    {
        return NumOps.GreaterThan(input, NumOps.Zero) ? NumOps.One : _alpha;
    }

    /// <summary>
    /// Calculates the derivative (gradient) of the Leaky ReLU function for a vector of values.
    /// </summary>
    /// <param name="input">The input vector at which to calculate the derivative.</param>
    /// <returns>A diagonal matrix containing the derivatives for each input value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how the output vector changes when we slightly change each input value.
    ///
    /// The result is a special matrix called a "Jacobian matrix" where:
    /// - Values on the main diagonal (top-left to bottom-right) are the derivatives for each input
    /// - All other values are 0
    ///
    /// This diagonal structure indicates that each output is affected only by its corresponding input,
    /// with no cross-interactions between different elements.
    ///
    /// For Leaky ReLU, each diagonal value will be either:
    /// - 1 (for inputs > 0)
    /// - alpha (for inputs = 0)
    /// </para>
    /// </remarks>
    public override Matrix<T> Derivative(Vector<T> input)
    {
        int size = input.Length;
        Matrix<T> jacobian = new Matrix<T>(size, size);

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                if (i == j)
                {
                    jacobian[i, j] = Derivative(input[i]);
                }
                else
                {
                    jacobian[i, j] = NumOps.Zero;
                }
            }
        }

        return jacobian;
    }

    /// <summary>
    /// Applies the Leaky ReLU activation function to each element in a tensor.
    /// </summary>
    /// <param name="input">The input tensor to activate.</param>
    /// <returns>A new tensor with the Leaky ReLU function applied to each element.</returns>
    public override Tensor<T> Activate(Tensor<T> input)
    {
        return Engine.LeakyReLU(input, _alpha);
    }

    /// <summary>
    /// Calculates the derivative of the Leaky ReLU function for each element in a tensor.
    /// </summary>
    /// <param name="input">The input tensor to calculate the derivative for.</param>
    /// <returns>A new tensor containing the derivatives of the Leaky ReLU function for each input element.</returns>
    public override Tensor<T> Derivative(Tensor<T> input)
    {
        Tensor<T> output = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            output[i] = Derivative(input[i]);
        }
        return output;
    }

    /// <summary>
    /// Calculates the backward pass gradient for Leaky ReLU using GPU-accelerated fused operation.
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
        return Engine.LeakyReluBackward(outputGradient, input, NumOps.ToDouble(_alpha));
    }

    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because gradient computation is fully implemented in TensorOperations.LeakyReLU.</value>
    /// <remarks>
    /// <para>
    /// LeakyReLU supports JIT compilation because:
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
    /// <returns>A new computation node with LeakyReLU activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps the LeakyReLU activation to TensorOperations&lt;T&gt;.LeakyReLU(input, alpha),
    /// which handles both forward and backward passes for JIT compilation.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        // Convert alpha to double for TensorOperations
        double alphaDouble = NumOps.ToDouble(_alpha);
        return TensorOperations<T>.LeakyReLU(input, alphaDouble);
    }

    #region GPU Training Support

    /// <summary>
    /// Gets whether LeakyReLU supports GPU-resident training.
    /// </summary>
    /// <value>True because LeakyReLU has GPU kernels for both forward and backward passes.</value>
    public override bool SupportsGpuTraining => true;

    /// <summary>
    /// Applies the Leaky ReLU activation function on GPU.
    /// </summary>
    /// <param name="backend">The GPU backend to use for execution.</param>
    /// <param name="input">The input GPU buffer.</param>
    /// <param name="output">The output GPU buffer to store the activated values.</param>
    /// <param name="size">The number of elements to process.</param>
    /// <remarks>
    /// LeakyReLU on GPU: output[i] = input[i] > 0 ? input[i] : alpha * input[i]
    /// </remarks>
    public override void ForwardGpu(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer output, int size)
    {
        float alpha = (float)NumOps.ToDouble(_alpha);
        backend.LeakyRelu(input, output, alpha, size);
    }

    /// <summary>
    /// Calculates the Leaky ReLU backward pass gradient on GPU.
    /// </summary>
    /// <param name="backend">The GPU backend to use for execution.</param>
    /// <param name="gradOutput">The gradient flowing back from the next layer.</param>
    /// <param name="input">The input buffer from the forward pass.</param>
    /// <param name="output">Not used for LeakyReLU (can be null). LeakyReLU backward uses forward input.</param>
    /// <param name="gradInput">The output buffer to store the input gradient.</param>
    /// <param name="size">The number of elements to process.</param>
    /// <remarks>
    /// LeakyReLU backward on GPU: gradInput[i] = gradOutput[i] * (input[i] > 0 ? 1 : alpha)
    /// </remarks>
    public override void BackwardGpu(IDirectGpuBackend backend, IGpuBuffer gradOutput, IGpuBuffer? input, IGpuBuffer? output, IGpuBuffer gradInput, int size)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input), "LeakyReLU backward requires the input from forward pass.");

        float alpha = (float)NumOps.ToDouble(_alpha);
        backend.LeakyReluBackward(gradOutput, input, gradInput, alpha, size);
    }

    #endregion
}
