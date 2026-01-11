using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Scaled Exponential Linear Unit (SELU) activation function.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SELU is an activation function that enables self-normalizing properties in neural networks.
/// It automatically ensures that the outputs maintain a mean of 0 and standard deviation of 1 
/// across the network, which helps with training stability and convergence.
/// </para>
/// <para>
/// <b>For Beginners:</b> SELU (Scaled Exponential Linear Unit) is a special activation function that 
/// helps neural networks train more effectively. Unlike simpler functions like ReLU, SELU has 
/// carefully chosen constants (alpha and lambda) that help keep the data flowing through your 
/// neural network well-balanced. This means your network can learn faster and more reliably 
/// without requiring extra normalization steps. Think of it as a self-regulating function that 
/// keeps your data in a "sweet spot" range as it passes through the network.
/// </para>
/// </remarks>
public class SELUActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// The scaling factor for the SELU function.
    /// </summary>
    /// <remarks>
    /// This specific value helps maintain the self-normalizing property of SELU networks.
    /// </remarks>
    private readonly T _lambda;

    /// <summary>
    /// The alpha parameter for the SELU function, which controls the negative saturation value.
    /// </summary>
    /// <remarks>
    /// This specific value works together with lambda to ensure self-normalization.
    /// </remarks>
    private readonly T _alpha;

    /// <summary>
    /// Initializes a new instance of the SELU activation function with predefined constants.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The constructor sets up the SELU function with two special numbers (lambda and alpha)
    /// that have been mathematically calculated to give SELU its self-normalizing properties.
    /// You don't need to change these values - they're carefully chosen to make neural networks work well.
    /// </remarks>
    public SELUActivation()
    {
        _lambda = NumOps.FromDouble(1.0507009873554804934193349852946);
        _alpha = NumOps.FromDouble(1.6732632423543772848170429916717);
    }

    /// <summary>
    /// Indicates whether this activation function supports scalar operations.
    /// </summary>
    /// <returns>Always returns true as SELU supports scalar operations.</returns>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the SELU activation function to a single input value.
    /// </summary>
    /// <param name="input">The input value to activate.</param>
    /// <returns>
    /// For positive inputs: lambda * input
    /// For negative inputs: lambda * alpha * (e^input - 1)
    /// </returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method processes a single number through the SELU function:
    /// - If the number is positive or zero, it multiplies it by lambda (about 1.05)
    /// - If the number is negative, it applies a special exponential formula that creates a smooth curve
    ///   that approaches a negative value as inputs become more negative
    /// 
    /// This combination helps keep the data flowing through your neural network balanced and well-behaved.
    /// </remarks>
    public override T Activate(T input)
    {
        if (NumOps.GreaterThanOrEquals(input, NumOps.Zero))
        {
            return NumOps.Multiply(_lambda, input);
        }
        else
        {
            T expTerm = NumOps.Subtract(NumOps.Exp(input), NumOps.One);
            return NumOps.Multiply(_lambda, NumOps.Multiply(_alpha, expTerm));
        }
    }

    /// <summary>
    /// Calculates the derivative of the SELU function for a single input value.
    /// </summary>
    /// <param name="input">The input value to calculate the derivative for.</param>
    /// <returns>
    /// For positive inputs: lambda
    /// For negative inputs: lambda * alpha * e^input
    /// </returns>
    /// <remarks>
    /// <b>For Beginners:</b> The derivative tells us how much the output changes when we slightly change the input.
    /// For SELU:
    /// - When the input is positive, the derivative is simply lambda (about 1.05), meaning the output
    ///   changes slightly faster than the input
    /// - When the input is negative, the derivative follows an exponential curve that gets smaller
    ///   as inputs become more negative
    /// 
    /// This derivative is important during the learning process when the neural network is adjusting its weights.
    /// </remarks>
    public override T Derivative(T input)
    {
        if (NumOps.GreaterThanOrEquals(input, NumOps.Zero))
        {
            return _lambda;
        }
        else
        {
            return NumOps.Multiply(_lambda, NumOps.Multiply(_alpha, NumOps.Exp(input)));
        }
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because gradient computation is fully implemented in TensorOperations.SELU.</value>
    /// <remarks>
    /// <para>
    /// SELU supports JIT compilation because:
    /// - The gradient computation (backward pass) is fully implemented in TensorOperations
    /// - Uses fixed λ ≈ 1.0507 and α ≈ 1.6733 constants for self-normalization
    /// - The gradient is λ for x >= 0, otherwise λ * α * e^x
    /// - It can be represented as a static computation graph node
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with SELU activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps the SELU activation to TensorOperations&lt;T&gt;.SELU(input),
    /// which handles both forward and backward passes for JIT compilation.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return TensorOperations<T>.SELU(input);
    }

    #region GPU Training Support

    /// <summary>
    /// Gets whether SELU supports GPU-resident training.
    /// </summary>
    /// <value>True because SELU has GPU kernels for both forward and backward passes.</value>
    public override bool SupportsGpuTraining => true;

    /// <summary>
    /// Applies the SELU activation function on GPU.
    /// </summary>
    /// <param name="backend">The GPU backend to use for execution.</param>
    /// <param name="input">The input GPU buffer.</param>
    /// <param name="output">The output GPU buffer to store the activated values.</param>
    /// <param name="size">The number of elements to process.</param>
    /// <remarks>
    /// SELU on GPU: output[i] = lambda * (input[i] >= 0 ? input[i] : alpha * (exp(input[i]) - 1))
    /// Uses fixed lambda ≈ 1.0507 and alpha ≈ 1.6733 for self-normalization.
    /// </remarks>
    public override void ForwardGpu(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer output, int size)
    {
        float alpha = (float)NumOps.ToDouble(_alpha);
        float lambda = (float)NumOps.ToDouble(_lambda);
        backend.Selu(input, output, alpha, lambda, size);
    }

    /// <summary>
    /// Calculates the SELU backward pass gradient on GPU.
    /// </summary>
    /// <param name="backend">The GPU backend to use for execution.</param>
    /// <param name="gradOutput">The gradient flowing back from the next layer.</param>
    /// <param name="input">The input buffer from the forward pass.</param>
    /// <param name="output">Not used for SELU (can be null). SELU backward uses forward input.</param>
    /// <param name="gradInput">The output buffer to store the input gradient.</param>
    /// <param name="size">The number of elements to process.</param>
    /// <remarks>
    /// SELU backward on GPU: gradInput[i] = gradOutput[i] * (input[i] >= 0 ? lambda : lambda * alpha * exp(input[i]))
    /// </remarks>
    public override void BackwardGpu(IDirectGpuBackend backend, IGpuBuffer gradOutput, IGpuBuffer? input, IGpuBuffer? output, IGpuBuffer gradInput, int size)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input), "SELU backward requires the input from forward pass.");

        float alpha = (float)NumOps.ToDouble(_alpha);
        float lambda = (float)NumOps.ToDouble(_lambda);
        backend.SeluBackward(gradOutput, input, gradInput, alpha, lambda, size);
    }

    #endregion
}
