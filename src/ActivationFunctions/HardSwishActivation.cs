using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Hard Swish activation function used in MobileNetV3.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Hard Swish is a computationally efficient approximation of the Swish activation function.
/// It is defined as: f(x) = x * min(max(0, x + 3), 6) / 6
/// </para>
/// <para>
/// <b>For Beginners:</b> Hard Swish combines the benefits of ReLU and Sigmoid-like activations.
///
/// The Swish function (x * sigmoid(x)) performs well in neural networks but requires
/// computing the expensive sigmoid function. Hard Swish approximates this using simpler operations:
///
/// - For inputs &lt; -3: output is 0 (similar to ReLU for negative values)
/// - For inputs &gt; 3: output equals the input (similar to identity for large positive values)
/// - For inputs between -3 and 3: smooth transition using a simple piecewise linear function
///
/// Hard Swish is particularly important in MobileNetV3 because:
/// - It is faster to compute than regular Swish on mobile devices
/// - It provides better accuracy than ReLU for deeper networks
/// - It is compatible with quantized inference (8-bit integer arithmetic)
/// </para>
/// </remarks>
public class HardSwishActivation<T> : ActivationFunctionBase<T>
{
    private readonly T _three;
    private readonly T _six;
    private readonly T _negThree;
    private readonly T _oneSixth;

    /// <summary>
    /// Initializes a new instance of the <see cref="HardSwishActivation{T}"/> class.
    /// </summary>
    public HardSwishActivation()
    {
        _three = NumOps.FromDouble(3.0);
        _six = NumOps.FromDouble(6.0);
        _negThree = NumOps.FromDouble(-3.0);
        _oneSixth = NumOps.FromDouble(1.0 / 6.0);
    }

    /// <summary>
    /// Indicates whether this activation function supports scalar operations.
    /// </summary>
    /// <returns>Always returns true as Hard Swish supports scalar operations.</returns>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the Hard Swish activation function to a single input value.
    /// </summary>
    /// <param name="input">The input value to activate.</param>
    /// <returns>x * min(max(0, x + 3), 6) / 6</returns>
    public override T Activate(T input)
    {
        T xPlusThree = NumOps.Add(input, _three);
        T clamped = MathHelper.Min(_six, MathHelper.Max(NumOps.Zero, xPlusThree));
        T scaled = NumOps.Multiply(clamped, _oneSixth);
        return NumOps.Multiply(input, scaled);
    }

    /// <summary>
    /// Calculates the derivative of the Hard Swish function for a single input value.
    /// </summary>
    /// <param name="input">The input value to calculate the derivative for.</param>
    /// <returns>The derivative value based on the input range.</returns>
    public override T Derivative(T input)
    {
        if (NumOps.LessThanOrEquals(input, _negThree))
        {
            return NumOps.Zero;
        }
        else if (NumOps.GreaterThanOrEquals(input, _three))
        {
            return NumOps.One;
        }
        else
        {
            T twoX = NumOps.Multiply(NumOps.FromDouble(2.0), input);
            T twoXPlusThree = NumOps.Add(twoX, _three);
            return NumOps.Multiply(twoXPlusThree, _oneSixth);
        }
    }

    /// <summary>
    /// Applies the Hard Swish activation function to each element in a vector.
    /// </summary>
    /// <param name="input">The input vector to activate.</param>
    /// <returns>A new vector with the Hard Swish function applied to each element.</returns>
    public override Vector<T> Activate(Vector<T> input)
    {
        return input.Transform(x =>
        {
            T xPlusThree = NumOps.Add(x, _three);
            T clamped = MathHelper.Min(_six, MathHelper.Max(NumOps.Zero, xPlusThree));
            T scaled = NumOps.Multiply(clamped, _oneSixth);
            return NumOps.Multiply(x, scaled);
        });
    }

    /// <summary>
    /// Calculates the Jacobian matrix of the Hard Swish function for a vector input.
    /// </summary>
    /// <param name="input">The input vector to calculate the derivative for.</param>
    /// <returns>A diagonal matrix where each diagonal element is the derivative of Hard Swish for the corresponding input element.</returns>
    public override Matrix<T> Derivative(Vector<T> input)
    {
        int n = input.Length;
        Matrix<T> jacobian = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            jacobian[i, i] = Derivative(input[i]);
        }

        return jacobian;
    }

    /// <summary>
    /// Applies the Hard Swish activation function to each element in a tensor.
    /// </summary>
    /// <param name="input">The input tensor to activate.</param>
    /// <returns>A new tensor with the Hard Swish function applied to each element.</returns>
    public override Tensor<T> Activate(Tensor<T> input)
    {
        return Engine.HardSwish(input);
    }

    /// <summary>
    /// Calculates the derivative of the Hard Swish function for each element in a tensor.
    /// </summary>
    /// <param name="input">The input tensor to calculate the derivative for.</param>
    /// <returns>A new tensor containing the derivatives of the Hard Swish function for each input element.</returns>
    public override Tensor<T> Derivative(Tensor<T> input)
    {
        return input.Transform((x, _) => Derivative(x));
    }

    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>False because TensorOperations.Minimum is not yet implemented.</value>
    public override bool SupportsJitCompilation => false;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with Hard Swish activation applied.</returns>
    /// <exception cref="NotSupportedException">Always thrown as JIT compilation is not supported.</exception>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        throw new NotSupportedException(
            "HardSwishActivation does not support JIT compilation. Use the standard Activate method instead.");
    }

    #region GPU Training Support

    /// <summary>
    /// Gets whether HardSwish supports GPU-resident training.
    /// </summary>
    /// <value>True because HardSwish has GPU kernels for both forward and backward passes.</value>
    public override bool SupportsGpuTraining => true;

    /// <summary>
    /// Applies the HardSwish activation function on GPU.
    /// </summary>
    /// <param name="backend">The GPU backend to use for execution.</param>
    /// <param name="input">The input GPU buffer.</param>
    /// <param name="output">The output GPU buffer to store the activated values.</param>
    /// <param name="size">The number of elements to process.</param>
    /// <remarks>
    /// HardSwish on GPU: output[i] = input[i] * min(max(0, input[i] + 3), 6) / 6
    /// </remarks>
    public override void ForwardGpu(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer output, int size)
    {
        backend.Hardswish(input, output, size);
    }

    /// <summary>
    /// Calculates the HardSwish backward pass gradient on GPU.
    /// </summary>
    /// <param name="backend">The GPU backend to use for execution.</param>
    /// <param name="gradOutput">The gradient flowing back from the next layer.</param>
    /// <param name="input">The input buffer from the forward pass.</param>
    /// <param name="output">Not used for HardSwish (can be null). HardSwish backward uses forward input.</param>
    /// <param name="gradInput">The output buffer to store the input gradient.</param>
    /// <param name="size">The number of elements to process.</param>
    /// <remarks>
    /// HardSwish backward on GPU:
    /// - For input &lt;= -3: gradInput[i] = 0
    /// - For input &gt;= 3: gradInput[i] = gradOutput[i]
    /// - Otherwise: gradInput[i] = gradOutput[i] * (2 * input[i] + 3) / 6
    /// </remarks>
    public override void BackwardGpu(IDirectGpuBackend backend, IGpuBuffer gradOutput, IGpuBuffer? input, IGpuBuffer? output, IGpuBuffer gradInput, int size)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input), "HardSwish backward requires the input from forward pass.");

        backend.HardswishBackward(gradOutput, input, gradInput, size);
    }

    #endregion
}
