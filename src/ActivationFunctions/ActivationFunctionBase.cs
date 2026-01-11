using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Base class for all activation functions used in neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Activation functions are mathematical operations that determine the output
/// of a neural network node. They introduce non-linearity into the network, allowing it to
/// learn complex patterns. Think of them as decision-makers that determine how strongly a
/// neuron "fires" based on its inputs.
/// 
/// Common activation functions include:
/// - Sigmoid: Outputs values between 0 and 1 (like probabilities)
/// - ReLU: Returns 0 for negative inputs, or the input value for positive inputs
/// - Tanh: Similar to sigmoid but outputs values between -1 and 1
/// 
/// The "derivative" methods are used during training to determine how to adjust the network's
/// weights to improve its accuracy.
/// </para>
/// </remarks>
public abstract class ActivationFunctionBase<T> : IActivationFunction<T>, IVectorActivationFunction<T>
{
    /// <summary>
    /// Provides mathematical operations for the numeric type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the global execution engine for vector operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Determines if the activation function supports operations on individual scalar values.
    /// </summary>
    /// <returns>True if scalar operations are supported; otherwise, false.</returns>
    protected abstract bool SupportsScalarOperations();

    /// <summary>
    /// Applies the activation function to a single input value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The activated output value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method transforms a single number using the activation function.
    /// The default implementation is the identity function (returns the input unchanged).
    /// Derived classes will override this with specific activation functions like sigmoid or ReLU.
    /// </para>
    /// </remarks>
    public virtual T Activate(T input)
    {
        return input; // Default to identity function
    }

    /// <summary>
    /// Calculates the derivative of the activation function for a single input value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The derivative value at the input point.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The derivative measures how much the activation function's output changes
    /// when its input changes slightly. This is essential for training neural networks through
    /// backpropagation. The default implementation returns 1, meaning the output changes at the
    /// same rate as the input.
    /// </para>
    /// </remarks>
    public virtual T Derivative(T input)
    {
        return NumOps.One; // Default to constant derivative of 1
    }

    /// <summary>
    /// Applies the activation function to each element in a vector.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>A new vector with the activation function applied to each element.</returns>
    public virtual Vector<T> Activate(Vector<T> input)
    {
        return input.Transform(Activate);
    }

    /// <summary>
    /// Calculates the derivative matrix for a vector input.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>A diagonal matrix containing derivatives for each input element.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a special matrix where the diagonal contains the derivatives
    /// for each input value. This matrix is used during backpropagation to efficiently calculate
    /// how errors propagate through the network.
    /// </para>
    /// </remarks>
    public virtual Matrix<T> Derivative(Vector<T> input)
    {
        return Matrix<T>.CreateDiagonal(input.Transform(Derivative));
    }

    /// <summary>
    /// Applies the activation function to each element in a tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A new tensor with the activation function applied to each element.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A tensor is a multi-dimensional array that can represent complex data
    /// structures like images (3D tensors) or video (4D tensors). This method applies the
    /// activation function to every single value in the tensor.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> Activate(Tensor<T> input)
    {
        Tensor<T> output = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            output[i] = Activate(input[i]);
        }

        return output;
    }

    /// <summary>
    /// Calculates the derivative for each element in a tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A new tensor containing derivatives for each input element.</returns>
    public virtual Tensor<T> Derivative(Tensor<T> input)
    {
        Tensor<T> output = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            output[i] = Derivative(input[i]);
        }

        return output;
    }

    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>False by default; derived classes override to return true when gradient is implemented.</value>
    /// <remarks>
    /// <para>
    /// The default implementation returns false, indicating the activation does not yet support
    /// JIT compilation. Derived classes should override this to return true once their gradient
    /// computation is fully implemented and tested.
    /// </para>
    /// </remarks>
    public virtual bool SupportsJitCompilation => false;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with the activation applied.</returns>
    /// <exception cref="NotSupportedException">Thrown because the default implementation does not support JIT compilation.</exception>
    /// <remarks>
    /// <para>
    /// The default implementation throws NotSupportedException. Derived classes must override
    /// this method to map their activation to the corresponding TensorOperations method.
    /// </para>
    /// <para>
    /// For example, ReLUActivation should return TensorOperations&lt;T&gt;.ReLU(input).
    /// </para>
    /// </remarks>
    public virtual ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        throw new NotSupportedException(
            $"{GetType().Name} does not support JIT compilation yet. " +
            $"SupportsJitCompilation = {SupportsJitCompilation}. " +
            $"Either the gradient computation is not implemented, or the activation uses " +
            $"operations not compatible with computation graphs.");
    }

    /// <summary>
    /// Calculates the backward pass gradient for this activation function.
    /// </summary>
    /// <param name="input">The input tensor that was used in the forward pass.</param>
    /// <param name="outputGradient">The gradient flowing back from the next layer.</param>
    /// <returns>The gradient with respect to the input.</returns>
    /// <remarks>
    /// <para>
    /// Default implementation assumes element-wise activation.
    /// Returns: Derivative(input) * outputGradient (element-wise).
    /// </para>
    /// </remarks>
    public virtual Tensor<T> Backward(Tensor<T> input, Tensor<T> outputGradient)
    {
        // Default behavior: Element-wise multiplication of derivative and gradient
        // This works for ReLU, Sigmoid, Tanh, etc.
        // Derived classes like Softmax MUST override this.

        var derivative = Derivative(input);
        return Engine.TensorMultiply(derivative, outputGradient);
    }

    #region GPU Training Support

    /// <summary>
    /// Gets whether this activation function supports GPU-resident training.
    /// </summary>
    /// <value>
    /// True if the activation can perform forward and backward passes entirely on GPU.
    /// Default is false; derived classes override this to return true if GPU kernels are available.
    /// </value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> GPU-resident training keeps all data on the graphics card during
    /// training, avoiding slow data transfers between CPU and GPU memory. This property indicates
    /// whether this activation function can participate in GPU-resident training.
    /// </para>
    /// </remarks>
    public virtual bool SupportsGpuTraining => false;

    /// <summary>
    /// Applies the activation function on GPU.
    /// </summary>
    /// <param name="backend">The GPU backend to use for execution.</param>
    /// <param name="input">The input GPU buffer.</param>
    /// <param name="output">The output GPU buffer to store the activated values.</param>
    /// <param name="size">The number of elements to process.</param>
    /// <exception cref="NotSupportedException">
    /// Thrown because this activation function does not support GPU execution.
    /// Override this method in derived classes to provide GPU support.
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the GPU-accelerated version of the Activate method.
    /// The default implementation throws an exception because most activation functions
    /// need to provide their own GPU kernel implementation.
    /// </para>
    /// </remarks>
    public virtual void ForwardGpu(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer output, int size)
    {
        throw new NotSupportedException(
            $"{GetType().Name} does not support GPU-resident training. " +
            $"SupportsGpuTraining = {SupportsGpuTraining}. " +
            "Use the CPU-based Activate() method instead or implement GPU support in the derived class.");
    }

    /// <summary>
    /// Calculates the backward pass gradient on GPU.
    /// </summary>
    /// <param name="backend">The GPU backend to use for execution.</param>
    /// <param name="gradOutput">The gradient flowing back from the next layer.</param>
    /// <param name="input">The input buffer from the forward pass (needed for ReLU, GELU, Swish, LeakyReLU).</param>
    /// <param name="output">The output buffer from the forward pass (needed for Sigmoid, Tanh).</param>
    /// <param name="gradInput">The output buffer to store the input gradient.</param>
    /// <param name="size">The number of elements to process.</param>
    /// <exception cref="NotSupportedException">
    /// Thrown because this activation function does not support GPU execution.
    /// Override this method in derived classes to provide GPU support.
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> During training, we need to compute gradients to update the network.
    /// The default implementation throws an exception because most activation functions
    /// need to provide their own GPU backward kernel implementation.
    /// </para>
    /// </remarks>
    public virtual void BackwardGpu(IDirectGpuBackend backend, IGpuBuffer gradOutput, IGpuBuffer? input, IGpuBuffer? output, IGpuBuffer gradInput, int size)
    {
        throw new NotSupportedException(
            $"{GetType().Name} does not support GPU-resident backward pass. " +
            $"SupportsGpuTraining = {SupportsGpuTraining}. " +
            "Use the CPU-based Backward() method instead or implement GPU support in the derived class.");
    }

    #endregion
}
