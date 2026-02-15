using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines an interface for activation functions used in neural networks and other machine learning algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> An activation function is like a decision-maker in a neural network.
/// 
/// Imagine each neuron (node) in a neural network receives a number as input. The activation 
/// function decides how strongly that neuron should "fire" or activate based on that input.
/// 
/// For example:
/// - If the input is very negative, the neuron might not activate at all (output = 0)
/// - If the input is very positive, the neuron might activate fully (output = 1)
/// - If the input is around zero, the neuron might activate partially
/// 
/// Different activation functions create different patterns of activation, which helps
/// neural networks learn different types of patterns in data. Common activation functions
/// include Sigmoid, ReLU (Rectified Linear Unit), and Tanh (Hyperbolic Tangent).
/// 
/// This interface defines the standard methods that all activation functions must implement.
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("ActivationFunction")]
public interface IActivationFunction<T>
{
    /// <summary>
    /// Applies the activation function to the input value.
    /// </summary>
    /// <param name="input">The input value to the activation function.</param>
    /// <returns>The activated output value.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method takes a number (which could be positive, negative, or zero)
    /// and transforms it according to the specific activation function's rule.
    /// 
    /// For example, with the ReLU activation function:
    /// - If input is negative, output is 0
    /// - If input is positive, output is the same as the input
    /// 
    /// This transformation helps neural networks model complex, non-linear relationships
    /// in data, which is essential for tasks like image recognition or language processing.
    /// </remarks>
    T Activate(T input);

    /// <summary>
    /// Calculates the derivative (slope) of the activation function at the given input value.
    /// </summary>
    /// <param name="input">The input value at which to calculate the derivative.</param>
    /// <returns>The derivative value of the activation function at the input point.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> The derivative tells us how quickly the activation function's output
    /// changes when we make a small change to the input.
    ///
    /// Think of it as the "slope" or "steepness" at a particular point on the activation function's curve.
    ///
    /// This is crucial for training neural networks because:
    /// - It helps determine how much to adjust the network's weights during learning
    /// - A higher derivative means a stronger signal for learning
    /// - A derivative of zero means no learning signal (which can be a problem known as "vanishing gradient")
    ///
    /// During training, the neural network uses this derivative to figure out how to adjust
    /// its internal parameters to improve its predictions.
    /// </remarks>
    T Derivative(T input);

    /// <summary>
    /// Applies the activation function to each element in a vector.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>A new vector with the activation function applied to each element.</returns>
    Vector<T> Activate(Vector<T> input);

    /// <summary>
    /// Calculates the derivative matrix for a vector input.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>A diagonal matrix containing derivatives for each input element.</returns>
    Matrix<T> Derivative(Vector<T> input);

    /// <summary>
    /// Applies the activation function to each element in a tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A new tensor with the activation function applied to each element.</returns>
    Tensor<T> Activate(Tensor<T> input);

    /// <summary>
    /// Calculates the derivative for each element in a tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A new tensor containing derivatives for each input element.</returns>
    Tensor<T> Derivative(Tensor<T> input);

    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True if the activation can be applied to computation graphs for JIT compilation.</value>
    /// <remarks>
    /// <para>
    /// Activation functions return false if:
    /// - Gradient computation (backward pass) is not yet implemented
    /// - The activation uses operations not supported by TensorOperations
    /// - The activation has dynamic behavior that cannot be represented in a static graph
    /// </para>
    /// <para>
    /// Once gradient computation is implemented and tested, set this to true.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> JIT (Just-In-Time) compilation is an advanced optimization technique
    /// that pre-compiles the neural network's operations into a faster execution graph.
    /// This property indicates whether this activation function is ready to be part of that
    /// optimized execution. If false, the activation will fall back to the standard execution path.
    /// </para>
    /// </remarks>
    bool SupportsJitCompilation { get; }

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with the activation applied.</returns>
    /// <exception cref="NotSupportedException">Thrown if SupportsJitCompilation is false.</exception>
    /// <remarks>
    /// <para>
    /// This method maps the activation to the corresponding TensorOperations method.
    /// For example, ReLU returns TensorOperations&lt;T&gt;.ReLU(input).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method adds the activation function to the computation graph,
    /// which is a data structure that represents all the operations in the neural network.
    /// The graph can then be optimized and executed more efficiently through JIT compilation.
    /// </para>
    /// </remarks>
    ComputationNode<T> ApplyToGraph(ComputationNode<T> input);

    /// <summary>
    /// Calculates the backward pass gradient for this activation function.
    /// </summary>
    /// <param name="input">The input tensor that was used in the forward pass.</param>
    /// <param name="outputGradient">The gradient flowing back from the next layer.</param>
    /// <returns>The gradient with respect to the input.</returns>
    /// <remarks>
    /// <para>
    /// This method computes dL/dx = dL/dy * dy/dx.
    /// For element-wise activations (ReLU, Sigmoid), this is element-wise multiplication.
    /// For vector activations (Softmax), this involves Jacobian multiplication.
    /// </para>
    /// </remarks>
    Tensor<T> Backward(Tensor<T> input, Tensor<T> outputGradient);

    #region GPU Training Support

    /// <summary>
    /// Gets whether this activation function supports GPU-resident training.
    /// </summary>
    /// <value>True if the activation can perform forward and backward passes entirely on GPU.</value>
    /// <remarks>
    /// <para>
    /// Activation functions return false if:
    /// - The GPU backend does not have kernels for this activation type
    /// - The activation has dynamic behavior that cannot be executed on GPU
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> GPU-resident training keeps all data on the graphics card during
    /// training, avoiding slow data transfers between CPU and GPU memory. This property indicates
    /// whether this activation function can participate in GPU-resident training.
    /// </para>
    /// </remarks>
    bool SupportsGpuTraining { get; }

    /// <summary>
    /// Applies the activation function on GPU.
    /// </summary>
    /// <param name="backend">The GPU backend to use for execution.</param>
    /// <param name="input">The input GPU buffer.</param>
    /// <param name="output">The output GPU buffer to store the activated values.</param>
    /// <param name="size">The number of elements to process.</param>
    /// <remarks>
    /// <para>
    /// This method applies the activation function entirely on GPU, avoiding CPU-GPU data transfers.
    /// The input and output buffers may be the same for in-place operations if supported.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the GPU-accelerated version of the Activate method.
    /// Instead of processing data on the CPU, this runs thousands of calculations in parallel
    /// on the GPU, making it much faster for large tensors.
    /// </para>
    /// </remarks>
    /// <exception cref="NotSupportedException">Thrown if SupportsGpuTraining is false.</exception>
    void ForwardGpu(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer output, int size);

    /// <summary>
    /// Calculates the backward pass gradient on GPU.
    /// </summary>
    /// <param name="backend">The GPU backend to use for execution.</param>
    /// <param name="gradOutput">The gradient flowing back from the next layer.</param>
    /// <param name="input">The input buffer from the forward pass (needed for ReLU, GELU, Swish, LeakyReLU).</param>
    /// <param name="output">The output buffer from the forward pass (needed for Sigmoid, Tanh).</param>
    /// <param name="gradInput">The output buffer to store the input gradient.</param>
    /// <param name="size">The number of elements to process.</param>
    /// <remarks>
    /// <para>
    /// This method computes the activation gradient entirely on GPU.
    /// Different activation functions require different cached values from forward pass:
    /// - ReLU, LeakyReLU, GELU, Swish: Need the input from forward pass
    /// - Sigmoid, Tanh: Need the output from forward pass
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> During training, we need to compute gradients to update the network.
    /// This method computes the gradient of the activation function on GPU, which is essential
    /// for efficient GPU-resident training.
    /// </para>
    /// </remarks>
    /// <exception cref="NotSupportedException">Thrown if SupportsGpuTraining is false.</exception>
    void BackwardGpu(IDirectGpuBackend backend, IGpuBuffer gradOutput, IGpuBuffer? input, IGpuBuffer? output, IGpuBuffer gradInput, int size);

    #endregion
}
