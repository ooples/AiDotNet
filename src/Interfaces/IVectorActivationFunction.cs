using AiDotNet.Autodiff;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines activation functions that operate on vectors and tensors in neural networks.
/// </summary>
/// <remarks>
/// Activation functions introduce non-linearity into neural networks, allowing them to learn
/// complex patterns in data. This interface provides methods to apply activation functions
/// to vectors and tensors, as well as calculate their derivatives for backpropagation.
/// 
/// <b>For Beginners:</b> Activation functions are like "decision makers" in neural networks.
/// 
/// Imagine you're deciding whether to go outside based on the temperature:
/// - If it's below 60�F, you definitely won't go (output = 0)
/// - If it's above 75�F, you definitely will go (output = 1)
/// - If it's between 60-75�F, you're somewhat likely to go (output between 0 and 1)
/// 
/// This is similar to how activation functions work. They take the input from previous
/// calculations in the neural network and transform it into an output that determines
/// how strongly a neuron "fires" or activates. Without activation functions, neural
/// networks would just be doing simple linear calculations and couldn't learn complex patterns.
/// 
/// Common activation functions include:
/// - Sigmoid: Outputs values between 0 and 1 (like our temperature example)
/// - ReLU: Outputs the input if positive, or zero if negative
/// - Tanh: Outputs values between -1 and 1
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public interface IVectorActivationFunction<T>
{
    /// <summary>
    /// Applies the activation function to each element in a vector.
    /// </summary>
    /// <remarks>
    /// This method transforms each value in the input vector according to the activation function.
    /// 
    /// <b>For Beginners:</b> This method takes a list of numbers (the input vector) and applies
    /// the same transformation to each number. For example, if using the ReLU activation function:
    /// 
    /// Input vector: [-2, 0, 3, -1, 5]
    /// Output vector: [0, 0, 3, 0, 5]
    /// 
    /// The ReLU function keeps positive values unchanged but changes negative values to zero.
    /// Different activation functions will transform the values differently.
    /// </remarks>
    /// <param name="input">The vector to apply the activation function to.</param>
    /// <returns>A new vector with the activation function applied to each element.</returns>
    Vector<T> Activate(Vector<T> input);

    /// <summary>
    /// Calculates the derivative of the activation function for each element in a vector.
    /// </summary>
    /// <remarks>
    /// This method computes how the activation function's output changes with respect to small
    /// changes in its input. This is essential for the backpropagation algorithm in neural networks.
    /// 
    /// <b>For Beginners:</b> The derivative tells us how sensitive the activation function is to changes
    /// in its input. This is crucial for the "learning" part of neural networks.
    /// 
    /// Think of it like this: If you slightly increase the temperature in our earlier example,
    /// how much more likely are you to go outside? The derivative gives us this rate of change.
    /// 
    /// For a vector input, this method returns a matrix where each element represents the
    /// derivative at the corresponding position in the input vector.
    /// </remarks>
    /// <param name="input">The vector to calculate derivatives for.</param>
    /// <returns>A matrix containing the derivatives of the activation function.</returns>
    Matrix<T> Derivative(Vector<T> input);

    /// <summary>
    /// Applies the activation function to each element in a tensor.
    /// </summary>
    /// <remarks>
    /// This method transforms each value in the input tensor according to the activation function.
    /// 
    /// <b>For Beginners:</b> A tensor is like a multi-dimensional array - think of it as a cube or
    /// higher-dimensional block of numbers. This method applies the same transformation to
    /// every number in that block.
    /// 
    /// For example, if you have image data (which can be represented as a 3D tensor with
    /// dimensions for height, width, and color channels), this method would apply the
    /// activation function to every pixel value in the image.
    /// </remarks>
    /// <param name="input">The tensor to apply the activation function to.</param>
    /// <returns>A new tensor with the activation function applied to each element.</returns>
    Tensor<T> Activate(Tensor<T> input);

    /// <summary>
    /// Calculates the derivative of the activation function for each element in a tensor.
    /// </summary>
    /// <remarks>
    /// This method computes the derivatives of the activation function for all elements in the input tensor.
    ///
    /// <b>For Beginners:</b> Similar to the vector version, this calculates how sensitive the activation
    /// function is to changes in each element of the input tensor. The difference is that this
    /// works with multi-dimensional data.
    ///
    /// For example, with image data, this would tell us how a small change in each pixel's value
    /// would affect the output of the activation function. This information is used during the
    /// learning process to adjust the neural network's parameters.
    /// </remarks>
    /// <param name="input">The tensor to calculate derivatives for.</param>
    /// <returns>A tensor containing the derivatives of the activation function.</returns>
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
    /// For example, Softmax returns TensorOperations&lt;T&gt;.Softmax(input).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method adds the activation function to the computation graph,
    /// which is a data structure that represents all the operations in the neural network.
    /// The graph can then be optimized and executed more efficiently through JIT compilation.
    /// </para>
    /// </remarks>
    ComputationNode<T> ApplyToGraph(ComputationNode<T> input);
}