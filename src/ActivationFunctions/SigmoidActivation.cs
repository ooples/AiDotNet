

using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Sigmoid activation function, one of the most common activation functions in neural networks.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Sigmoid function maps any input value to an output between 0 and 1, creating an S-shaped curve.
/// It's often used in the output layer of binary classification problems or in hidden layers of neural networks.
/// </para>
/// <para>
/// <b>For Beginners:</b> The Sigmoid function is like a "squashing" function that takes any number (from negative
/// infinity to positive infinity) and converts it to a value between 0 and 1. This is useful in neural networks
/// because it helps transform unbounded values into a probability-like range. The function creates an S-shaped
/// curve that approaches 0 for very negative inputs and approaches 1 for very positive inputs, with a smooth
/// transition in between. However, one limitation is that for extreme values (very large positive or negative),
/// the gradient becomes very small, which can slow down learning in deep networks (known as the "vanishing
/// gradient problem").
/// </para>
/// </remarks>
public class SigmoidActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Indicates whether this activation function supports scalar operations.
    /// </summary>
    /// <returns>Always returns true as Sigmoid supports scalar operations.</returns>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the Sigmoid activation function to a single input value.
    /// </summary>
    /// <param name="input">The input value to activate.</param>
    /// <returns>The Sigmoid of the input: 1 / (1 + e^(-input)).</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method calculates the Sigmoid value for a single number. The formula 1/(1+e^(-x)) 
    /// creates that S-shaped curve that squashes any input to be between 0 and 1. When the input is 0, 
    /// the output is exactly 0.5. As inputs get more positive, the output approaches 1, and as inputs get 
    /// more negative, the output approaches 0.
    /// </remarks>
    public override T Activate(T input)
    {
        return NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, NumOps.Exp(NumOps.Negate(input))));
    }

    /// <summary>
    /// Calculates the derivative of the Sigmoid function for a single input value.
    /// </summary>
    /// <param name="input">The input value to calculate the derivative for.</param>
    /// <returns>The derivative of Sigmoid at the input: sigmoid(input) * (1 - sigmoid(input)).</returns>
    /// <remarks>
    /// <b>For Beginners:</b> The derivative tells us how much the Sigmoid output changes when we slightly change the input.
    /// For the Sigmoid function, the derivative has a simple formula: sigmoid(x) * (1 - sigmoid(x)).
    /// This means the derivative is largest (0.25) when the input is 0, and gets smaller as the input moves away from 0
    /// in either direction. This property can cause the "vanishing gradient problem" in deep neural networks,
    /// where the learning signal becomes too weak for neurons in early layers.
    /// </remarks>
    public override T Derivative(T input)
    {
        T sigmoid = Activate(input);
        return NumOps.Multiply(sigmoid, NumOps.Subtract(NumOps.One, sigmoid));
    }

    /// <summary>
    /// Applies the Sigmoid activation function to each element in a vector using SIMD optimization.
    /// </summary>
    /// <param name="input">The vector of input values.</param>
    /// <returns>A new vector with the Sigmoid function applied to each element.</returns>
    /// <remarks>
    /// <para>
    /// This implementation uses TensorPrimitivesHelper for SIMD-optimized operations (3-6× speedup for float).
    /// For arrays with fewer than 16 elements, it falls back to manual loops.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method applies the Sigmoid function to a whole list of numbers at once
    /// using hardware acceleration, making it much faster than processing each number separately.
    ///
    /// For example, if you have a vector [-2, -1, 0, 1, 2]:
    /// - The output would be approximately [0.12, 0.27, 0.50, 0.73, 0.88]
    /// - All values are computed in parallel using SIMD instructions
    /// </para>
    /// </remarks>
    public override Vector<T> Activate(Vector<T> input)
    {
        // Use SIMD-optimized Sigmoid (3-6× speedup for float)
        return TensorPrimitivesHelper<T>.Sigmoid(input);
    }

    /// <summary>
    /// Applies the Sigmoid activation function to each element in a tensor.
    /// </summary>
    /// <param name="input">The input tensor to activate.</param>
    /// <returns>A new tensor with the Sigmoid function applied to each element.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> A tensor is a multi-dimensional array or a container for data.
    /// This method applies the Sigmoid function to every single value in that container,
    /// regardless of its position or dimension.
    /// </remarks>
    public override Tensor<T> Activate(Tensor<T> input)
    {
        return Engine.Sigmoid(input);
    }

    /// <summary>
    /// Calculates the Jacobian matrix of the Sigmoid function for a vector input.
    /// </summary>
    /// <param name="input">The vector of input values.</param>
    /// <returns>A diagonal matrix where each diagonal element is the derivative of the Sigmoid function at the corresponding input.</returns>
    /// <remarks>
    /// <para>
    /// The Jacobian matrix represents how each output element changes with respect to each input element.
    /// For element-wise functions like Sigmoid, this is a diagonal matrix.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how sensitive each output is to changes in the input,
    /// but for a whole list of numbers at once. The result is a special matrix (called a diagonal matrix)
    /// that shows the rate of change for each input value. This is important during the learning process
    /// when the neural network is adjusting its weights. Don't worry too much about the mathematical details -
    /// this is handled automatically during training.
    /// </para>
    /// </remarks>
    public override Matrix<T> Derivative(Vector<T> input)
    {
        Vector<T> sigmoid = Activate(input);
        return Matrix<T>.CreateDiagonal(sigmoid.Transform(s => NumOps.Multiply(s, NumOps.Subtract(NumOps.One, s))));
    }

    /// <summary>
    /// Calculates the derivative of the Sigmoid function for each element in a tensor.
    /// </summary>
    /// <param name="input">The input tensor to calculate the derivative for.</param>
    /// <returns>A new tensor containing the derivatives of the Sigmoid function for each input element.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method calculates how sensitive each output value is to small changes
    /// in the corresponding input value, for every value in the tensor. For Sigmoid, the derivative is
    /// sigmoid(x) * (1 - sigmoid(x)).
    /// </remarks>
    public override Tensor<T> Derivative(Tensor<T> input)
    {
        var sigmoid = Activate(input);
        var oneMinusSigmoid = Engine.TensorSubtract(Tensor<T>.CreateDefault(input.Shape, NumOps.One), sigmoid);
        return Engine.TensorMultiply(sigmoid, oneMinusSigmoid);
    }

    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because Sigmoid gradient computation is fully implemented and tested.</value>
    /// <remarks>
    /// <para>
    /// Sigmoid supports JIT compilation because:
    /// - The gradient computation (backward pass) is fully implemented in TensorOperations
    /// - The operation is well-defined and differentiable
    /// - It can be represented as a static computation graph node
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with Sigmoid activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps the Sigmoid activation to TensorOperations&lt;T&gt;.Sigmoid(input),
    /// which handles both forward and backward passes for JIT compilation.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return TensorOperations<T>.Sigmoid(input);
    }
}
