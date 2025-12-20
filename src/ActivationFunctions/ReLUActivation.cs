using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Rectified Linear Unit (ReLU) activation function.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The ReLU function is one of the most commonly used activation functions in neural networks.
/// It outputs the input directly if it is positive, otherwise, it outputs zero.
/// </para>
/// <para>
/// <b>For Beginners:</b> ReLU (Rectified Linear Unit) is like a filter that only lets positive values 
/// pass through unchanged, while changing all negative values to zero. Think of it as a function 
/// that "turns off" neurons that have negative values and keeps positive ones as they are.
/// This helps neural networks learn more effectively by introducing non-linearity.
/// </para>
/// </remarks>
public class ReLUActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Indicates whether this activation function supports scalar operations.
    /// </summary>
    /// <returns>Always returns true as ReLU supports scalar operations.</returns>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the ReLU activation function to a single input value.
    /// </summary>
    /// <param name="input">The input value to activate.</param>
    /// <returns>The input value if it's greater than zero, otherwise zero.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method takes a single number and returns that same number 
    /// if it's positive, or returns zero if it's negative or zero.
    /// </remarks>
    public override T Activate(T input)
    {
        return NumOps.GreaterThan(input, NumOps.Zero) ? input : NumOps.Zero;
    }

    /// <summary>
    /// Calculates the derivative of the ReLU function for a single input value.
    /// </summary>
    /// <param name="input">The input value to calculate the derivative for.</param>
    /// <returns>1 if the input is greater than zero, otherwise 0.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> The derivative tells us how much the output changes when we slightly change the input.
    /// For ReLU, the derivative is 1 for positive inputs (meaning the output changes at the same rate as the input),
    /// and 0 for negative inputs (meaning the output doesn't change at all when the input changes).
    /// </remarks>
    public override T Derivative(T input)
    {
        return NumOps.GreaterThan(input, NumOps.Zero) ? NumOps.One : NumOps.Zero;
    }

    /// <summary>
    /// Applies the ReLU activation function to each element in a vector.
    /// </summary>
    /// <param name="input">The input vector to activate.</param>
    /// <returns>A new vector with the ReLU function applied to each element.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method takes a list of numbers (a vector) and applies the ReLU function
    /// to each number in the list, returning a new list of the same size.
    /// </remarks>
    public override Vector<T> Activate(Vector<T> input)
    {
        return input.Transform(x => MathHelper.Max(NumOps.Zero, x));
    }

    /// <summary>
    /// Calculates the Jacobian matrix of the ReLU function for a vector input.
    /// </summary>
    /// <param name="input">The input vector to calculate the derivative for.</param>
    /// <returns>A diagonal matrix where each diagonal element is the derivative of ReLU for the corresponding input element.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method creates a special matrix (called a Jacobian) that represents how each output
    /// element changes with respect to each input element. For ReLU, this is a diagonal matrix (only has values along
    /// the main diagonal) where each value is either 1 (if the corresponding input was positive) or 0 (if negative).
    /// </remarks>
    public override Matrix<T> Derivative(Vector<T> input)
    {
        int n = input.Length;
        Matrix<T> jacobian = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            jacobian[i, i] = NumOps.GreaterThan(input[i], NumOps.Zero) ? NumOps.One : NumOps.Zero;
        }

        return jacobian;
    }

    /// <summary>
    /// Applies the ReLU activation function to each element in a tensor.
    /// </summary>
    /// <param name="input">The input tensor to activate.</param>
    /// <returns>A new tensor with the ReLU function applied to each element.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> A tensor is like a multi-dimensional array or a container for data.
    /// This method applies the ReLU function to every single value in that container,
    /// regardless of its position or dimension.
    /// </remarks>
    public override Tensor<T> Activate(Tensor<T> input)
    {
        return Engine.ReLU(input);
    }

    /// <summary>
    /// Calculates the derivative of the ReLU function for each element in a tensor.
    /// </summary>
    /// <param name="input">The input tensor to calculate the derivative for.</param>
    /// <returns>A new tensor containing the derivatives of the ReLU function for each input element.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method calculates how sensitive each output value is to small changes
    /// in the corresponding input value, for every value in the tensor. For ReLU, this means
    /// replacing positive values with 1 and negative values with 0.
    /// </remarks>
    public override Tensor<T> Derivative(Tensor<T> input)
    {
        return Engine.TensorGreaterThan(input, NumOps.Zero);
    }

    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because ReLU gradient computation is fully implemented and tested.</value>
    /// <remarks>
    /// <para>
    /// ReLU supports JIT compilation because:
    /// - The gradient computation (backward pass) is fully implemented in TensorOperations
    /// - The operation is simple and efficient (max(0, x))
    /// - It can be represented as a static computation graph node
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with ReLU activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps the ReLU activation to TensorOperations&lt;T&gt;.ReLU(input),
    /// which handles both forward and backward passes for JIT compilation.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return TensorOperations<T>.ReLU(input);
    }
}
