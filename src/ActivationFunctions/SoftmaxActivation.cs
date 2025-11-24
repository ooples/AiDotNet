using AiDotNet.Helpers;

using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Softmax activation function, which converts a vector of real numbers into a probability distribution.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Softmax function takes a vector of real numbers and normalizes it into a probability distribution,
/// where each value is between 0 and 1, and all values sum to 1. It's defined as:
/// softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j in the vector.
/// </para>
/// <para>
/// <b>For Beginners:</b> Softmax is commonly used in the output layer of neural networks for classification problems.
/// Think of it as a way to convert raw scores (called "logits") into probabilities. For example, if you're
/// classifying images into 3 categories (cat, dog, bird), the neural network might output raw scores like
/// [2.5, 1.2, 0.8]. Softmax converts these to probabilities like [0.65, 0.22, 0.13], which sum to 1.0 (or 100%).
/// This makes it easy to interpret the highest value as the model's prediction (in this case, "cat" with 65% confidence).
///
/// Unlike other activation functions that work on single values, Softmax needs to see all values at once because
/// it normalizes them relative to each other.
/// </para>
/// </remarks>
public class SoftmaxActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Applies the Softmax activation function to a vector of input values.
    /// </summary>
    /// <param name="input">The vector of input values.</param>
    /// <returns>A vector of probabilities that sum to 1.</returns>
    /// <remarks>
    /// <para>
    /// The implementation uses TensorPrimitivesHelper for SIMD-optimized Exp and Sum operations (5-10Ã— speedup for float),
    /// then divides each value by the sum to ensure the output values sum to 1.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method transforms a list of numbers into probabilities by:
    /// 1. Calculating e^x for each number (which makes all values positive) - SIMD optimized!
    /// 2. Adding up all these e^x values to get a total - SIMD optimized!
    /// 3. Dividing each e^x by this total
    ///
    /// The result is a list of numbers between 0 and 1 that add up to exactly 1 (or 100%).
    /// The largest input value will produce the largest probability, but the exact values
    /// depend on the relative differences between all inputs.
    /// </para>
    /// </remarks>
    public override Vector<T> Activate(Vector<T> input)
    {
        // Use TensorPrimitivesHelper for SIMD-optimized Exp (5-10Ã— speedup for float)
        var expVector = TensorPrimitivesHelper<T>.Exp(input);

        // Use TensorPrimitivesHelper for SIMD-optimized Sum (8-12Ã— speedup for float)
        T sum = TensorPrimitivesHelper<T>.Sum(expVector);

        // Create sum vector for vectorized division
        var sumVector = new Vector<T>(Enumerable.Repeat(sum, expVector.Length).ToArray());

        // Use TensorPrimitivesHelper for SIMD-optimized Divide (5-10Ã— speedup for float)
        return TensorPrimitivesHelper<T>.Divide(expVector, sumVector);
    }

    /// <summary>
    /// Calculates the Jacobian matrix of the Softmax function for a vector input.
    /// </summary>
    /// <param name="input">The vector of input values.</param>
    /// <returns>A matrix representing the partial derivatives of each output with respect to each input.</returns>
    /// <remarks>
    /// <para>
    /// The Jacobian matrix for Softmax has a special structure:
    /// - For diagonal elements (i=j): J[i,i] = softmax(x_i) * (1 - softmax(x_i))
    /// - For off-diagonal elements (i?j): J[i,j] = -softmax(x_i) * softmax(x_j)
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The derivative of Softmax is more complex than other activation functions because
    /// changing one input affects all outputs. This method creates a grid (matrix) that shows how each 
    /// output probability changes when you slightly change each input value.
    /// 
    /// The Jacobian matrix is important during neural network training (backpropagation) to determine
    /// how to adjust the network's weights. You don't need to understand the mathematical details to use
    /// Softmax - the library handles these calculations automatically when training your model.
    /// </para>
    /// </remarks>
    public override Matrix<T> Derivative(Vector<T> input)
    {
        Vector<T> softmaxOutput = Activate(input);
        int size = softmaxOutput.Length;
        Matrix<T> jacobian = new Matrix<T>(size, size);

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                if (i == j)
                {
                    jacobian[i, j] = NumOps.Multiply(softmaxOutput[i], NumOps.Subtract(NumOps.One, softmaxOutput[i]));
                }
                else
                {
                    jacobian[i, j] = NumOps.Multiply(NumOps.Negate(softmaxOutput[i]), softmaxOutput[j]);
                }
            }
        }

        return jacobian;
    }

    /// <summary>
    /// Indicates whether this activation function supports scalar operations.
    /// </summary>
    /// <returns>Always returns false as Softmax only operates on vectors, not individual values.</returns>
    /// <remarks>
    /// <para>
    /// Softmax is inherently a vector operation because it normalizes values relative to each other.
    /// It cannot be applied to a single scalar value in isolation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Unlike functions like ReLU or Sigmoid that can work on single numbers,
    /// Softmax needs to see all numbers at once to calculate probabilities. This is because
    /// each output probability depends on all input values. This method returns false to indicate
    /// that you should only use Softmax with vectors (arrays of numbers), not individual values.
    /// </para>
    /// </remarks>
    protected override bool SupportsScalarOperations() => false;


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>False because gradient computation is not yet implemented.</value>
    /// <remarks>
    /// <para>
    /// This activation does not yet support JIT compilation because the gradient
    /// computation (backward pass) has not been implemented in TensorOperations.Softmax.
    /// </para>
    /// <para>
    /// To enable JIT support:
    /// 1. Implement the backward pass in TensorOperations.Softmax
    /// 2. Test the gradient computation
    /// 3. Change SupportsJitCompilation to return true
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => false;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with Softmax activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <exception cref="NotSupportedException">Thrown because gradient is not implemented.</exception>
    /// <remarks>
    /// <para>
    /// This method would map the activation to TensorOperations&lt;T&gt;.Softmax(input)
    /// once the gradient computation is implemented.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        throw new NotSupportedException(
            $"SoftmaxActivation does not support JIT compilation yet. " +
            $"The gradient computation (backward pass) has not been implemented in TensorOperations.Softmax. " +
            $"Once gradients are implemented, this activation can be used in JIT-compiled computation graphs.");
    }
}