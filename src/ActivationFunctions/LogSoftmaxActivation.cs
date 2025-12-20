

using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the LogSoftmax activation function for neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The LogSoftmax function combines two operations:
///
/// 1. First, it applies the "softmax" function, which converts a vector of numbers into probabilities
///    (values between 0 and 1 that sum to 1).
/// 2. Then, it takes the natural logarithm of these probabilities.
///
/// This function is commonly used in the final layer of neural networks for classification problems,
/// especially when combined with Negative Log-Likelihood loss. It helps with:
///
/// - Numerical stability (preventing extremely small or large numbers)
/// - Making the math work better during training
/// - Producing outputs that work well for calculating classification probabilities
///
/// Unlike most activation functions, LogSoftmax operates on vectors (collections of numbers) rather than
/// individual values, because it needs to consider all outputs together to calculate probabilities.
/// </para>
/// </remarks>
public class LogSoftmaxActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Determines if the activation function supports operations on individual scalar values.
    /// </summary>
    /// <returns>False, as LogSoftmax requires a vector of values to operate.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This returns false because LogSoftmax needs to look at all values together
    /// to calculate probabilities. It can't process just one number at a time.
    /// </para>
    /// </remarks>
    protected override bool SupportsScalarOperations() => false;

    /// <summary>
    /// Applies the LogSoftmax activation function to a vector of values.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>A vector with LogSoftmax applied.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes a collection of numbers and transforms them using the LogSoftmax function.
    ///
    /// The implementation uses TensorPrimitivesHelper for SIMD-optimized operations and a numerically stable approach:
    /// 1. First, it finds the maximum value in the input (to prevent overflow) - SIMD optimized!
    /// 2. It subtracts this maximum from all values before applying exponential functions - SIMD optimized!
    /// 3. It calculates the sum of these adjusted exponential values - SIMD optimized!
    /// 4. Finally, it computes log(sum) + max and subtracts this from each input - SIMD optimized!
    ///
    /// This approach helps avoid extremely large or small numbers that could cause calculation errors.
    /// </para>
    /// </remarks>
    public override Vector<T> Activate(Vector<T> input)
    {
        // Use SIMD-optimized Max (8-12Ã— speedup for float)
        T maxInput = TensorPrimitivesHelper<T>.Max(input);

        // Subtract max from all elements (for numerical stability)
        var maxVector = new Vector<T>(Enumerable.Repeat(maxInput, input.Length).ToArray());
        var shifted = TensorPrimitivesHelper<T>.Subtract(input, maxVector);

        // Apply Exp using SIMD (3-6Ã— speedup for float)
        var shiftedExp = TensorPrimitivesHelper<T>.Exp(shifted);

        // Use SIMD-optimized Sum (8-12Ã— speedup for float)
        T sumExp = TensorPrimitivesHelper<T>.Sum(shiftedExp);
        T logSumExp = NumOps.Add(NumericalStabilityHelper.SafeLog(sumExp), maxInput);

        // Subtract logSumExp from each element using SIMD
        var logSumExpVector = new Vector<T>(Enumerable.Repeat(logSumExp, input.Length).ToArray());
        return TensorPrimitivesHelper<T>.Subtract(input, logSumExpVector);
    }

    /// <summary>
    /// Calculates the derivative (Jacobian matrix) of the LogSoftmax function for a vector input.
    /// </summary>
    /// <param name="input">The input vector at which to calculate the derivative.</param>
    /// <returns>A Jacobian matrix representing the derivative of LogSoftmax.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The derivative of LogSoftmax is more complex than most activation functions
    /// because each output depends on all inputs.
    /// 
    /// This method calculates a special matrix called a "Jacobian matrix" that shows how each 
    /// output changes when each input changes slightly. For LogSoftmax:
    /// 
    /// - Diagonal elements (where i=j): 1 - softmax(i)
    /// - Off-diagonal elements: -softmax(j)
    /// 
    /// This matrix is essential for backpropagation during neural network training, as it helps
    /// determine how to adjust the weights to improve the network's performance.
    /// </para>
    /// </remarks>
    public override Matrix<T> Derivative(Vector<T> input)
    {
        Vector<T> softmax = input.Transform(NumOps.Exp);
        T sum = softmax.Sum();
        softmax = softmax.Transform(x => NumOps.Divide(x, sum));

        int n = input.Length;
        Matrix<T> jacobian = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                {
                    jacobian[i, j] = NumOps.Subtract(NumOps.One, softmax[i]);
                }
                else
                {
                    jacobian[i, j] = NumOps.Negate(softmax[j]);
                }
            }
        }

        return jacobian;
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because TensorOperations.LogSoftmax provides full forward and backward pass support.</value>
    /// <remarks>
    /// <para>
    /// LogSoftmax supports JIT compilation with numerically stable gradient computation.
    /// The backward pass efficiently computes gradients: gradient - softmax * sum(gradient).
    /// </para>
    /// <para>
    /// Note: Currently implemented for 2D tensors (batch, features) along axis=-1.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with LogSoftmax activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps to TensorOperations&lt;T&gt;.LogSoftmax(input) which handles both
    /// forward and backward passes for JIT compilation with numerical stability.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return TensorOperations<T>.LogSoftmax(input);
    }
}
