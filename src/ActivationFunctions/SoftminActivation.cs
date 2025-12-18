using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Softmin activation function, which is the opposite of Softmax and highlights the smallest values in a vector.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Softmin function takes a vector of real numbers and transforms it into a probability distribution
/// that emphasizes smaller values. It's defined as: softmin(x_i) = exp(-x_i) / sum(exp(-x_j)) for all j in the vector.
/// </para>
/// <para>
/// <b>For Beginners:</b> While Softmax emphasizes the largest values in a vector, Softmin does the opposite - it gives
/// more weight to smaller values. Think of it as a "smooth minimum" function. For example, if you have scores
/// [5, 2, 8], Softmax would highlight 8 (the maximum), but Softmin would highlight 2 (the minimum).
/// 
/// Softmin is less commonly used than Softmax but can be useful in scenarios where you want to identify or
/// emphasize the smallest values in a set, such as finding the closest points in certain distance-based algorithms
/// or when you want to assign higher probabilities to smaller values.
/// 
/// Like Softmax, Softmin outputs values between 0 and 1 that sum to 1, creating a probability distribution.
/// </para>
/// </remarks>
public class SoftminActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Indicates whether this activation function supports scalar operations.
    /// </summary>
    /// <returns>Always returns false as Softmin only operates on vectors, not individual values.</returns>
    /// <remarks>
    /// <para>
    /// Softmin is inherently a vector operation because it normalizes values relative to each other.
    /// It cannot be applied to a single scalar value in isolation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Like Softmax, Softmin needs to see all numbers at once to calculate its output.
    /// This is because each output value depends on all input values. This method returns false to indicate
    /// that you should only use Softmin with vectors (arrays of numbers), not individual values.
    /// </para>
    /// </remarks>
    protected override bool SupportsScalarOperations() => false;

    /// <summary>
    /// Applies the Softmin activation function to a vector of input values.
    /// </summary>
    /// <param name="input">The vector of input values.</param>
    /// <returns>A vector of probabilities that sum to 1, with higher probabilities for smaller input values.</returns>
    /// <remarks>
    /// <para>
    /// The implementation first negates each input value (turning x into -x), then computes the exponential of each,
    /// and finally divides each by the sum of all exponentials to ensure the output values sum to 1.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method transforms a list of numbers into probabilities by:
    /// 1. Flipping the sign of each number (making positive numbers negative and vice versa)
    /// 2. Calculating e^x for each of these negated numbers
    /// 3. Adding up all these e^x values to get a total
    /// 4. Dividing each e^x by this total
    /// 
    /// The result is a list of numbers between 0 and 1 that add up to exactly 1 (or 100%).
    /// The smallest original input value will produce the largest probability in the output.
    /// </para>
    /// </remarks>
    public override Vector<T> Activate(Vector<T> input)
    {
        Vector<T> negInput = input.Transform(NumOps.Negate);
        Vector<T> expValues = negInput.Transform(NumOps.Exp);
        T sum = expValues.Sum();

        return expValues.Transform(x => NumOps.Divide(x, sum));
    }

    /// <summary>
    /// Calculates the Jacobian matrix of the Softmin function for a vector input.
    /// </summary>
    /// <param name="input">The vector of input values.</param>
    /// <returns>A matrix representing the partial derivatives of each output with respect to each input.</returns>
    /// <remarks>
    /// <para>
    /// The Jacobian matrix for Softmin has a structure similar to Softmax:
    /// - For diagonal elements (i=j): J[i,i] = softmin(x_i) * (1 - softmin(x_i))
    /// - For off-diagonal elements (i?j): J[i,j] = -softmin(x_i) * softmin(x_j)
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The derivative of Softmin shows how the output probabilities change when you slightly
    /// change each input value. This creates a grid (matrix) where each cell shows how changing one input
    /// affects one output.
    /// 
    /// This is important during neural network training to determine how to adjust the network's weights.
    /// Don't worry about understanding the mathematical details - the library handles these calculations
    /// automatically when training your model.
    /// 
    /// The key difference from Softmax's derivative is that Softmin's derivative reflects how changes
    /// affect the emphasis on smaller values rather than larger ones.
    /// </para>
    /// </remarks>
    public override Matrix<T> Derivative(Vector<T> input)
    {
        Vector<T> softmin = Activate(input);
        int n = softmin.Length;
        Matrix<T> jacobian = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                {
                    jacobian[i, j] = NumOps.Multiply(softmin[i], NumOps.Subtract(NumOps.One, softmin[i]));
                }
                else
                {
                    jacobian[i, j] = NumOps.Negate(NumOps.Multiply(softmin[i], softmin[j]));
                }
            }
        }

        return jacobian;
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because TensorOperations.Softmin provides full forward and backward pass support.</value>
    /// <remarks>
    /// <para>
    /// Softmin supports JIT compilation with full gradient computation.
    /// The backward pass computes gradients similar to softmax but with negation for the input transformation.
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
    /// <returns>A new computation node with Softmin activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps to TensorOperations&lt;T&gt;.Softmin(input) which handles both
    /// forward and backward passes for JIT compilation.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return TensorOperations<T>.Softmin(input);
    }
}
