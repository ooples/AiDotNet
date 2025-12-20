using AiDotNet.Autodiff;


namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the LogSoftmin activation function for neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The LogSoftmin function is similar to LogSoftmax, but focuses on the smallest values 
/// instead of the largest values in a vector.
/// 
/// It works in two steps:
/// 1. First, it applies the "softmin" function, which gives more weight to smaller numbers in the input
///    (the opposite of softmax, which emphasizes larger numbers).
/// 2. Then, it takes the natural logarithm of these values.
/// 
/// While LogSoftmax is used to highlight the largest values (useful for finding the most likely class in 
/// classification), LogSoftmin can be useful when you want to focus on the smallest values in your data.
/// 
/// Like LogSoftmax, this function operates on vectors (collections of numbers) rather than individual values,
/// because it needs to consider all values together to determine their relative importance.
/// </para>
/// </remarks>
public class LogSoftminActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Determines if the activation function supports operations on individual scalar values.
    /// </summary>
    /// <returns>False, as LogSoftmin requires a vector of values to operate.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This returns false because LogSoftmin needs to compare all values in a collection
    /// to determine which ones are smallest. It can't process just one number at a time.
    /// </para>
    /// </remarks>
    protected override bool SupportsScalarOperations() => false;

    /// <summary>
    /// Applies the LogSoftmin activation function to a vector of values.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>A vector with LogSoftmin applied.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes a collection of numbers and transforms them using the LogSoftmin function.
    /// 
    /// The implementation uses a numerically stable approach:
    /// 1. First, it finds the minimum value in the input
    /// 2. It calculates exp(min - x) for each value x, which gives higher results for smaller input values
    /// 3. It sums these exponential values
    /// 4. Finally, it computes log(sum) - min and subtracts this from the negative of each input
    /// 
    /// This approach helps avoid extremely large or small numbers that could cause calculation errors,
    /// while emphasizing the smallest values in the input vector.
    /// </para>
    /// </remarks>
    public override Vector<T> Activate(Vector<T> input)
    {
        T minInput = input.Min();
        Vector<T> shiftedExp = input.Transform(x => NumOps.Exp(NumOps.Subtract(minInput, x)));
        T sumExp = shiftedExp.Sum();
        T logSumExp = NumOps.Add(NumericalStabilityHelper.SafeLog(sumExp), NumOps.Negate(minInput));

        return input.Transform(x => NumOps.Subtract(NumOps.Negate(x), logSumExp));
    }

    /// <summary>
    /// Calculates the derivative (Jacobian matrix) of the LogSoftmin function for a vector input.
    /// </summary>
    /// <param name="input">The input vector at which to calculate the derivative.</param>
    /// <returns>A Jacobian matrix representing the derivative of LogSoftmin.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The derivative helps us understand how the output changes when we slightly change the input.
    /// For LogSoftmin, this is complex because changing one input value affects all outputs.
    /// 
    /// This method calculates a special matrix called a "Jacobian matrix" that shows how each 
    /// output changes when each input changes slightly. For LogSoftmin:
    /// 
    /// - Diagonal elements (where i=j): softmin(i) - 1
    /// - Off-diagonal elements: softmin(j)
    /// 
    /// This matrix is essential during neural network training, as it helps determine how to adjust
    /// the network's weights to improve performance. The structure of this matrix reflects how
    /// the LogSoftmin function emphasizes smaller values in the input.
    /// </para>
    /// </remarks>
    public override Matrix<T> Derivative(Vector<T> input)
    {
        Vector<T> softmin = Activate(input).Transform(NumOps.Exp);
        int n = input.Length;
        Matrix<T> jacobian = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                {
                    jacobian[i, j] = NumOps.Subtract(softmin[i], NumOps.One);
                }
                else
                {
                    jacobian[i, j] = softmin[j];
                }
            }
        }

        return jacobian;
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because TensorOperations.LogSoftmin provides full forward and backward pass support.</value>
    /// <remarks>
    /// <para>
    /// LogSoftmin supports JIT compilation with numerically stable gradient computation.
    /// The backward pass efficiently computes gradients similar to LogSoftmax but for the minimum-focused variant.
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
    /// <returns>A new computation node with LogSoftmin activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps to TensorOperations&lt;T&gt;.LogSoftmin(input) which handles both
    /// forward and backward passes for JIT compilation.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return TensorOperations<T>.LogSoftmin(input);
    }
}
