namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the LogSoftmax activation function for neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// For Beginners: The LogSoftmax function combines two operations:
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
    /// For Beginners: This returns false because LogSoftmax needs to look at all values together
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
    /// For Beginners: This method takes a collection of numbers and transforms them using the LogSoftmax function.
    /// 
    /// The implementation uses a numerically stable approach:
    /// 1. First, it finds the maximum value in the input (to prevent overflow)
    /// 2. It subtracts this maximum from all values before applying exponential functions
    /// 3. It calculates the sum of these adjusted exponential values
    /// 4. Finally, it computes log(sum) + max and subtracts this from each input
    /// 
    /// This approach helps avoid extremely large or small numbers that could cause calculation errors.
    /// </para>
    /// </remarks>
    public override Vector<T> Activate(Vector<T> input)
    {
        T maxInput = input.Max();
        Vector<T> shiftedExp = input.Transform(x => NumOps.Exp(NumOps.Subtract(x, maxInput)));
        T sumExp = shiftedExp.Sum();
        T logSumExp = NumOps.Add(NumOps.Log(sumExp), maxInput);

        return input.Transform(x => NumOps.Subtract(x, logSumExp));
    }

    /// <summary>
    /// Calculates the derivative (Jacobian matrix) of the LogSoftmax function for a vector input.
    /// </summary>
    /// <param name="input">The input vector at which to calculate the derivative.</param>
    /// <returns>A Jacobian matrix representing the derivative of LogSoftmax.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: The derivative of LogSoftmax is more complex than most activation functions
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
}