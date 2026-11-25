using AiDotNet.Attributes;
using AiDotNet.Enums;


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
[ActivationCategory(ActivationCategory.Normalization)]
[ActivationTask(ActivationTask.NormalizationOutput)]
[ActivationProperty(IsMonotonic = false, ZeroPreserving = false, IsBounded = false, IsVectorActivation = true, Cost = ComputeCost.High)]
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
        if (input.Length == 0)
        {
            return new Vector<T>(0);
        }

        T minInput = input.Min();
        Vector<T> shiftedExp = input.Transform(x => NumOps.Exp(NumOps.Subtract(minInput, x)));
        T sumExp = shiftedExp.Sum();
        T logSumExp = NumOps.Add(NumericalStabilityHelper.SafeLog(sumExp), NumOps.Negate(minInput));

        return input.Transform(x => NumOps.Subtract(NumOps.Negate(x), logSumExp));
    }

    /// <summary>
    /// Applies LogSoftmin to a tensor via engine primitives so the gradient tape records
    /// every step. LogSoftmin(x) = LogSoftmax(-x), which is a single negate followed by
    /// Engine.TensorLogSoftmax.
    /// </summary>
    public override Tensor<T> Activate(Tensor<T> input) =>
        Engine.TensorLogSoftmax(Engine.TensorNegate(input), axis: -1);

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
        if (input.Length == 0)
        {
            return new Matrix<T>(0, 0);
        }

        Vector<T> softmin = Activate(input).Transform(NumOps.Exp);
        // J[i, i] = s[i] - 1; J[i, j] = s[j], so every off-diagonal row is the softmin
        var engine = Engine;
        if (UseScalarJacobian(input.Length))
        {
            return ScalarLogSoftmaxJacobian(softmin, NumOps.Negate(NumOps.One));
        }

        var probabilities = Tensor<T>.FromVector(softmin);
        // Every row is softmin: ones (x) softmin, with softmin - 1 on the diagonal
        return OuterWithDiagonal(
            engine.TensorOnesLike(probabilities), probabilities, engine.TensorSubtractScalar(probabilities, NumOps.One));
    }

}
