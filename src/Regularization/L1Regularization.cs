namespace AiDotNet.Regularization;

/// <summary>
/// Implements L1 regularization (also known as Lasso), a technique that adds a penalty equal to the
/// absolute value of the magnitude of coefficients.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The type of input data structure.</typeparam>
/// <typeparam name="TOutput">The type of output data structure.</typeparam>
/// <remarks>
/// <para>
/// L1 regularization adds a penalty term to the loss function equal to the sum of the absolute values
/// of the model coefficients, multiplied by a regularization strength parameter. This encourages sparse
/// models by driving some coefficients to exactly zero, effectively performing feature selection.
/// </para>
/// <para><b>For Beginners:</b> L1 regularization helps create simpler models by completely removing less important features.
/// 
/// Think of it like a strict budget committee:
/// - It forces the model to focus only on the most important features
/// - Less important features get their coefficients reduced to exactly zero
/// - This means some features are completely eliminated from the model
/// </para>
/// </remarks>
public class L1Regularization<T, TInput, TOutput> : RegularizationBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the L1Regularization class with optional custom options.
    /// </summary>
    /// <param name="options">
    /// Configuration options for regularization including strength. If not provided,
    /// default values will be used (strength = 0.1).
    /// </param>
    /// <remarks>
    /// <para>
    /// This constructor creates an L1 regularization instance with the specified options or default values.
    /// For L1 regularization, the L1Ratio is always set to 1.0, as it represents pure L1 regularization.
    /// </para>
    /// </remarks>
    public L1Regularization(RegularizationOptions? options = null)
        : base(options ?? new RegularizationOptions
        {
            Type = RegularizationType.L1,
            Strength = 0.1, // Default L1 regularization strength
            L1Ratio = 1.0  // For L1, this should always be 1.0
        })
    {
    }

    /// <summary>
    /// Adjusts the gradient vector to account for L1 regularization during optimization.
    /// </summary>
    /// <param name="gradient">The original gradient vector from the loss function.</param>
    /// <param name="coefficients">The current coefficient vector.</param>
    /// <returns>The regularized gradient vector.</returns>
    /// <remarks>
    /// <para>
    /// This method modifies the gradient vector during optimization to account for the L1 regularization
    /// penalty. It adds the subdifferential of the L1 norm (the sign of each coefficient multiplied by
    /// the regularization strength) to the original gradient, steering the optimization toward sparse solutions.
    /// </para>
    /// </remarks>
    public override TOutput Regularize(TOutput gradient, TOutput coefficients)
    {
        if (gradient is Vector<T> gradientVector && coefficients is Vector<T> coefficientVector)
        {
            var regularizationStrength = NumOps.FromDouble(Options.Strength);
            var result = gradientVector.Add(coefficientVector.Transform(c =>
                NumOps.Multiply(regularizationStrength, NumOps.SignOrZero(c))
            ));

            return (TOutput)(object)result;
        }
        else if (gradient is Tensor<T> gradientTensor && coefficients is Tensor<T> coefficientTensor)
        {
            var regularizationStrength = NumOps.FromDouble(Options.Strength);
            var gradientFlattenedVector = gradientTensor.ToVector();
            var coefficientFlattenedVector = coefficientTensor.ToVector();

            var result = gradientFlattenedVector.Add(coefficientFlattenedVector.Transform(c =>
                NumOps.Multiply(regularizationStrength, NumOps.SignOrZero(c))
            ));

            // Convert back to tensor with the same shape as the gradient tensor
            var resultTensor = Tensor<T>.FromVector(result);
            if (gradientTensor.Shape.Length > 1)
            {
                resultTensor = resultTensor.Reshape(gradientTensor.Shape);
            }

            return (TOutput)(object)resultTensor;
        }

        throw new InvalidOperationException(
            $"Unsupported output types {typeof(TOutput).Name} for L1 regularization gradient. " +
            $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
    }

    /// <summary>
    /// Applies L1 regularization to a matrix.
    /// </summary>
    /// <param name="data">The matrix data to regularize.</param>
    /// <returns>The regularized matrix data.</returns>
    /// <remarks>
    /// <para>
    /// This method applies L1 regularization to a matrix by performing soft thresholding on each element.
    /// Elements with absolute values less than the regularization strength are set to zero,
    /// while those above the threshold are shrunk toward zero by the regularization strength amount.
    /// </para>
    /// </remarks>
    public override Matrix<T> Regularize(Matrix<T> data)
    {
        var regularizationStrength = NumOps.FromDouble(Options.Strength);
        var result = new Matrix<T>(data.Rows, data.Columns);

        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                var value = data[i, j];
                var sub = NumOps.Subtract(NumOps.Abs(value), regularizationStrength);
                result[i, j] = NumOps.Multiply(
                    NumOps.SignOrZero(value),
                    NumOps.GreaterThan(sub, NumOps.Zero) ? sub : NumOps.Zero
                );
            }
        }

        return result;
    }

    /// <summary>
    /// Applies L1 regularization to a vector.
    /// </summary>
    /// <param name="data">The vector data to regularize.</param>
    /// <returns>The regularized vector data.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the soft thresholding operation for L1 regularization on vector data.
    /// Elements with absolute values less than the regularization strength are set to zero,
    /// while those above the threshold are shrunk toward zero by the regularization strength amount.
    /// </para>
    /// </remarks>
    public override Vector<T> Regularize(Vector<T> data)
    {
        var regularizationStrength = NumOps.FromDouble(Options.Strength);
        var result = new Vector<T>(data.Length);

        for (int i = 0; i < data.Length; i++)
        {
            var value = data[i];
            var sub = NumOps.Subtract(NumOps.Abs(value), regularizationStrength);
            result[i] = NumOps.Multiply(
                NumOps.SignOrZero(value),
                NumOps.GreaterThan(sub, NumOps.Zero) ? sub : NumOps.Zero
            );
        }

        return result;
    }
}
