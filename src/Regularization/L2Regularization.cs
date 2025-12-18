namespace AiDotNet.Regularization;

/// <summary>
/// Implements L2 regularization (also known as Ridge), a technique that adds a penalty equal to the
/// square of the magnitude of coefficients.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The type of input data structure.</typeparam>
/// <typeparam name="TOutput">The type of output data structure.</typeparam>
/// <remarks>
/// <para>
/// L2 regularization adds a penalty term to the loss function equal to the sum of the squared values
/// of the model coefficients, multiplied by a regularization strength parameter. This encourages smaller,
/// more evenly distributed coefficient values, which helps prevent overfitting and improves model stability.
/// </para>
/// <para><b>For Beginners:</b> L2 regularization helps create smoother models by making all coefficients smaller.
/// 
/// Think of it like a gentle pull that shrinks all coefficients proportionally:
/// - It doesn't eliminate features entirely (unlike L1 regularization)
/// - It reduces the impact of all features by making their coefficients smaller
/// - It particularly penalizes large coefficient values
/// </para>
/// </remarks>
public class L2Regularization<T, TInput, TOutput> : RegularizationBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the L2Regularization class with optional custom options.
    /// </summary>
    /// <param name="options">
    /// Configuration options for regularization including strength. If not provided,
    /// default values will be used (strength = 0.01).
    /// </param>
    /// <remarks>
    /// <para>
    /// This constructor creates an L2 regularization instance with the specified options or default values.
    /// For L2 regularization, the L1Ratio is always set to 0.0, as it represents pure L2 regularization.
    /// Note that the default strength for L2 regularization (0.01) is typically smaller than for L1
    /// because it's applied to squared values.
    /// </para>
    /// </remarks>
    public L2Regularization(RegularizationOptions? options = null) : base(options ?? new RegularizationOptions
    {
        Type = RegularizationType.L2,
        Strength = 0.01, // Default L2 regularization strength
        L1Ratio = 0.0  // For L2, this should always be 0.0
    })
    {
    }

    /// <summary>
    /// Adjusts the gradient vector to account for L2 regularization during optimization.
    /// </summary>
    /// <param name="gradient">The original gradient vector from the loss function.</param>
    /// <param name="coefficients">The current coefficient vector.</param>
    /// <returns>The regularized gradient vector.</returns>
    /// <remarks>
    /// <para>
    /// This method modifies the gradient vector during optimization to account for the L2 regularization
    /// penalty. It adds the derivative of the L2 penalty term (the coefficients multiplied by the regularization
    /// strength) to the original gradient, steering the optimization toward solutions with smaller coefficient values.
    /// </para>
    /// </remarks>
    public override TOutput Regularize(TOutput gradient, TOutput coefficients)
    {
        var regularizationStrength = NumOps.FromDouble(Options.Strength);

        if (gradient is Vector<T> gradientVector && coefficients is Vector<T> coefficientVector)
        {
            var result = gradientVector.Add(coefficientVector.Multiply(regularizationStrength));
            return (TOutput)(object)result;
        }
        else if (gradient is Tensor<T> gradientTensor && coefficients is Tensor<T> coefficientTensor)
        {
            // Convert tensors to vectors for calculation
            var gradientFlattenedVector = gradientTensor.ToVector();
            var coefficientFlattenedVector = coefficientTensor.ToVector();

            // Apply regularization to gradient
            var result = gradientFlattenedVector.Add(coefficientFlattenedVector.Multiply(regularizationStrength));

            // Convert back to tensor with the same shape as the gradient tensor
            var resultTensor = Tensor<T>.FromVector(result);
            if (gradientTensor.Shape.Length > 1)
            {
                resultTensor = resultTensor.Reshape(gradientTensor.Shape);
            }

            return (TOutput)(object)resultTensor;
        }

        throw new InvalidOperationException(
            $"Unsupported output types {typeof(TOutput).Name} for L2 regularization gradient. " +
            $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
    }

    /// <summary>
    /// Applies L2 regularization to a matrix.
    /// </summary>
    /// <param name="data">The matrix data to regularize.</param>
    /// <returns>The regularized matrix data.</returns>
    /// <remarks>
    /// <para>
    /// This method implements L2 regularization for matrix data by uniformly shrinking all values.
    /// Each element is multiplied by a factor slightly less than 1, determined by the 
    /// regularization strength. This causes all values to become smaller, which helps
    /// prevent overfitting in the model.
    /// </para>
    /// </remarks>
    public override Matrix<T> Regularize(Matrix<T> data)
    {
        var regularizationStrength = NumOps.FromDouble(Options.Strength);
        var shrinkageFactor = NumOps.Subtract(NumOps.One, regularizationStrength);
        var result = new Matrix<T>(data.Rows, data.Columns);

        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                result[i, j] = NumOps.Multiply(data[i, j], shrinkageFactor);
            }
        }

        return result;
    }

    /// <summary>
    /// Applies L2 regularization to a vector.
    /// </summary>
    /// <param name="data">The vector data to regularize.</param>
    /// <returns>The regularized vector data.</returns>
    /// <remarks>
    /// <para>
    /// This method implements L2 regularization for vector data by uniformly shrinking all values.
    /// Each element is multiplied by a factor slightly less than 1, determined by the 
    /// regularization strength. This approach reduces the magnitude of all coefficients
    /// proportionally without eliminating any features entirely.
    /// </para>
    /// </remarks>
    public override Vector<T> Regularize(Vector<T> data)
    {
        var regularizationStrength = NumOps.FromDouble(Options.Strength);
        var shrinkageFactor = NumOps.Subtract(NumOps.One, regularizationStrength);
        var result = new Vector<T>(data.Length);

        for (int i = 0; i < data.Length; i++)
        {
            result[i] = NumOps.Multiply(data[i], shrinkageFactor);
        }

        return result;
    }
}
