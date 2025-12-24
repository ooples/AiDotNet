namespace AiDotNet.Normalizers;

/// <summary>
/// A normalizer that does not modify the data but maintains the same interface as other normalizers.
/// </summary>
/// <remarks>
/// <para>
/// The NoNormalizer is a pass-through implementation that does not alter the input values.
/// It returns the original values unchanged while still conforming to the INormalizer interface.
/// This implementation is useful when normalization is optional in an algorithm, or when comparing
/// normalized and unnormalized results.
/// </para>
/// <para>
/// Use cases for NoNormalizer include:
/// - Testing the effects of normalization by comparing with non-normalized data
/// - Working with data that is already in an appropriate scale
/// - Algorithm implementations that make normalization optional
/// - Ensuring consistent interfaces across a system even when normalization isn't needed
/// </para>
/// <para><b>For Beginners:</b> This normalizer doesn't actually normalize anything.
/// 
/// Think of NoNormalizer as a "do nothing" option:
/// - While other normalizers transform data to different scales
/// - This one simply passes the original values through unchanged
/// - It's like having a "raw" or "as is" option when processing data
/// 
/// This is useful when:
/// - You want to compare the effects of normalization versus no normalization
/// - Your data is already in a good range for your needs
/// - You're building a system where normalization is optional but the interface should be consistent
/// 
/// It's similar to having an "identity function" that returns exactly what you put into it.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The type of input data structure.</typeparam>
/// <typeparam name="TOutput">The type of output data structure.</typeparam>
public class NoNormalizer<T, TInput, TOutput> : NormalizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="NoNormalizer{T, TInput, TOutput}"/> class.
    /// </summary>
    /// <remarks>
    /// This constructor creates a new NoNormalizer and initializes the numeric operations
    /// provider for the specified type T through the base class constructor.
    /// </remarks>
    public NoNormalizer() : base()
    {
        // Base constructor already initializes NumOps
    }

    /// <summary>
    /// Returns the input data unchanged along with minimal normalization parameters.
    /// </summary>
    /// <param name="data">The input data to "normalize" (which remains unchanged).</param>
    /// <returns>
    /// A tuple containing the original data and basic normalization parameters with Method set to None.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method returns the original data without any transformation, along with
    /// a NormalizationParameters object that indicates no normalization was performed.
    /// This preserves the interface expected by calling code while performing no actual normalization.
    /// </para>
    /// <para><b>For Beginners:</b> This method just returns your data exactly as you provided it.
    /// 
    /// When you call this method:
    /// - The data you input is the same data you get back
    /// - No calculations or transformations are performed
    /// - The method also returns information indicating that no normalization was performed
    /// 
    /// For example, if your data was [10, 20, 30, 40]:
    /// - It remains [10, 20, 30, 40] after "normalization"
    /// 
    /// This is used when you want to maintain the same code structure across your system,
    /// even when no normalization is needed for a particular case.
    /// </para>
    /// </remarks>
    public override (TOutput, NormalizationParameters<T>) NormalizeOutput(TOutput data)
    {
        // Since we're just passing through the data, we don't need type-specific handling
        // as long as the input and output types match, but for consistency with other normalizers,
        // we'll do basic type checking
        if (data is Vector<T> || data is Tensor<T>)
        {
            return (data, new NormalizationParameters<T> { Method = NormalizationMethod.None });
        }

        throw new InvalidOperationException(
            $"Unsupported data type {typeof(TOutput).Name}. " +
            $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
    }

    /// <summary>
    /// Returns the input data unchanged along with minimal normalization parameters for each column.
    /// </summary>
    /// <param name="data">The input data to "normalize" (which remains unchanged).</param>
    /// <returns>
    /// A tuple containing the original data and a list of basic normalization parameters for each column, all with Method set to None.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method returns the original data without any transformation, along with
    /// a list of NormalizationParameters objects (one for each column) that indicate no normalization was performed.
    /// This preserves the interface expected by calling code while performing no actual normalization.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns your table of data exactly as you provided it.
    /// 
    /// When you call this method:
    /// - The matrix (table of data) you input is the same one you get back
    /// - No calculations or transformations are performed on any columns
    /// - The method also returns information for each column indicating that no normalization was performed
    /// 
    /// This is used when you want to maintain the same code structure across your system,
    /// even when no normalization is needed for a particular dataset.
    /// </para>
    /// </remarks>
    public override (TInput, List<NormalizationParameters<T>>) NormalizeInput(TInput data)
    {
        if (data is Matrix<T> matrix)
        {
            var parameters = Enumerable.Repeat(new NormalizationParameters<T> { Method = NormalizationMethod.None }, matrix.Columns).ToList();
            return (data, parameters);
        }
        else if (data is Tensor<T> tensor && tensor.Shape.Length == 2)
        {
            int columns = tensor.Shape[1];
            var parameters = Enumerable.Repeat(new NormalizationParameters<T> { Method = NormalizationMethod.None }, columns).ToList();
            return (data, parameters);
        }

        throw new InvalidOperationException(
            $"Unsupported data type {typeof(TInput).Name}. " +
            $"Supported types are Matrix<{typeof(T).Name}> and 2D Tensor<{typeof(T).Name}>.");
    }

    /// <summary>
    /// Returns the input data unchanged, as no denormalization is needed when no normalization was performed.
    /// </summary>
    /// <param name="data">The data to "denormalize" (which remains unchanged).</param>
    /// <param name="parameters">The normalization parameters (which are ignored).</param>
    /// <returns>The original data unchanged.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the input data without any transformation, ignoring the provided parameters.
    /// Since no normalization was performed, no denormalization is needed.
    /// </para>
    /// <para><b>For Beginners:</b> This method also does nothing, just returning your data as is.
    /// 
    /// When you call this method:
    /// - The data you input is the same data you get back
    /// - The normalization parameters are ignored since they weren't used for anything
    /// 
    /// For example, if your "normalized" data was [10, 20, 30, 40]:
    /// - It remains [10, 20, 30, 40] after "denormalization"
    /// 
    /// This is the counterpart to the Normalize method, maintaining the expected interface
    /// while performing no actual transformation.
    /// </para>
    /// </remarks>
    public override TOutput Denormalize(TOutput data, NormalizationParameters<T> parameters)
    {
        // Simple pass-through, just checking that data is a supported type
        if (data is Vector<T> || data is Tensor<T>)
        {
            return data;
        }

        throw new InvalidOperationException(
            $"Unsupported data type {typeof(TOutput).Name}. " +
            $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
    }

    /// <summary>
    /// Returns the input coefficients unchanged, as no adjustment is needed when no normalization was performed.
    /// </summary>
    /// <param name="coefficients">The coefficients to "denormalize" (which remain unchanged).</param>
    /// <param name="xParams">The normalization parameters for input features (which are ignored).</param>
    /// <param name="yParams">The normalization parameters for the output variable (which are ignored).</param>
    /// <returns>The original coefficients unchanged.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the input coefficients without any transformation, ignoring the provided parameters.
    /// Since no normalization was performed on the data used to train the model that produced these coefficients,
    /// no adjustment of the coefficients is needed.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns model weights exactly as they are.
    /// 
    /// When you build a prediction model:
    /// - The model learns weights (coefficients) based on your data
    /// - Since no normalization was applied to the data, the weights are already in the correct scale
    /// - This method simply returns the coefficients unchanged
    /// 
    /// This maintains consistency with other normalizers, which might need to adjust
    /// coefficients to work with original data scales.
    /// </para>
    /// </remarks>
    public override TOutput Denormalize(TOutput coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        // Simple pass-through, just checking that coefficients is a supported type
        if (coefficients is Vector<T> || coefficients is Tensor<T>)
        {
            return coefficients;
        }

        throw new InvalidOperationException(
            $"Unsupported coefficients type {typeof(TOutput).Name}. " +
            $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
    }

    /// <summary>
    /// Calculates the appropriate y-intercept for a model trained on the original, unnormalized data.
    /// </summary>
    /// <param name="xMatrix">The original input feature matrix.</param>
    /// <param name="y">The original output vector.</param>
    /// <param name="coefficients">The regression coefficients.</param>
    /// <param name="xParams">The normalization parameters for input features (which are ignored).</param>
    /// <param name="yParams">The normalization parameters for the output variable (which are ignored).</param>
    /// <returns>The calculated y-intercept for the model.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the y-intercept for a regression model by using the MathHelper utility.
    /// Since no normalization was performed, the calculation uses the original data directly.
    /// This ensures that predictions using the model will be accurate across the range of the original data.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the baseline value for predictions with original data.
    /// 
    /// In a prediction model, the y-intercept is the starting point:
    /// - It's what you predict when all features are at their reference values
    /// - With no normalization, this baseline is calculated directly from the original data
    /// 
    /// This calculation ensures that your model's predictions are properly calibrated
    /// to match your original data scale. Unlike other normalizers that need special
    /// adjustments, this can directly calculate the intercept from the original data.
    /// </para>
    /// </remarks>
    public override T Denormalize(TInput xMatrix, TOutput y, TOutput coefficients,
        List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        // Extract vectors from inputs for calculation
        if (xMatrix is Matrix<T> matrix && coefficients is Vector<T> coeffVector && y is Vector<T> yVector)
        {
            return MathHelper.CalculateYIntercept(matrix, yVector, coeffVector);
        }
        else if (xMatrix is Tensor<T> xTensor && xTensor.Shape.Length == 2)
        {
            // Convert tensor to matrix for calculation
            var rows = xTensor.Shape[0];
            var cols = xTensor.Shape[1];
            var newMatrix = new Matrix<T>(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    newMatrix[i, j] = xTensor[i, j];
                }
            }

            // Extract vectors from coefficients and y
            Vector<T> coeffVector2;
            if (coefficients is Vector<T> vector)
            {
                coeffVector2 = vector;
            }
            else if (coefficients is Tensor<T> coeffTensor)
            {
                coeffVector2 = coeffTensor.ToVector();
            }
            else
            {
                throw new InvalidOperationException(
                    $"Unsupported coefficients type {coefficients?.GetType().Name}. " +
                    $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
            }

            Vector<T> yVector2;
            if (y is Vector<T> yVec)
            {
                yVector2 = yVec;
            }
            else if (y is Tensor<T> yTensor)
            {
                yVector2 = yTensor.ToVector();
            }
            else
            {
                throw new InvalidOperationException(
                    $"Unsupported y type {y?.GetType().Name}. " +
                    $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
            }

            return MathHelper.CalculateYIntercept(newMatrix, yVector2, coeffVector2);
        }

        throw new InvalidOperationException(
            $"Unsupported input types. xMatrix: {xMatrix?.GetType().Name}, " +
            $"y: {y?.GetType().Name}, coefficients: {coefficients?.GetType().Name}. " +
            $"Expected Matrix<T> or 2D Tensor<T> for xMatrix, and Vector<T> or Tensor<T> for y and coefficients.");
    }
}
