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
public class NoNormalizer<T> : INormalizer<T>
{
    /// <summary>
    /// Returns the input vector unchanged along with minimal normalization parameters.
    /// </summary>
    /// <param name="vector">The input vector to "normalize" (which remains unchanged).</param>
    /// <returns>
    /// A tuple containing the original vector and basic normalization parameters with Method set to None.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method returns the original vector without any transformation, along with
    /// a NormalizationParameters object that indicates no normalization was performed.
    /// This preserves the interface expected by calling code while performing no actual normalization.
    /// </para>
    /// <para><b>For Beginners:</b> This method just returns your data exactly as you provided it.
    /// 
    /// When you call this method:
    /// - The vector you input is the same vector you get back
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
    public (Vector<T>, NormalizationParameters<T>) NormalizeVector(Vector<T> vector)
    {
        return (vector, new NormalizationParameters<T> { Method = NormalizationMethod.None });
    }

    /// <summary>
    /// Returns the input matrix unchanged along with minimal normalization parameters for each column.
    /// </summary>
    /// <param name="matrix">The input matrix to "normalize" (which remains unchanged).</param>
    /// <returns>
    /// A tuple containing the original matrix and a list of basic normalization parameters for each column, all with Method set to None.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method returns the original matrix without any transformation, along with
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
    public (Matrix<T>, List<NormalizationParameters<T>>) NormalizeMatrix(Matrix<T> matrix)
    {
        var parameters = Enumerable.Repeat(new NormalizationParameters<T> { Method = NormalizationMethod.None }, matrix.Columns).ToList();
        return (matrix, parameters);
    }

    /// <summary>
    /// Returns the input vector unchanged, as no denormalization is needed when no normalization was performed.
    /// </summary>
    /// <param name="vector">The vector to "denormalize" (which remains unchanged).</param>
    /// <param name="parameters">The normalization parameters (which are ignored).</param>
    /// <returns>The original vector unchanged.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the input vector without any transformation, ignoring the provided parameters.
    /// Since no normalization was performed, no denormalization is needed.
    /// </para>
    /// <para><b>For Beginners:</b> This method also does nothing, just returning your data as is.
    /// 
    /// When you call this method:
    /// - The vector you input is the same vector you get back
    /// - The normalization parameters are ignored since they weren't used for anything
    /// 
    /// For example, if your "normalized" data was [10, 20, 30, 40]:
    /// - It remains [10, 20, 30, 40] after "denormalization"
    /// 
    /// This is the counterpart to the NormalizeVector method, maintaining the expected interface
    /// while performing no actual transformation.
    /// </para>
    /// </remarks>
    public Vector<T> DenormalizeVector(Vector<T> vector, NormalizationParameters<T> parameters)
    {
        return vector;
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
    public Vector<T> DenormalizeCoefficients(Vector<T> coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        return coefficients;
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
    public T DenormalizeYIntercept(Matrix<T> xMatrix, Vector<T> y, Vector<T> coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        return MathHelper.CalculateYIntercept(xMatrix, y, coefficients);
    }
}