namespace AiDotNet.Normalizers;

/// <summary>
/// Normalizes data by adjusting its contrast globally based on the mean and standard deviation.
/// </summary>
/// <remarks>
/// <para>
/// The GlobalContrastNormalizer transforms data by centering it around the mean and scaling by the standard deviation,
/// then shifting it to typically fall within the [0, 1] range. This normalization enhances the contrast of the data
/// by ensuring a standardized distribution while maintaining relative relationships between values.
/// </para>
/// <para>
/// The transformation applies the formula: normalized = ((original - mean) / (2 * stdDev)) + 0.5
/// </para>
/// <para>
/// This normalization method is particularly useful for:
/// - Image processing where contrast enhancement is desired
/// - Feature normalization in machine learning to improve training stability
/// - Data with naturally occurring normal distributions
/// </para>
/// <para><b>For Beginners:</b> Global contrast normalization is like adjusting the brightness and contrast on a TV.
/// 
/// Think of it as improving the clarity of your data:
/// - First, it finds the average value (like the middle brightness level)
/// - Then, it measures how spread out your values are (the contrast)
/// - Finally, it adjusts all values so they're centered around the middle and properly spaced out
/// - The result is values that typically fall between 0 and 1, with 0.5 being the new average
/// 
/// For example, if you have temperature readings that are clustered together:
/// - Original temperatures: [68°F, 70°F, 71°F, 69°F, 72°F]
/// - After normalization, they might become: [0.3, 0.5, 0.6, 0.4, 0.7]
/// - Now the differences between temperatures are more visible and standardized
/// 
/// This is particularly useful when you want to highlight subtle differences in your data
/// or when combining different types of data that need to be on a comparable scale.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GlobalContrastNormalizer<T> : INormalizer<T>
{
    /// <summary>
    /// The numeric operations provider for the specified type T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds a reference to an object that provides operations for the numeric type T.
    /// These operations include addition, subtraction, multiplication, division, and statistical calculations
    /// like mean and standard deviation that are needed for global contrast normalization.
    /// </para>
    /// <para><b>For Beginners:</b> This is like having a specialized calculator for the type of numbers you're using.
    /// 
    /// Since this normalizer needs to perform various mathematical operations (addition, subtraction,
    /// multiplication, division, square roots, etc.) on different types of numbers, it uses this
    /// helper to ensure the calculations work correctly regardless of whether you're using
    /// decimals, doubles, or other numeric types.
    /// </para>
    /// </remarks>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="GlobalContrastNormalizer{T}"/> class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor creates a new GlobalContrastNormalizer and initializes the numeric operations
    /// provider for the specified type T. No additional parameters are required as the normalizer
    /// calculates the necessary statistics (mean and standard deviation) from the data itself.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up your contrast normalization system.
    /// 
    /// When you create a new GlobalContrastNormalizer:
    /// - It prepares the mathematical tools needed for the normalization
    /// - No additional settings are needed because the normalizer will automatically calculate
    ///   the appropriate statistics (average and spread) based on your actual data
    /// 
    /// It's like turning on your TV's auto-contrast feature - it will analyze the content and
    /// adjust the settings automatically to improve visibility.
    /// </para>
    /// </remarks>
    public GlobalContrastNormalizer()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Normalizes a vector using the global contrast approach.
    /// </summary>
    /// <param name="vector">The input vector to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized vector and the normalization parameters, which include the mean and standard deviation.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method normalizes a vector by:
    /// 1. Computing the mean (average) of all values
    /// 2. Computing the standard deviation to measure data spread
    /// 3. Standardizing each value using the formula: ((value - mean) / (2 * stdDev)) + 0.5
    /// 
    /// The resulting values will typically fall within the range [0, 1], with 0.5 representing the mean.
    /// Values that were more than 2 standard deviations from the mean in the original data may fall outside this range.
    /// 
    /// The normalization parameters include the mean and standard deviation, which are needed for later denormalization.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts your data to highlight important differences.
    /// 
    /// The process works like this:
    /// 1. First, it calculates the average value in your data
    /// 2. Then, it measures how spread out your values are (the standard deviation)
    /// 3. For each value, it:
    ///    - Subtracts the average (centering around zero)
    ///    - Divides by twice the spread (scaling the contrast)
    ///    - Adds 0.5 (shifting to center around 0.5 instead of 0)
    /// 
    /// After normalization:
    /// - Values near the original average will be close to 0.5
    /// - Values that were higher than average will be greater than 0.5
    /// - Values that were lower than average will be less than 0.5
    /// - Most values will fall between 0 and 1
    /// 
    /// For example, if your income data was [30K, 45K, 50K, 55K, 70K]:
    /// - The average is 50K
    /// - After normalization, it might become [0.2, 0.4, 0.5, 0.6, 0.8]
    /// - Now you can easily see that 30K is 0.3 below average, and 70K is 0.3 above average
    /// </para>
    /// </remarks>
    public (Vector<T>, NormalizationParameters<T>) NormalizeVector(Vector<T> vector)
    {
        T mean = vector.Average();
        T variance = vector.Select(x => _numOps.Multiply(_numOps.Subtract(x, mean), _numOps.Subtract(x, mean))).Average();
        T stdDev = _numOps.Sqrt(variance);
        var normalizedVector = vector.Transform(x => 
            _numOps.Add(
                _numOps.Divide(
                    _numOps.Subtract(x, mean),
                    _numOps.Multiply(_numOps.FromDouble(2), stdDev)
                ),
                _numOps.FromDouble(0.5)
            )
        );
        var parameters = new NormalizationParameters<T> { Mean = mean, StdDev = stdDev, Method = NormalizationMethod.GlobalContrast };
        return (normalizedVector, parameters);
    }

    /// <summary>
    /// Normalizes a matrix using the global contrast approach, applying normalization separately to each column.
    /// </summary>
    /// <param name="matrix">The input matrix to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized matrix and a list of normalization parameters for each column.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method normalizes each column of the matrix independently using the global contrast approach.
    /// It treats each column as a separate feature that needs its own mean and standard deviation calculation,
    /// since different features may have different distributions and scales.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies contrast normalization to a table of data, one column at a time.
    /// 
    /// When working with a table (matrix) of data:
    /// - Each column might represent a different type of information (age, income, blood pressure, etc.)
    /// - Each column needs its own normalization because the averages and spreads differ
    /// - This method processes each column separately using the vector normalization method
    /// 
    /// For example, with a table of health metrics:
    /// - Column 1 (ages) might have values around 40 years with small variations
    /// - Column 2 (blood pressure) might have values around 120 with larger variations
    /// - Each column gets its own appropriate adjustment for average and spread
    /// 
    /// The method returns:
    /// - A new table with all values contrast-normalized to highlight important differences
    /// - The statistics for each column, so you can convert back to original values later if needed
    /// </para>
    /// </remarks>
    public (Matrix<T>, List<NormalizationParameters<T>>) NormalizeMatrix(Matrix<T> matrix)
    {
        var normalizedMatrix = Matrix<T>.CreateZeros(matrix.Rows, matrix.Columns);
        var parametersList = new List<NormalizationParameters<T>>();
        for (int i = 0; i < matrix.Columns; i++)
        {
            var column = matrix.GetColumn(i);
            var (normalizedColumn, parameters) = NormalizeVector(column);
            normalizedMatrix.SetColumn(i, normalizedColumn);
            parametersList.Add(parameters);
        }
        return (normalizedMatrix, parametersList);
    }

    /// <summary>
    /// Denormalizes a vector using the provided normalization parameters.
    /// </summary>
    /// <param name="vector">The normalized vector to denormalize.</param>
    /// <param name="parameters">The normalization parameters containing mean and standard deviation.</param>
    /// <returns>A denormalized vector with values converted back to their original scale.</returns>
    /// <remarks>
    /// <para>
    /// This method reverses the global contrast normalization by applying the inverse of the original formula:
    /// original = ((normalized - 0.5) * 2 * stdDev) + mean
    /// 
    /// This transformation restores the values to their original scale and distribution.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts your normalized values back to their original scale.
    /// 
    /// The process reverses the normalization steps:
    /// 1. First, it subtracts 0.5 (to center around zero instead of 0.5)
    /// 2. Then, it multiplies by twice the standard deviation (restoring the original spread)
    /// 3. Finally, it adds the original mean (putting values back to their original center)
    /// 
    /// For example, if your normalized data was [0.2, 0.4, 0.5, 0.6, 0.8] with mean = 50K and standard deviation = 15K:
    /// - The denormalized values would be approximately [30K, 45K, 50K, 55K, 70K]
    /// 
    /// This allows you to go back to the original values after performing calculations or analysis
    /// on the normalized data.
    /// </para>
    /// </remarks>
    public Vector<T> DenormalizeVector(Vector<T> vector, NormalizationParameters<T> parameters)
    {
        return vector
            .Transform(x => _numOps.Subtract(x, _numOps.FromDouble(0.5)))
            .Transform(x => _numOps.Multiply(x, _numOps.Multiply(_numOps.FromDouble(2), parameters.StdDev)))
            .Transform(x => _numOps.Add(x, parameters.Mean));
    }

    /// <summary>
    /// Denormalizes coefficients from a regression model that was trained on global contrast normalized data.
    /// </summary>
    /// <param name="coefficients">The regression coefficients to denormalize.</param>
    /// <param name="xParams">The normalization parameters for the input features.</param>
    /// <param name="yParams">The normalization parameters for the output variable.</param>
    /// <returns>Denormalized coefficients that can be applied to original, unnormalized data.</returns>
    /// <remarks>
    /// <para>
    /// This method adjusts regression coefficients to work with original unnormalized data.
    /// For global contrast normalization, this involves multiplying each coefficient by the ratio of
    /// the output scale (2 * output stdDev) to the corresponding input scale (2 * input stdDev).
    /// This adjustment accounts for the different scales that were applied to each feature and the output during normalization.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts prediction model weights for use with original values.
    /// 
    /// When you build a model using normalized data:
    /// - The model learns weights (coefficients) based on the standardized values
    /// - To use this model with original, unprocessed data, you need to adjust these weights
    /// 
    /// This method performs that adjustment by:
    /// - Calculating how much each input feature was scaled during normalization
    /// - Calculating how much the output was scaled during normalization
    /// - Adjusting each coefficient by the ratio of these scaling factors
    /// 
    /// For example, if:
    /// - An input feature's standard deviation was 10 (meaning it was divided by 20 during normalization)
    /// - The output's standard deviation was 5 (meaning it was divided by 10 during normalization)
    /// - The model learned a coefficient of 2.0 for this feature on normalized data
    /// 
    /// The denormalized coefficient would be 2.0 × (10 ÷ 20) = 1.0
    /// 
    /// This ensures that predictions made using original data will be properly scaled.
    /// </para>
    /// </remarks>
    public Vector<T> DenormalizeCoefficients(Vector<T> coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        var scalingFactors = xParams.Select(p => 
            _numOps.Divide(
                _numOps.Multiply(_numOps.FromDouble(2), yParams.StdDev),
                _numOps.Multiply(_numOps.FromDouble(2), p.StdDev)
            )
        ).ToArray();
        return coefficients.PointwiseMultiply(Vector<T>.FromArray(scalingFactors));
    }

    /// <summary>
    /// Denormalizes the y-intercept from a regression model that was trained on global contrast normalized data.
    /// </summary>
    /// <param name="xMatrix">The original input feature matrix.</param>
    /// <param name="y">The original output vector.</param>
    /// <param name="coefficients">The regression coefficients.</param>
    /// <param name="xParams">The normalization parameters for the input features.</param>
    /// <param name="yParams">The normalization parameters for the output variable.</param>
    /// <returns>A denormalized y-intercept that can be used with original, unnormalized data.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the appropriate y-intercept for a model trained on normalized data
    /// but applied to unnormalized data. The calculation accounts for the shifts in both the input features
    /// and the output variable that occurred during normalization.
    /// </para>
    /// <para><b>For Beginners:</b> This method determines the correct starting point for predictions with original data.
    /// 
    /// In a prediction model, the y-intercept is the baseline value:
    /// - It's the predicted value when all features are zero
    /// - When using normalized data, this baseline needs to be adjusted for original data
    /// 
    /// The calculation is more complex than for other normalizers because:
    /// - The global contrast normalization involves both scaling (by standard deviation) and shifting (by mean and adding 0.5)
    /// - Each feature has its own mean and standard deviation
    /// - The output also has its own mean and standard deviation
    /// 
    /// The method performs this complex adjustment to ensure that:
    /// - If you input the original feature values into your model
    /// - Using the denormalized coefficients and intercept
    /// - You'll get predictions in the original scale of your output variable
    /// 
    /// This makes your model usable with raw, unprocessed data while maintaining the accuracy
    /// gained from training on normalized data.
    /// </para>
    /// </remarks>
    public T DenormalizeYIntercept(Matrix<T> xMatrix, Vector<T> y, Vector<T> coefficients, 
        List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        T denormalizedIntercept = _numOps.Subtract(
            yParams.Mean,
            _numOps.Multiply(
                _numOps.FromDouble(0.5),
                _numOps.Multiply(_numOps.FromDouble(2), yParams.StdDev)
            )
        );
        for (int i = 0; i < coefficients.Length; i++)
        {
            T term1 = _numOps.Multiply(
                xParams[i].Mean,
                _numOps.Divide(
                    _numOps.Multiply(_numOps.FromDouble(2), yParams.StdDev),
                    _numOps.Multiply(_numOps.FromDouble(2), xParams[i].StdDev)
                )
            );
            T term2 = _numOps.Multiply(_numOps.FromDouble(0.5), _numOps.Multiply(_numOps.FromDouble(2), yParams.StdDev));
            T difference = _numOps.Subtract(term1, term2);
            T product = _numOps.Multiply(coefficients[i], difference);
            denormalizedIntercept = _numOps.Subtract(denormalizedIntercept, product);
        }

        return denormalizedIntercept;
    }
}