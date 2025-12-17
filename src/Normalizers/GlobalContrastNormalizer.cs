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
/// <typeparam name="TInput">The type of input data structure.</typeparam>
/// <typeparam name="TOutput">The type of output data structure.</typeparam>
public class GlobalContrastNormalizer<T, TInput, TOutput> : NormalizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="GlobalContrastNormalizer{T, TInput, TOutput}"/> class.
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
    public GlobalContrastNormalizer() : base()
    {
        // Base constructor already initializes NumOps
    }

    /// <summary>
    /// Normalizes output data using the global contrast approach.
    /// </summary>
    /// <param name="data">The input data to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized data and the normalization parameters, which include the mean and standard deviation.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method normalizes data by:
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
    public override (TOutput, NormalizationParameters<T>) NormalizeOutput(TOutput data)
    {
        if (data is Vector<T> vector)
        {
            T mean = vector.Average();
            T variance = vector.Select(x => NumOps.Multiply(NumOps.Subtract(x, mean), NumOps.Subtract(x, mean))).Average();
            T stdDev = NumOps.Sqrt(variance);
            var normalizedVector = vector.Transform(x =>
                NumOps.Add(
                    NumOps.Divide(
                        NumOps.Subtract(x, mean),
                        NumOps.Multiply(NumOps.FromDouble(2), stdDev)
                    ),
                    NumOps.FromDouble(0.5)
                )
            );
            var parameters = new NormalizationParameters<T> { Mean = mean, StdDev = stdDev, Method = NormalizationMethod.GlobalContrast };
            return ((TOutput)(object)normalizedVector, parameters);
        }
        else if (data is Tensor<T> tensor)
        {
            // Flatten tensor to apply global contrast normalization
            var flattenedTensor = tensor.ToVector();

            T mean = flattenedTensor.Average();
            T variance = flattenedTensor.Select(x => NumOps.Multiply(NumOps.Subtract(x, mean), NumOps.Subtract(x, mean))).Average();
            T stdDev = NumOps.Sqrt(variance);

            var normalizedVector = flattenedTensor.Transform(x =>
                NumOps.Add(
                    NumOps.Divide(
                        NumOps.Subtract(x, mean),
                        NumOps.Multiply(NumOps.FromDouble(2), stdDev)
                    ),
                    NumOps.FromDouble(0.5)
                )
            );

            // Convert back to tensor with the same shape
            var normalizedTensor = Tensor<T>.FromVector(normalizedVector);
            if (tensor.Shape.Length > 1)
            {
                normalizedTensor = normalizedTensor.Reshape(tensor.Shape);
            }

            var parameters = new NormalizationParameters<T> { Mean = mean, StdDev = stdDev, Method = NormalizationMethod.GlobalContrast };
            return ((TOutput)(object)normalizedTensor, parameters);
        }

        throw new InvalidOperationException(
            $"Unsupported data type {typeof(TOutput).Name}. " +
            $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
    }

    /// <summary>
    /// Normalizes input data using the global contrast approach, applying normalization separately to each column.
    /// </summary>
    /// <param name="data">The input data to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized data and a list of normalization parameters for each column.
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
    public override (TInput, List<NormalizationParameters<T>>) NormalizeInput(TInput data)
    {
        if (data is Matrix<T> matrix)
        {
            var normalizedMatrix = Matrix<T>.CreateZeros(matrix.Rows, matrix.Columns);
            var parametersList = new List<NormalizationParameters<T>>();

            for (int i = 0; i < matrix.Columns; i++)
            {
                var column = matrix.GetColumn(i);
                // Convert column to TOutput for normalize method
                var (normalizedColumn, parameters) = NormalizeOutput((TOutput)(object)column);
                // Convert back to Vector<T>
                if (normalizedColumn is Vector<T> normalizedVector)
                {
                    normalizedMatrix.SetColumn(i, normalizedVector);
                    parametersList.Add(parameters);
                }
                else
                {
                    throw new InvalidOperationException(
                        $"Expected Vector<{typeof(T).Name}> but got {normalizedColumn?.GetType().Name}.");
                }
            }

            return ((TInput)(object)normalizedMatrix, parametersList);
        }
        else if (data is Tensor<T> tensor && tensor.Shape.Length == 2)
        {
            // Convert 2D tensor to matrix for column-wise normalization
            var rows = tensor.Shape[0];
            var cols = tensor.Shape[1];
            var newMatrix = new Matrix<T>(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    newMatrix[i, j] = tensor[i, j];
                }
            }

            // Normalize each column separately
            var normalizedColumns = new List<Vector<T>>();
            var parametersList = new List<NormalizationParameters<T>>();

            for (int i = 0; i < cols; i++)
            {
                var column = newMatrix.GetColumn(i);
                // Convert column to TOutput for normalize method
                var (normalizedColumn, parameters) = NormalizeOutput((TOutput)(object)column);
                // Convert back to Vector<T>
                if (normalizedColumn is Vector<T> normalizedVector)
                {
                    normalizedColumns.Add(normalizedVector);
                    parametersList.Add(parameters);
                }
                else
                {
                    throw new InvalidOperationException(
                        $"Expected Vector<{typeof(T).Name}> but got {normalizedColumn?.GetType().Name}.");
                }
            }

            // Convert back to tensor
            var normalizedMatrix = Matrix<T>.FromColumnVectors(normalizedColumns);
            var normalizedTensor = new Tensor<T>(new[] { normalizedMatrix.Rows, normalizedMatrix.Columns }, normalizedMatrix);

            return ((TInput)(object)normalizedTensor, parametersList);
        }

        throw new InvalidOperationException(
            $"Unsupported data type {typeof(TInput).Name}. " +
            $"Supported types are Matrix<{typeof(T).Name}> and 2D Tensor<{typeof(T).Name}>.");
    }

    /// <summary>
    /// Reverses the normalization of data using the provided normalization parameters.
    /// </summary>
    /// <param name="data">The normalized data to denormalize.</param>
    /// <param name="parameters">The normalization parameters containing mean and standard deviation.</param>
    /// <returns>A denormalized data with values converted back to their original scale.</returns>
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
    public override TOutput Denormalize(TOutput data, NormalizationParameters<T> parameters)
    {
        if (data is Vector<T> vector)
        {
            var denormalizedVector = vector
                .Transform(x => NumOps.Subtract(x, NumOps.FromDouble(0.5)))
                .Transform(x => NumOps.Multiply(x, NumOps.Multiply(NumOps.FromDouble(2), parameters.StdDev)))
                .Transform(x => NumOps.Add(x, parameters.Mean));

            return (TOutput)(object)denormalizedVector;
        }
        else if (data is Tensor<T> tensor)
        {
            // Flatten tensor for denormalization
            var flattenedTensor = tensor.ToVector();

            var denormalizedVector = flattenedTensor
                .Transform(x => NumOps.Subtract(x, NumOps.FromDouble(0.5)))
                .Transform(x => NumOps.Multiply(x, NumOps.Multiply(NumOps.FromDouble(2), parameters.StdDev)))
                .Transform(x => NumOps.Add(x, parameters.Mean));

            // Convert back to tensor with the same shape
            var denormalizedTensor = Tensor<T>.FromVector(denormalizedVector);
            if (tensor.Shape.Length > 1)
            {
                denormalizedTensor = denormalizedTensor.Reshape(tensor.Shape);
            }

            return (TOutput)(object)denormalizedTensor;
        }

        throw new InvalidOperationException(
            $"Unsupported data type {typeof(TOutput).Name}. " +
            $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
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
    public override TOutput Denormalize(TOutput coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        if (coefficients is Vector<T> vector)
        {
            var scalingFactors = xParams.Select(p =>
                NumOps.Divide(
                    NumOps.Multiply(NumOps.FromDouble(2), yParams.StdDev),
                    NumOps.Multiply(NumOps.FromDouble(2), p.StdDev)
                )
            ).ToArray();

            var denormalizedCoefficients = vector.PointwiseMultiply(Vector<T>.FromArray(scalingFactors));
            return (TOutput)(object)denormalizedCoefficients;
        }
        else if (coefficients is Tensor<T> tensor)
        {
            // Flatten tensor for denormalization
            var flattenedTensor = tensor.ToVector();

            var scalingFactors = xParams.Select(p =>
                NumOps.Divide(
                    NumOps.Multiply(NumOps.FromDouble(2), yParams.StdDev),
                    NumOps.Multiply(NumOps.FromDouble(2), p.StdDev)
                )
            ).ToArray();

            var denormalizedVector = flattenedTensor.PointwiseMultiply(Vector<T>.FromArray(scalingFactors));

            // Convert back to tensor with the same shape
            var denormalizedTensor = Tensor<T>.FromVector(denormalizedVector);
            if (tensor.Shape.Length > 1)
            {
                denormalizedTensor = denormalizedTensor.Reshape(tensor.Shape);
            }

            return (TOutput)(object)denormalizedTensor;
        }

        throw new InvalidOperationException(
            $"Unsupported coefficients type {typeof(TOutput).Name}. " +
            $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
    }

    /// <summary>
    /// Calculates the denormalized Y-intercept (constant term) for a linear model.
    /// </summary>
    /// <param name="xMatrix">The original input feature matrix.</param>
    /// <param name="y">The original target vector.</param>
    /// <param name="coefficients">The model coefficients.</param>
    /// <param name="xParams">The normalization parameters for the input features.</param>
    /// <param name="yParams">The normalization parameters for the output variable.</param>
    /// <returns>The denormalized Y-intercept for use with non-normalized data.</returns>
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
    public override T Denormalize(TInput xMatrix, TOutput y, TOutput coefficients,
        List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        // Extract vector from coefficients
        Vector<T> coefficientsVector;
        if (coefficients is Vector<T> vector)
        {
            coefficientsVector = vector;
        }
        else if (coefficients is Tensor<T> tensor)
        {
            coefficientsVector = tensor.ToVector();
        }
        else
        {
            throw new InvalidOperationException(
                $"Unsupported coefficients type {typeof(TOutput).Name}. " +
                $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
        }

        // Calculate y-intercept
        T denormalizedIntercept = NumOps.Subtract(
            yParams.Mean,
            NumOps.Multiply(
                NumOps.FromDouble(0.5),
                NumOps.Multiply(NumOps.FromDouble(2), yParams.StdDev)
            )
        );

        for (int i = 0; i < coefficientsVector.Length; i++)
        {
            T term1 = NumOps.Multiply(
                xParams[i].Mean,
                NumOps.Divide(
                    NumOps.Multiply(NumOps.FromDouble(2), yParams.StdDev),
                    NumOps.Multiply(NumOps.FromDouble(2), xParams[i].StdDev)
                )
            );
            T term2 = NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Multiply(NumOps.FromDouble(2), yParams.StdDev));
            T difference = NumOps.Subtract(term1, term2);
            T product = NumOps.Multiply(coefficientsVector[i], difference);
            denormalizedIntercept = NumOps.Subtract(denormalizedIntercept, product);
        }

        return denormalizedIntercept;
    }
}
