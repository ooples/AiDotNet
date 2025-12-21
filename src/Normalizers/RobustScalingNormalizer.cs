namespace AiDotNet.Normalizers;

/// <summary>
/// Normalizes data using robust scaling based on median and interquartile range (IQR).
/// </summary>
/// <remarks>
/// <para>
/// The RobustScalingNormalizer transforms data by subtracting the median and dividing by the
/// interquartile range (IQR). Unlike mean-variance normalization, this approach is robust to
/// outliers because it uses statistics (median, IQR) that are less sensitive to extreme values.
/// </para>
/// <para>
/// The transformation formula is: normalized = (original - median) / IQR
/// </para>
/// <para>
/// This normalization method is particularly useful for:
/// - Datasets with outliers or skewed distributions
/// - Features where extreme values should not overly influence the scaling
/// - Machine learning algorithms sensitive to the scale of input features
/// - Situations where a robust, outlier-resistant transformation is needed
/// </para>
/// <para><b>For Beginners:</b> Robust scaling is like using the median clothing size instead of the average.
/// 
/// Think of it as a way to standardize data that isn't fooled by extreme values:
/// - Instead of using the mean (average), which can be pulled toward outliers, it uses the median (middle value)
/// - Instead of using standard deviation to measure spread, it uses the interquartile range (IQR)
/// - The IQR is the range between the 25th and 75th percentiles, covering the middle 50% of your data
/// 
/// For example, with income data:
/// - Regular scaling might be thrown off by a few billionaires in the dataset
/// - Robust scaling focuses on the middle range where most people's incomes fall
/// - Original incomes: [$30K, $45K, $60K, $75K, $5M]
/// - After robust scaling: [-1.0, -0.33, 0.33, 1.0, 162.67]
/// 
/// Notice how the first four values are nicely spread between -1 and 1, while the outlier ($5M)
/// gets a very large value but doesn't compress everyone else together.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The type of input data structure.</typeparam>
/// <typeparam name="TOutput">The type of output data structure.</typeparam>
public class RobustScalingNormalizer<T, TInput, TOutput> : NormalizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="RobustScalingNormalizer{T, TInput, TOutput}"/> class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor creates a new RobustScalingNormalizer and initializes the numeric operations
    /// provider for the specified type T. No additional parameters are required as the normalizer
    /// calculates the necessary statistics (median and IQR) from the data itself.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up your outlier-resistant scaling system.
    /// 
    /// When you create a new RobustScalingNormalizer:
    /// - It prepares the mathematical tools needed for outlier-resistant normalization
    /// - No additional settings are needed because the normalizer will automatically calculate
    ///   the appropriate statistics (median and IQR) based on your actual data
    /// 
    /// It's like setting up a sizing system that will automatically adjust to the most
    /// representative central range of your data, ignoring extreme values.
    /// </para>
    /// </remarks>
    public RobustScalingNormalizer() : base()
    {
        // Base constructor already initializes NumOps
    }

    /// <summary>
    /// Normalizes output data using the robust scaling approach.
    /// </summary>
    /// <param name="data">The input data to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized data and the normalization parameters, which include the median and IQR.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method normalizes data by:
    /// 1. Computing the median of all values
    /// 2. Computing the 25th and 75th percentiles to determine the interquartile range (IQR)
    /// 3. Standardizing each value using the formula: (value - median) / IQR
    /// 
    /// If the IQR is zero (which can happen if many values are identical), it is set to 1 to avoid division by zero.
    /// 
    /// The normalization parameters include the median and IQR, which are needed for denormalization.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts your data to a scale centered around the middle value.
    /// 
    /// The process works like this:
    /// 1. First, it finds the median (middle value) in your data
    /// 2. Then, it calculates the interquartile range (the spread of the middle 50% of values)
    /// 3. For each value, it:
    ///    - Subtracts the median (centering around zero)
    ///    - Divides by the IQR (scaling to a standard spread)
    /// 
    /// After normalization:
    /// - Values near the median will be close to 0
    /// - Values at the 25th percentile will be around -1
    /// - Values at the 75th percentile will be around 1
    /// - Outliers can have values much larger than 1 or smaller than -1
    /// 
    /// This approach works well when you have extreme values that shouldn't overly influence your scaling.
    /// </para>
    /// </remarks>
    public override (TOutput, NormalizationParameters<T>) NormalizeOutput(TOutput data)
    {
        if (data is Vector<T> vector)
        {
            T median = StatisticsHelper<T>.CalculateMedian(vector);
            T q1 = StatisticsHelper<T>.CalculateQuantile(vector, NumOps.FromDouble(0.25));
            T q3 = StatisticsHelper<T>.CalculateQuantile(vector, NumOps.FromDouble(0.75));
            T iqr = NumOps.Subtract(q3, q1);
            if (NumOps.Equals(iqr, NumOps.Zero)) iqr = NumOps.One;

            var normalizedVector = vector.Subtract(median).Divide(iqr);
            var parameters = new NormalizationParameters<T> { Median = median, IQR = iqr, Method = NormalizationMethod.RobustScaling };
            return ((TOutput)(object)normalizedVector, parameters);
        }
        else if (data is Tensor<T> tensor)
        {
            // Flatten tensor to apply robust scaling normalization
            var flattenedTensor = tensor.ToVector();

            T median = StatisticsHelper<T>.CalculateMedian(flattenedTensor);
            T q1 = StatisticsHelper<T>.CalculateQuantile(flattenedTensor, NumOps.FromDouble(0.25));
            T q3 = StatisticsHelper<T>.CalculateQuantile(flattenedTensor, NumOps.FromDouble(0.75));
            T iqr = NumOps.Subtract(q3, q1);
            if (NumOps.Equals(iqr, NumOps.Zero)) iqr = NumOps.One;

            var normalizedVector = flattenedTensor.Subtract(median).Divide(iqr);

            // Convert back to tensor with the same shape
            var normalizedTensor = Tensor<T>.FromVector(normalizedVector);
            if (tensor.Shape.Length > 1)
            {
                normalizedTensor = normalizedTensor.Reshape(tensor.Shape);
            }

            var parameters = new NormalizationParameters<T> { Median = median, IQR = iqr, Method = NormalizationMethod.RobustScaling };
            return ((TOutput)(object)normalizedTensor, parameters);
        }

        throw new InvalidOperationException(
            $"Unsupported data type {typeof(TOutput).Name}. " +
            $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
    }

    /// <summary>
    /// Normalizes input data using the robust scaling approach, applying normalization separately to each column.
    /// </summary>
    /// <param name="data">The input data to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized data and a list of normalization parameters for each column.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method normalizes each column of the matrix independently using the robust scaling approach.
    /// It treats each column as a separate feature that needs its own median and IQR calculation,
    /// since different features may have different distributions and scales.
    /// </para>
    /// <para><b>For Beginners:</b> This method robustly scales a table of data, one column at a time.
    /// 
    /// When working with a table (matrix) of data:
    /// - Each column might represent a different type of information (age, income, blood pressure, etc.)
    /// - Each column needs its own normalization because the distributions differ
    /// - This method processes each column separately using the vector normalization method
    /// 
    /// For example, with a table of financial metrics:
    /// - Column 1 (salaries) might have some very high executive compensations
    /// - Column 2 (employee counts) might have some small departments and some huge ones
    /// - Each column gets its own robust scaling based on its median and middle spread
    /// 
    /// The method returns:
    /// - A new table with all values robustly normalized
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
    /// <param name="parameters">The normalization parameters containing the median and IQR.</param>
    /// <returns>A denormalized data with values converted back to their original scale.</returns>
    /// <remarks>
    /// <para>
    /// This method reverses the robust scaling normalization by applying the inverse of the original formula:
    /// original = (normalized * IQR) + median
    /// 
    /// This transformation restores the values to their original scale and distribution.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts your robustly scaled values back to their original scale.
    /// 
    /// The process reverses the normalization steps:
    /// 1. First, it multiplies each value by the original IQR (restoring the original spread)
    /// 2. Then, it adds the original median (putting values back to their original center)
    /// 
    /// For example, if your normalized data was [-1.0, -0.33, 0.33, 1.0] with median = $60K and IQR = $30K:
    /// - The denormalized values would be:
    ///   * -1.0 × $30K + $60K = $30K
    ///   * -0.33 × $30K + $60K = $50K
    ///   * 0.33 × $30K + $60K = $70K
    ///   * 1.0 × $30K + $60K = $90K
    /// 
    /// This allows you to go back to the original measurements after performing calculations
    /// or analysis on the normalized data.
    /// </para>
    /// </remarks>
    public override TOutput Denormalize(TOutput data, NormalizationParameters<T> parameters)
    {
        if (data is Vector<T> vector)
        {
            var denormalizedVector = vector.Multiply(parameters.IQR).Add(parameters.Median);
            return (TOutput)(object)denormalizedVector;
        }
        else if (data is Tensor<T> tensor)
        {
            // Flatten tensor for denormalization
            var flattenedTensor = tensor.ToVector();

            var denormalizedVector = flattenedTensor.Multiply(parameters.IQR).Add(parameters.Median);

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
    /// Denormalizes coefficients from a regression model that was trained on robustly scaled data.
    /// </summary>
    /// <param name="coefficients">The regression coefficients to denormalize.</param>
    /// <param name="xParams">The normalization parameters for the input features.</param>
    /// <param name="yParams">The normalization parameters for the output variable.</param>
    /// <returns>Denormalized coefficients that can be applied to original, unnormalized data.</returns>
    /// <remarks>
    /// <para>
    /// This method adjusts regression coefficients to work with original unnormalized data.
    /// For robust scaling, this involves multiplying each coefficient by the ratio of
    /// the output IQR to the corresponding input IQR.
    /// This adjustment accounts for the different scales that were applied to each feature
    /// and the output during normalization.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts prediction model weights for use with original values.
    /// 
    /// When you build a model using normalized data:
    /// - The model learns weights (coefficients) based on the robustly scaled values
    /// - To use this model with original, unscaled data, you need to adjust these weights
    /// 
    /// This method performs that adjustment by:
    /// - Multiplying each coefficient by the ratio of output IQR to input IQR
    /// 
    /// For example, if:
    /// - An input feature's IQR was $30K (meaning it was divided by $30K during normalization)
    /// - The output's IQR was $10K (meaning it was divided by $10K during normalization)
    /// - The model learned a coefficient of 0.9 for this feature on normalized data
    /// 
    /// The denormalized coefficient would be 0.9 × ($10K ÷ $30K) = 0.3
    /// 
    /// This ensures that predictions made using original data will be properly scaled.
    /// </para>
    /// </remarks>
    public override TOutput Denormalize(TOutput coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        if (coefficients is Vector<T> vector)
        {
            var denormalizedCoefficients = vector.PointwiseMultiply(Vector<T>.FromArray(xParams.Select(p =>
                NumOps.Divide(yParams.IQR, p.IQR)).ToArray()));

            return (TOutput)(object)denormalizedCoefficients;
        }
        else if (coefficients is Tensor<T> tensor)
        {
            // Flatten tensor for denormalization
            var flattenedTensor = tensor.ToVector();

            var denormalizedVector = flattenedTensor.PointwiseMultiply(Vector<T>.FromArray(xParams.Select(p =>
                NumOps.Divide(yParams.IQR, p.IQR)).ToArray()));

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
    /// - It's what you predict when all input features are at their median values
    /// - When using robustly normalized data, this baseline needs to be adjusted for original data
    /// 
    /// The calculation is complex because:
    /// - During normalization, we shifted each feature by subtracting its median
    /// - We also shifted the output by subtracting its median
    /// - The model's coefficients have been adjusted for these changes
    /// 
    /// The method calculates the correct intercept to ensure that:
    /// - If you input the median value for each feature
    /// - The prediction will be the median output value
    /// 
    /// This makes the model properly calibrated for use with the original, unnormalized data.
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

        // Calculate denormalized intercept
        T denormalizedIntercept = yParams.Median;
        for (int i = 0; i < coefficientsVector.Length; i++)
        {
            T term = NumOps.Multiply(coefficientsVector[i], xParams[i].Median);
            term = NumOps.Multiply(term, NumOps.Divide(yParams.IQR, xParams[i].IQR));
            denormalizedIntercept = NumOps.Subtract(denormalizedIntercept, term);
        }

        return denormalizedIntercept;
    }
}
