namespace AiDotNet.Normalizers;

/// <summary>
/// Normalizes data by standardizing it to have zero mean and unit variance.
/// </summary>
/// <remarks>
/// <para>
/// The MeanVarianceNormalizer transforms data by subtracting the mean and dividing by the standard deviation.
/// This process, also known as z-score normalization or standardization, centers the data around zero and
/// scales it to have a standard deviation of one. After normalization, the data follows a distribution with
/// zero mean and unit variance.
/// </para>
/// <para>
/// The transformation formula is: normalized = (original - mean) / standard_deviation
/// </para>
/// <para>
/// This normalization method is particularly useful for:
/// - Features with normal or Gaussian distributions
/// - Machine learning algorithms that assume data is centered around zero
/// - Comparing features with different units or scales
/// - Improving convergence in gradient-based optimization algorithms
/// </para>
/// <para><b>For Beginners:</b> Mean-variance normalization is like creating a standard scale to compare different measurements.
/// 
/// Think of it as converting different units into a universal scale:
/// - First, it finds the average (mean) value in your data
/// - Then, it measures how spread out your values are (the standard deviation)
/// - Finally, it expresses each value as "how many standard deviations away from the average" it is
/// 
/// For example, if you have student test scores:
/// - Original scores: [70, 80, 90, 100]
/// - Mean = 85, Standard deviation = ~12.9
/// - Normalized scores: [-1.16, -0.39, 0.39, 1.16]
/// 
/// After normalization:
/// - The average value becomes 0
/// - Most values fall between -3 and +3
/// - A value of +1 means "one standard deviation above average"
/// - A value of -2 means "two standard deviations below average"
/// 
/// This makes it easy to compare data from different sources or with different units.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The type of input data structure.</typeparam>
/// <typeparam name="TOutput">The type of output data structure.</typeparam>
public class MeanVarianceNormalizer<T, TInput, TOutput> : NormalizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MeanVarianceNormalizer{T, TInput, TOutput}"/> class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor creates a new MeanVarianceNormalizer and initializes the numeric operations
    /// provider for the specified type T. No additional parameters are required as the normalizer
    /// calculates the necessary statistics (mean and standard deviation) from the data itself.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up your standardization system.
    /// 
    /// When you create a new MeanVarianceNormalizer:
    /// - It prepares the mathematical tools needed for standardization
    /// - No additional settings are needed because the normalizer will automatically calculate
    ///   the appropriate statistics (average and spread) based on your actual data
    /// 
    /// It's like setting up a measurement system that will automatically calibrate itself
    /// to whatever data you provide.
    /// </para>
    /// </remarks>
    public MeanVarianceNormalizer() : base()
    {
        // Base constructor already initializes NumOps
    }

    /// <summary>
    /// Normalizes output data using the mean-variance approach.
    /// </summary>
    /// <param name="data">The input data to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized data and the normalization parameters, which include the mean and standard deviation.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method normalizes data by:
    /// 1. Computing the mean (average) of all values
    /// 2. Computing the variance and standard deviation to measure data spread
    /// 3. Standardizing each value using the formula: (value - mean) / standard_deviation
    /// 
    /// The resulting values will have a mean of 0 and a standard deviation of 1.
    /// The normalization parameters include the mean and standard deviation, which are needed for later denormalization.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts your data to a standard scale centered at zero.
    /// 
    /// The process works like this:
    /// 1. First, it calculates the average value in your data
    /// 2. Then, it measures how spread out your values are (the standard deviation)
    /// 3. For each value, it:
    ///    - Subtracts the average (centering around zero)
    ///    - Divides by the standard deviation (scaling to a standard spread)
    /// 
    /// After normalization:
    /// - Values near the original average will be close to 0
    /// - Values that were higher than average will be positive
    /// - Values that were lower than average will be negative
    /// - Most values will typically fall between -3 and +3
    /// 
    /// For example, if your height data was [150cm, 165cm, 180cm, 195cm]:
    /// - The average is 172.5cm
    /// - After normalization, it might become [-1.5, -0.5, 0.5, 1.5]
    /// - Now you can easily see that 150cm is 1.5 standard deviations below average
    /// </para>
    /// </remarks>
    public override (TOutput, NormalizationParameters<T>) NormalizeOutput(TOutput data)
    {
        if (data is Vector<T> vector)
        {
            T mean = StatisticsHelper<T>.CalculateMean(vector);
            T variance = StatisticsHelper<T>.CalculateVariance(vector, mean);
            T stdDev = NumOps.Sqrt(variance);
            var normalizedVector = vector.Transform(x => NumOps.Divide(NumOps.Subtract(x, mean), stdDev));
            var parameters = new NormalizationParameters<T> { Mean = mean, StdDev = stdDev, Method = NormalizationMethod.MeanVariance };
            return ((TOutput)(object)normalizedVector, parameters);
        }
        else if (data is Tensor<T> tensor)
        {
            // Flatten tensor to apply mean-variance normalization
            var flattenedTensor = tensor.ToVector();

            T mean = StatisticsHelper<T>.CalculateMean(flattenedTensor);
            T variance = StatisticsHelper<T>.CalculateVariance(flattenedTensor, mean);
            T stdDev = NumOps.Sqrt(variance);

            var normalizedVector = flattenedTensor.Transform(x => NumOps.Divide(NumOps.Subtract(x, mean), stdDev));

            // Convert back to tensor with the same shape
            var normalizedTensor = Tensor<T>.FromVector(normalizedVector);
            if (tensor.Shape.Length > 1)
            {
                normalizedTensor = normalizedTensor.Reshape(tensor.Shape);
            }

            var parameters = new NormalizationParameters<T> { Mean = mean, StdDev = stdDev, Method = NormalizationMethod.MeanVariance };
            return ((TOutput)(object)normalizedTensor, parameters);
        }

        throw new InvalidOperationException(
            $"Unsupported data type {typeof(TOutput).Name}. " +
            $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
    }

    /// <summary>
    /// Normalizes input data using the mean-variance approach, applying normalization separately to each column.
    /// </summary>
    /// <param name="data">The input data to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized data and a list of normalization parameters for each column.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method normalizes each column of the matrix independently using the mean-variance approach.
    /// It treats each column as a separate feature that needs its own mean and standard deviation calculation,
    /// since different features may have different distributions and scales.
    /// </para>
    /// <para><b>For Beginners:</b> This method standardizes a table of data, one column at a time.
    /// 
    /// When working with a table (matrix) of data:
    /// - Each column might represent a different type of information (height, weight, age, etc.)
    /// - Each column needs its own normalization because the averages and spreads differ
    /// - This method processes each column separately using the vector normalization method
    /// 
    /// For example, with a table of health metrics:
    /// - Column 1 (heights) might be around 170cm with a spread of 15cm
    /// - Column 2 (weights) might be around 70kg with a spread of 12kg
    /// - Each column gets its own appropriate adjustment for average and spread
    /// 
    /// The method returns:
    /// - A new table with all values standardized to have mean 0 and standard deviation 1
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
    /// This method reverses the mean-variance normalization by applying the inverse of the original formula:
    /// original = (normalized * standard_deviation) + mean
    /// 
    /// This transformation restores the values to their original scale and distribution.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts your standardized values back to their original scale.
    /// 
    /// The process reverses the normalization steps:
    /// 1. First, it multiplies each value by the original standard deviation (restoring the original spread)
    /// 2. Then, it adds the original mean (putting values back to their original center)
    /// 
    /// For example, if your normalized data was [-1.5, -0.5, 0.5, 1.5] with mean = 172.5cm and standard deviation = 15cm:
    /// - The denormalized values would be [150cm, 165cm, 180cm, 195cm]
    /// 
    /// This allows you to go back to the original measurements after performing calculations
    /// or analysis on the normalized data.
    /// </para>
    /// </remarks>
    public override TOutput Denormalize(TOutput data, NormalizationParameters<T> parameters)
    {
        if (data is Vector<T> vector)
        {
            var denormalizedVector = vector.Multiply(parameters.StdDev).Add(parameters.Mean);
            return (TOutput)(object)denormalizedVector;
        }
        else if (data is Tensor<T> tensor)
        {
            // Flatten tensor for denormalization
            var flattenedTensor = tensor.ToVector();

            var denormalizedVector = flattenedTensor.Multiply(parameters.StdDev).Add(parameters.Mean);

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
    /// Denormalizes coefficients from a regression model that was trained on mean-variance normalized data.
    /// </summary>
    /// <param name="coefficients">The regression coefficients to denormalize.</param>
    /// <param name="xParams">The normalization parameters for the input features.</param>
    /// <param name="yParams">The normalization parameters for the output variable.</param>
    /// <returns>Denormalized coefficients that can be applied to original, unnormalized data.</returns>
    /// <remarks>
    /// <para>
    /// This method adjusts regression coefficients to work with original unnormalized data.
    /// For mean-variance normalization, this involves scaling each coefficient by the ratio of
    /// the output standard deviation to the corresponding input standard deviation.
    /// This adjustment accounts for the different scales that were applied to each feature
    /// and the output during normalization.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts prediction model weights for use with original values.
    /// 
    /// When you build a model using normalized data:
    /// - The model learns weights (coefficients) based on the standardized values
    /// - To use this model with original, unstandardized data, you need to adjust these weights
    /// 
    /// This method performs that adjustment by:
    /// - Calculating how much each input feature was scaled during normalization
    /// - Calculating how much the output was scaled during normalization
    /// - Adjusting each coefficient by the ratio of these scaling factors
    /// 
    /// For example, if:
    /// - An input feature's standard deviation was 15 (meaning it was divided by 15 during normalization)
    /// - The output's standard deviation was 5 (meaning it was divided by 5 during normalization)
    /// - The model learned a coefficient of 0.3 for this feature on normalized data
    /// 
    /// The denormalized coefficient would be 0.3 × (5 ÷ 15) = 0.1
    /// 
    /// This ensures that predictions made using original data will be properly scaled.
    /// </para>
    /// </remarks>
    public override TOutput Denormalize(TOutput coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        if (coefficients is Vector<T> vector)
        {
            var denormalizedCoefficients = vector.PointwiseMultiply(Vector<T>.FromArray(xParams.Select(p =>
                NumOps.Divide(yParams.StdDev, p.StdDev)).ToArray()));

            return (TOutput)(object)denormalizedCoefficients;
        }
        else if (coefficients is Tensor<T> tensor)
        {
            // Flatten tensor for denormalization
            var flattenedTensor = tensor.ToVector();

            var denormalizedVector = flattenedTensor.PointwiseMultiply(Vector<T>.FromArray(xParams.Select(p =>
                NumOps.Divide(yParams.StdDev, p.StdDev)).ToArray()));

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
    /// - It's the predicted value when all features are at their average values
    /// - When using standardized data, this baseline needs to be adjusted for original data
    /// 
    /// The calculation is complex because:
    /// - During standardization, we shifted each feature by subtracting its mean
    /// - We also shifted the output by subtracting its mean
    /// - The model's coefficients were scaled by the ratio of standard deviations
    /// 
    /// The method calculates the correct intercept to ensure that:
    /// - If you input the average value for each feature
    /// - The prediction will be the average output value
    /// 
    /// This makes the model properly calibrated for use with the original, unstandardized data.
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
        T denormalizedIntercept = yParams.Mean;
        for (int i = 0; i < coefficientsVector.Length; i++)
        {
            T term = NumOps.Multiply(coefficientsVector[i], xParams[i].Mean);
            term = NumOps.Multiply(term, NumOps.Divide(yParams.StdDev, xParams[i].StdDev));
            denormalizedIntercept = NumOps.Subtract(denormalizedIntercept, term);
        }
        return denormalizedIntercept;
    }
}
