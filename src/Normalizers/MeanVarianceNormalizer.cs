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
public class MeanVarianceNormalizer<T> : INormalizer<T>
{
    /// <summary>
    /// The numeric operations provider for the specified type T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds a reference to an object that provides operations for the numeric type T.
    /// These operations include addition, subtraction, multiplication, division, and statistical
    /// calculations needed for mean-variance normalization.
    /// </para>
    /// <para><b>For Beginners:</b> This is like having a specialized calculator that works with the type of numbers you're using.
    /// 
    /// Since this normalizer needs to perform various mathematical operations (subtraction, division,
    /// square root, etc.) on different types of numbers, it uses this helper to ensure
    /// the calculations work correctly regardless of whether you're using decimals, doubles, or other numeric types.
    /// </para>
    /// </remarks>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="MeanVarianceNormalizer{T}"/> class.
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
    public MeanVarianceNormalizer()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Normalizes a vector using the mean-variance approach.
    /// </summary>
    /// <param name="vector">The input vector to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized vector and the normalization parameters, which include the mean and standard deviation.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method normalizes a vector by:
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
    public (Vector<T>, NormalizationParameters<T>) NormalizeVector(Vector<T> vector)
    {
        T mean = StatisticsHelper<T>.CalculateMean(vector);
        T variance = StatisticsHelper<T>.CalculateVariance(vector, mean);
        T stdDev = _numOps.Sqrt(variance);
        var normalizedVector = vector.Transform(x => _numOps.Divide(_numOps.Subtract(x, mean), stdDev));
        var parameters = new NormalizationParameters<T> { Mean = mean, StdDev = stdDev, Method = NormalizationMethod.MeanVariance };
        return (normalizedVector, parameters);
    }

    /// <summary>
    /// Normalizes a matrix using the mean-variance approach, applying normalization separately to each column.
    /// </summary>
    /// <param name="matrix">The input matrix to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized matrix and a list of normalization parameters for each column.
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
    public Vector<T> DenormalizeVector(Vector<T> vector, NormalizationParameters<T> parameters)
    {
        return vector.Multiply(parameters.StdDev).Add(parameters.Mean);
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
    public Vector<T> DenormalizeCoefficients(Vector<T> coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        return coefficients.PointwiseMultiply(Vector<T>.FromArray(xParams.Select(p => _numOps.Divide(yParams.StdDev, p.StdDev)).ToArray()));
    }

    /// <summary>
    /// Denormalizes the y-intercept from a regression model that was trained on mean-variance normalized data.
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
    public T DenormalizeYIntercept(Matrix<T> xMatrix, Vector<T> y, Vector<T> coefficients, 
        List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        T denormalizedIntercept = yParams.Mean;
        for (int i = 0; i < coefficients.Length; i++)
        {
            T term = _numOps.Multiply(coefficients[i], xParams[i].Mean);
            term = _numOps.Multiply(term, _numOps.Divide(yParams.StdDev, xParams[i].StdDev));
            denormalizedIntercept = _numOps.Subtract(denormalizedIntercept, term);
        }
        return denormalizedIntercept;
    }
}