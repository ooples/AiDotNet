namespace AiDotNet.Normalizers;

/// <summary>
/// Normalizes the data by subtracting the mean from each value and dividing by the standard deviation.
/// </summary>
/// <remarks>
/// <para>
/// Z-Score normalization (standardization) transforms data to have a mean of 0 and a standard deviation of 1.
/// This process is important for many machine learning algorithms as it puts different features on comparable scales,
/// which can improve convergence speed and model performance.
/// </para>
/// <para><b>For Beginners:</b> This class transforms your data so all values are on a standard scale.
/// 
/// Think of it like converting test scores to a standard scale:
/// - If the average score on a test is 75 with a standard deviation of 10
/// - And you scored 85
/// - Your Z-score would be 1.0, meaning you scored 1 standard deviation above average
/// 
/// Benefits of Z-Score normalization:
/// - Makes different features comparable (like comparing height and weight)
/// - Helps machine learning algorithms work better and faster
/// - Makes the model less sensitive to the scale of input features
/// 
/// For example, in a dataset with house prices and number of rooms, prices might be in hundreds
/// of thousands while rooms might be 1-10. Z-Score normalization puts both on the same scale.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class ZScoreNormalizer<T> : INormalizer<T>
{
    /// <summary>
    /// Provides operations for mathematical calculations on the generic type T.
    /// </summary>
    /// <remarks>
    /// This field holds the numeric operations (addition, subtraction, multiplication, etc.)
    /// appropriate for the generic type T.
    /// </remarks>
    private readonly INumericOperations<T> _numOps;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="ZScoreNormalizer{T}"/> class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The constructor automatically acquires the appropriate numeric operations for the specified type T
    /// (such as float, double, or decimal) using the MathHelper utility.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new ZScoreNormalizer.
    /// 
    /// When you create a new ZScoreNormalizer:
    /// - It automatically sets up the correct math operations for your chosen number type (T)
    /// - You don't need to provide any additional parameters
    /// - It's ready to use immediately for normalizing data
    /// 
    /// For example: var normalizer = new ZScoreNormalizer&lt;double&gt;();
    /// </para>
    /// </remarks>
    public ZScoreNormalizer()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }
    
    /// <summary>
    /// Normalizes a vector using Z-Score normalization.
    /// </summary>
    /// <param name="vector">The vector to normalize.</param>
    /// <returns>
    /// A tuple containing:
    /// - The normalized vector where each value has been transformed to its Z-score
    /// - The normalization parameters (mean and standard deviation) used for the transformation
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method calculates the mean and standard deviation of the input vector, then transforms each value
    /// using the formula: (value - mean) / standardDeviation. The normalization parameters are returned
    /// alongside the normalized vector to enable denormalization later if needed.
    /// </para>
    /// <para><b>For Beginners:</b> This method transforms a list of numbers to their Z-scores.
    /// 
    /// What happens in this method:
    /// 1. It calculates the average (mean) of all values
    /// 2. It calculates how spread out the values are (standard deviation)
    /// 3. It transforms each value using: (value - mean) / standardDeviation
    /// 4. It returns both the transformed values and the parameters used (mean and standard deviation)
    /// 
    /// For example, if your data is [2, 4, 6, 8, 10]:
    /// - Mean = 6
    /// - Standard deviation = 3.16
    /// - Z-scores would be approximately [-1.26, -0.63, 0, 0.63, 1.26]
    /// 
    /// Keeping the mean and standard deviation lets you convert back to the original values later.
    /// </para>
    /// </remarks>
    public (Vector<T>, NormalizationParameters<T>) NormalizeVector(Vector<T> vector)
    {
        T mean = StatisticsHelper<T>.CalculateMean(vector);
        T variance = StatisticsHelper<T>.CalculateVariance(vector, mean);
        T stdDev = _numOps.Sqrt(variance);
        Vector<T> normalizedVector = vector.Transform(x => _numOps.Divide(_numOps.Subtract(x, mean), stdDev));
        return (normalizedVector, new NormalizationParameters<T> { Method = NormalizationMethod.ZScore, Mean = mean, StdDev = stdDev });
    }
    
    /// <summary>
    /// Normalizes each column in a matrix using Z-Score normalization.
    /// </summary>
    /// <param name="matrix">The matrix to normalize.</param>
    /// <returns>
    /// A tuple containing:
    /// - The normalized matrix where each column has been independently normalized
    /// - A list of normalization parameters (mean and standard deviation) for each column
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method normalizes each column of the input matrix independently, treating each column as a separate feature.
    /// For each column, the mean and standard deviation are calculated, and values are transformed using the
    /// Z-Score formula. The normalization parameters for each column are returned to enable denormalization.
    /// </para>
    /// <para><b>For Beginners:</b> This method normalizes a table of data, column by column.
    /// 
    /// In machine learning, a matrix often represents:
    /// - Multiple data points (rows)
    /// - Multiple features for each data point (columns)
    /// 
    /// This method:
    /// 1. Takes each column separately (each feature)
    /// 2. Normalizes it using Z-Score normalization
    /// 3. Returns the normalized matrix and parameters for each column
    /// 
    /// For example, in a dataset of houses, columns might be price, size, and age.
    /// Each column gets normalized independently because each feature has its own scale and distribution.
    /// </para>
    /// </remarks>
    public (Matrix<T>, List<NormalizationParameters<T>>) NormalizeMatrix(Matrix<T> matrix)
    {
        var normalizedColumns = new List<Vector<T>>();
        var parameters = new List<NormalizationParameters<T>>();
        for (int i = 0; i < matrix.Columns; i++)
        {
            var (normalizedColumn, columnParams) = NormalizeVector(matrix.GetColumn(i));
            normalizedColumns.Add(normalizedColumn);
            parameters.Add(columnParams);
        }
        return (Matrix<T>.FromColumnVectors(normalizedColumns), parameters);
    }
    
    /// <summary>
    /// Reverses the normalization process, converting Z-scores back to the original scale.
    /// </summary>
    /// <param name="vector">The normalized vector (containing Z-scores).</param>
    /// <param name="parameters">The normalization parameters that were used during normalization.</param>
    /// <returns>A vector with values converted back to their original scale.</returns>
    /// <remarks>
    /// <para>
    /// This method transforms normalized values back to their original scale using the formula:
    /// (normalizedValue * standardDeviation) + mean. The normalization parameters (mean and standard deviation)
    /// must be the same ones used during the normalization process.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts Z-scores back to their original values.
    /// 
    /// When you have normalized data and want to return to the original scale:
    /// 1. You need the Z-scores (normalized values)
    /// 2. You need the mean and standard deviation used for normalization
    /// 3. The formula is: originalValue = (zScore * standardDeviation) + mean
    /// 
    /// For example, if your Z-scores were [-1.26, -0.63, 0, 0.63, 1.26]:
    /// - With mean = 6 and standard deviation = 3.16
    /// - Original values would be approximately [2, 4, 6, 8, 10]
    /// 
    /// This is useful when you need to present results in the original units or scale.
    /// </para>
    /// </remarks>
    public Vector<T> DenormalizeVector(Vector<T> vector, NormalizationParameters<T> parameters)
    {
        return vector.Transform(x => _numOps.Add(_numOps.Multiply(x, parameters.StdDev), parameters.Mean));
    }
    
    /// <summary>
    /// Adjusts model coefficients to account for the normalization of input and output variables.
    /// </summary>
    /// <param name="coefficients">The coefficients obtained from model training on normalized data.</param>
    /// <param name="xParams">The normalization parameters used for each input feature.</param>
    /// <param name="yParams">The normalization parameters used for the output variable.</param>
    /// <returns>Coefficients adjusted to work with non-normalized data.</returns>
    /// <remarks>
    /// <para>
    /// When a model is trained on normalized data, its coefficients need to be adjusted to work correctly with
    /// non-normalized data. This method performs that adjustment by scaling each coefficient by the ratio of 
    /// the input feature's standard deviation to the output variable's standard deviation.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts model weights when moving from normalized to original data.
    /// 
    /// When you train a model (like linear regression) on normalized data:
    /// - The model learns coefficients (weights) that work with normalized values
    /// - To use the model with original (non-normalized) data, these coefficients need adjustment
    /// - This method performs that adjustment
    /// 
    /// For example, in a house price prediction model:
    /// - If you trained the model on normalized data (house size, number of rooms, etc.)
    /// - But want to predict prices using raw input values
    /// - This method transforms the coefficients so they work correctly
    /// 
    /// This saves you from having to normalize every new input before making predictions.
    /// </para>
    /// </remarks>
    public Vector<T> DenormalizeCoefficients(Vector<T> coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        return coefficients.PointwiseMultiply(Vector<T>.FromEnumerable(xParams.Select(p => _numOps.Divide(p.StdDev, yParams.StdDev))));
    }
    
    /// <summary>
    /// Calculates the appropriate y-intercept for a model trained on normalized data.
    /// </summary>
    /// <param name="xMatrix">The original input matrix before normalization.</param>
    /// <param name="y">The original output vector before normalization.</param>
    /// <param name="coefficients">The coefficients obtained from model training on normalized data.</param>
    /// <param name="xParams">The normalization parameters used for each input feature.</param>
    /// <param name="yParams">The normalization parameters used for the output variable.</param>
    /// <returns>The y-intercept adjusted to work with non-normalized data.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the appropriate y-intercept (constant term) for a model trained on normalized data
    /// so that it can be used with non-normalized data. It uses the means of the input and output variables
    /// along with the denormalized coefficients to compute the correct intercept term.
    /// </para>
    /// <para><b>For Beginners:</b> This method finds the correct starting point (y-intercept) for your model.
    /// 
    /// The y-intercept is the predicted value when all inputs are zero.
    /// 
    /// When moving from normalized to original data:
    /// - The y-intercept needs special calculation
    /// - This method computes the right value using:
    ///   - The original data means
    ///   - The adjusted coefficients
    ///   - The normalization parameters
    /// 
    /// For example, in a linear regression model for house prices:
    /// - The y-intercept might represent the "base price" before considering features
    /// - This method ensures this base price is correct when using non-normalized inputs
    /// 
    /// Together with denormalized coefficients, this gives you a complete model that works with original data.
    /// </para>
    /// </remarks>
    public T DenormalizeYIntercept(Matrix<T> xMatrix, Vector<T> y, Vector<T> coefficients, 
        List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        T yMean = yParams.Mean;
        var xMeans = Vector<T>.FromEnumerable(xParams.Select(p => p.Mean));
        var denormalizedCoefficients = DenormalizeCoefficients(coefficients, xParams, yParams);
         
        return _numOps.Subtract(yMean, xMeans.DotProduct(denormalizedCoefficients));
    }
}