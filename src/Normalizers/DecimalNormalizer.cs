namespace AiDotNet.Normalizers;

/// <summary>
/// Normalizes the data by dividing each value by the smallest multiple of 10 that is greater than the largest value.
/// </summary>
/// <remarks>
/// <para>
/// The DecimalNormalizer scales values by powers of 10 to ensure that all normalized values fall between -1 and 1.
/// It finds the smallest power of 10 that is greater than the largest absolute value in the data and divides
/// all values by this scale factor. This approach preserves the relative decimal positions and sign of the values.
/// </para>
/// <para>
/// This normalization method is particularly useful for:
/// - Datasets where preserving decimal places is important
/// - Financial data where values need to be comparable in terms of magnitude
/// - Scenarios where the interpretation of normalized values should be intuitive
/// </para>
/// <para><b>For Beginners:</b> Decimal normalization is like adjusting all numbers to show them in the right decimal places.
/// 
/// Think of it as working with a cash register display that only shows two decimal places:
/// - If you're dealing with large amounts (hundreds or thousands of dollars), you need to adjust the display
/// - This normalizer figures out whether your values are in ones, tens, hundreds, thousands, etc.
/// - Then it divides everything by the appropriate power of 10 so all values are between -1 and 1
/// 
/// For example:
/// - If your largest value is 750, it divides everything by 1,000
/// - So 750 becomes 0.75, 42 becomes 0.042, and so on
/// - Now all values have been "shifted" to the same decimal scale
/// 
/// This is useful when you want to keep the relative sizes clear and the decimal places meaningful.
/// Unlike some other normalizers that squeeze values into specific ranges, this one preserves
/// the intuitive understanding of your data's magnitude.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class DecimalNormalizer<T> : INormalizer<T>
{
    /// <summary>
    /// The numeric operations provider for the specified type T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds a reference to an object that provides operations for the numeric type T.
    /// These operations include addition, subtraction, multiplication, division, and comparisons,
    /// which are needed for the decimal normalization calculations.
    /// </para>
    /// <para><b>For Beginners:</b> This is like having a calculator that works with whatever number type you're using.
    /// 
    /// Since this normalizer can work with different types of numbers (integers, decimals, etc.),
    /// it needs a way to perform math operations on these numbers. This field provides those capabilities,
    /// similar to having a specialized calculator that knows how to handle the specific type of numbers
    /// you're working with.
    /// </para>
    /// </remarks>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="DecimalNormalizer{T}"/> class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor creates a new DecimalNormalizer and initializes the numeric operations
    /// provider for the specified type T. No additional parameters are required as the normalizer
    /// automatically detects the appropriate scale.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up your decimal normalization system.
    /// 
    /// When you create a new DecimalNormalizer:
    /// - It gets ready to work with the type of numbers you'll be processing
    /// - It prepares the mathematical operations needed for normalization
    /// - No additional settings are needed because the normalizer will automatically figure out
    ///   the right scale based on your data
    /// 
    /// It's like setting up your calculator before starting your calculations.
    /// </para>
    /// </remarks>
    public DecimalNormalizer()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Normalizes a vector using the decimal approach.
    /// </summary>
    /// <param name="vector">The input vector to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized vector and the normalization parameters, which include the scale factor.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method normalizes a vector by:
    /// 1. Finding the largest absolute value in the vector
    /// 2. Determining the smallest power of 10 that is greater than this maximum value
    /// 3. Dividing all values in the vector by this scale factor
    /// 
    /// The normalization parameters include the scale factor, which is needed for later denormalization.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts all your numbers to an appropriate decimal scale.
    /// 
    /// The process works like this:
    /// 1. First, it finds the largest value in your data (ignoring negative signs)
    /// 2. Then, it figures out which power of 10 is just large enough to make this value less than 1
    ///    - If the largest value is 456, it would use 1,000
    ///    - If the largest value is 5.6, it would use 10
    /// 3. Finally, it divides all your values by this scale factor
    /// 
    /// The method returns:
    /// - Your transformed data with each value divided by the scale factor
    /// - The scale factor itself, so you can convert back to original values later
    /// 
    /// For example, if your data was [42, 125, 7, -89]:
    /// - The largest absolute value is 125
    /// - The scale factor would be 1,000
    /// - The normalized values would be [0.042, 0.125, 0.007, -0.089]
    /// </para>
    /// </remarks>
    public (Vector<T>, NormalizationParameters<T>) NormalizeVector(Vector<T> vector)
    {
        T maxAbs = vector.AbsoluteMaximum();
        T scale = _numOps.One;
        T ten = _numOps.FromDouble(10);
        while (_numOps.GreaterThanOrEquals(maxAbs, scale))
        {
            scale = _numOps.Multiply(scale, ten);
        }
        var normalizedVector = vector.Transform(x => _numOps.Divide(x, scale));
        var parameters = new NormalizationParameters<T>
        {
            Method = NormalizationMethod.Decimal,
            Scale = scale
        };
        return (normalizedVector, parameters);
    }

    /// <summary>
    /// Normalizes a matrix using the decimal approach, applying normalization separately to each column.
    /// </summary>
    /// <param name="matrix">The input matrix to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized matrix and a list of normalization parameters for each column.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method normalizes each column of the matrix independently using the decimal approach.
    /// It treats each column as a separate feature that needs its own scaling factor, since different
    /// features may have different ranges and magnitudes.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies decimal scaling to a table of data, one column at a time.
    /// 
    /// When working with a table (matrix) of data:
    /// - Each column might represent a different type of information (age, income, height, etc.)
    /// - Each column needs its own scaling factor because the ranges and magnitudes differ
    /// - This method processes each column separately using the vector normalization method
    /// 
    /// For example, with a table of people's information:
    /// - Column 1 (ages) might be in the tens (scale factor 100)
    /// - Column 2 (incomes) might be in the thousands (scale factor 10,000)
    /// - Each column gets its own appropriate scale factor
    /// 
    /// The method returns:
    /// - A new table with all values scaled to appropriate decimal places
    /// - A separate scale factor for each column, so you can convert back to original values later
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
    /// <param name="parameters">The normalization parameters containing the scale factor.</param>
    /// <returns>A denormalized vector with values converted back to their original scale.</returns>
    /// <remarks>
    /// <para>
    /// This method converts normalized values back to their original scale by multiplying each value
    /// by the scale factor that was used during normalization. Unlike some other normalization methods,
    /// decimal normalization can be perfectly reversed to recover the exact original values.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts your scaled numbers back to their original values.
    /// 
    /// The process is simple:
    /// - Take each normalized value
    /// - Multiply it by the scale factor that was used during normalization
    /// - The result is the original value
    /// 
    /// For example, if your normalized data was [0.042, 0.125, 0.007, -0.089] with a scale factor of 1,000:
    /// - The denormalized values would be [42, 125, 7, -89]
    /// 
    /// One advantage of decimal normalization is that the conversion back is exact - you get precisely
    /// the original values, without any loss of information.
    /// </para>
    /// </remarks>
    public Vector<T> DenormalizeVector(Vector<T> vector, NormalizationParameters<T> parameters)
    {
        return vector.Transform(x => _numOps.Multiply(x, parameters.Scale));
    }

    /// <summary>
    /// Denormalizes coefficients from a regression model that was trained on decimal-normalized data.
    /// </summary>
    /// <param name="coefficients">The regression coefficients to denormalize.</param>
    /// <param name="xParams">The normalization parameters for the input features.</param>
    /// <param name="yParams">The normalization parameters for the output variable.</param>
    /// <returns>Denormalized coefficients that can be applied to original, unnormalized data.</returns>
    /// <remarks>
    /// <para>
    /// This method adjusts regression coefficients to work with original unnormalized data.
    /// For decimal normalization, this involves multiplying each coefficient by the ratio of
    /// the output scale to the corresponding input scale. This adjustment accounts for the
    /// different scales that were applied to each feature and the output during normalization.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts prediction model weights to work with original values instead of scaled values.
    /// 
    /// When you've built a prediction model using normalized data:
    /// - The model's weights (coefficients) were learned based on the scaled values
    /// - To use this model with original unscaled data, you need to adjust these weights
    /// - This method converts the weights to work with the original scale of your data
    /// 
    /// For example, if:
    /// - Your input feature was scaled by dividing by 1,000
    /// - Your output was scaled by dividing by 100
    /// - The model learned a coefficient of 2.5 for this feature
    /// 
    /// The denormalized coefficient would be 2.5 × (100 ÷ 1,000) = 0.25
    /// 
    /// This adjustment ensures that when you multiply the original unscaled input by this new coefficient,
    /// you get the correct prediction in the original output scale.
    /// </para>
    /// </remarks>
    public Vector<T> DenormalizeCoefficients(Vector<T> coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        var scalingFactors = xParams.Select(p => _numOps.Divide(yParams.Scale, p.Scale)).ToArray();
        return coefficients.PointwiseMultiply(Vector<T>.FromArray(scalingFactors));
    }

    /// <summary>
    /// Denormalizes the y-intercept from a regression model that was trained on decimal-normalized data.
    /// </summary>
    /// <param name="xMatrix">The original input feature matrix.</param>
    /// <param name="y">The original output vector.</param>
    /// <param name="coefficients">The regression coefficients.</param>
    /// <param name="xParams">The normalization parameters for the input features.</param>
    /// <param name="yParams">The normalization parameters for the output variable.</param>
    /// <returns>A denormalized y-intercept that can be used with original, unnormalized data.</returns>
    /// <remarks>
    /// <para>
    /// This method returns zero as the y-intercept for models trained on decimal-normalized data.
    /// For decimal normalization, the y-intercept in the original scale is always zero because
    /// the normalization only involves scaling by a constant factor, which doesn't introduce any shift
    /// in the intercept.
    /// </para>
    /// <para><b>For Beginners:</b> This method determines the baseline value for predictions with original data.
    /// 
    /// In a prediction model, the y-intercept is the starting point:
    /// - It's the predicted value when all features are set to zero
    /// 
    /// For decimal normalization specifically:
    /// - The y-intercept is always zero in the original scale
    /// - This is because decimal normalization only involves multiplication/division, not addition/subtraction
    /// - A zero input will still be zero after scaling, so the origin point doesn't change
    /// 
    /// This is different from other normalization methods that might shift values, which would require
    /// a non-zero intercept adjustment.
    /// </para>
    /// </remarks>
    public T DenormalizeYIntercept(Matrix<T> xMatrix, Vector<T> y, Vector<T> coefficients,
        List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        return _numOps.Zero; // The y-intercept for decimal normalization is always 0
    }
}