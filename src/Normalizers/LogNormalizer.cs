namespace AiDotNet.Normalizers;

/// <summary>
/// Normalizes the data by taking the natural log of each value.
/// </summary>
/// <remarks>
/// <para>
/// The LogNormalizer transforms data using logarithmic scaling, which compresses the range of
/// values and is especially useful for data that spans multiple orders of magnitude. It shifts
/// negative values to ensure all inputs are positive (required for logarithm), then applies
/// logarithmic scaling and normalizes the results to the [0, 1] range.
/// </para>
/// <para>
/// The transformation scales logarithmically, meaning that ratios in the original data are
/// preserved as differences in the normalized data. This is particularly useful for data
/// with exponential growth patterns or multiplicative relationships.
/// </para>
/// <para>
/// This normalization method is effective for:
/// - Data with exponential growth or decay patterns
/// - Values that span several orders of magnitude
/// - Financial data, population figures, or other naturally log-distributed phenomena
/// - Situations where ratios between values are more meaningful than absolute differences
/// </para>
/// <para><b>For Beginners:</b> Log normalization is like using a special ruler that measures percentages rather than absolute amounts.
/// 
/// Think of it as a way to handle data where some values are MUCH larger than others:
/// - With regular measurement, going from 1 to 10 and from 10 to 100 would look very different
/// - But with logarithmic measurement, both represent a "10 times increase" and would appear as equal steps
/// - This makes patterns easier to see when dealing with very wide-ranging values
/// 
/// For example, with population data:
/// - Original: [1,000, 10,000, 100,000, 1,000,000]
/// - After log normalization: [0.0, 0.33, 0.67, 1.0]
/// 
/// Now each step represents a 10× increase, making it easier to compare growth rates across
/// different scales. This is particularly useful when percentage changes or multiplicative
/// relationships are more important than absolute differences.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class LogNormalizer<T> : INormalizer<T>
{
    /// <summary>
    /// The numeric operations provider for the specified type T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds a reference to an object that provides operations for the numeric type T.
    /// These operations include logarithm, exponentiation, addition, subtraction, and comparisons,
    /// which are essential for the logarithmic normalization process.
    /// </para>
    /// <para><b>For Beginners:</b> This is like having a specialized calculator that works with the type of numbers you're using.
    /// 
    /// Since this normalizer needs to perform logarithm calculations and other mathematical operations
    /// on different types of numbers, it uses this helper to ensure the calculations work correctly
    /// regardless of whether you're using decimals, doubles, or other numeric types.
    /// </para>
    /// </remarks>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="LogNormalizer{T}"/> class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor creates a new LogNormalizer and initializes the numeric operations
    /// provider for the specified type T. No additional parameters are required as the normalizer
    /// determines the necessary shifts and scaling from the data itself.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up your logarithmic measurement system.
    /// 
    /// When you create a new LogNormalizer:
    /// - It prepares the mathematical tools needed for logarithmic transformation
    /// - No additional settings are needed because the normalizer will automatically calculate
    ///   the appropriate shifts and scales based on your actual data
    /// 
    /// It's like getting a logarithmic ruler ready before measuring, letting it automatically
    /// adjust to the range of items you're measuring.
    /// </para>
    /// </remarks>
    public LogNormalizer()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Normalizes a vector using the logarithmic approach.
    /// </summary>
    /// <param name="vector">The input vector to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized vector and the normalization parameters, which include the minimum and maximum values and the shift.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method normalizes a vector by:
    /// 1. Finding the minimum and maximum values
    /// 2. Shifting all values if necessary to ensure they are positive (required for logarithm)
    /// 3. Taking the natural logarithm of each shifted value
    /// 4. Scaling the logarithmic values to the range [0, 1]
    /// 
    /// The normalization preserves the logarithmic relationship between values, making multiplicative
    /// relationships more apparent. Any values that become non-positive after shifting are set to zero.
    /// 
    /// The normalization parameters include the minimum, maximum, and shift values, which are needed for denormalization.
    /// </para>
    /// <para><b>For Beginners:</b> This method transforms your data to highlight multiplicative relationships.
    /// 
    /// The process works like this:
    /// 1. First, it finds the smallest and largest values in your data
    /// 2. If there are negative values, it shifts all values to make them positive
    ///    (because logarithms only work with positive numbers)
    /// 3. Then, it takes the logarithm of each value, which compresses large ranges
    /// 4. Finally, it scales these logarithmic values between 0 and 1
    /// 
    /// For example, with exponential data [2, 20, 200, 2000]:
    /// - After calculating logarithms: [0.69, 3.0, 5.3, 7.6]
    /// - After scaling to [0, 1]: [0.0, 0.33, 0.67, 1.0]
    /// 
    /// This transformation preserves the fact that each value is 10 times the previous one,
    /// which appears as equal steps in the normalized data.
    /// </para>
    /// </remarks>
    public (Vector<T>, NormalizationParameters<T>) NormalizeVector(Vector<T> vector)
    {
        T min = vector.Min();
        T max = vector.Max();
        T shift = _numOps.GreaterThan(min, _numOps.Zero) ? _numOps.Zero : _numOps.Add(_numOps.Negate(min), _numOps.One);
        var normalizedVector = vector.Transform(x =>
        {
            T shiftedValue = _numOps.Add(x, shift);
            return _numOps.GreaterThan(shiftedValue, _numOps.Zero) 
                ? _numOps.Divide(
                    _numOps.Subtract(_numOps.Log(shiftedValue), _numOps.Log(_numOps.Add(min, shift))),
                    _numOps.Subtract(_numOps.Log(_numOps.Add(max, shift)), _numOps.Log(_numOps.Add(min, shift))))
                : _numOps.Zero;
        });
        var parameters = new NormalizationParameters<T>
        {
            Method = NormalizationMethod.Log,
            Min = min,
            Max = max,
            Shift = shift
        };
        return (normalizedVector, parameters);
    }

    /// <summary>
    /// Normalizes a matrix using the logarithmic approach, applying normalization separately to each column.
    /// </summary>
    /// <param name="matrix">The input matrix to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized matrix and a list of normalization parameters for each column.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method normalizes each column of the matrix independently using the logarithmic approach.
    /// It treats each column as a separate feature that needs its own min, max, and shift values,
    /// since different features may have different ranges and distributions.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies logarithmic scaling to a table of data, one column at a time.
    /// 
    /// When working with a table (matrix) of data:
    /// - Each column might represent a different type of information (population, income, company size, etc.)
    /// - Each column needs its own normalization because the ranges differ
    /// - This method processes each column separately using the vector normalization method
    /// 
    /// For example, with a table of company data:
    /// - Column 1 (employees) might range from 10 to 100,000
    /// - Column 2 (revenue) might range from $100,000 to $1 billion
    /// - Each column gets its own appropriate logarithmic scaling
    /// 
    /// The method returns:
    /// - A new table with all values logarithmically normalized to the [0, 1] range
    /// - The parameters for each column, so you can convert back to original values later if needed
    /// </para>
    /// </remarks>
    public (Matrix<T>, List<NormalizationParameters<T>>) NormalizeMatrix(Matrix<T> matrix)
    {
        var normalizedColumns = new List<Vector<T>>();
        var parametersList = new List<NormalizationParameters<T>>();
        for (int i = 0; i < matrix.Columns; i++)
        {
            var column = matrix.GetColumn(i);
            var (normalizedColumn, parameters) = NormalizeVector(column);
            normalizedColumns.Add(normalizedColumn);
            parametersList.Add(parameters);
        }
        var normalizedMatrix = Matrix<T>.FromColumnVectors(normalizedColumns);
        return (normalizedMatrix, parametersList);
    }

    /// <summary>
    /// Denormalizes a vector using the provided normalization parameters.
    /// </summary>
    /// <param name="normalizedVector">The normalized vector to denormalize.</param>
    /// <param name="parameters">The normalization parameters containing min, max, and shift values.</param>
    /// <returns>A denormalized vector with values converted back to their original scale.</returns>
    /// <remarks>
    /// <para>
    /// This method reverses the logarithmic normalization by:
    /// 1. Scaling the normalized values back to the logarithmic range
    /// 2. Taking the exponential to reverse the logarithm transformation
    /// 3. Removing the shift that was applied to make values positive
    /// 
    /// This series of steps restores the values to their original exponential scale and distribution.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts your normalized values back to their original scale.
    /// 
    /// The process reverses all the normalization steps:
    /// 1. First, it rescales the normalized values back to their logarithmic range
    /// 2. Then, it applies the exponential function (the opposite of logarithm)
    /// 3. Finally, it removes any shift that was added to make values positive
    /// 
    /// For example, if your normalized data was [0.0, 0.33, 0.67, 1.0]:
    /// - After rescaling to logarithmic range: [0.69, 3.0, 5.3, 7.6]
    /// - After taking exponentials: [2, 20, 200, 2000]
    /// - After removing any shift: back to the original values
    /// 
    /// This allows you to recover the original exponential data after performing calculations
    /// or analysis on the normalized values.
    /// </para>
    /// </remarks>
    public Vector<T> DenormalizeVector(Vector<T> normalizedVector, NormalizationParameters<T> parameters)
    {
        return normalizedVector.Transform(x =>
        {
            T expValue = _numOps.Exp(_numOps.Add(
                _numOps.Multiply(x, _numOps.Subtract(
                    _numOps.Log(_numOps.Add(parameters.Max, parameters.Shift)),
                    _numOps.Log(_numOps.Add(parameters.Min, parameters.Shift)))),
                _numOps.Log(_numOps.Add(parameters.Min, parameters.Shift))));
            return _numOps.Subtract(expValue, parameters.Shift);
        });
    }

    /// <summary>
    /// Denormalizes coefficients from a regression model that was trained on logarithmically normalized data.
    /// </summary>
    /// <param name="coefficients">The regression coefficients to denormalize.</param>
    /// <param name="xParams">The normalization parameters for the input features.</param>
    /// <param name="yParams">The normalization parameters for the output variable.</param>
    /// <returns>Denormalized coefficients for use with logarithmically transformed data.</returns>
    /// <remarks>
    /// <para>
    /// This method adjusts regression coefficients to work with logarithmically transformed original data.
    /// It scales each coefficient by the ratio of the logarithmic ranges of the output and input features.
    /// 
    /// Note that these denormalized coefficients should be applied to logarithmically transformed inputs,
    /// not to the raw original data, due to the non-linear nature of the logarithm transformation.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts prediction model weights for use with logarithmically transformed data.
    /// 
    /// When you build a model using logarithmically normalized data:
    /// - The model learns weights (coefficients) based on the normalized log values
    /// - To use this model with log-transformed (but not normalized) data, you need to adjust these weights
    /// 
    /// This method adjusts the coefficients based on:
    /// - The logarithmic range of each input feature
    /// - The logarithmic range of the output variable
    /// 
    /// Important note: Because logarithmic transformation is non-linear, you can't simply apply these
    /// coefficients to the original raw data. Instead, to use these coefficients:
    /// 1. Take the logarithm of your input data
    /// 2. Apply these denormalized coefficients
    /// 3. Exponentiate the result to get predictions in the original scale
    /// 
    /// This reflects the complexity of working with logarithmic transformations in predictive models.
    /// </para>
    /// </remarks>
    public Vector<T> DenormalizeCoefficients(Vector<T> coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        return coefficients.PointwiseMultiply(Vector<T>.FromEnumerable(
            xParams.Select(p => _numOps.Divide(
                _numOps.Subtract(_numOps.Log(_numOps.Add(yParams.Max, yParams.Shift)), _numOps.Log(_numOps.Add(yParams.Min, yParams.Shift))),
                _numOps.Subtract(_numOps.Log(_numOps.Add(p.Max, p.Shift)), _numOps.Log(_numOps.Add(p.Min, p.Shift)))))));
    }

    /// <summary>
    /// Denormalizes the y-intercept from a regression model that was trained on logarithmically normalized data.
    /// </summary>
    /// <param name="xMatrix">The original input feature matrix.</param>
    /// <param name="y">The original output vector.</param>
    /// <param name="coefficients">The regression coefficients.</param>
    /// <param name="xParams">The normalization parameters for the input features.</param>
    /// <param name="yParams">The normalization parameters for the output variable.</param>
    /// <returns>A denormalized y-intercept for use with the original data.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates an appropriate y-intercept for a model trained on normalized data
    /// but applied to original data. It uses the denormalized coefficients and the means of the
    /// original features and output to derive an intercept that maintains the correct predicted
    /// values when used with the original data.
    /// </para>
    /// <para><b>For Beginners:</b> This method determines the correct starting point for predictions with original data.
    /// 
    /// In a prediction model, the y-intercept is the baseline value:
    /// - It's the predicted output when all inputs are at their average values
    /// - When using logarithmic normalization, this baseline needs special calculation
    /// 
    /// The method works by:
    /// 1. First denormalizing the coefficients to work with original data
    /// 2. Finding the average value of each input feature
    /// 3. Finding the average value of the output
    /// 4. Calculating an intercept that ensures predictions using average inputs yield the average output
    /// 
    /// This approach ensures that when you use:
    /// - Your original input data
    /// - With the denormalized coefficients
    /// - And this denormalized intercept
    /// 
    /// The predictions will be properly calibrated to match your original data scale.
    /// </para>
    /// </remarks>
    public T DenormalizeYIntercept(Matrix<T> xMatrix, Vector<T> y, Vector<T> coefficients,
        List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        var denormalizedCoefficients = DenormalizeCoefficients(coefficients, xParams, yParams);
        var meanX = Vector<T>.FromEnumerable(xMatrix.GetColumns().Select(col => col.Mean()));
        var meanY = y.Mean();
        T intercept = meanY;
        for (int i = 0; i < coefficients.Length; i++)
        {
            intercept = _numOps.Subtract(intercept, _numOps.Multiply(denormalizedCoefficients[i], meanX[i]));
        }
        return intercept;
    }
}