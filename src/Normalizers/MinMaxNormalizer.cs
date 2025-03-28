namespace AiDotNet.Normalizers;

/// <summary>
/// Normalizes the data by
/// 1) Subtracting the minimum value from each value
/// 2) Dividing each value from step #1 by the absolute difference between the maximum and minimum values
/// </summary>
/// <remarks>
/// <para>
/// The MinMaxNormalizer scales data to a fixed range, typically [0, 1], by applying a linear transformation.
/// It first shifts the data by subtracting the minimum value, then scales it by dividing by the range
/// (maximum minus minimum). This ensures that the smallest value becomes 0 and the largest becomes 1,
/// with all other values linearly distributed between these extremes.
/// </para>
/// <para>
/// The transformation formula is: normalized = (original - min) / (max - min)
/// </para>
/// <para>
/// This normalization method is particularly useful for:
/// - Algorithms that require input values in a specific range, like neural networks with sigmoid activation
/// - Visualization purposes where a fixed scale is desired
/// - Features with different units or scales that need to be compared
/// - Data where the relative minimum and maximum values are meaningful
/// </para>
/// <para><b>For Beginners:</b> Min-max normalization is like converting grades to percentages.
/// 
/// Think of it as scaling values to fit on a scale from 0 to 100%:
/// - The minimum value becomes 0% (or 0.0)
/// - The maximum value becomes 100% (or 1.0)
/// - Everything else is spaced proportionally between these extremes
/// 
/// For example, if you have test scores:
/// - Original scores: [60, 75, 90, 100]
/// - Min = 60, Max = 100
/// - Normalized scores: [0.0, 0.375, 0.75, 1.0]
/// 
/// After normalization:
/// - The lowest value is always 0
/// - The highest value is always 1
/// - The spacing between values remains proportional to the original data
/// 
/// This is useful when you need values within a specific range, like percentages
/// or probability values between 0 and 1.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MinMaxNormalizer<T> : INormalizer<T>
{
    /// <summary>
    /// The numeric operations provider for the specified type T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds a reference to an object that provides operations for the numeric type T.
    /// These operations include addition, subtraction, multiplication, division, and comparisons,
    /// which are needed for min-max normalization calculations.
    /// </para>
    /// <para><b>For Beginners:</b> This is like having a specialized calculator that works with whatever number type you're using.
    /// 
    /// Since this normalizer can work with different types of numbers (integers, decimals, etc.),
    /// it needs a way to perform math operations on these numbers. This field provides those capabilities,
    /// like a specialized calculator for the specific type of numbers being processed.
    /// </para>
    /// </remarks>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="MinMaxNormalizer{T}"/> class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor creates a new MinMaxNormalizer and initializes the numeric operations
    /// provider for the specified type T. No additional parameters are required as the normalizer
    /// determines the minimum and maximum values from the data itself.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up your percentage-scaling system.
    /// 
    /// When you create a new MinMaxNormalizer:
    /// - It prepares the mathematical tools needed for scaling values to the [0, 1] range
    /// - No additional settings are needed because the normalizer will automatically find
    ///   the minimum and maximum from your actual data
    /// 
    /// It's like setting up a grading system that will automatically adjust to the lowest
    /// and highest scores in a class.
    /// </para>
    /// </remarks>
    public MinMaxNormalizer()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Normalizes a vector using the min-max approach.
    /// </summary>
    /// <param name="vector">The input vector to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized vector and the normalization parameters, which include the minimum and maximum values.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method normalizes a vector by:
    /// 1. Finding the minimum and maximum values in the vector
    /// 2. Applying the formula: (value - min) / (max - min) to each element
    /// 
    /// The resulting values will fall within the range [0, 1], with the minimum value becoming 0
    /// and the maximum value becoming 1.
    /// 
    /// The normalization parameters include the minimum and maximum values, which are needed for denormalization.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts your data to a 0-1 scale.
    /// 
    /// The process works like this:
    /// 1. First, it finds the smallest and largest values in your data
    /// 2. Then, for each value, it:
    ///    - Subtracts the minimum (shifting the scale to start at 0)
    ///    - Divides by the range (max - min) to make the maximum value 1
    /// 
    /// After normalization:
    /// - The smallest value becomes exactly 0
    /// - The largest value becomes exactly 1
    /// - Everything else is proportionally spaced between 0 and 1
    /// 
    /// For example, if your test scores were [60, 75, 90, 100]:
    /// - The minimum is 60 and the maximum is 100
    /// - After normalization, they become [0.0, 0.375, 0.75, 1.0]
    /// - Now you can easily see that 75 is 37.5% of the way from the minimum to the maximum
    /// </para>
    /// </remarks>
    public (Vector<T>, NormalizationParameters<T>) NormalizeVector(Vector<T> vector)
    {
        T min = vector.Min();
        T max = vector.Max();
        var normalized = vector.Transform(x => _numOps.Divide(_numOps.Subtract(x, min), _numOps.Subtract(max, min)));
        return (normalized, new NormalizationParameters<T> { Method = NormalizationMethod.MinMax, Min = min, Max = max });
    }

    /// <summary>
    /// Normalizes a matrix using the min-max approach, applying normalization separately to each column.
    /// </summary>
    /// <param name="matrix">The input matrix to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized matrix and a list of normalization parameters for each column.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method normalizes each column of the matrix independently using the min-max approach.
    /// It treats each column as a separate feature that needs its own min and max values,
    /// since different features may have different ranges.
    /// </para>
    /// <para><b>For Beginners:</b> This method scales each column of a data table to the 0-1 range.
    /// 
    /// When working with a table (matrix) of data:
    /// - Each column might represent a different type of information (height, weight, price, etc.)
    /// - Each column needs its own scaling because the ranges differ
    /// - This method processes each column separately using the vector normalization method
    /// 
    /// For example, with a table of product data:
    /// - Column 1 (weights) might range from 0.5kg to 10kg
    /// - Column 2 (prices) might range from $5 to $500
    /// - Each column gets its own appropriate scaling to the 0-1 range
    /// 
    /// The method returns:
    /// - A new table with all values scaled to between 0 and 1
    /// - The minimum and maximum for each column, so you can convert back to original values later if needed
    /// </para>
    /// </remarks>
    public (Matrix<T>, List<NormalizationParameters<T>>) NormalizeMatrix(Matrix<T> matrix)
    {
        var normalizedColumns = new List<Vector<T>>();
        var parameters = new List<NormalizationParameters<T>>();
        for (int i = 0; i < matrix.Columns; i++)
        {
            var column = matrix.GetColumn(i);
            var (normalizedColumn, param) = NormalizeVector(column);
            normalizedColumns.Add(normalizedColumn);
            parameters.Add(param);
        }

        return (Matrix<T>.FromColumnVectors(normalizedColumns), parameters);
    }

    /// <summary>
    /// Denormalizes a vector using the provided normalization parameters.
    /// </summary>
    /// <param name="vector">The normalized vector to denormalize.</param>
    /// <param name="parameters">The normalization parameters containing the minimum and maximum values.</param>
    /// <returns>A denormalized vector with values converted back to their original scale.</returns>
    /// <remarks>
    /// <para>
    /// This method reverses the min-max normalization by applying the inverse of the original formula:
    /// original = normalized * (max - min) + min
    /// 
    /// This transformation restores the values to their original scale and range.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts your scaled values back to their original range.
    /// 
    /// The process reverses the normalization steps:
    /// 1. First, it multiplies each value by the original range (max - min)
    /// 2. Then, it adds the original minimum value
    /// 
    /// For example, if your normalized values were [0.0, 0.375, 0.75, 1.0] with min = 60 and max = 100:
    /// - The range is (100 - 60) = 40
    /// - The denormalized values would be:
    ///   * 0.0 × 40 + 60 = 60
    ///   * 0.375 × 40 + 60 = 75
    ///   * 0.75 × 40 + 60 = 90
    ///   * 1.0 × 40 + 60 = 100
    /// 
    /// This recovers the original test scores: [60, 75, 90, 100]
    /// </para>
    /// </remarks>
    public Vector<T> DenormalizeVector(Vector<T> vector, NormalizationParameters<T> parameters)
    {
        return vector.Transform(x => _numOps.Add(_numOps.Multiply(x, _numOps.Subtract(parameters.Max, parameters.Min)), parameters.Min));
    }

    /// <summary>
    /// Denormalizes coefficients from a regression model that was trained on min-max normalized data.
    /// </summary>
    /// <param name="coefficients">The regression coefficients to denormalize.</param>
    /// <param name="xParams">The normalization parameters for the input features.</param>
    /// <param name="yParams">The normalization parameters for the output variable.</param>
    /// <returns>Denormalized coefficients that can be applied to original, unnormalized data.</returns>
    /// <remarks>
    /// <para>
    /// This method adjusts regression coefficients to work with original unnormalized data.
    /// For min-max normalization, this involves scaling each coefficient by the ratio of
    /// the output range to the corresponding input range.
    /// This adjustment accounts for the different scales that were applied to each feature
    /// and the output during normalization.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts prediction model weights for use with original values.
    /// 
    /// When you build a model using normalized data:
    /// - The model learns weights (coefficients) based on the 0-1 scaled values
    /// - To use this model with original, unscaled data, you need to adjust these weights
    /// 
    /// This method adjusts each coefficient by:
    /// - Multiplying it by the output range (max - min)
    /// - Dividing it by the input feature's range
    /// 
    /// For example, if:
    /// - Your input feature ranged from 60 to 100 (range = 40)
    /// - Your output ranged from 10 to 50 (range = 40)
    /// - The model learned a coefficient of 0.75 for this feature on normalized data
    /// 
    /// The denormalized coefficient would be 0.75 × (40 ÷ 40) = 0.75
    /// 
    /// But if the ranges were different (say input range was 100 and output range was 50),
    /// the denormalized coefficient would be adjusted accordingly.
    /// </para>
    /// </remarks>
    public Vector<T> DenormalizeCoefficients(Vector<T> coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        var denormalizedCoefficients = new T[coefficients.Length];
        for (int i = 0; i < coefficients.Length; i++)
        {
            denormalizedCoefficients[i] = _numOps.Divide(
                _numOps.Multiply(coefficients[i], _numOps.Subtract(yParams.Max, yParams.Min)),
                _numOps.Subtract(xParams[i].Max, xParams[i].Min)
            );
        }

        return Vector<T>.FromArray(denormalizedCoefficients);
    }

    /// <summary>
    /// Denormalizes the y-intercept from a regression model that was trained on min-max normalized data.
    /// </summary>
    /// <param name="x">The original input feature matrix.</param>
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
    /// - It's what you predict when all input features are at their minimum values
    /// - When using normalized data, this baseline needs to be adjusted for original data
    /// 
    /// The calculation is complex because:
    /// - During normalization, we shifted and scaled each feature
    /// - We also shifted and scaled the output
    /// - The model's coefficients have been adjusted for these changes
    /// 
    /// The method calculates the correct intercept to ensure that:
    /// - If you input the minimum value for each feature
    /// - You should get the minimum output value, adjusted by the appropriate coefficient effects
    /// 
    /// This allows the model to make correct predictions across the entire range of the original data.
    /// </para>
    /// </remarks>
    public T DenormalizeYIntercept(Matrix<T> x, Vector<T> y, Vector<T> coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        T yIntercept = yParams.Min;
        for (int i = 0; i < coefficients.Length; i++)
        {
            yIntercept = _numOps.Subtract(yIntercept, 
                _numOps.Divide(
                    _numOps.Multiply(
                        _numOps.Multiply(coefficients[i], xParams[i].Min),
                        _numOps.Subtract(yParams.Max, yParams.Min)
                    ),
                    _numOps.Subtract(xParams[i].Max, xParams[i].Min)
                )
            );
        }

        return yIntercept;
    }
}