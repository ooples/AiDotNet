namespace AiDotNet.Normalizers;

/// <summary>
/// Normalizes vectors using the Lp-norm, dividing each element by the vector's p-norm.
/// </summary>
/// <remarks>
/// <para>
/// The LpNormNormalizer scales vectors by dividing each element by the Lp-norm of the vector.
/// The Lp-norm is a generalization of different vector norms based on the parameter p:
/// - p = 1: Manhattan (L1) norm (sum of absolute values)
/// - p = 2: Euclidean (L2) norm (square root of sum of squares)
/// - p = ∞: Maximum (L∞) norm (maximum absolute value)
/// </para>
/// <para>
/// This normalization preserves the direction of the vector while scaling its magnitude.
/// After normalization, the Lp-norm of the resulting vector equals 1.
/// </para>
/// <para>
/// This normalization method is particularly useful for:
/// - Feature normalization in machine learning models
/// - Ensuring consistent scaling across feature vectors
/// - Distance calculations in various algorithms
/// - Applications where vector direction is more important than magnitude
/// </para>
/// <para><b>For Beginners:</b> Lp-norm normalization is like scaling a map while preserving its proportions.
/// 
/// Think of a vector as an arrow pointing in a specific direction:
/// - The Lp-norm is a way to measure the "length" of this arrow
/// - This normalizer divides each component of the arrow by its length
/// - The result is an arrow pointing in the same direction, but with a standard length of 1
/// 
/// Different values of p provide different ways to measure the arrow's length:
/// - p = 1: Measures length as the sum of absolute values (like walking along city blocks)
/// - p = 2: Measures length as the straight-line distance (like the crow flies)
/// - Higher p values: Increasingly emphasize the largest component
/// 
/// For example, normalizing the vector [3, 4] with p = 2 (Euclidean norm):
/// - The norm is sqrt(3² + 4²) = sqrt(25) = 5
/// - The normalized vector is [3/5, 4/5] = [0.6, 0.8]
/// - This new vector points in the same direction but has a length of 1
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class LpNormNormalizer<T> : INormalizer<T>
{
    /// <summary>
    /// The p parameter that defines which norm to use.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value determines which Lp-norm is calculated:
    /// - p = 1: Manhattan (L1) norm
    /// - p = 2: Euclidean (L2) norm
    /// - As p approaches infinity, the norm approaches the maximum absolute value (L∞ norm)
    /// </para>
    /// <para><b>For Beginners:</b> The p value determines how the "length" of a vector is measured.
    /// 
    /// Different p values lead to different ways of measuring distance:
    /// - p = 1: Like measuring distance by walking along city blocks
    /// - p = 2: Like measuring distance "as the crow flies" (straight line)
    /// - Large p values: Increasingly dominated by the largest dimension
    /// 
    /// For many applications, p = 2 (Euclidean norm) is most common because it represents
    /// the familiar straight-line distance in geometric space.
    /// </para>
    /// </remarks>
    private readonly T _p;

    /// <summary>
    /// The numeric operations provider for the specified type T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds a reference to an object that provides operations for the numeric type T.
    /// These operations include power, absolute value, addition, division, and other calculations
    /// needed for computing the Lp-norm.
    /// </para>
    /// <para><b>For Beginners:</b> This is like having a specialized calculator that works with the type of numbers you're using.
    /// 
    /// Since this normalizer needs to perform various mathematical operations (power, absolute value,
    /// addition, division, etc.) on different types of numbers, it uses this helper to ensure
    /// the calculations work correctly regardless of whether you're using decimals, doubles, or other numeric types.
    /// </para>
    /// </remarks>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="LpNormNormalizer{T}"/> class with the specified p value.
    /// </summary>
    /// <param name="p">The p value that determines which norm to use.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new LpNormNormalizer with the specified p value and initializes
    /// the numeric operations provider for the type T. Common values for p are 1 (Manhattan norm),
    /// 2 (Euclidean norm), and large values to approximate the maximum norm.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up your vector normalization system with a specific way to measure vector length.
    /// 
    /// When you create a new LpNormNormalizer:
    /// - You specify the p value, which determines how vector length is calculated
    /// - Common choices are p = 1 (Manhattan/taxicab distance) or p = 2 (straight-line distance)
    /// - The normalizer prepares the mathematical tools needed for these calculations
    /// 
    /// It's like choosing which ruler (straight or Manhattan-style) you'll use to measure distances
    /// before standardizing them.
    /// </para>
    /// </remarks>
    public LpNormNormalizer(T p)
    {
        _p = p;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Normalizes a vector using the Lp-norm.
    /// </summary>
    /// <param name="vector">The input vector to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized vector and the normalization parameters, which include the scale (norm) and p value.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method normalizes a vector by:
    /// 1. Computing the Lp-norm of the vector: (sum of |x_i|^p)^(1/p)
    /// 2. Dividing each element by this norm
    /// 
    /// The resulting vector has the same direction as the original but with an Lp-norm of 1.
    /// The norm is saved in the normalization parameters for later denormalization.
    /// </para>
    /// <para><b>For Beginners:</b> This method standardizes a vector to have a length of 1 while keeping its direction.
    /// 
    /// The process works like this:
    /// 1. First, it measures the "length" of your vector using the p-norm
    ///    - Takes the absolute value of each element
    ///    - Raises each to the power of p
    ///    - Adds them all together
    ///    - Takes the (1/p)th root of the sum
    /// 2. Then, it divides each element of the vector by this length
    /// 
    /// For example, with vector [3, 4] and p = 2:
    /// - The norm is sqrt(3² + 4²) = sqrt(25) = 5
    /// - The normalized vector is [3/5, 4/5] = [0.6, 0.8]
    /// 
    /// After normalization:
    /// - The vector points in the same direction
    /// - The vector has a length of exactly 1 using the p-norm
    /// - The relative proportions between elements are preserved
    /// </para>
    /// </remarks>
    public (Vector<T>, NormalizationParameters<T>) NormalizeVector(Vector<T> vector)
    {
        T sum = vector.Select(x => _numOps.Power(_numOps.Abs(x), _p)).Aggregate(_numOps.Zero, _numOps.Add);
        T norm = _numOps.Power(sum, _numOps.Divide(_numOps.One, _p));
        var normalizedVector = vector.Transform(x => _numOps.Divide(x, norm));
        var parameters = new NormalizationParameters<T> { Scale = norm, P = _p, Method = NormalizationMethod.LpNorm };
        return (normalizedVector, parameters);
    }

    /// <summary>
    /// Normalizes a matrix using the Lp-norm approach, applying normalization separately to each column.
    /// </summary>
    /// <param name="matrix">The input matrix to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized matrix and a list of normalization parameters for each column.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method normalizes each column of the matrix independently using the Lp-norm approach.
    /// It treats each column as a separate feature vector that needs its own normalization.
    /// </para>
    /// <para><b>For Beginners:</b> This method standardizes each column of a data table to have a length of 1.
    /// 
    /// When working with a table (matrix) of data:
    /// - Each column is treated as a separate vector
    /// - Each column gets normalized independently using the method described above
    /// - This ensures that each feature (column) has a consistent scale
    /// 
    /// For example, with a table of measurements where:
    /// - Column 1 might represent heights
    /// - Column 2 might represent weights
    /// - Each column is scaled to have a p-norm of 1, while preserving the relationships within that column
    /// 
    /// The method returns:
    /// - A new table with all columns normalized to unit length
    /// - The norm values for each column, so you can convert back to original values later if needed
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
    /// <param name="parameters">The normalization parameters containing the scale (norm).</param>
    /// <returns>A denormalized vector with values converted back to their original scale.</returns>
    /// <remarks>
    /// <para>
    /// This method reverses the Lp-norm normalization by multiplying each element by the original norm.
    /// This restores the vector to its original magnitude while maintaining its direction.
    /// </para>
    /// <para><b>For Beginners:</b> This method restores the original length of a normalized vector.
    /// 
    /// The process is straightforward:
    /// - Take each element in the normalized vector
    /// - Multiply it by the original length (norm) that was saved during normalization
    /// 
    /// For example, if your normalized vector was [0.6, 0.8] with an original norm of 5:
    /// - The denormalized vector would be [0.6 × 5, 0.8 × 5] = [3, 4]
    /// 
    /// This restores the vector to its original scale while maintaining its direction and the
    /// proportional relationships between its elements.
    /// </para>
    /// </remarks>
    public Vector<T> DenormalizeVector(Vector<T> vector, NormalizationParameters<T> parameters)
    {
        return vector.Transform(x => _numOps.Multiply(x, parameters.Scale));
    }

    /// <summary>
    /// Denormalizes coefficients from a regression model that was trained on Lp-norm normalized data.
    /// </summary>
    /// <param name="coefficients">The regression coefficients to denormalize.</param>
    /// <param name="xParams">The normalization parameters for the input features.</param>
    /// <param name="yParams">The normalization parameters for the output variable.</param>
    /// <returns>Denormalized coefficients that can be applied to original, unnormalized data.</returns>
    /// <remarks>
    /// <para>
    /// This method adjusts regression coefficients to work with original unnormalized data.
    /// For Lp-norm normalization, this involves scaling each coefficient by the ratio of
    /// the output norm to the corresponding input norm.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts prediction model weights for use with original values.
    /// 
    /// When you build a model using normalized data:
    /// - The model learns weights (coefficients) based on the normalized values
    /// - To use this model with original, unnormalized data, you need to adjust these weights
    /// 
    /// This method performs that adjustment by:
    /// - Multiplying each coefficient by the ratio of output norm to input norm
    /// 
    /// For example, if:
    /// - An input feature was normalized by dividing by 5
    /// - The output was normalized by dividing by 10
    /// - The model learned a coefficient of 2.0 for this feature on normalized data
    /// 
    /// The denormalized coefficient would be 2.0 × (10 ÷ 5) = 4.0
    /// 
    /// This ensures that predictions made using original data will be properly scaled.
    /// </para>
    /// </remarks>
    public Vector<T> DenormalizeCoefficients(Vector<T> coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        var scalingFactors = xParams.Select(p => _numOps.Divide(yParams.Scale, p.Scale)).ToArray();
        return coefficients.PointwiseMultiply(Vector<T>.FromArray(scalingFactors));
    }

    /// <summary>
    /// Denormalizes the y-intercept from a regression model that was trained on Lp-norm normalized data.
    /// </summary>
    /// <param name="xMatrix">The original input feature matrix.</param>
    /// <param name="y">The original output vector.</param>
    /// <param name="coefficients">The regression coefficients.</param>
    /// <param name="xParams">The normalization parameters for the input features.</param>
    /// <param name="yParams">The normalization parameters for the output variable.</param>
    /// <returns>A denormalized y-intercept that can be used with original, unnormalized data.</returns>
    /// <remarks>
    /// <para>
    /// This method returns zero as the y-intercept for models trained on Lp-norm normalized data.
    /// For Lp-norm normalization, the y-intercept in the original scale is always zero because
    /// the normalization only involves scaling by a constant factor, which doesn't introduce any shift
    /// in the intercept.
    /// </para>
    /// <para><b>For Beginners:</b> This method determines the baseline value for predictions with original data.
    /// 
    /// In a prediction model, the y-intercept is the starting point:
    /// - It's the predicted value when all features are set to zero
    /// 
    /// For Lp-norm normalization specifically:
    /// - The y-intercept is always zero in the original scale
    /// - This is because Lp-norm normalization only involves multiplication/division, not addition/subtraction
    /// - A zero input will still be zero after scaling, so the origin point doesn't change
    /// 
    /// This is different from other normalization methods that might shift values, which would require
    /// a non-zero intercept adjustment.
    /// </para>
    /// </remarks>
    public T DenormalizeYIntercept(Matrix<T> xMatrix, Vector<T> y, Vector<T> coefficients, 
        List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        return _numOps.Zero;
    }
}