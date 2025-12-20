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
/// - p = 8: Maximum (L8) norm (maximum absolute value)
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
/// <typeparam name="TInput">The type of input data structure.</typeparam>
/// <typeparam name="TOutput">The type of output data structure.</typeparam>
public class LpNormNormalizer<T, TInput, TOutput> : NormalizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The p parameter that defines which norm to use.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value determines which Lp-norm is calculated:
    /// - p = 1: Manhattan (L1) norm
    /// - p = 2: Euclidean (L2) norm
    /// - As p approaches infinity, the norm approaches the maximum absolute value (L8 norm)
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
    /// Initializes a new instance of the <see cref="LpNormNormalizer{T, TInput, TOutput}"/> class with the specified p value.
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
    public LpNormNormalizer(T p) : base()
    {
        _p = p;
    }

    /// <summary>
    /// Normalizes output data using the Lp-norm.
    /// </summary>
    /// <param name="data">The input data to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized data and the normalization parameters, which include the scale (norm) and p value.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method normalizes data by:
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
    public override (TOutput, NormalizationParameters<T>) NormalizeOutput(TOutput data)
    {
        if (data is Vector<T> vector)
        {
            T sum = vector.Select(x => NumOps.Power(NumOps.Abs(x), _p)).Aggregate(NumOps.Zero, NumOps.Add);
            T norm = NumOps.Power(sum, NumOps.Divide(NumOps.One, _p));
            var normalizedVector = vector.Transform(x => NumOps.Divide(x, norm));
            var parameters = new NormalizationParameters<T> { Scale = norm, P = _p, Method = NormalizationMethod.LpNorm };
            return ((TOutput)(object)normalizedVector, parameters);
        }
        else if (data is Tensor<T> tensor)
        {
            // Flatten tensor to apply Lp-norm normalization
            var flattenedTensor = tensor.ToVector();

            T sum = flattenedTensor.Select(x => NumOps.Power(NumOps.Abs(x), _p)).Aggregate(NumOps.Zero, NumOps.Add);
            T norm = NumOps.Power(sum, NumOps.Divide(NumOps.One, _p));
            var normalizedVector = flattenedTensor.Transform(x => NumOps.Divide(x, norm));

            // Convert back to tensor with the same shape
            var normalizedTensor = Tensor<T>.FromVector(normalizedVector);
            if (tensor.Shape.Length > 1)
            {
                normalizedTensor = normalizedTensor.Reshape(tensor.Shape);
            }

            var parameters = new NormalizationParameters<T> { Scale = norm, P = _p, Method = NormalizationMethod.LpNorm };
            return ((TOutput)(object)normalizedTensor, parameters);
        }

        throw new InvalidOperationException(
            $"Unsupported data type {typeof(TOutput).Name}. " +
            $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
    }

    /// <summary>
    /// Normalizes input data using the Lp-norm approach, applying normalization separately to each column.
    /// </summary>
    /// <param name="data">The input data to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized data and a list of normalization parameters for each column.
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
    /// <param name="parameters">The normalization parameters containing the scale (norm).</param>
    /// <returns>A denormalized data with values converted back to their original scale.</returns>
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
    public override TOutput Denormalize(TOutput data, NormalizationParameters<T> parameters)
    {
        if (data is Vector<T> vector)
        {
            var denormalizedVector = vector.Transform(x => NumOps.Multiply(x, parameters.Scale));
            return (TOutput)(object)denormalizedVector;
        }
        else if (data is Tensor<T> tensor)
        {
            // Flatten tensor for denormalization
            var flattenedTensor = tensor.ToVector();

            var denormalizedVector = flattenedTensor.Transform(x => NumOps.Multiply(x, parameters.Scale));

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
    public override TOutput Denormalize(TOutput coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        if (coefficients is Vector<T> vector)
        {
            var scalingFactors = xParams.Select(p => NumOps.Divide(yParams.Scale, p.Scale)).ToArray();
            var denormalizedCoefficients = vector.PointwiseMultiply(Vector<T>.FromArray(scalingFactors));
            return (TOutput)(object)denormalizedCoefficients;
        }
        else if (coefficients is Tensor<T> tensor)
        {
            // Flatten tensor for denormalization
            var flattenedTensor = tensor.ToVector();

            var scalingFactors = xParams.Select(p => NumOps.Divide(yParams.Scale, p.Scale)).ToArray();
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
    public override T Denormalize(TInput xMatrix, TOutput y, TOutput coefficients,
        List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        return NumOps.Zero;
    }
}
