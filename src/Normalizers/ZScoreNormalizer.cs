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
/// <typeparam name="TInput">The type of input data structure.</typeparam>
/// <typeparam name="TOutput">The type of output data structure.</typeparam>
public class ZScoreNormalizer<T, TInput, TOutput> : NormalizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ZScoreNormalizer{T, TInput, TOutput}"/> class.
    /// </summary>
    public ZScoreNormalizer() : base()
    {
    }

    /// <summary>
    /// Normalizes output data using Z-Score normalization.
    /// </summary>
    /// <param name="data">The output data to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized data and the normalization parameters, which include the mean and standard deviation.
    /// </returns>
    public override (TOutput, NormalizationParameters<T>) NormalizeOutput(TOutput data)
    {
        if (data is Vector<T> vector)
        {
            T mean = StatisticsHelper<T>.CalculateMean(vector);
            T variance = StatisticsHelper<T>.CalculateVariance(vector, mean);
            T stdDev = NumOps.Sqrt(variance);

            Vector<T> normalizedVector = vector.Transform(x =>
                NumOps.Divide(NumOps.Subtract(x, mean), stdDev)
            );

            return ((TOutput)(object)normalizedVector, new NormalizationParameters<T>
            {
                Method = NormalizationMethod.ZScore,
                Mean = mean,
                StdDev = stdDev
            });
        }
        else if (data is Tensor<T> tensor)
        {
            // For tensors, calculate mean and stddev from flattened data
            var flattenedVector = tensor.ToVector();
            T mean = StatisticsHelper<T>.CalculateMean(flattenedVector);
            T variance = StatisticsHelper<T>.CalculateVariance(flattenedVector, mean);
            T stdDev = NumOps.Sqrt(variance);

            // Create normalized tensor with the same shape
            var normalizedTensor = tensor.Transform(x =>
                NumOps.Divide(NumOps.Subtract(x, mean), stdDev)
            );

            return ((TOutput)(object)normalizedTensor, new NormalizationParameters<T>
            {
                Method = NormalizationMethod.ZScore,
                Mean = mean,
                StdDev = stdDev
            });
        }

        throw new InvalidOperationException(
            $"Unsupported output type {typeof(TOutput).Name} for Z-Score normalization. " +
            $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
    }

    /// <summary>
    /// Normalizes input data using Z-Score normalization, applying normalization separately to each column.
    /// </summary>
    /// <param name="data">The input data to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized data and a list of normalization parameters for each column.
    /// </returns>
    public override (TInput, List<NormalizationParameters<T>>) NormalizeInput(TInput data)
    {
        if (data is Matrix<T> matrix)
        {
            var normalizedColumns = new List<Vector<T>>();
            var parameters = new List<NormalizationParameters<T>>();

            for (int i = 0; i < matrix.Columns; i++)
            {
                var column = matrix.GetColumn(i);
                var (normalizedColumn, columnParams) = NormalizeVector(column);
                normalizedColumns.Add(normalizedColumn);
                parameters.Add(columnParams);
            }

            return ((TInput)(object)Matrix<T>.FromColumnVectors(normalizedColumns), parameters);
        }
        else if (data is Tensor<T> tensor)
        {
            // For 2D tensor, normalize each column (dimension 1)
            if (tensor.Shape.Length != 2)
            {
                throw new InvalidOperationException(
                    "Z-Score normalization for tensors requires a 2D tensor (matrix-like).");
            }

            var parameters = new List<NormalizationParameters<T>>();
            var normalizedTensor = new Tensor<T>(tensor.Shape);

            // Process each column
            for (int i = 0; i < tensor.Shape[1]; i++)
            {
                // Extract column as vector
                var column = new Vector<T>(tensor.Shape[0]);
                for (int j = 0; j < tensor.Shape[0]; j++)
                {
                    column[j] = tensor[j, i];
                }

                // Normalize column
                var (normalizedColumn, columnParams) = NormalizeVector(column);

                // Put normalized column back into tensor
                for (int j = 0; j < tensor.Shape[0]; j++)
                {
                    normalizedTensor[j, i] = normalizedColumn[j];
                }

                parameters.Add(columnParams);
            }

            return ((TInput)(object)normalizedTensor, parameters);
        }

        throw new InvalidOperationException(
            $"Unsupported input type {typeof(TInput).Name} for Z-Score normalization. " +
            $"Supported types are Matrix<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
    }

    /// <summary>
    /// Helper method to normalize a vector using Z-Score normalization.
    /// </summary>
    private (Vector<T>, NormalizationParameters<T>) NormalizeVector(Vector<T> vector)
    {
        T mean = StatisticsHelper<T>.CalculateMean(vector);
        T variance = StatisticsHelper<T>.CalculateVariance(vector, mean);
        T stdDev = NumOps.Sqrt(variance);

        Vector<T> normalizedVector = vector.Transform(x =>
            NumOps.Divide(NumOps.Subtract(x, mean), stdDev)
        );

        return (normalizedVector, new NormalizationParameters<T>
        {
            Method = NormalizationMethod.ZScore,
            Mean = mean,
            StdDev = stdDev
        });
    }

    /// <summary>
    /// Denormalizes data using the provided normalization parameters.
    /// </summary>
    /// <param name="data">The normalized data to denormalize.</param>
    /// <param name="parameters">The normalization parameters containing the mean and standard deviation.</param>
    /// <returns>A denormalized data with values converted back to their original scale.</returns>
    public override TOutput Denormalize(TOutput data, NormalizationParameters<T> parameters)
    {
        if (data is Vector<T> vector)
        {
            var denormalized = vector.Transform(x =>
                NumOps.Add(NumOps.Multiply(x, parameters.StdDev), parameters.Mean)
            );

            return (TOutput)(object)denormalized;
        }
        else if (data is Tensor<T> tensor)
        {
            var denormalizedTensor = tensor.Transform(x =>
                NumOps.Add(NumOps.Multiply(x, parameters.StdDev), parameters.Mean)
            );

            return (TOutput)(object)denormalizedTensor;
        }

        throw new InvalidOperationException(
            $"Unsupported output type {typeof(TOutput).Name} for Z-Score denormalization. " +
            $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
    }

    /// <summary>
    /// Denormalizes coefficients from a regression model that was trained on Z-Score normalized data.
    /// </summary>
    /// <param name="coefficients">The regression coefficients to denormalize.</param>
    /// <param name="xParams">The normalization parameters for the input features.</param>
    /// <param name="yParams">The normalization parameters for the output variable.</param>
    /// <returns>Denormalized coefficients that can be applied to original, unnormalized data.</returns>
    public override TOutput Denormalize(TOutput coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        if (coefficients is Vector<T> coefs)
        {
            var denormalizedCoefs = coefs.PointwiseMultiply(
                Vector<T>.FromEnumerable(xParams.Select(p =>
                    NumOps.Divide(p.StdDev, yParams.StdDev)
                ))
            );

            return (TOutput)(object)denormalizedCoefs;
        }
        else if (coefficients is Tensor<T> tensor)
        {
            // For tensor coefficients, we flatten them, denormalize, and then reshape
            var flattenedVector = tensor.ToVector();

            if (flattenedVector.Length != xParams.Count)
            {
                throw new InvalidOperationException(
                    "Number of coefficients does not match the number of input features.");
            }

            var denormalizedCoefs = flattenedVector.PointwiseMultiply(
                Vector<T>.FromEnumerable(xParams.Select(p =>
                    NumOps.Divide(p.StdDev, yParams.StdDev)
                ))
            );

            var result = Tensor<T>.FromVector(denormalizedCoefs);

            // If the original tensor had a specific shape, try to reshape the result
            if (tensor.Shape.Length > 1)
            {
                result = result.Reshape(tensor.Shape);
            }

            return (TOutput)(object)result;
        }

        throw new InvalidOperationException(
            $"Unsupported coefficient type {typeof(TOutput).Name} for Z-Score denormalization. " +
            $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
    }

    /// <summary>
    /// Denormalizes the y-intercept from a regression model that was trained on Z-Score normalized data.
    /// </summary>
    /// <param name="xMatrix">The original input feature matrix.</param>
    /// <param name="y">The original output vector.</param>
    /// <param name="coefficients">The regression coefficients.</param>
    /// <param name="xParams">The normalization parameters for the input features.</param>
    /// <param name="yParams">The normalization parameters for the output variable.</param>
    /// <returns>A denormalized y-intercept that can be used with original, unnormalized data.</returns>
    public override T Denormalize(TInput xMatrix, TOutput y, TOutput coefficients,
        List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        if (coefficients is Vector<T> coefs)
        {
            T yMean = yParams.Mean;
            var xMeans = Vector<T>.FromEnumerable(xParams.Select(p => p.Mean));
            var denormalizedCoefs = coefs.PointwiseMultiply(
                Vector<T>.FromEnumerable(xParams.Select(p =>
                    NumOps.Divide(p.StdDev, yParams.StdDev)
                ))
            );

            return NumOps.Subtract(yMean, xMeans.DotProduct(denormalizedCoefs));
        }

        // Default fallback if coefficient type is not a vector
        return NumOps.Zero;
    }
}
