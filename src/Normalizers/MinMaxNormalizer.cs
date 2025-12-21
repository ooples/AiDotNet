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
/// <para><b>For Beginners:</b> Min-max normalization is like converting grades to percentages.
/// 
/// Think of it as scaling values to fit on a scale from 0 to 100%:
/// - The minimum value becomes 0% (or 0.0)
/// - The maximum value becomes 100% (or 1.0)
/// - Everything else is spaced proportionally between these extremes
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The type of input data structure.</typeparam>
/// <typeparam name="TOutput">The type of output data structure.</typeparam>
public class MinMaxNormalizer<T, TInput, TOutput> : NormalizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MinMaxNormalizer{T, TInput, TOutput}"/> class.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up your percentage-scaling system.
    /// No additional settings are needed because the normalizer will automatically find
    /// the minimum and maximum from your actual data.
    /// </para>
    /// </remarks>
    public MinMaxNormalizer() : base()
    {
    }

    /// <summary>
    /// Normalizes output data to a standard range.
    /// </summary>
    /// <param name="data">The data to normalize.</param>
    /// <returns>A tuple containing the normalized data and the normalization parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method converts your data to a 0-1 scale.
    /// The smallest value becomes 0, the largest becomes 1, and everything else is proportionally distributed.
    /// </para>
    /// </remarks>
    public override (TOutput, NormalizationParameters<T>) NormalizeOutput(TOutput data)
    {
        if (data is Vector<T> vector)
        {
            T min = vector.Min();
            T max = vector.Max();

            var normalized = vector.Transform(x =>
                NumOps.Divide(NumOps.Subtract(x, min), NumOps.Subtract(max, min)));

            var parameters = new NormalizationParameters<T>
            {
                Method = NormalizationMethod.MinMax,
                Min = min,
                Max = max
            };

            return ((TOutput)(object)normalized, parameters);
        }
        else if (data is Tensor<T> tensor)
        {
            // Flatten tensor to apply min-max normalization
            var flattenedTensor = tensor.ToVector();
            T min = flattenedTensor.Min();
            T max = flattenedTensor.Max();

            var normalized = flattenedTensor.Transform(x =>
                NumOps.Divide(NumOps.Subtract(x, min), NumOps.Subtract(max, min)));

            // Convert back to tensor with the same shape
            var normalizedTensor = Tensor<T>.FromVector(normalized);
            if (tensor.Shape.Length > 1)
            {
                normalizedTensor = normalizedTensor.Reshape(tensor.Shape);
            }

            var parameters = new NormalizationParameters<T>
            {
                Method = NormalizationMethod.MinMax,
                Min = min,
                Max = max
            };

            return ((TOutput)(object)normalizedTensor, parameters);
        }

        throw new InvalidOperationException(
            $"Unsupported data type {typeof(TOutput).Name}. " +
            $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
    }

    /// <summary>
    /// Normalizes input data to a standard range.
    /// </summary>
    /// <param name="data">The data to normalize.</param>
    /// <returns>A tuple containing the normalized data and a list of normalization parameters for each feature.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method normalizes multiple features at once.
    /// Each feature (column) is normalized separately to its own 0-1 range.
    /// </para>
    /// </remarks>
    public override (TInput, List<NormalizationParameters<T>>) NormalizeInput(TInput data)
    {
        if (data is Matrix<T> matrix)
        {
            var normalizedColumns = new List<Vector<T>>();
            var parameters = new List<NormalizationParameters<T>>();

            for (int i = 0; i < matrix.Columns; i++)
            {
                var column = matrix.GetColumn(i);
                T min = column.Min();
                T max = column.Max();

                var normalizedColumn = column.Transform(x =>
                    NumOps.Divide(NumOps.Subtract(x, min), NumOps.Subtract(max, min)));

                normalizedColumns.Add(normalizedColumn);
                parameters.Add(new NormalizationParameters<T>
                {
                    Method = NormalizationMethod.MinMax,
                    Min = min,
                    Max = max
                });
            }

            var normalizedMatrix = Matrix<T>.FromColumnVectors(normalizedColumns);
            return ((TInput)(object)normalizedMatrix, parameters);
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
            var parameters = new List<NormalizationParameters<T>>();

            for (int i = 0; i < cols; i++)
            {
                var column = newMatrix.GetColumn(i);
                T min = column.Min();
                T max = column.Max();

                var normalizedColumn = column.Transform(x =>
                    NumOps.Divide(NumOps.Subtract(x, min), NumOps.Subtract(max, min)));

                normalizedColumns.Add(normalizedColumn);
                parameters.Add(new NormalizationParameters<T>
                {
                    Method = NormalizationMethod.MinMax,
                    Min = min,
                    Max = max
                });
            }

            // Convert back to tensor
            var normalizedMatrix = Matrix<T>.FromColumnVectors(normalizedColumns);
            var normalizedTensor = new Tensor<T>(new[] { normalizedMatrix.Rows, normalizedMatrix.Columns }, normalizedMatrix);

            return ((TInput)(object)normalizedTensor, parameters);
        }

        throw new InvalidOperationException(
            $"Unsupported data type {typeof(TInput).Name}. " +
            $"Supported types are Matrix<{typeof(T).Name}> and 2D Tensor<{typeof(T).Name}>.");
    }

    /// <summary>
    /// Reverses the normalization of data using the original normalization parameters.
    /// </summary>
    /// <param name="data">The normalized data to denormalize.</param>
    /// <param name="parameters">The normalization parameters used during normalization.</param>
    /// <returns>The denormalized data in its original scale.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method converts your scaled values back to their original range
    /// by multiplying by the original range (max - min) and adding the minimum value.
    /// </para>
    /// </remarks>
    public override TOutput Denormalize(TOutput data, NormalizationParameters<T> parameters)
    {
        if (data is Vector<T> vector)
        {
            var denormalized = vector.Transform(x =>
                NumOps.Add(
                    NumOps.Multiply(x, NumOps.Subtract(parameters.Max, parameters.Min)),
                    parameters.Min
                ));

            return (TOutput)(object)denormalized;
        }
        else if (data is Tensor<T> tensor)
        {
            // Flatten tensor for denormalization
            var flattenedTensor = tensor.ToVector();

            var denormalized = flattenedTensor.Transform(x =>
                NumOps.Add(
                    NumOps.Multiply(x, NumOps.Subtract(parameters.Max, parameters.Min)),
                    parameters.Min
                ));

            // Convert back to tensor with the same shape
            var denormalizedTensor = Tensor<T>.FromVector(denormalized);
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
    /// Denormalizes model coefficients to make them applicable to non-normalized input data.
    /// </summary>
    /// <param name="coefficients">The model coefficients from a model trained on normalized data.</param>
    /// <param name="xParams">The normalization parameters used for the input features.</param>
    /// <param name="yParams">The normalization parameters used for the target variable.</param>
    /// <returns>Denormalized coefficients for use with original, non-normalized data.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This adjusts model weights to work with your original data
    /// by scaling each coefficient by the ratio of output range to input range.
    /// </para>
    /// </remarks>
    public override TOutput Denormalize(TOutput coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        if (coefficients is Vector<T> vector)
        {
            var denormalizedCoefficients = new T[vector.Length];

            for (int i = 0; i < vector.Length; i++)
            {
                denormalizedCoefficients[i] = NumOps.Divide(
                    NumOps.Multiply(vector[i], NumOps.Subtract(yParams.Max, yParams.Min)),
                    NumOps.Subtract(xParams[i].Max, xParams[i].Min)
                );
            }

            return (TOutput)(object)Vector<T>.FromArray(denormalizedCoefficients);
        }
        else if (coefficients is Tensor<T> tensor)
        {
            // Flatten tensor for denormalization
            var flattenedTensor = tensor.ToVector();
            var denormalizedCoefficients = new T[flattenedTensor.Length];

            for (int i = 0; i < flattenedTensor.Length; i++)
            {
                denormalizedCoefficients[i] = NumOps.Divide(
                    NumOps.Multiply(flattenedTensor[i], NumOps.Subtract(yParams.Max, yParams.Min)),
                    NumOps.Subtract(xParams[i].Max, xParams[i].Min)
                );
            }

            // Convert back to tensor with the same shape
            var denormalizedVector = Vector<T>.FromArray(denormalizedCoefficients);
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
    /// <param name="yParams">The normalization parameters for the target variable.</param>
    /// <returns>The denormalized Y-intercept for use with non-normalized data.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This calculates the starting point for your model's predictions
    /// when using the original, unnormalized data.
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

        // Calculate y-intercept
        T yIntercept = yParams.Min;

        for (int i = 0; i < coefficientsVector.Length; i++)
        {
            yIntercept = NumOps.Subtract(yIntercept,
                NumOps.Divide(
                    NumOps.Multiply(
                        NumOps.Multiply(coefficientsVector[i], xParams[i].Min),
                        NumOps.Subtract(yParams.Max, yParams.Min)
                    ),
                    NumOps.Subtract(xParams[i].Max, xParams[i].Min)
                )
            );
        }

        return yIntercept;
    }
}
