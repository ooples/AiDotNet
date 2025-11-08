namespace AiDotNet.Normalizers;

/// <summary>
/// Scales features to the range [-1, 1] by dividing by the maximum absolute value.
/// </summary>
/// <remarks>
/// <para>
/// The MaxAbsScaler scales data to the range [-1, 1] by dividing each value by the maximum absolute
/// value in the dataset. Unlike min-max scaling, this method preserves the sign of values and maintains
/// zeros, making it particularly well-suited for sparse data where many values are zero.
/// </para>
/// <para>
/// The transformation formula is: scaled = value / max(|values|)
/// </para>
/// <para>
/// This method is ideal when:
/// - Working with sparse data that contains many zeros
/// - You need to preserve the sign of values (positive/negative)
/// - Your algorithm requires values in the [-1, 1] range
/// - You want a simple, efficient scaling method
/// </para>
/// <para><b>For Beginners:</b> MaxAbsScaler is like scaling a thermometer reading.
///
/// Think of it as finding the most extreme reading (hottest or coldest) and using that as your reference:
/// - Find the largest absolute value (ignoring whether it's positive or negative)
/// - Divide all values by this number
/// - Results fall between -1 and 1
/// - Zero stays zero (important for sparse data)
/// - Positive values stay positive, negative values stay negative
///
/// For example, if your data is: [-50, 0, 0, 75, 100]
/// - The maximum absolute value is 100
/// - After scaling: [-0.5, 0, 0, 0.75, 1.0]
/// - Notice how zeros are preserved and signs are maintained
///
/// This is especially useful for:
/// - Sparse matrices where most values are zero
/// - Data that already centers around zero
/// - Algorithms that work well with symmetric ranges like [-1, 1]
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The type of input data structure.</typeparam>
/// <typeparam name="TOutput">The type of output data structure.</typeparam>
public class MaxAbsScaler<T, TInput, TOutput> : NormalizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MaxAbsScaler{T, TInput, TOutput}"/> class.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up your max-abs scaling system.
    /// No additional settings are needed because the scaler will automatically find
    /// the maximum absolute value from your actual data.
    /// </para>
    /// </remarks>
    public MaxAbsScaler() : base()
    {
    }

    /// <summary>
    /// Normalizes output data by scaling to the range [-1, 1].
    /// </summary>
    /// <param name="data">The data to normalize.</param>
    /// <returns>A tuple containing the normalized data and the normalization parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method converts your data to a [-1, 1] scale.
    /// The value with the largest absolute magnitude becomes either 1 or -1 (keeping its sign),
    /// and everything else is proportionally scaled.
    /// </para>
    /// </remarks>
    public override (TOutput, NormalizationParameters<T>) NormalizeOutput(TOutput data)
    {
        if (data is Vector<T> vector)
        {
            // Find the maximum absolute value
            T maxAbs = NumOps.Zero;
            for (int i = 0; i < vector.Length; i++)
            {
                T absValue = NumOps.Abs(vector[i]);
                if (NumOps.GreaterThan(absValue, maxAbs))
                {
                    maxAbs = absValue;
                }
            }

            // Avoid division by zero
            if (NumOps.Equals(maxAbs, NumOps.Zero))
            {
                maxAbs = NumOps.One;
            }

            // Scale by dividing by max absolute value
            var normalized = vector.Transform(x => NumOps.Divide(x, maxAbs));

            var parameters = new NormalizationParameters<T>
            {
                Method = NormalizationMethod.MaxAbsScaler,
                MaxAbs = maxAbs
            };

            return ((TOutput)(object)normalized, parameters);
        }
        else if (data is Tensor<T> tensor)
        {
            // Flatten tensor to apply max-abs scaling
            var flattenedTensor = tensor.ToVector();

            // Find the maximum absolute value
            T maxAbs = NumOps.Zero;
            for (int i = 0; i < flattenedTensor.Length; i++)
            {
                T absValue = NumOps.Abs(flattenedTensor[i]);
                if (NumOps.GreaterThan(absValue, maxAbs))
                {
                    maxAbs = absValue;
                }
            }

            // Avoid division by zero
            if (NumOps.Equals(maxAbs, NumOps.Zero))
            {
                maxAbs = NumOps.One;
            }

            // Scale by dividing by max absolute value
            var normalized = flattenedTensor.Transform(x => NumOps.Divide(x, maxAbs));

            // Convert back to tensor with the same shape
            var normalizedTensor = Tensor<T>.FromVector(normalized);
            if (tensor.Shape.Length > 1)
            {
                normalizedTensor = normalizedTensor.Reshape(tensor.Shape);
            }

            var parameters = new NormalizationParameters<T>
            {
                Method = NormalizationMethod.MaxAbsScaler,
                MaxAbs = maxAbs
            };

            return ((TOutput)(object)normalizedTensor, parameters);
        }

        throw new InvalidOperationException(
            $"Unsupported data type {typeof(TOutput).Name}. " +
            $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
    }

    /// <summary>
    /// Normalizes input data by scaling each feature independently to the range [-1, 1].
    /// </summary>
    /// <param name="data">The data to normalize.</param>
    /// <returns>A tuple containing the normalized data and a list of normalization parameters for each feature.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method normalizes multiple features at once.
    /// Each feature (column) is normalized separately to its own [-1, 1] range based on
    /// its maximum absolute value.
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

                // Find the maximum absolute value for this column
                T maxAbs = NumOps.Zero;
                for (int j = 0; j < column.Length; j++)
                {
                    T absValue = NumOps.Abs(column[j]);
                    if (NumOps.GreaterThan(absValue, maxAbs))
                    {
                        maxAbs = absValue;
                    }
                }

                // Avoid division by zero
                if (NumOps.Equals(maxAbs, NumOps.Zero))
                {
                    maxAbs = NumOps.One;
                }

                // Scale the column
                var normalizedColumn = column.Transform(x => NumOps.Divide(x, maxAbs));

                normalizedColumns.Add(normalizedColumn);
                parameters.Add(new NormalizationParameters<T>
                {
                    Method = NormalizationMethod.MaxAbsScaler,
                    MaxAbs = maxAbs
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

                // Find the maximum absolute value for this column
                T maxAbs = NumOps.Zero;
                for (int j = 0; j < column.Length; j++)
                {
                    T absValue = NumOps.Abs(column[j]);
                    if (NumOps.GreaterThan(absValue, maxAbs))
                    {
                        maxAbs = absValue;
                    }
                }

                // Avoid division by zero
                if (NumOps.Equals(maxAbs, NumOps.Zero))
                {
                    maxAbs = NumOps.One;
                }

                // Scale the column
                var normalizedColumn = column.Transform(x => NumOps.Divide(x, maxAbs));

                normalizedColumns.Add(normalizedColumn);
                parameters.Add(new NormalizationParameters<T>
                {
                    Method = NormalizationMethod.MaxAbsScaler,
                    MaxAbs = maxAbs
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
    /// by multiplying by the original maximum absolute value.
    /// </para>
    /// </remarks>
    public override TOutput Denormalize(TOutput data, NormalizationParameters<T> parameters)
    {
        if (data is Vector<T> vector)
        {
            var denormalized = vector.Transform(x => NumOps.Multiply(x, parameters.MaxAbs));
            return (TOutput)(object)denormalized;
        }
        else if (data is Tensor<T> tensor)
        {
            // Flatten tensor for denormalization
            var flattenedTensor = tensor.ToVector();
            var denormalized = flattenedTensor.Transform(x => NumOps.Multiply(x, parameters.MaxAbs));

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
    /// by scaling each coefficient by the ratio of output max-abs to input max-abs.
    /// </para>
    /// </remarks>
    public override TOutput Denormalize(TOutput coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        if (coefficients is Vector<T> vector)
        {
            var denormalizedCoefficients = new T[vector.Length];

            for (int i = 0; i < vector.Length; i++)
            {
                // coefficient_denorm = coefficient_norm * (maxAbs_y / maxAbs_x)
                denormalizedCoefficients[i] = NumOps.Divide(
                    NumOps.Multiply(vector[i], yParams.MaxAbs),
                    xParams[i].MaxAbs
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
                    NumOps.Multiply(flattenedTensor[i], yParams.MaxAbs),
                    xParams[i].MaxAbs
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
        // For MaxAbsScaler, the intercept is zero because the scaling is centered at zero
        // and doesn't involve any shifting. The transformation is purely multiplicative.
        // Since x_scaled = x / maxAbs_x and y_scaled = y / maxAbs_y,
        // and the model is: y_scaled = intercept_scaled + sum(coef_scaled * x_scaled)
        // When converted back: y = intercept * maxAbs_y + sum(coef * x)
        // For a model trained on scaled data with no shift, intercept should be 0
        return NumOps.Zero;
    }
}
