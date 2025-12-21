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
/// <typeparam name="TInput">The type of input data structure.</typeparam>
/// <typeparam name="TOutput">The type of output data structure.</typeparam>
public class LogNormalizer<T, TInput, TOutput> : NormalizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="LogNormalizer{T, TInput, TOutput}"/> class.
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
    public LogNormalizer() : base()
    {
    }

    /// <summary>
    /// Normalizes output data using the logarithmic approach.
    /// </summary>
    /// <param name="data">The input data to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized data and the normalization parameters, which include the minimum and maximum values and the shift.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method normalizes data by:
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
    public override (TOutput, NormalizationParameters<T>) NormalizeOutput(TOutput data)
    {
        if (data is Vector<T> vector)
        {
            T min = vector.Min();
            T max = vector.Max();
            T shift = NumOps.GreaterThan(min, NumOps.Zero) ? NumOps.Zero : NumOps.Add(NumOps.Negate(min), NumOps.One);
            var normalizedVector = vector.Transform(x =>
            {
                T shiftedValue = NumOps.Add(x, shift);
                return NumOps.GreaterThan(shiftedValue, NumOps.Zero)
                    ? NumOps.Divide(
                        NumOps.Subtract(NumOps.Log(shiftedValue), NumOps.Log(NumOps.Add(min, shift))),
                        NumOps.Subtract(NumOps.Log(NumOps.Add(max, shift)), NumOps.Log(NumOps.Add(min, shift))))
                    : NumOps.Zero;
            });
            var parameters = new NormalizationParameters<T>
            {
                Method = NormalizationMethod.Log,
                Min = min,
                Max = max,
                Shift = shift
            };
            return ((TOutput)(object)normalizedVector, parameters);
        }
        else if (data is Tensor<T> tensor)
        {
            // Flatten tensor to apply log normalization
            var flattenedTensor = tensor.ToVector();

            T min = flattenedTensor.Min();
            T max = flattenedTensor.Max();
            T shift = NumOps.GreaterThan(min, NumOps.Zero) ? NumOps.Zero : NumOps.Add(NumOps.Negate(min), NumOps.One);

            var normalizedVector = flattenedTensor.Transform(x =>
            {
                T shiftedValue = NumOps.Add(x, shift);
                return NumOps.GreaterThan(shiftedValue, NumOps.Zero)
                    ? NumOps.Divide(
                        NumOps.Subtract(NumOps.Log(shiftedValue), NumOps.Log(NumOps.Add(min, shift))),
                        NumOps.Subtract(NumOps.Log(NumOps.Add(max, shift)), NumOps.Log(NumOps.Add(min, shift))))
                    : NumOps.Zero;
            });

            // Convert back to tensor with the same shape
            var normalizedTensor = Tensor<T>.FromVector(normalizedVector);
            if (tensor.Shape.Length > 1)
            {
                normalizedTensor = normalizedTensor.Reshape(tensor.Shape);
            }

            var parameters = new NormalizationParameters<T>
            {
                Method = NormalizationMethod.Log,
                Min = min,
                Max = max,
                Shift = shift
            };

            return ((TOutput)(object)normalizedTensor, parameters);
        }

        throw new InvalidOperationException(
            $"Unsupported data type {typeof(TOutput).Name}. " +
            $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
    }

    /// <summary>
    /// Normalizes input data using the logarithmic approach, applying normalization separately to each column.
    /// </summary>
    /// <param name="data">The input data to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized data and a list of normalization parameters for each column.
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
    /// <param name="parameters">The normalization parameters containing min, max, and shift values.</param>
    /// <returns>A denormalized data with values converted back to their original scale.</returns>
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
    public override TOutput Denormalize(TOutput data, NormalizationParameters<T> parameters)
    {
        if (data is Vector<T> vector)
        {
            var denormalizedVector = vector.Transform(x =>
            {
                T expValue = NumOps.Exp(NumOps.Add(
                    NumOps.Multiply(x, NumOps.Subtract(
                        NumOps.Log(NumOps.Add(parameters.Max, parameters.Shift)),
                        NumOps.Log(NumOps.Add(parameters.Min, parameters.Shift)))),
                    NumOps.Log(NumOps.Add(parameters.Min, parameters.Shift))));
                return NumOps.Subtract(expValue, parameters.Shift);
            });

            return (TOutput)(object)denormalizedVector;
        }
        else if (data is Tensor<T> tensor)
        {
            // Flatten tensor for denormalization
            var flattenedTensor = tensor.ToVector();

            var denormalizedVector = flattenedTensor.Transform(x =>
            {
                T expValue = NumOps.Exp(NumOps.Add(
                    NumOps.Multiply(x, NumOps.Subtract(
                        NumOps.Log(NumOps.Add(parameters.Max, parameters.Shift)),
                        NumOps.Log(NumOps.Add(parameters.Min, parameters.Shift)))),
                    NumOps.Log(NumOps.Add(parameters.Min, parameters.Shift))));
                return NumOps.Subtract(expValue, parameters.Shift);
            });

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
    public override TOutput Denormalize(TOutput coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        if (coefficients is Vector<T> vector)
        {
            var denormalizedCoefficients = vector.PointwiseMultiply(Vector<T>.FromEnumerable(
                xParams.Select(p => NumOps.Divide(
                    NumOps.Subtract(NumOps.Log(NumOps.Add(yParams.Max, yParams.Shift)), NumOps.Log(NumOps.Add(yParams.Min, yParams.Shift))),
                    NumOps.Subtract(NumOps.Log(NumOps.Add(p.Max, p.Shift)), NumOps.Log(NumOps.Add(p.Min, p.Shift)))))));

            return (TOutput)(object)denormalizedCoefficients;
        }
        else if (coefficients is Tensor<T> tensor)
        {
            // Flatten tensor for denormalization
            var flattenedTensor = tensor.ToVector();

            var denormalizedVector = flattenedTensor.PointwiseMultiply(Vector<T>.FromEnumerable(
                xParams.Select(p => NumOps.Divide(
                    NumOps.Subtract(NumOps.Log(NumOps.Add(yParams.Max, yParams.Shift)), NumOps.Log(NumOps.Add(yParams.Min, yParams.Shift))),
                    NumOps.Subtract(NumOps.Log(NumOps.Add(p.Max, p.Shift)), NumOps.Log(NumOps.Add(p.Min, p.Shift)))))));

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
    public override T Denormalize(TInput xMatrix, TOutput y, TOutput coefficients,
        List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        // Extract vectors from inputs
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

        Vector<T> yVector;
        if (y is Vector<T> yVec)
        {
            yVector = yVec;
        }
        else if (y is Tensor<T> yTensor)
        {
            yVector = yTensor.ToVector();
        }
        else
        {
            throw new InvalidOperationException(
                $"Unsupported y type {typeof(TOutput).Name}. " +
                $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
        }

        // Get columns from xMatrix
        IEnumerable<Vector<T>> columns;
        if (xMatrix is Matrix<T> matrix)
        {
            columns = matrix.GetColumns();
        }
        else if (xMatrix is Tensor<T> xTensor && xTensor.Shape.Length == 2)
        {
            var rows = xTensor.Shape[0];
            var cols = xTensor.Shape[1];
            var newMatrix = new Matrix<T>(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    newMatrix[i, j] = xTensor[i, j];
                }
            }

            columns = newMatrix.GetColumns();
        }
        else
        {
            throw new InvalidOperationException(
                $"Unsupported xMatrix type {typeof(TInput).Name}. " +
                $"Supported types are Matrix<{typeof(T).Name}> and 2D Tensor<{typeof(T).Name}>.");
        }

        // Calculate denormalized intercept
        TOutput denormalizedCoefficients = Denormalize(coefficients, xParams, yParams);
        Vector<T> denormalizedCoefficientsVector;

        // Convert to Vector<T> based on the type
        if (denormalizedCoefficients is Vector<T> vector2)
        {
            denormalizedCoefficientsVector = vector2;
        }
        else if (denormalizedCoefficients is Tensor<T> tensor)
        {
            denormalizedCoefficientsVector = tensor.ToVector();
        }
        else
        {
            throw new InvalidOperationException(
                $"Unsupported denormalized coefficients type {typeof(TOutput).Name}. " +
                $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
        }

        var meanX = Vector<T>.FromEnumerable(columns.Select(col => col.Mean()));
        var meanY = yVector.Mean();

        // Start with mean of target variable
        T intercept = meanY;

        // For log normalization, we need to adjust for the effect of each coefficient on the mean values
        for (int i = 0; i < denormalizedCoefficientsVector.Length; i++)
        {
            // For logarithmic scaling, we need to apply logarithmic transformation to the mean values
            // before using them in the intercept calculation
            T logMeanX = NumOps.GreaterThan(meanX[i], NumOps.Zero)
                ? NumOps.Log(NumOps.Add(meanX[i], xParams[i].Shift))
                : NumOps.Zero;

            intercept = NumOps.Subtract(
                intercept,
                NumOps.Multiply(denormalizedCoefficientsVector[i], logMeanX)
            );
        }

        return intercept;
    }
}
