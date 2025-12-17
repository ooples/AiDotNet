namespace AiDotNet.Normalizers;

/// <summary>
/// Normalizes data by taking the logarithm and then applying mean-variance normalization.
/// </summary>
/// <remarks>
/// <para>
/// The LogMeanVarianceNormalizer combines logarithmic transformation with standardization to handle data that
/// spans multiple orders of magnitude or has a skewed distribution. It first applies a logarithm transformation
/// to compress the range of values, then centers and scales the transformed data based on the mean and
/// standard deviation.
/// </para>
/// <para>
/// The transformation occurs in several steps:
/// 1. Shift values to ensure all are positive (if needed)
/// 2. Apply a logarithm transformation
/// 3. Compute the mean and standard deviation of the log-transformed data
/// 4. Standardize the log-transformed data by subtracting the mean and dividing by the standard deviation
/// </para>
/// <para>
/// This normalization method is particularly useful for:
/// - Data with exponential growth or decay
/// - Values that span several orders of magnitude
/// - Financial data, population growth, or other naturally log-distributed phenomena
/// - Dealing with data that has positive skew
/// </para>
/// <para><b>For Beginners:</b> Log-mean-variance normalization is like a zoom lens that works better for exponential data.
/// 
/// Think of this as a special way to normalize data when values are extremely spread out:
/// - Regular data normalization works well when values are somewhat evenly spread
/// - But when some values are MUCH larger than others (like 1, 10, 100, 1000), regular normalization can lose information
/// - This method first uses logarithms to "compress" these wide-ranging values
/// - Then it applies standard normalization techniques to the compressed data
/// 
/// For example, with population data of cities:
/// - Original populations: [5,000, 50,000, 500,000, 5,000,000]
/// - After taking logarithms: [3.7, 4.7, 5.7, 6.7] (log10 values)
/// - After standardization: [-1.5, -0.5, 0.5, 1.5]
/// 
/// This is particularly useful for data where ratios (like "10 times bigger") are more meaningful 
/// than absolute differences (like "9,000 more units").
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The type of input data structure.</typeparam>
/// <typeparam name="TOutput">The type of output data structure.</typeparam>
public class LogMeanVarianceNormalizer<T, TInput, TOutput> : NormalizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// A small value added to prevent numerical issues with logarithms of zero or very small numbers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This epsilon value is added to inputs before taking logarithms to avoid errors when processing
    /// zeros or very small values. It's typically set to a very small positive number (like 1e-10)
    /// that doesn't significantly affect the results but prevents numerical instability.
    /// </para>
    /// <para><b>For Beginners:</b> This is a tiny safety cushion for mathematical operations.
    /// 
    /// Logarithms can't handle zero or negative numbers, so:
    /// - This small value is added to make sure we never try to take the logarithm of zero
    /// - It's so small (0.0000000001) that it doesn't meaningfully change your data
    /// - It's like adding a drop of water to an empty cup just to make sure it's not completely dry
    /// 
    /// This prevents errors in the mathematics while preserving the meaning of your data.
    /// </para>
    /// </remarks>
    private readonly T _epsilon;

    /// <summary>
    /// Initializes a new instance of the <see cref="LogMeanVarianceNormalizer{T, TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="numOps">Optional numeric operations provider. If not provided, a default one will be created.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new LogMeanVarianceNormalizer and initializes the numeric operations
    /// provider for the specified type T, either using the provided one or creating a default.
    /// It also initializes the epsilon value used to prevent issues with logarithms of very small numbers.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up your logarithmic normalization system.
    /// 
    /// When you create a new LogMeanVarianceNormalizer:
    /// - It prepares the mathematical tools needed for the normalization
    /// - It sets up a tiny safety value (epsilon) to prevent mathematical errors
    /// - You can optionally provide your own calculator (numOps), or it will use a default one
    /// 
    /// It's like preparing a special camera with logarithmic zoom before taking pictures of objects
    /// that range from microscopic to gigantic in size.
    /// </para>
    /// </remarks>
    public LogMeanVarianceNormalizer() : base()
    {
        _epsilon = NumOps.FromDouble(1e-10);
    }

    /// <summary>
    /// Normalizes output data using the log-mean-variance approach.
    /// </summary>
    /// <param name="data">The input data to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized data and the normalization parameters, which include the shift, mean, and standard deviation.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method normalizes data by:
    /// 1. Shifting values if necessary to ensure all are positive (required for logarithm)
    /// 2. Applying a logarithm transformation to compress the range
    /// 3. Computing the mean and standard deviation of the log-transformed values
    /// 4. Standardizing the log-transformed values using the mean and standard deviation
    /// 
    /// The result is a vector where even widely varying values are well-distributed.
    /// Any potential NaN values (from logarithm issues) are replaced with zeros.
    /// 
    /// The normalization parameters include the shift, mean, and standard deviation, which are needed for denormalization.
    /// </para>
    /// <para><b>For Beginners:</b> This method transforms your data to highlight patterns in highly varied values.
    /// 
    /// The process works like this:
    /// 1. First, it adjusts your data to make sure all values are positive (required for logarithms)
    /// 2. Then, it takes the logarithm of each value, which compresses very large differences
    /// 3. Next, it calculates the average and spread of these logarithmic values
    /// 4. Finally, it standardizes the values by centering them around zero and scaling by the spread
    /// 5. Any problem values are cleaned up by replacing them with zeros
    /// 
    /// After normalization:
    /// - Values that were extremely spread out are now more evenly distributed
    /// - Patterns and relationships become more visible when plotted or analyzed
    /// - The data is centered around zero with most values falling between -3 and 3
    /// 
    /// For example, with exponential data [2, 20, 200, 2000]:
    /// - After logarithmic transformation: [0.69, 3.0, 5.3, 7.6]
    /// - After standardization, something like: [-1.5, -0.5, 0.5, 1.5]
    /// 
    /// This transformation preserves the important relationships in highly varied data.
    /// </para>
    /// </remarks>
    public override (TOutput, NormalizationParameters<T>) NormalizeOutput(TOutput data)
    {
        if (data is Vector<T> vector)
        {
            T minValue = vector.Min();
            T shift = NumOps.GreaterThan(minValue, NumOps.Zero)
                ? NumOps.Zero
                : NumOps.Add(NumOps.Add(NumOps.Negate(minValue), NumOps.One), _epsilon);
            var logVector = vector.Transform(x => NumOps.Log(NumOps.Add(x, shift)));
            T mean = logVector.Average();

            T variance = logVector.Select(x => NumOps.Power(NumOps.Subtract(x, mean), NumOps.FromDouble(2))).Average();
            T stdDev = NumOps.Sqrt(NumOps.GreaterThan(variance, _epsilon) ? variance : _epsilon);
            var normalizedVector = logVector.Transform(x => NumOps.Divide(NumOps.Subtract(x, mean), stdDev));
            normalizedVector = normalizedVector.Transform(x => NumOps.IsNaN(x) ? NumOps.Zero : x);
            var parameters = new NormalizationParameters<T>(NumOps)
            {
                Method = NormalizationMethod.LogMeanVariance,
                Shift = shift,
                Mean = mean,
                StdDev = stdDev
            };
            return ((TOutput)(object)normalizedVector, parameters);
        }
        else if (data is Tensor<T> tensor)
        {
            // Flatten tensor to apply log-mean-variance normalization
            var flattenedTensor = tensor.ToVector();

            T minValue = flattenedTensor.Min();
            T shift = NumOps.GreaterThan(minValue, NumOps.Zero)
                ? NumOps.Zero
                : NumOps.Add(NumOps.Add(NumOps.Negate(minValue), NumOps.One), _epsilon);

            var logVector = flattenedTensor.Transform(x => NumOps.Log(NumOps.Add(x, shift)));
            T mean = logVector.Average();

            T variance = logVector.Select(x => NumOps.Power(NumOps.Subtract(x, mean), NumOps.FromDouble(2))).Average();
            T stdDev = NumOps.Sqrt(NumOps.GreaterThan(variance, _epsilon) ? variance : _epsilon);
            var normalizedVector = logVector.Transform(x => NumOps.Divide(NumOps.Subtract(x, mean), stdDev));
            normalizedVector = normalizedVector.Transform(x => NumOps.IsNaN(x) ? NumOps.Zero : x);

            // Convert back to tensor with the same shape
            var normalizedTensor = Tensor<T>.FromVector(normalizedVector);
            if (tensor.Shape.Length > 1)
            {
                normalizedTensor = normalizedTensor.Reshape(tensor.Shape);
            }

            var parameters = new NormalizationParameters<T>(NumOps)
            {
                Method = NormalizationMethod.LogMeanVariance,
                Shift = shift,
                Mean = mean,
                StdDev = stdDev
            };

            return ((TOutput)(object)normalizedTensor, parameters);
        }

        throw new InvalidOperationException(
            $"Unsupported data type {typeof(TOutput).Name}. " +
            $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
    }

    /// <summary>
    /// Normalizes input data using the log-mean-variance approach, applying normalization separately to each column.
    /// </summary>
    /// <param name="data">The input data to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized data and a list of normalization parameters for each column.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method normalizes each column of the matrix independently using the log-mean-variance approach.
    /// It treats each column as a separate feature that needs its own shift, mean, and standard deviation,
    /// since different features may have different distributions and scales.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies logarithmic normalization to a table of data, one column at a time.
    /// 
    /// When working with a table (matrix) of data:
    /// - Each column might represent a different type of information (population, GDP, area, etc.)
    /// - Each column needs its own normalization because the ranges and distributions differ
    /// - This method processes each column separately using the vector normalization method
    /// 
    /// For example, with a table of country statistics:
    /// - Column 1 (population) might range from thousands to billions
    /// - Column 2 (GDP) might range from millions to trillions
    /// - Each column gets its own appropriate logarithmic transformation
    /// 
    /// The method returns:
    /// - A new table with all values logarithmically normalized
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
    /// <param name="parameters">The normalization parameters containing shift, mean, and standard deviation.</param>
    /// <returns>A denormalized data with values converted back to their original scale.</returns>
    /// <remarks>
    /// <para>
    /// This method reverses the log-mean-variance normalization by:
    /// 1. Reversing the standardization step by multiplying by the standard deviation and adding the mean
    /// 2. Reversing the logarithm transformation by applying exponentiation
    /// 3. Reversing the initial shift by subtracting the shift value
    /// 
    /// This series of steps restores the values to their original scale and distribution.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts your normalized values back to their original scale.
    /// 
    /// The process reverses all the normalization steps in reverse order:
    /// 1. First, it multiplies by the standard deviation and adds the mean (undoing the standardization)
    /// 2. Then, it takes the exponential of each value (undoing the logarithm)
    /// 3. Finally, it subtracts the shift value that was added initially
    /// 
    /// For example, if your normalized data was [-1.5, -0.5, 0.5, 1.5]:
    /// - After undoing standardization: [0.69, 3.0, 5.3, 7.6]
    /// - After taking exponentials: [2, 20, 200, 2000]
    /// - After removing any shift: back to the original values
    /// 
    /// This allows you to recover the original data after performing calculations or analysis
    /// on the normalized values.
    /// </para>
    /// </remarks>
    public override TOutput Denormalize(TOutput data, NormalizationParameters<T> parameters)
    {
        if (data is Vector<T> vector)
        {
            var denormalizedVector = vector
                .Multiply(parameters.StdDev)
                .Add(parameters.Mean)
                .Transform(x => NumOps.Subtract(NumOps.Exp(x), parameters.Shift));

            return (TOutput)(object)denormalizedVector;
        }
        else if (data is Tensor<T> tensor)
        {
            // Flatten tensor for denormalization
            var flattenedTensor = tensor.ToVector();

            var denormalizedVector = flattenedTensor
                .Multiply(parameters.StdDev)
                .Add(parameters.Mean)
                .Transform(x => NumOps.Subtract(NumOps.Exp(x), parameters.Shift));

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
    /// Denormalizes coefficients from a regression model that was trained on log-mean-variance normalized data.
    /// </summary>
    /// <param name="coefficients">The regression coefficients to denormalize.</param>
    /// <param name="xParams">The normalization parameters for the input features.</param>
    /// <param name="yParams">The normalization parameters for the output variable.</param>
    /// <returns>Denormalized coefficients that can be applied to log-transformed original data.</returns>
    /// <remarks>
    /// <para>
    /// This method adjusts regression coefficients to work with log-transformed original data.
    /// For log-mean-variance normalization, this involves scaling each coefficient by the ratio of
    /// the output standard deviation to the corresponding input standard deviation.
    /// 
    /// Note that these denormalized coefficients should be applied to log-transformed inputs, not to
    /// the raw original data, due to the non-linear nature of the logarithm transformation.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts prediction model weights for use with log-transformed original values.
    /// 
    /// When you build a model using logarithmically normalized data:
    /// - The model learns weights (coefficients) based on the standardized log values
    /// - To use this model with log-transformed (but not standardized) data, you need to adjust these weights
    /// 
    /// This method performs that adjustment by:
    /// - Calculating how much each input feature's log values were scaled
    /// - Calculating how much the output's log values were scaled
    /// - Adjusting each coefficient by the ratio of these scaling factors
    /// 
    /// Important: Unlike simpler normalizers, you can't directly use these coefficients with the original raw data.
    /// Because logarithm transformation is non-linear, you would need to:
    /// 1. Take the logarithm of your input data
    /// 2. Apply these denormalized coefficients
    /// 3. Exponentiate the result to get predictions in the original scale
    /// 
    /// This reflects the complexity of working with logarithmic transformations.
    /// </para>
    /// </remarks>
    public override TOutput Denormalize(TOutput coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        if (coefficients is Vector<T> vector)
        {
            var denormalizedCoefficients = vector.PointwiseMultiply(
                Vector<T>.FromArray(xParams.Select(p =>
                    NumOps.Divide(yParams.StdDev, NumOps.GreaterThan(p.StdDev, _epsilon) ? p.StdDev : _epsilon)
                ).ToArray())
            );

            return (TOutput)(object)denormalizedCoefficients;
        }
        else if (coefficients is Tensor<T> tensor)
        {
            // Flatten tensor for denormalization
            var flattenedTensor = tensor.ToVector();

            var denormalizedVector = flattenedTensor.PointwiseMultiply(
                Vector<T>.FromArray(xParams.Select(p =>
                    NumOps.Divide(yParams.StdDev, NumOps.GreaterThan(p.StdDev, _epsilon) ? p.StdDev : _epsilon)
                ).ToArray())
            );

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
    /// This method calculates the appropriate y-intercept for a model trained on normalized data
    /// but applied to log-transformed (but not standardized) original data. The calculation accounts
    /// for the shifts in both the input features and the output variable that occurred during normalization.
    /// 
    /// Like with coefficient denormalization, this y-intercept should be used with log-transformed inputs,
    /// not with the raw original data, due to the non-linear nature of the logarithm transformation.
    /// </para>
    /// <para><b>For Beginners:</b> This method determines the correct starting point for predictions with log-transformed data.
    /// 
    /// In a prediction model, the y-intercept is the baseline value:
    /// - It's the predicted output when all inputs are zero
    /// - When using logarithmic normalization, this baseline needs special handling
    /// 
    /// The calculation is complex because:
    /// - The logarithmic transformation is non-linear
    /// - Each feature has its own mean and standard deviation in log space
    /// - The output also has its own mean and standard deviation in log space
    /// 
    /// The method calculates the correct intercept to use with log-transformed data:
    /// 1. It accounts for the means of all log-transformed features
    /// 2. It adjusts for the scaling factors used during normalization
    /// 3. It applies the exponential function and removes the shift to get back to the original scale
    /// 
    /// As with the denormalized coefficients, this intercept is meant to be used in a model that:
    /// - Takes the logarithm of input features
    /// - Applies the denormalized coefficients and intercept
    /// - Exponentiates the result to get predictions in the original scale
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
        T denormalizedLogIntercept = yParams.Mean;
        for (int i = 0; i < coefficientsVector.Length; i++)
        {
            denormalizedLogIntercept = NumOps.Subtract(denormalizedLogIntercept,
                NumOps.Multiply(coefficientsVector[i],
                    NumOps.Divide(
                        NumOps.Multiply(xParams[i].Mean, yParams.StdDev),
                        NumOps.GreaterThan(xParams[i].StdDev, _epsilon) ? xParams[i].StdDev : _epsilon
                    )
                )
            );
        }

        return NumOps.Subtract(NumOps.Exp(denormalizedLogIntercept), yParams.Shift);
    }
}
