namespace AiDotNet.Normalizers;

/// <summary>
/// Represents a normalizer that uses data binning to transform values into discrete ranges.
/// </summary>
/// <remarks>
/// <para>
/// The BinningNormalizer divides data into a fixed number of bins (categories) based on their values.
/// It transforms each value to a normalized value representing the bin it belongs to, creating a 
/// discrete representation of continuous data. This approach can be useful for handling outliers
/// and non-linear relationships in the data.
/// </para>
/// <para>
/// The binning process works by:
/// - Sorting the input values
/// - Creating equally spaced quantile bins (default is 10 bins)
/// - Mapping each value to its corresponding bin
/// - Normalizing bin indices to the range [0, 1]
/// </para>
/// <para><b>For Beginners:</b> A BinningNormalizer is like sorting items into different sized containers.
/// 
/// Think of binning as organizing your data like groceries on shelves:
/// - Instead of keeping the exact price of each item, you group them into price ranges
/// - For example, items costing $0-$5 go on shelf 1, $5-$10 on shelf 2, and so on
/// - The shelves (bins) are sized to contain roughly equal numbers of items
/// - After binning, you just need to know which shelf an item is on, not its exact price
/// 
/// This is useful when:
/// - You care more about the general range of a value than its exact number
/// - Your data has unusual values that might skew your analysis
/// - You want to simplify complex data into more manageable categories
/// 
/// For instance, instead of working with exact ages (23, 37, 82, etc.), binning might group them
/// into categories like "young adult," "middle-aged," and "senior."
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The type of input data structure.</typeparam>
/// <typeparam name="TOutput">The type of output data structure.</typeparam>
public class BinningNormalizer<T, TInput, TOutput> : NormalizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The default number of bins to use for normalization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constant defines the number of bins that will be created when dividing the data range.
    /// The default value of 10 provides a reasonable balance between granularity and simplification.
    /// </para>
    /// <para><b>For Beginners:</b> This is like deciding how many shelves to use in your grocery store.
    /// 
    /// Having 10 bins means:
    /// - Your data will be divided into 10 categories
    /// - Each category will contain roughly the same number of data points
    /// - More bins give you more detail but less simplification
    /// - Fewer bins give you more simplification but less detail
    /// 
    /// 10 bins is usually a good starting point for most datasets.
    /// </para>
    /// </remarks>
    private const int DefaultBinCount = 10;

    /// <summary>
    /// Initializes a new instance of the <see cref="BinningNormalizer{T, TInput, TOutput}"/> class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor creates a new BinningNormalizer and initializes the numeric operations
    /// provider for the specified type T. No additional parameters are required as the normalizer
    /// uses the default bin count.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up your binning system with default settings.
    /// 
    /// When you create a new BinningNormalizer:
    /// - It automatically sets up to use 10 bins (categories)
    /// - It figures out what kind of numbers you're working with
    /// - It gets ready to sort your data into these bins
    /// 
    /// It's like setting up shelves in a store before you start organizing products.
    /// </para>
    /// </remarks>
    public BinningNormalizer() : base()
    {
        // Base constructor already initializes NumOps
    }

    /// <summary>
    /// Normalizes a vector using the binning approach.
    /// </summary>
    /// <param name="data">The input vector to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized vector and the normalization parameters, which include the bin boundaries.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method normalizes a vector by:
    /// 1. Sorting the values to determine the distribution
    /// 2. Creating equally spaced quantile bins based on the sorted values
    /// 3. Assigning each value to a bin and returning the normalized bin index (between 0 and 1)
    /// 
    /// The normalization parameters include the bin boundaries, which are needed for later denormalization.
    /// </para>
    /// <para><b>For Beginners:</b> This method sorts your data into bins and gives each value a shelf number.
    /// 
    /// The process works like this:
    /// 1. First, it sorts all your values from smallest to largest
    /// 2. Then, it creates 10 bins (shelves) that will each contain roughly the same number of items
    /// 3. For each value in your original data:
    ///    - It figures out which bin the value belongs in
    ///    - It assigns a normalized value based on the bin number (bin 0 ? 0.0, bin 9 ? 1.0)
    /// 
    /// The method returns:
    /// - Your transformed data with each value replaced by its normalized bin value
    /// - Information about the bin boundaries so you can reverse the process later
    /// 
    /// For example, if you had ages [5, 18, 25, 40, 62, 71], after normalization they might become
    /// values like [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] representing which bin they fall into.
    /// </para>
    /// </remarks>
    public override (TOutput, NormalizationParameters<T>) NormalizeOutput(TOutput data)
    {
        if (data is Vector<T> vector)
        {
            var sortedVector = vector.ToArray();
            Array.Sort(sortedVector);

            var bins = new List<T>();
            for (int i = 0; i <= DefaultBinCount; i++)
            {
                int index = Convert.ToInt32(NumOps.Multiply(NumOps.FromDouble((double)i / DefaultBinCount), NumOps.FromDouble(sortedVector.Length - 1)));
                bins.Add(sortedVector[index]);
            }

            var normalizedVector = vector.Transform(x =>
            {
                int binIndex = bins.FindIndex(b => NumOps.LessThanOrEquals(x, b));
                return NumOps.Divide(NumOps.FromDouble(binIndex == -1 ? DefaultBinCount - 1 : binIndex), NumOps.FromDouble(DefaultBinCount - 1));
            });

            var parameters = new NormalizationParameters<T> { Method = NormalizationMethod.Binning, Bins = bins };
            return ((TOutput)(object)normalizedVector, parameters);
        }
        else if (data is Tensor<T> tensor)
        {
            // Flatten tensor to apply binning normalization
            var flattenedTensor = tensor.ToVector();
            var sortedVector = flattenedTensor.ToArray();
            Array.Sort(sortedVector);

            var bins = new List<T>();
            for (int i = 0; i <= DefaultBinCount; i++)
            {
                int index = Convert.ToInt32(NumOps.Multiply(NumOps.FromDouble((double)i / DefaultBinCount), NumOps.FromDouble(sortedVector.Length - 1)));
                bins.Add(sortedVector[index]);
            }

            var normalizedVector = flattenedTensor.Transform(x =>
            {
                int binIndex = bins.FindIndex(b => NumOps.LessThanOrEquals(x, b));
                return NumOps.Divide(NumOps.FromDouble(binIndex == -1 ? DefaultBinCount - 1 : binIndex), NumOps.FromDouble(DefaultBinCount - 1));
            });

            // Convert back to tensor with the same shape
            var normalizedTensor = Tensor<T>.FromVector(normalizedVector);
            if (tensor.Shape.Length > 1)
            {
                normalizedTensor = normalizedTensor.Reshape(tensor.Shape);
            }

            var parameters = new NormalizationParameters<T> { Method = NormalizationMethod.Binning, Bins = bins };
            return ((TOutput)(object)normalizedTensor, parameters);
        }

        throw new InvalidOperationException(
            $"Unsupported data type {typeof(TOutput).Name}. " +
            $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
    }

    /// <summary>
    /// Normalizes a matrix using the binning approach, applying normalization separately to each column.
    /// </summary>
    /// <param name="data">The input matrix to normalize.</param>
    /// <returns>
    /// A tuple containing the normalized matrix and a list of normalization parameters for each column.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method normalizes each column of the matrix independently using the binning approach.
    /// It treats each column as a separate feature that needs its own binning strategy, since different
    /// features may have different distributions and ranges.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies binning to a table of data, one column at a time.
    /// 
    /// When working with a table (matrix) of data:
    /// - Each column might represent a different type of information (age, height, income, etc.)
    /// - Each column needs its own binning strategy because the ranges and distributions differ
    /// - This method processes each column separately using the vector normalization method
    /// 
    /// For example, with a table of people's information:
    /// - Column 1 (ages) might be binned into age ranges
    /// - Column 2 (heights) would be binned into height ranges
    /// - Each column gets its own set of bins appropriate for its values
    /// 
    /// The method returns:
    /// - A new table with all values replaced by their bin positions (0.0 to 1.0)
    /// - A separate set of bin boundaries for each column, so you can reverse the process later
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
    /// Denormalizes a vector using the provided normalization parameters.
    /// </summary>
    /// <param name="data">The normalized vector to denormalize.</param>
    /// <param name="parameters">The normalization parameters containing bin information.</param>
    /// <returns>A denormalized vector with values approximated from the bins.</returns>
    /// <remarks>
    /// <para>
    /// This method converts normalized bin indices back to approximate original values.
    /// Since binning is a lossy transformation (multiple original values map to the same bin),
    /// the denormalization process returns an approximation, typically the average value of the bin.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts shelf numbers back to approximate original values.
    /// 
    /// When you denormalize binned data:
    /// - You're trying to go from bin numbers (0.0 to 1.0) back to something close to the original values
    /// - Since multiple original values were grouped into the same bin, you can't recover exact original values
    /// - Instead, the method returns a representative value for each bin (typically the average of the bin's boundaries)
    /// 
    /// For example:
    /// - If age 25 and age 30 both got binned to 0.4 during normalization
    /// - When denormalizing 0.4, you might get back 27.5 (the average of those ages)
    /// - This is an approximation, not the exact original value
    /// 
    /// This loss of precision is the trade-off for the simplification that binning provides.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">
    /// Thrown when the parameters do not contain valid bin information.
    /// </exception>
    public override TOutput Denormalize(TOutput data, NormalizationParameters<T> parameters)
    {
        if (parameters.Bins == null || parameters.Bins.Count == 0)
        {
            throw new ArgumentException("Invalid bin parameters. Bins list is null or empty.");
        }

        if (data is Vector<T> vector)
        {
            var denormalizedVector = vector.Transform(x =>
            {
                // Ensure x is within [0, 1] range
                var min = NumOps.LessThan(NumOps.One, x) ? NumOps.One : x;
                x = NumOps.GreaterThan(NumOps.Zero, min) ? NumOps.Zero : min;

                // Calculate the bin index
                int binIndex = Convert.ToInt32(NumOps.Multiply(x, NumOps.FromDouble(parameters.Bins.Count - 1)));

                // Ensure binIndex is within valid range
                binIndex = Math.Max(0, Math.Min(parameters.Bins.Count - 2, binIndex));

                // Return the average of the current bin and the next bin
                return NumOps.Divide(NumOps.Add(parameters.Bins[binIndex], parameters.Bins[binIndex + 1]), NumOps.FromDouble(2));
            });

            return (TOutput)(object)denormalizedVector;
        }
        else if (data is Tensor<T> tensor)
        {
            // Flatten tensor for denormalization
            var flattenedTensor = tensor.ToVector();

            var denormalizedVector = flattenedTensor.Transform(x =>
            {
                // Ensure x is within [0, 1] range
                var min = NumOps.LessThan(NumOps.One, x) ? NumOps.One : x;
                x = NumOps.GreaterThan(NumOps.Zero, min) ? NumOps.Zero : min;

                // Calculate the bin index
                int binIndex = Convert.ToInt32(NumOps.Multiply(x, NumOps.FromDouble(parameters.Bins.Count - 1)));

                // Ensure binIndex is within valid range
                binIndex = Math.Max(0, Math.Min(parameters.Bins.Count - 2, binIndex));

                // Return the average of the current bin and the next bin
                return NumOps.Divide(NumOps.Add(parameters.Bins[binIndex], parameters.Bins[binIndex + 1]), NumOps.FromDouble(2));
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
    /// Denormalizes coefficients from a regression model that was trained on binned data.
    /// </summary>
    /// <param name="coefficients">The regression coefficients to denormalize.</param>
    /// <param name="xParams">The normalization parameters for the input features.</param>
    /// <param name="yParams">The normalization parameters for the output variable.</param>
    /// <returns>Denormalized coefficients that can be applied to original, unbinned data.</returns>
    /// <remarks>
    /// <para>
    /// This method attempts to adjust regression coefficients to work with original unbinned data.
    /// Denormalizing coefficients for binned data is complex and may not always produce ideal results,
    /// as binning is a non-linear transformation that changes the relationship between variables.
    /// This implementation uses a simplified approach based on the range of each variable.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts prediction model weights to work with original values instead of bin numbers.
    /// 
    /// When you've built a prediction model using binned data:
    /// - The model's weights (coefficients) were learned based on bin numbers (0.0 to 1.0)
    /// - To use this model with unbinned data, you need to adjust these weights
    /// - This method attempts to convert the weights to work with the original scale of your data
    /// 
    /// However, there's an important limitation:
    /// - Since binning groups many values together, some information is permanently lost
    /// - This means the conversion back to original scales is an approximation
    /// - The adjusted weights might not work as well as a model originally trained on unbinned data
    /// 
    /// Think of it like translating between languages - some meaning inevitably gets lost or changed.
    /// </para>
    /// </remarks>
    public override TOutput Denormalize(TOutput coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        if (coefficients is Vector<T> vector)
        {
            var denormalizedCoefficients = vector.PointwiseMultiply(Vector<T>.FromArray(xParams.Select((p, i) =>
                NumOps.Divide(
                    NumOps.Subtract(yParams.Bins.Last(), yParams.Bins.First()),
                    NumOps.Subtract(p.Bins.Last(), p.Bins.First())
                )).ToArray()));

            return (TOutput)(object)denormalizedCoefficients;
        }
        else if (coefficients is Tensor<T> tensor)
        {
            // Flatten tensor for denormalization
            var flattenedTensor = tensor.ToVector();

            var denormalizedVector = flattenedTensor.PointwiseMultiply(Vector<T>.FromArray(xParams.Select((p, i) =>
                NumOps.Divide(
                    NumOps.Subtract(yParams.Bins.Last(), yParams.Bins.First()),
                    NumOps.Subtract(p.Bins.Last(), p.Bins.First())
                )).ToArray()));

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
    /// Denormalizes the y-intercept from a regression model that was trained on binned data.
    /// </summary>
    /// <param name="xMatrix">The original input feature matrix.</param>
    /// <param name="y">The original output vector.</param>
    /// <param name="coefficients">The regression coefficients.</param>
    /// <param name="xParams">The normalization parameters for the input features.</param>
    /// <param name="yParams">The normalization parameters for the output variable.</param>
    /// <returns>A denormalized y-intercept that can be used with original, unbinned data.</returns>
    /// <remarks>
    /// <para>
    /// This method attempts to adjust the y-intercept of a regression model to work with original unbinned data.
    /// Like coefficient denormalization, this is a complex task for binned data and may not always produce
    /// ideal results. This implementation uses a simplified approach based on the midpoints of the bin ranges.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts the baseline value of a prediction model to work with original values.
    /// 
    /// In a prediction model, the y-intercept is like the starting point or baseline:
    /// - It's the predicted value when all features are set to zero
    /// - When using binned data, this baseline needs to be adjusted to work with original values
    /// - This method calculates an approximate y-intercept for the unbinned data
    /// 
    /// Like with coefficient denormalization:
    /// - This is an approximation because binning loses some information
    /// - The adjusted baseline might not be as accurate as one from a model trained on unbinned data
    /// - It uses the average values of bins to estimate a reasonable baseline
    /// 
    /// Think of it as figuring out where to start your predictions when switching from working
    /// with simplified bin numbers back to complex original values.
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
        T denormalizedIntercept = NumOps.Divide(NumOps.Add(yParams.Bins.First(), yParams.Bins.Last()), NumOps.FromDouble(2));
        for (int i = 0; i < coefficientsVector.Length; i++)
        {
            denormalizedIntercept = NumOps.Subtract(denormalizedIntercept,
                NumOps.Multiply(coefficientsVector[i],
                    NumOps.Divide(NumOps.Add(xParams[i].Bins.First(), xParams[i].Bins.Last()), NumOps.FromDouble(2))));
        }

        return denormalizedIntercept;
    }
}
