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
public class BinningNormalizer<T> : INormalizer<T>
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
    /// The numeric operations provider for the specified type T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds a reference to an object that provides operations for the numeric type T.
    /// These operations include addition, subtraction, multiplication, division, and comparisons,
    /// which are needed for the binning calculations.
    /// </para>
    /// <para><b>For Beginners:</b> This is like having a calculator that works with whatever number type you're using.
    /// 
    /// Since this normalizer can work with different types of numbers (integers, decimals, etc.),
    /// it needs a way to perform math operations on these numbers. This field provides those capabilities,
    /// like a specialized calculator for the specific type of numbers being processed.
    /// </para>
    /// </remarks>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="BinningNormalizer{T}"/> class.
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
    public BinningNormalizer()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Normalizes a vector using the binning approach.
    /// </summary>
    /// <param name="vector">The input vector to normalize.</param>
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
    ///    - It assigns a normalized value based on the bin number (bin 0 → 0.0, bin 9 → 1.0)
    /// 
    /// The method returns:
    /// - Your transformed data with each value replaced by its normalized bin value
    /// - Information about the bin boundaries so you can reverse the process later
    /// 
    /// For example, if you had ages [5, 18, 25, 40, 62, 71], after normalization they might become
    /// values like [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] representing which bin they fall into.
    /// </para>
    /// </remarks>
    public (Vector<T>, NormalizationParameters<T>) NormalizeVector(Vector<T> vector)
    {
        var sortedVector = vector.ToArray();
        Array.Sort(sortedVector);

        var bins = new List<T>();
        for (int i = 0; i <= DefaultBinCount; i++)
        {
            int index = Convert.ToInt32(_numOps.Multiply(_numOps.FromDouble((double)i / DefaultBinCount), _numOps.FromDouble(sortedVector.Length - 1)));
            bins.Add(sortedVector[index]);
        }

        var normalizedVector = vector.Transform(x => 
        {
            int binIndex = bins.FindIndex(b => _numOps.LessThanOrEquals(x, b));
            return _numOps.Divide(_numOps.FromDouble(binIndex == -1 ? DefaultBinCount - 1 : binIndex), _numOps.FromDouble(DefaultBinCount - 1));
        });

        var parameters = new NormalizationParameters<T> { Method = NormalizationMethod.Binning, Bins = bins };
        return (normalizedVector, parameters);
    }

    /// <summary>
    /// Normalizes a matrix using the binning approach, applying normalization separately to each column.
    /// </summary>
    /// <param name="matrix">The input matrix to normalize.</param>
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
    public Vector<T> DenormalizeVector(Vector<T> vector, NormalizationParameters<T> parameters)
    {
        if (parameters.Bins == null || parameters.Bins.Count == 0)
        {
            throw new ArgumentException("Invalid bin parameters. Bins list is null or empty.");
        }

        return vector.Transform(x => 
        {
            // Ensure x is within [0, 1] range
            var min = _numOps.LessThan(_numOps.One, x) ? _numOps.One : x;
            x = _numOps.GreaterThan(_numOps.Zero, min) ? _numOps.Zero : min;
        
            // Calculate the bin index
            int binIndex = Convert.ToInt32(_numOps.Multiply(x, _numOps.FromDouble(parameters.Bins.Count - 1)));
        
            // Ensure binIndex is within valid range
            binIndex = Math.Max(0, Math.Min(parameters.Bins.Count - 2, binIndex));

            // Return the average of the current bin and the next bin
            return _numOps.Divide(_numOps.Add(parameters.Bins[binIndex], parameters.Bins[binIndex + 1]), _numOps.FromDouble(2));
        });
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
    public Vector<T> DenormalizeCoefficients(Vector<T> coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        // Denormalizing coefficients for binning is complex and may not always be meaningful.
        // This is a simplified approach that may not be suitable for all use cases.
        return coefficients.PointwiseMultiply(Vector<T>.FromArray(xParams.Select((p, i) => 
            _numOps.Divide(
                _numOps.Subtract(yParams.Bins.Last(), yParams.Bins.First()),
                _numOps.Subtract(p.Bins.Last(), p.Bins.First())
            )).ToArray()));
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
    public T DenormalizeYIntercept(Matrix<T> xMatrix, Vector<T> y, Vector<T> coefficients, 
        List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        // Denormalizing y-intercept for binning is complex and may not always be meaningful.
        // This is a simplified approach that may not be suitable for all use cases.
        T denormalizedIntercept = _numOps.Divide(_numOps.Add(yParams.Bins.First(), yParams.Bins.Last()), _numOps.FromDouble(2));
        for (int i = 0; i < coefficients.Length; i++)
        {
            denormalizedIntercept = _numOps.Subtract(denormalizedIntercept, 
                _numOps.Multiply(coefficients[i], 
                    _numOps.Divide(_numOps.Add(xParams[i].Bins.First(), xParams[i].Bins.Last()), _numOps.FromDouble(2))));
        }

        return denormalizedIntercept;
    }
}