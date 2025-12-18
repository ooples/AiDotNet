namespace AiDotNet.OutlierRemoval;

/// <summary>
/// Implements outlier detection and removal based on the Z-Score method.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The type of input data structure.</typeparam>
/// <typeparam name="TOutput">The type of output data structure.</typeparam>
/// <remarks>
/// <para>
/// This class provides functionality to identify and remove outliers from a dataset using the Z-Score method.
/// The Z-Score approach measures how many standard deviations a data point is from the mean of the dataset.
/// Points with a Z-Score exceeding a specified threshold are considered outliers. This method assumes that
/// the data follows a relatively normal distribution.
/// </para>
/// <para><b>For Beginners:</b> This class helps you find and remove unusual data points from your dataset.
/// 
/// Imagine you collected height measurements for a group of people. Most heights will be close to the average,
/// but there might be a few very unusual values (like someone recorded as 10 feet tall), which could be errors
/// or just very rare cases.
/// 
/// The Z-Score method works like this:
/// - It calculates the average (mean) of all your data points
/// - It determines how spread out the data is (standard deviation)
/// - For each point, it calculates how far that point is from the average, in terms of standard deviations
/// - If a point is too far away (beyond the threshold you set), it's considered an outlier
/// 
/// For example, with a threshold of 3, any point more than 3 standard deviations from the mean would be
/// considered an outlier and removed from the dataset.
/// </para>
/// </remarks>
public class ZScoreOutlierRemoval<T, TInput, TOutput> : IOutlierRemoval<T, TInput, TOutput>
{
    /// <summary>
    /// The threshold value for determining outliers. Points with a Z-Score exceeding this threshold
    /// (in absolute value) are considered outliers.
    /// </summary>
    private readonly T _threshold;

    /// <summary>
    /// The numeric operations provider for type T, used for mathematical calculations.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="ZScoreOutlierRemoval{T, TInput, TOutput}"/> class with the specified threshold.
    /// </summary>
    /// <param name="threshold">The threshold value for outlier detection, default is 3.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the outlier removal algorithm with a specified threshold. The threshold
    /// determines how extreme a value must be to be considered an outlier. A common default value is 3,
    /// meaning data points with a Z-Score greater than 3 (or less than -3) are considered outliers.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new outlier detector with a specified sensitivity level.
    /// 
    /// The threshold parameter controls how sensitive the outlier detection is:
    /// - A lower threshold (like 2.0) will be more aggressive in labeling points as outliers
    /// - A higher threshold (like 4.0) will be more lenient, only marking very extreme values as outliers
    /// 
    /// In statistics, a threshold of 3 is commonly used, which means points that are more than 3 standard
    /// deviations away from the average are considered outliers. With a normal distribution, this would
    /// flag approximately 0.3% of the data as outliers.
    /// </para>
    /// </remarks>
    public ZScoreOutlierRemoval(double threshold = 3)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _threshold = _numOps.FromDouble(threshold);
    }

    /// <summary>
    /// Removes outliers from the provided inputs and outputs based on the Z-Score algorithm.
    /// </summary>
    /// <param name="inputs">The matrix of input data, where each row represents a data point and each column represents a feature.</param>
    /// <param name="outputs">The vector of output values corresponding to each row in the inputs matrix.</param>
    /// <returns>A tuple containing the cleaned inputs (as a matrix) and cleaned outputs (as a vector) with outliers removed.</returns>
    /// <remarks>
    /// <para>
    /// This method identifies outliers by examining each feature (column) in the inputs matrix. For each feature,
    /// it calculates the mean and standard deviation, then computes the Z-Score for every data point. If a data point
    /// has a Z-Score exceeding the threshold for any feature, it is considered an outlier and is removed from both
    /// the inputs and outputs.
    /// </para>
    /// <para><b>For Beginners:</b> This method analyzes your data and removes any unusual points.
    /// 
    /// Here's how it works:
    /// 1. It looks at each characteristic (column) of your data separately
    /// 2. For each characteristic, it:
    ///    - Calculates the average value
    ///    - Determines how spread out the values are (standard deviation)
    ///    - Computes a Z-Score for each point (how many standard deviations from the average)
    /// 3. If a data point has an unusually high or low Z-Score for ANY characteristic, it's considered an outlier
    /// 4. All outliers are removed, and the cleaned data is returned
    /// 
    /// For example, if you have data about houses with columns for size, price, and age, a house might
    /// be considered an outlier if it has a normal size and age but an extremely high price (high Z-Score for price).
    /// </para>
    /// </remarks>
    public (TInput CleanedInputs, TOutput CleanedOutputs) RemoveOutliers(TInput inputs, TOutput outputs)
    {
        // Convert to concrete types
        var (inputMatrix, outputVector) = OutlierRemovalHelper<T, TInput, TOutput>.ConvertToMatrixVector(inputs, outputs);

        var cleanedInputs = new List<Vector<T>>();
        var cleanedOutputs = new List<T>();

        for (int i = 0; i < outputVector.Length; i++)
        {
            bool isOutlier = false;

            for (int j = 0; j < inputMatrix.Columns; j++)
            {
                var column = inputMatrix.GetColumn(j);
                (var mean, var std) = StatisticsHelper<T>.CalculateMeanAndStandardDeviation(column);

                // If standard deviation is zero (all values are the same), skip z-score check
                // as no point can be an outlier when all values are identical
                if (_numOps.Equals(std, _numOps.Zero))
                {
                    continue;
                }

                var zScore = _numOps.Divide(_numOps.Subtract(column[i], mean), std);

                if (_numOps.GreaterThan(_numOps.Abs(zScore), _threshold))
                {
                    isOutlier = true;
                    break;
                }
            }

            if (!isOutlier)
            {
                cleanedInputs.Add(inputMatrix.GetRow(i));
                cleanedOutputs.Add(outputVector[i]);
            }
        }

        var cleanedInputMatrix = new Matrix<T>(cleanedInputs);
        var cleanedOutputVector = new Vector<T>(cleanedOutputs);

        // Convert back to original types
        return OutlierRemovalHelper<T, TInput, TOutput>.ConvertToOriginalTypes(
            cleanedInputMatrix,
            cleanedOutputVector,
            typeof(TInput),
            typeof(TOutput));
    }
}
