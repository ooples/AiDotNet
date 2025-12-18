namespace AiDotNet.OutlierRemoval;

/// <summary>
/// Implements outlier detection and removal based on the Median Absolute Deviation (MAD) method.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The type of input data structure.</typeparam>
/// <typeparam name="TOutput">The type of output data structure.</typeparam>
/// <remarks>
/// <para>
/// This class provides functionality to identify and remove outliers from a dataset using the Median Absolute Deviation (MAD) method.
/// The MAD method is more robust to outliers than standard deviation-based approaches since it uses medians instead of means.
/// The algorithm calculates a modified Z-score for each data point and compares it against a threshold to determine if the point
/// is an outlier.
/// </para>
/// <para><b>For Beginners:</b> This class helps you find and remove "unusual" data points from your dataset.
/// 
/// Imagine you have a collection of measurements, like people's heights. Most measurements will be clustered around
/// a typical value, but occasionally there might be mistakes or very unusual cases (like someone who is 9 feet tall).
/// These unusual values are called "outliers" and can negatively affect your analysis.
/// 
/// This class uses a technique called "Median Absolute Deviation" (MAD) to identify outliers:
/// - It finds the middle value (median) of your data
/// - It calculates how far each point is from this middle value
/// - It finds the middle value of these distances (the MAD)
/// - It calculates a score for each point that represents how unusual it is
/// - If this score is higher than a threshold, the point is considered an outlier
/// 
/// This approach is particularly good because it's not easily fooled by the outliers themselves,
/// unlike methods that use averages.
/// </para>
/// </remarks>
public class MADOutlierRemoval<T, TInput, TOutput> : IOutlierRemoval<T, TInput, TOutput>
{
    /// <summary>
    /// The threshold value for determining outliers. Points with a modified Z-score exceeding this threshold
    /// are considered outliers.
    /// </summary>
    private readonly T _threshold;

    /// <summary>
    /// The numeric operations provider for type T, used for mathematical calculations.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="MADOutlierRemoval{T, TInput, TOutput}"/> class with an optional threshold.
    /// </summary>
    /// <param name="threshold">The threshold value for outlier detection. If not specified, a default value of 3.5 will be used.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the outlier removal algorithm with a specified threshold. The threshold
    /// determines how extreme a value must be to be considered an outlier. Higher thresholds are more
    /// conservative, identifying fewer points as outliers.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new outlier detector with a specified sensitivity level.
    /// 
    /// The threshold parameter controls how sensitive the outlier detection is:
    /// - A lower threshold (like 2.0) will be more aggressive in labeling points as outliers
    /// - A higher threshold (like 5.0) will be more lenient, only marking very extreme values as outliers
    /// 
    /// If you don't specify a threshold, the default value of 3.5 is used, which is a common choice
    /// in statistical practice for outlier detection.
    /// </para>
    /// </remarks>
    public MADOutlierRemoval(double threshold = 3.5)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _threshold = _numOps.FromDouble(threshold);
    }

    /// <summary>
    /// Removes outliers from the provided inputs and outputs based on the MAD algorithm.
    /// </summary>
    /// <param name="inputs">The matrix of input data, where each row represents a data point and each column represents a feature.</param>
    /// <param name="outputs">The vector of output values corresponding to each row in the inputs matrix.</param>
    /// <returns>A tuple containing the cleaned inputs (as a matrix) and cleaned outputs (as a vector) with outliers removed.</returns>
    /// <remarks>
    /// <para>
    /// This method identifies outliers by examining each feature (column) in the inputs matrix. For each feature,
    /// it calculates the modified Z-score for every data point. If a data point has a modified Z-score exceeding the
    /// threshold for any feature, it is considered an outlier and is removed from both the inputs and outputs.
    /// The modified Z-score is calculated as 0.6745 * (x - median) / MAD, where MAD is the median absolute deviation.
    /// </para>
    /// <para><b>For Beginners:</b> This method analyzes your data and removes any unusual points.
    /// 
    /// Here's how it works:
    /// 1. It looks at each characteristic (column) of your data separately
    /// 2. For each characteristic, it:
    ///    - Finds the middle value (median)
    ///    - Measures how far each point is from this middle value
    ///    - Calculates a score that indicates how unusual each point is
    /// 3. If a data point has an unusually high score for ANY characteristic, it's considered an outlier
    /// 4. All outliers are removed, and the cleaned data is returned
    /// 
    /// For example, if you have data about houses with columns for size, price, and age, a house might
    /// be considered an outlier if it has a normal size and age but an extremely high price.
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
                var median = StatisticsHelper<T>.CalculateMedian(column);
                var deviations = column.Select(x => _numOps.Abs(_numOps.Subtract(x, median)));
                var mad = StatisticsHelper<T>.CalculateMedian(deviations);
                var modifiedZScore = _numOps.Divide(_numOps.Multiply(_numOps.FromDouble(0.6745), _numOps.Subtract(column[i], median)), mad);

                if (_numOps.GreaterThan(_numOps.Abs(modifiedZScore), _threshold))
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
