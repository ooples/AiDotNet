namespace AiDotNet.OutlierRemoval;

/// <summary>
/// Implements a threshold-based method for removing outliers from datasets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The type of input data structure.</typeparam>
/// <typeparam name="TOutput">The type of output data structure.</typeparam>
/// <remarks>
/// This class uses the Median Absolute Deviation (MAD) method to identify and remove outliers.
/// 
/// <b>For Beginners:</b> Outliers are data points that differ significantly from other observations in your dataset.
/// They can negatively impact your model's performance by skewing the results. This class helps
/// identify and remove these unusual data points before training your model.
/// </remarks>
public class ThresholdOutlierRemoval<T, TInput, TOutput> : IOutlierRemoval<T, TInput, TOutput>
{
    private readonly T _threshold;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Creates a new instance of the threshold-based outlier removal algorithm.
    /// </summary>
    /// <param name="threshold">
    /// The threshold multiplier used to determine what constitutes an outlier.
    /// Higher values are more lenient (remove fewer points), while lower values are stricter.
    /// If not specified, a default value of 3.0 is used.
    /// </param>
    /// <remarks>
    /// <b>For Beginners:</b> The threshold determines how sensitive the outlier detection is.
    /// A common rule of thumb is to use 3.0, which means data points more than 3 times
    /// the median deviation from the median are considered outliers. If your data naturally
    /// has more variation, you might want to use a higher threshold.
    /// </remarks>
    public ThresholdOutlierRemoval(double threshold = 3)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _threshold = _numOps.FromDouble(threshold);
    }

    /// <summary>
    /// Removes outliers from the input data based on the threshold criterion.
    /// </summary>
    /// <param name="inputs">
    /// The input feature matrix where each row represents a data point and each column represents a feature.
    /// </param>
    /// <param name="outputs">
    /// The target values vector where each element corresponds to a row in the input matrix.
    /// </param>
    /// <returns>
    /// A tuple containing:
    /// - CleanedInputs: A matrix of input features with outliers removed
    /// - CleanedOutputs: A vector of output values corresponding to the cleaned inputs
    /// </returns>
    /// <remarks>
    /// This method uses the Median Absolute Deviation (MAD) approach to detect outliers:
    /// 1. For each feature (column), it calculates the median value
    /// 2. It computes how far each data point deviates from this median
    /// 3. It calculates the median of these deviations (the MAD)
    /// 4. Points that deviate more than (threshold × MAD) from the median are considered outliers
    /// 
    /// <b>For Beginners:</b> This method examines each feature (column) in your data separately.
    /// It finds the middle value (median) for each feature, then measures how far each data point
    /// is from this middle value. If any point is too far away (based on your threshold),
    /// that entire row of data is considered an outlier and removed from both inputs and outputs.
    /// </remarks>
    public (TInput CleanedInputs, TOutput CleanedOutputs) RemoveOutliers(TInput inputs, TOutput outputs)
    {
        // Convert to concrete types
        var (inputMatrix, outputVector) = OutlierRemovalHelper<T, TInput, TOutput>.ConvertToMatrixVector(inputs, outputs);

        var cleanedInputs = new List<Vector<T>>();
        var cleanedOutputs = new List<T>();
        var includeRow = new bool[inputMatrix.Rows];

        // Initialize all rows as included
        for (int i = 0; i < inputMatrix.Rows; i++)
        {
            includeRow[i] = true;
        }

        for (int j = 0; j < inputMatrix.Columns; j++)
        {
            var column = inputMatrix.GetColumn(j);
            var median = StatisticsHelper<T>.CalculateMedian(column);
            var deviations = column.Select(x => _numOps.Abs(_numOps.Subtract(x, median))).OrderBy(x => x).ToList();
            var medianDeviation = deviations[deviations.Count / 2];
            var threshold = _numOps.Multiply(_threshold, medianDeviation);

            for (int i = 0; i < inputMatrix.Rows; i++)
            {
                if (_numOps.GreaterThan(_numOps.Abs(_numOps.Subtract(inputMatrix[i, j], median)), threshold))
                {
                    includeRow[i] = false;
                }
            }
        }

        // Collect all rows that are still marked as included
        for (int i = 0; i < inputMatrix.Rows; i++)
        {
            if (includeRow[i])
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
