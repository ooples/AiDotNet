namespace AiDotNet.OutlierRemoval;

/// <summary>
/// Implements the Interquartile Range (IQR) method for removing outliers from datasets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The type of input data structure.</typeparam>
/// <typeparam name="TOutput">The type of output data structure.</typeparam>
/// <remarks>
/// This class uses the IQR method, a robust statistical technique for identifying outliers.
/// 
/// <b>For Beginners:</b> Outliers are unusual data points that are significantly different from most of your data.
/// These unusual points can negatively affect your machine learning models. The IQR method is a common
/// statistical approach that uses quartiles (which divide your data into four equal parts) to identify
/// which data points should be considered outliers.
/// </remarks>
public class IQROutlierRemoval<T, TInput, TOutput> : IOutlierRemoval<T, TInput, TOutput>
{
    private readonly T _iqrMultiplier;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Creates a new instance of the IQR-based outlier removal algorithm.
    /// </summary>
    /// <param name="iqrMultiplier">
    /// The multiplier applied to the IQR to determine the outlier threshold.
    /// If not specified, a default value of 1.5 is used, which is the standard in statistics.
    /// </param>
    /// <remarks>
    /// <b>For Beginners:</b> The IQR multiplier determines how strict the outlier detection will be.
    /// The standard value of 1.5 means that any data point below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
    /// is considered an outlier. A higher multiplier (like 2.0 or 3.0) will be more lenient and
    /// identify fewer outliers, while a lower value will be stricter.
    /// 
    /// Q1 is the first quartile (25th percentile) and Q3 is the third quartile (75th percentile).
    /// The IQR is the range between Q1 and Q3, representing the middle 50% of your data.
    /// </remarks>
    public IQROutlierRemoval(double iqrMultiplier = 1.5)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _iqrMultiplier = _numOps.FromDouble(iqrMultiplier);
    }

    /// <summary>
    /// Removes outliers from the input data using the IQR method.
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
    /// This method applies the IQR outlier detection technique to each feature (column) in your data:
    /// 1. For each feature, it calculates Q1 (25th percentile) and Q3 (75th percentile)
    /// 2. It computes the IQR as Q3 - Q1
    /// 3. It defines the lower bound as Q1 - (multiplier × IQR) and upper bound as Q3 + (multiplier × IQR)
    /// 4. Any data point outside these bounds for any feature is considered an outlier
    /// 
    /// <b>For Beginners:</b> This method examines each feature (column) in your data separately.
    /// It finds the range where the middle 50% of your data falls (the IQR), then extends this range
    /// in both directions by a factor of your multiplier. Any data point that falls outside this
    /// extended range for ANY feature is considered an outlier and removed from both inputs and outputs.
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
                var quartiles = new Quartile<T>(column);
                var q1 = quartiles.Q1;
                var q3 = quartiles.Q3;
                var iqr = _numOps.Subtract(q3, q1);
                var lowerBound = _numOps.Subtract(q1, _numOps.Multiply(_iqrMultiplier, iqr));
                var upperBound = _numOps.Add(q3, _numOps.Multiply(_iqrMultiplier, iqr));

                if (_numOps.LessThan(column[i], lowerBound) || _numOps.GreaterThan(column[i], upperBound))
                {
                    isOutlier = true;
                    break;
                }
            }

            if (!isOutlier)
            {
                cleanedInputs.Add(new Vector<T>(inputMatrix.GetRow(i)));
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
