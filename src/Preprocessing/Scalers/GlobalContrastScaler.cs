using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Scalers;

/// <summary>
/// Scales features by adjusting contrast based on mean and standard deviation.
/// </summary>
/// <remarks>
/// <para>
/// Global contrast scaling transforms data using the formula: (x - mean) / (2 * stdDev) + 0.5
/// This centers values around 0.5 and typically results in values between 0 and 1.
/// </para>
/// <para><b>For Beginners:</b> This scaler improves the "contrast" of your data:
/// - It centers values around 0.5 (the new average)
/// - Values above average become > 0.5, below average become &lt; 0.5
/// - Most values end up between 0 and 1
///
/// Example: [68, 70, 71, 69, 72] → [0.3, 0.5, 0.6, 0.4, 0.7]
/// Now the differences between values are more visible.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class GlobalContrastScaler<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private Vector<T>? _mean;
    private Vector<T>? _stdDev;

    /// <summary>
    /// Gets the mean for each feature computed during fitting.
    /// </summary>
    public Vector<T>? Mean => _mean;

    /// <summary>
    /// Gets the standard deviation for each feature computed during fitting.
    /// </summary>
    public Vector<T>? StdDev => _stdDev;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="GlobalContrastScaler{T}"/>.
    /// </summary>
    /// <param name="columnIndices">The column indices to scale, or null for all columns.</param>
    public GlobalContrastScaler(int[]? columnIndices = null)
        : base(columnIndices)
    {
    }

    /// <summary>
    /// Computes the mean and standard deviation for each feature from the training data.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    protected override void FitCore(Matrix<T> data)
    {
        int numColumns = data.Columns;
        var columnsToProcess = GetColumnsToProcess(numColumns);

        // Initialize arrays for all columns
        var means = new T[numColumns];
        var stdDevs = new T[numColumns];

        // Default values for columns not processed (no transformation)
        for (int i = 0; i < numColumns; i++)
        {
            means[i] = NumOps.Zero;
            stdDevs[i] = NumOps.One;
        }

        // Compute parameters for specified columns
        foreach (var colIdx in columnsToProcess)
        {
            var column = data.GetColumn(colIdx);

            // Calculate mean
            T sum = NumOps.Zero;
            for (int i = 0; i < column.Length; i++)
            {
                sum = NumOps.Add(sum, column[i]);
            }
            T mean = NumOps.Divide(sum, NumOps.FromDouble(column.Length));

            // Calculate variance and std dev
            T varianceSum = NumOps.Zero;
            for (int i = 0; i < column.Length; i++)
            {
                T diff = NumOps.Subtract(column[i], mean);
                varianceSum = NumOps.Add(varianceSum, NumOps.Multiply(diff, diff));
            }
            T variance = NumOps.Divide(varianceSum, NumOps.FromDouble(column.Length));
            T stdDev = NumOps.Sqrt(variance);

            // Prevent division by zero
            if (NumOps.Compare(stdDev, NumOps.Zero) == 0)
            {
                stdDev = NumOps.One;
            }

            means[colIdx] = mean;
            stdDevs[colIdx] = stdDev;
        }

        _mean = new Vector<T>(means);
        _stdDev = new Vector<T>(stdDevs);
    }

    /// <summary>
    /// Transforms the data by applying global contrast scaling.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The scaled data with values typically between 0 and 1.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_mean is null || _stdDev is null)
        {
            throw new InvalidOperationException("Scaler has not been fitted.");
        }

        int numRows = data.Rows;
        int numColumns = data.Columns;
        var result = new T[numRows, numColumns];
        var columnsToProcess = GetColumnsToProcess(numColumns);
        var processSet = new HashSet<int>(columnsToProcess);
        T two = NumOps.FromDouble(2.0);
        T half = NumOps.FromDouble(0.5);

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numColumns; j++)
            {
                T value = data[i, j];

                if (processSet.Contains(j))
                {
                    // (x - mean) / (2 * stdDev) + 0.5
                    T centered = NumOps.Subtract(value, _mean[j]);
                    T scaled = NumOps.Divide(centered, NumOps.Multiply(two, _stdDev[j]));
                    value = NumOps.Add(scaled, half);
                }

                result[i, j] = value;
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Reverses the global contrast scaling transformation.
    /// </summary>
    /// <param name="data">The scaled data.</param>
    /// <returns>The original-scale data.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_mean is null || _stdDev is null)
        {
            throw new InvalidOperationException("Scaler has not been fitted.");
        }

        int numRows = data.Rows;
        int numColumns = data.Columns;
        var result = new T[numRows, numColumns];
        var columnsToProcess = GetColumnsToProcess(numColumns);
        var processSet = new HashSet<int>(columnsToProcess);
        T two = NumOps.FromDouble(2.0);
        T half = NumOps.FromDouble(0.5);

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numColumns; j++)
            {
                T value = data[i, j];

                if (processSet.Contains(j))
                {
                    // Reverse: (x - 0.5) * 2 * stdDev + mean
                    T shifted = NumOps.Subtract(value, half);
                    T unscaled = NumOps.Multiply(shifted, NumOps.Multiply(two, _stdDev[j]));
                    value = NumOps.Add(unscaled, _mean[j]);
                }

                result[i, j] = value;
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names.</param>
    /// <returns>The same feature names (GlobalContrastScaler doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
