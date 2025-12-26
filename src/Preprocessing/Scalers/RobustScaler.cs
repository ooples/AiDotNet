using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Scalers;

/// <summary>
/// Scales features using statistics that are robust to outliers.
/// </summary>
/// <remarks>
/// <para>
/// Robust scaling removes the median and scales data according to the interquartile range (IQR).
/// The IQR is the range between the 25th percentile (Q1) and 75th percentile (Q3).
/// Unlike StandardScaler, RobustScaler uses statistics that are less affected by outliers.
/// </para>
/// <para><b>For Beginners:</b> This scaler is like StandardScaler but better handles outliers:
/// - Uses median (middle value) instead of mean (average)
/// - Uses IQR (spread of middle 50%) instead of standard deviation
///
/// Why this matters:
/// - Mean and std are heavily influenced by extreme values
/// - Median and IQR ignore extreme values
///
/// Example: If most house prices are $100K-$500K but a few are $10M,
/// RobustScaler won't let those mansions distort the scaling.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class RobustScaler<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private Vector<T>? _median;
    private Vector<T>? _iqr;
    private readonly double _quantileRangeMin;
    private readonly double _quantileRangeMax;
    private readonly bool _withCentering;
    private readonly bool _withScaling;

    /// <summary>
    /// Gets the median of each feature computed during fitting.
    /// </summary>
    public Vector<T>? Median => _median;

    /// <summary>
    /// Gets the interquartile range (IQR) of each feature computed during fitting.
    /// </summary>
    public Vector<T>? InterquartileRange => _iqr;

    /// <summary>
    /// Gets whether this scaler centers the data (subtracts median).
    /// </summary>
    public bool WithCentering => _withCentering;

    /// <summary>
    /// Gets whether this scaler scales the data (divides by IQR).
    /// </summary>
    public bool WithScaling => _withScaling;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="RobustScaler{T}"/> with default settings.
    /// </summary>
    /// <param name="withCentering">If true, center the data by subtracting the median. Default is true.</param>
    /// <param name="withScaling">If true, scale the data by dividing by the IQR. Default is true.</param>
    /// <param name="columnIndices">The column indices to scale, or null for all columns.</param>
    public RobustScaler(bool withCentering = true, bool withScaling = true, int[]? columnIndices = null)
        : this(25.0, 75.0, withCentering, withScaling, columnIndices)
    {
    }

    /// <summary>
    /// Creates a new instance of <see cref="RobustScaler{T}"/> with custom quantile range.
    /// </summary>
    /// <param name="quantileRangeMin">The lower quantile (0-100). Default is 25 (Q1).</param>
    /// <param name="quantileRangeMax">The upper quantile (0-100). Default is 75 (Q3).</param>
    /// <param name="withCentering">If true, center the data by subtracting the median.</param>
    /// <param name="withScaling">If true, scale the data by dividing by the IQR.</param>
    /// <param name="columnIndices">The column indices to scale, or null for all columns.</param>
    public RobustScaler(
        double quantileRangeMin,
        double quantileRangeMax,
        bool withCentering = true,
        bool withScaling = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (quantileRangeMin < 0 || quantileRangeMin > 100)
        {
            throw new ArgumentOutOfRangeException(
                nameof(quantileRangeMin), "Quantile must be between 0 and 100.");
        }

        if (quantileRangeMax < 0 || quantileRangeMax > 100)
        {
            throw new ArgumentOutOfRangeException(
                nameof(quantileRangeMax), "Quantile must be between 0 and 100.");
        }

        if (quantileRangeMin >= quantileRangeMax)
        {
            throw new ArgumentException(
                $"Quantile range minimum ({quantileRangeMin}) must be less than maximum ({quantileRangeMax}).");
        }

        _quantileRangeMin = quantileRangeMin;
        _quantileRangeMax = quantileRangeMax;
        _withCentering = withCentering;
        _withScaling = withScaling;
    }

    /// <summary>
    /// Computes the median and IQR of each feature from the training data.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    protected override void FitCore(Matrix<T> data)
    {
        int numColumns = data.Columns;
        var columnsToProcess = GetColumnsToProcess(numColumns);

        // Initialize arrays for all columns
        var medians = new T[numColumns];
        var iqrs = new T[numColumns];

        // Default values for columns not processed
        for (int i = 0; i < numColumns; i++)
        {
            medians[i] = NumOps.Zero;
            iqrs[i] = NumOps.One;
        }

        // Compute median and IQR for specified columns
        foreach (var colIdx in columnsToProcess)
        {
            var column = data.GetColumn(colIdx);
            var sorted = SortColumn(column);

            if (_withCentering)
            {
                medians[colIdx] = ComputeQuantile(sorted, 50.0);
            }

            if (_withScaling)
            {
                T q1 = ComputeQuantile(sorted, _quantileRangeMin);
                T q3 = ComputeQuantile(sorted, _quantileRangeMax);
                T iqr = NumOps.Subtract(q3, q1);

                // Prevent division by zero - if IQR is zero, use 1 (no scaling)
                iqrs[colIdx] = NumOps.Compare(iqr, NumOps.Zero) == 0 ? NumOps.One : iqr;
            }
        }

        _median = new Vector<T>(medians);
        _iqr = new Vector<T>(iqrs);
    }

    /// <summary>
    /// Transforms the data by applying robust scaling.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The scaled data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_median is null || _iqr is null)
        {
            throw new InvalidOperationException("Scaler has not been fitted.");
        }

        int numRows = data.Rows;
        int numColumns = data.Columns;
        var result = new T[numRows, numColumns];
        var columnsToProcess = GetColumnsToProcess(numColumns);
        var processSet = new HashSet<int>(columnsToProcess);

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numColumns; j++)
            {
                T value = data[i, j];

                if (processSet.Contains(j))
                {
                    if (_withCentering)
                    {
                        value = NumOps.Subtract(value, _median[j]);
                    }

                    if (_withScaling)
                    {
                        value = NumOps.Divide(value, _iqr[j]);
                    }
                }

                result[i, j] = value;
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Reverses the robust scaling transformation.
    /// </summary>
    /// <param name="data">The scaled data.</param>
    /// <returns>The original-scale data.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_median is null || _iqr is null)
        {
            throw new InvalidOperationException("Scaler has not been fitted.");
        }

        int numRows = data.Rows;
        int numColumns = data.Columns;
        var result = new T[numRows, numColumns];
        var columnsToProcess = GetColumnsToProcess(numColumns);
        var processSet = new HashSet<int>(columnsToProcess);

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numColumns; j++)
            {
                T value = data[i, j];

                if (processSet.Contains(j))
                {
                    if (_withScaling)
                    {
                        value = NumOps.Multiply(value, _iqr[j]);
                    }

                    if (_withCentering)
                    {
                        value = NumOps.Add(value, _median[j]);
                    }
                }

                result[i, j] = value;
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Sorts a column vector for quantile computation.
    /// </summary>
    private T[] SortColumn(Vector<T> column)
    {
        var sorted = new T[column.Length];
        for (int i = 0; i < column.Length; i++)
        {
            sorted[i] = column[i];
        }

        Array.Sort(sorted, (a, b) => NumOps.Compare(a, b));
        return sorted;
    }

    /// <summary>
    /// Computes a quantile from sorted data using linear interpolation.
    /// </summary>
    /// <param name="sorted">The sorted data.</param>
    /// <param name="percentile">The percentile (0-100).</param>
    /// <returns>The quantile value.</returns>
    private T ComputeQuantile(T[] sorted, double percentile)
    {
        if (sorted.Length == 0)
        {
            return NumOps.Zero;
        }

        if (sorted.Length == 1)
        {
            return sorted[0];
        }

        // Convert percentile to index
        double index = (percentile / 100.0) * (sorted.Length - 1);
        int lowerIndex = (int)Math.Floor(index);
        int upperIndex = (int)Math.Ceiling(index);

        if (lowerIndex == upperIndex)
        {
            return sorted[lowerIndex];
        }

        // Linear interpolation between adjacent values
        double fraction = index - lowerIndex;
        T lower = sorted[lowerIndex];
        T upper = sorted[upperIndex];

        T diff = NumOps.Subtract(upper, lower);
        T interpolated = NumOps.Add(lower, NumOps.Multiply(diff, NumOps.FromDouble(fraction)));

        return interpolated;
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names.</param>
    /// <returns>The same feature names (RobustScaler doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
