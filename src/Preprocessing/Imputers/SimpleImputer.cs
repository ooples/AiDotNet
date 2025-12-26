using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Imputers;

/// <summary>
/// Specifies the strategy for imputing missing values.
/// </summary>
public enum ImputationStrategy
{
    /// <summary>
    /// Replace missing values with the mean of each feature.
    /// </summary>
    Mean,

    /// <summary>
    /// Replace missing values with the median of each feature.
    /// </summary>
    Median,

    /// <summary>
    /// Replace missing values with the most frequent value of each feature.
    /// </summary>
    MostFrequent,

    /// <summary>
    /// Replace missing values with a constant value.
    /// </summary>
    Constant
}

/// <summary>
/// Imputes missing values using simple strategies like mean, median, or constant.
/// </summary>
/// <remarks>
/// <para>
/// SimpleImputer fills in missing values (represented as NaN) using simple strategies:
/// - Mean: Replace with column mean
/// - Median: Replace with column median
/// - MostFrequent: Replace with most common value
/// - Constant: Replace with a specified value
/// </para>
/// <para><b>For Beginners:</b> This transformer fills in gaps in your data:
/// - If you have missing ages, replace them with average age (Mean)
/// - If you have missing incomes with outliers, use median income (Median)
/// - If you have missing categories, use most common category (MostFrequent)
/// - Or fill with a specific value like 0 or -1 (Constant)
///
/// Example with Mean strategy:
/// [1, 2, NaN, 4, 5] → [1, 2, 3, 4, 5] (NaN replaced with mean=3)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class SimpleImputer<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly ImputationStrategy _strategy;
    private readonly T? _fillValue;
    private readonly T _missingValue;

    // Fitted parameters: statistics for each column
    private Vector<T>? _statistics;

    /// <summary>
    /// Gets the imputation strategy used.
    /// </summary>
    public ImputationStrategy Strategy => _strategy;

    /// <summary>
    /// Gets the fill value used for Constant strategy.
    /// </summary>
    public T? FillValue => _fillValue;

    /// <summary>
    /// Gets the value considered as missing.
    /// </summary>
    public T MissingValue => _missingValue;

    /// <summary>
    /// Gets the computed statistics for each feature.
    /// </summary>
    public Vector<T>? Statistics => _statistics;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    /// <remarks>
    /// Inverse transform is not supported because we don't know which values were originally missing.
    /// </remarks>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="SimpleImputer{T}"/>.
    /// </summary>
    /// <param name="strategy">The imputation strategy. Defaults to Mean.</param>
    /// <param name="fillValue">The fill value for Constant strategy. Ignored for other strategies.</param>
    /// <param name="columnIndices">The column indices to impute, or null for all columns.</param>
    public SimpleImputer(
        ImputationStrategy strategy = ImputationStrategy.Mean,
        T? fillValue = default,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _strategy = strategy;
        _fillValue = fillValue;
        _missingValue = NumOps.FromDouble(double.NaN);

        if (strategy == ImputationStrategy.Constant && fillValue is null)
        {
            throw new ArgumentException("fillValue must be provided when using Constant strategy.", nameof(fillValue));
        }
    }

    /// <summary>
    /// Computes the statistics for each feature from the training data.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    protected override void FitCore(Matrix<T> data)
    {
        int numColumns = data.Columns;
        var columnsToProcess = GetColumnsToProcess(numColumns);
        var stats = new T[numColumns];

        // Default value for columns not processed
        for (int i = 0; i < numColumns; i++)
        {
            stats[i] = NumOps.Zero;
        }

        foreach (var col in columnsToProcess)
        {
            var column = data.GetColumn(col);
            stats[col] = ComputeStatistic(column);
        }

        _statistics = new Vector<T>(stats);
    }

    private T ComputeStatistic(Vector<T> column)
    {
        // Filter out missing values
        var validValues = new List<T>();
        for (int i = 0; i < column.Length; i++)
        {
            if (!IsMissing(column[i]))
            {
                validValues.Add(column[i]);
            }
        }

        if (validValues.Count == 0)
        {
            // All values are missing, return default
            return _strategy == ImputationStrategy.Constant && _fillValue is not null
                ? _fillValue
                : NumOps.Zero;
        }

        switch (_strategy)
        {
            case ImputationStrategy.Mean:
                return ComputeMean(validValues);
            case ImputationStrategy.Median:
                return ComputeMedian(validValues);
            case ImputationStrategy.MostFrequent:
                return ComputeMostFrequent(validValues);
            case ImputationStrategy.Constant:
                return _fillValue ?? NumOps.Zero;
            default:
                throw new ArgumentException($"Unknown imputation strategy: {_strategy}");
        }
    }

    private T ComputeMean(List<T> values)
    {
        T sum = NumOps.Zero;
        foreach (var value in values)
        {
            sum = NumOps.Add(sum, value);
        }
        return NumOps.Divide(sum, NumOps.FromDouble(values.Count));
    }

    private T ComputeMedian(List<T> values)
    {
        var sorted = values.ToArray();
        Array.Sort(sorted, (a, b) => NumOps.Compare(a, b));

        int mid = sorted.Length / 2;
        if (sorted.Length % 2 == 0)
        {
            return NumOps.Divide(NumOps.Add(sorted[mid - 1], sorted[mid]), NumOps.FromDouble(2.0));
        }
        return sorted[mid];
    }

    private T ComputeMostFrequent(List<T> values)
    {
        var counts = new Dictionary<double, int>();
        foreach (var value in values)
        {
            double key = NumOps.ToDouble(value);
            if (counts.ContainsKey(key))
            {
                counts[key]++;
            }
            else
            {
                counts[key] = 1;
            }
        }

        double mostFrequent = 0;
        int maxCount = 0;
        foreach (var kvp in counts)
        {
            if (kvp.Value > maxCount)
            {
                maxCount = kvp.Value;
                mostFrequent = kvp.Key;
            }
        }

        return NumOps.FromDouble(mostFrequent);
    }

    private bool IsMissing(T value)
    {
        return NumOps.IsNaN(value);
    }

    /// <summary>
    /// Transforms the data by imputing missing values.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The data with missing values imputed.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_statistics is null)
        {
            throw new InvalidOperationException("Imputer has not been fitted.");
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

                if (processSet.Contains(j) && IsMissing(value))
                {
                    value = _statistics[j];
                }

                result[i, j] = value;
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Inverse transformation is not supported for imputation.
    /// </summary>
    /// <param name="data">The imputed data.</param>
    /// <returns>Never returns - always throws.</returns>
    /// <exception cref="NotSupportedException">Always thrown because we don't track which values were missing.</exception>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException(
            "SimpleImputer does not support inverse transformation. " +
            "We don't track which values were originally missing.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names.</param>
    /// <returns>The same feature names (SimpleImputer doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
