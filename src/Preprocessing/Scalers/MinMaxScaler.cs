using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Scalers;

/// <summary>
/// Scales features to a given range, typically [0, 1].
/// </summary>
/// <remarks>
/// <para>
/// Min-max scaling transforms features by scaling each feature to a given range.
/// The default range is [0, 1]. The transformation is: X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
/// </para>
/// <para><b>For Beginners:</b> This scaler squishes all your data into a specific range:
/// - The smallest value becomes the minimum of the range (default 0)
/// - The largest value becomes the maximum of the range (default 1)
/// - Everything else is proportionally scaled in between
///
/// This is useful when:
/// - Your algorithm requires data in a specific range
/// - You want to preserve the relationships between values
/// - You don't want outliers to heavily influence the scaling
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class MinMaxScaler<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private Vector<T>? _dataMin;
    private Vector<T>? _dataMax;
    private readonly T _featureRangeMin;
    private readonly T _featureRangeMax;

    /// <summary>
    /// Gets the minimum value of each feature computed during fitting.
    /// </summary>
    public Vector<T>? DataMin => _dataMin;

    /// <summary>
    /// Gets the maximum value of each feature computed during fitting.
    /// </summary>
    public Vector<T>? DataMax => _dataMax;

    /// <summary>
    /// Gets the minimum value of the target feature range.
    /// </summary>
    public T FeatureRangeMin => _featureRangeMin;

    /// <summary>
    /// Gets the maximum value of the target feature range.
    /// </summary>
    public T FeatureRangeMax => _featureRangeMax;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="MinMaxScaler{T}"/> with default range [0, 1].
    /// </summary>
    /// <param name="columnIndices">The column indices to scale, or null for all columns.</param>
    public MinMaxScaler(int[]? columnIndices = null)
        : this(0.0, 1.0, columnIndices)
    {
    }

    /// <summary>
    /// Creates a new instance of <see cref="MinMaxScaler{T}"/> with a custom range.
    /// </summary>
    /// <param name="featureRangeMin">The minimum value of the target range.</param>
    /// <param name="featureRangeMax">The maximum value of the target range.</param>
    /// <param name="columnIndices">The column indices to scale, or null for all columns.</param>
    public MinMaxScaler(double featureRangeMin, double featureRangeMax, int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (featureRangeMin >= featureRangeMax)
        {
            throw new ArgumentException(
                $"Feature range minimum ({featureRangeMin}) must be less than maximum ({featureRangeMax}).");
        }

        _featureRangeMin = NumOps.FromDouble(featureRangeMin);
        _featureRangeMax = NumOps.FromDouble(featureRangeMax);
    }

    /// <summary>
    /// Computes the minimum and maximum of each feature from the training data.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    protected override void FitCore(Matrix<T> data)
    {
        int numColumns = data.Columns;
        var columnsToProcess = GetColumnsToProcess(numColumns);

        // Initialize arrays for all columns
        var mins = new T[numColumns];
        var maxs = new T[numColumns];

        // Default values for columns not processed
        for (int i = 0; i < numColumns; i++)
        {
            mins[i] = NumOps.Zero;
            maxs[i] = NumOps.One;
        }

        // Compute min and max for specified columns
        foreach (var colIdx in columnsToProcess)
        {
            var column = data.GetColumn(colIdx);

            T min = column[0];
            T max = column[0];

            for (int i = 1; i < column.Length; i++)
            {
                if (NumOps.Compare(column[i], min) < 0)
                {
                    min = column[i];
                }
                if (NumOps.Compare(column[i], max) > 0)
                {
                    max = column[i];
                }
            }

            mins[colIdx] = min;
            maxs[colIdx] = max;
        }

        _dataMin = new Vector<T>(mins);
        _dataMax = new Vector<T>(maxs);
    }

    /// <summary>
    /// Transforms the data by applying min-max scaling.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The scaled data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_dataMin is null || _dataMax is null)
        {
            throw new InvalidOperationException("Scaler has not been fitted.");
        }

        int numRows = data.Rows;
        int numColumns = data.Columns;
        var result = new T[numRows, numColumns];
        var columnsToProcess = GetColumnsToProcess(numColumns);
        var processSet = new HashSet<int>(columnsToProcess);

        // Compute range for feature scaling: (featureMax - featureMin)
        T featureRange = NumOps.Subtract(_featureRangeMax, _featureRangeMin);

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numColumns; j++)
            {
                T value = data[i, j];

                if (processSet.Contains(j))
                {
                    T dataRange = NumOps.Subtract(_dataMax[j], _dataMin[j]);

                    // Handle case where min == max (constant column)
                    if (NumOps.Compare(dataRange, NumOps.Zero) == 0)
                    {
                        // If constant, map to middle of feature range
                        value = NumOps.Divide(
                            NumOps.Add(_featureRangeMin, _featureRangeMax),
                            NumOps.FromDouble(2.0));
                    }
                    else
                    {
                        // X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
                        T normalized = NumOps.Divide(
                            NumOps.Subtract(value, _dataMin[j]),
                            dataRange);
                        value = NumOps.Add(
                            NumOps.Multiply(normalized, featureRange),
                            _featureRangeMin);
                    }
                }

                result[i, j] = value;
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Reverses the min-max scaling transformation.
    /// </summary>
    /// <param name="data">The scaled data.</param>
    /// <returns>The original-scale data.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_dataMin is null || _dataMax is null)
        {
            throw new InvalidOperationException("Scaler has not been fitted.");
        }

        int numRows = data.Rows;
        int numColumns = data.Columns;
        var result = new T[numRows, numColumns];
        var columnsToProcess = GetColumnsToProcess(numColumns);
        var processSet = new HashSet<int>(columnsToProcess);

        // Compute range for feature scaling: (featureMax - featureMin)
        T featureRange = NumOps.Subtract(_featureRangeMax, _featureRangeMin);

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numColumns; j++)
            {
                T value = data[i, j];

                if (processSet.Contains(j))
                {
                    T dataRange = NumOps.Subtract(_dataMax[j], _dataMin[j]);

                    // Handle case where min == max (constant column)
                    if (NumOps.Compare(dataRange, NumOps.Zero) == 0)
                    {
                        // Return the constant value
                        value = _dataMin[j];
                    }
                    else
                    {
                        // X = (X_scaled - min) / (max - min) * (X_max - X_min) + X_min
                        T normalized = NumOps.Divide(
                            NumOps.Subtract(value, _featureRangeMin),
                            featureRange);
                        value = NumOps.Add(
                            NumOps.Multiply(normalized, dataRange),
                            _dataMin[j]);
                    }
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
    /// <returns>The same feature names (MinMaxScaler doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
