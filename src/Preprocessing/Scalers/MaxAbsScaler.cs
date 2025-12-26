using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Scalers;

/// <summary>
/// Scales each feature by its maximum absolute value.
/// </summary>
/// <remarks>
/// <para>
/// Max absolute scaling transforms each feature by dividing by its maximum absolute value,
/// resulting in features with values in the range [-1, 1]. This scaler does not shift or center
/// the data, so it preserves sparsity.
/// </para>
/// <para><b>For Beginners:</b> This scaler divides each feature by its largest absolute value:
/// - The largest value (positive or negative) becomes 1 or -1
/// - All other values are proportionally scaled
/// - Zero values remain zero (preserves sparsity)
///
/// This is useful when:
/// - You have sparse data and want to preserve zeros
/// - You don't want to center your data
/// - You want values bounded between -1 and 1
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class MaxAbsScaler<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private Vector<T>? _maxAbs;

    /// <summary>
    /// Gets the maximum absolute value of each feature computed during fitting.
    /// </summary>
    public Vector<T>? MaxAbsolute => _maxAbs;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="MaxAbsScaler{T}"/>.
    /// </summary>
    /// <param name="columnIndices">The column indices to scale, or null for all columns.</param>
    public MaxAbsScaler(int[]? columnIndices = null)
        : base(columnIndices)
    {
    }

    /// <summary>
    /// Computes the maximum absolute value of each feature from the training data.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    protected override void FitCore(Matrix<T> data)
    {
        int numColumns = data.Columns;
        var columnsToProcess = GetColumnsToProcess(numColumns);

        // Initialize array for all columns
        var maxAbsValues = new T[numColumns];

        // Default value for columns not processed (1.0 = no scaling)
        for (int i = 0; i < numColumns; i++)
        {
            maxAbsValues[i] = NumOps.One;
        }

        // Compute max absolute value for specified columns
        foreach (var colIdx in columnsToProcess)
        {
            var column = data.GetColumn(colIdx);

            T maxAbs = NumOps.Zero;

            for (int i = 0; i < column.Length; i++)
            {
                T absValue = NumOps.Abs(column[i]);
                if (NumOps.Compare(absValue, maxAbs) > 0)
                {
                    maxAbs = absValue;
                }
            }

            // Prevent division by zero - if max abs is zero, use 1 (no scaling)
            maxAbsValues[colIdx] = NumOps.Compare(maxAbs, NumOps.Zero) == 0 ? NumOps.One : maxAbs;
        }

        _maxAbs = new Vector<T>(maxAbsValues);
    }

    /// <summary>
    /// Transforms the data by applying max absolute scaling.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The scaled data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_maxAbs is null)
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
                    // Scale by dividing by max absolute value
                    value = NumOps.Divide(value, _maxAbs[j]);
                }

                result[i, j] = value;
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Reverses the max absolute scaling transformation.
    /// </summary>
    /// <param name="data">The scaled data.</param>
    /// <returns>The original-scale data.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_maxAbs is null)
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
                    // Reverse by multiplying by max absolute value
                    value = NumOps.Multiply(value, _maxAbs[j]);
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
    /// <returns>The same feature names (MaxAbsScaler doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
