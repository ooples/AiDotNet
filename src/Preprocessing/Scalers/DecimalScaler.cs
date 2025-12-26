using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Scalers;

/// <summary>
/// Scales features by dividing by the smallest power of 10 greater than the max absolute value.
/// </summary>
/// <remarks>
/// <para>
/// Decimal scaling transforms values to fall between -1 and 1 by dividing by an appropriate power of 10.
/// This preserves the relative decimal positions and signs of values.
/// </para>
/// <para><b>For Beginners:</b> This scaler adjusts numbers to show them in appropriate decimal places:
/// - If your largest value is 750, it divides everything by 1,000
/// - So 750 becomes 0.75, 42 becomes 0.042, etc.
/// - All values end up between -1 and 1
///
/// This is useful when you want to keep relative sizes clear and decimal places meaningful.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class DecimalScaler<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private Vector<T>? _scale;

    /// <summary>
    /// Gets the power-of-10 scale factor for each feature computed during fitting.
    /// </summary>
    public Vector<T>? Scale => _scale;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="DecimalScaler{T}"/>.
    /// </summary>
    /// <param name="columnIndices">The column indices to scale, or null for all columns.</param>
    public DecimalScaler(int[]? columnIndices = null)
        : base(columnIndices)
    {
    }

    /// <summary>
    /// Computes the power-of-10 scale factor for each feature from the training data.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    protected override void FitCore(Matrix<T> data)
    {
        int numColumns = data.Columns;
        var columnsToProcess = GetColumnsToProcess(numColumns);

        // Initialize array for all columns
        var scales = new T[numColumns];

        // Default value for columns not processed (1.0 = no scaling)
        for (int i = 0; i < numColumns; i++)
        {
            scales[i] = NumOps.One;
        }

        T ten = NumOps.FromDouble(10.0);

        // Compute scale for specified columns
        foreach (var colIdx in columnsToProcess)
        {
            var column = data.GetColumn(colIdx);

            // Find max absolute value
            T maxAbs = NumOps.Zero;
            for (int i = 0; i < column.Length; i++)
            {
                T absValue = NumOps.Abs(column[i]);
                if (NumOps.Compare(absValue, maxAbs) > 0)
                {
                    maxAbs = absValue;
                }
            }

            // Find smallest power of 10 greater than or equal to maxAbs
            T scale = NumOps.One;
            while (NumOps.Compare(maxAbs, scale) >= 0)
            {
                scale = NumOps.Multiply(scale, ten);
            }

            scales[colIdx] = scale;
        }

        _scale = new Vector<T>(scales);
    }

    /// <summary>
    /// Transforms the data by applying decimal scaling.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The scaled data with values between -1 and 1.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_scale is null)
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
                    value = NumOps.Divide(value, _scale[j]);
                }

                result[i, j] = value;
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Reverses the decimal scaling transformation.
    /// </summary>
    /// <param name="data">The scaled data.</param>
    /// <returns>The original-scale data.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_scale is null)
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
                    value = NumOps.Multiply(value, _scale[j]);
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
    /// <returns>The same feature names (DecimalScaler doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
