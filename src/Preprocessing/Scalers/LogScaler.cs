using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Scalers;

/// <summary>
/// Applies logarithmic transformation to features, useful for data spanning multiple orders of magnitude.
/// </summary>
/// <remarks>
/// <para>
/// Log scaling transforms data using natural logarithm, which compresses the range of values.
/// It shifts negative values to ensure all inputs are positive, then applies log scaling.
/// This is particularly useful for exponentially distributed data.
/// </para>
/// <para><b>For Beginners:</b> Log normalization measures percentages rather than absolute amounts:
/// - With regular measurement, going from 1 to 10 and from 10 to 100 look very different
/// - With logarithmic measurement, both represent a "10× increase" and appear as equal steps
///
/// Example: [1,000, 10,000, 100,000, 1,000,000] becomes evenly spaced values.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class LogScaler<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private Vector<T>? _shift;
    private Vector<T>? _logMin;
    private Vector<T>? _logRange;

    /// <summary>
    /// Gets the shift applied to each feature to ensure positive values.
    /// </summary>
    public Vector<T>? Shift => _shift;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="LogScaler{T}"/>.
    /// </summary>
    /// <param name="columnIndices">The column indices to scale, or null for all columns.</param>
    public LogScaler(int[]? columnIndices = null)
        : base(columnIndices)
    {
    }

    /// <summary>
    /// Computes the shift and log range for each feature from the training data.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    protected override void FitCore(Matrix<T> data)
    {
        int numColumns = data.Columns;
        var columnsToProcess = GetColumnsToProcess(numColumns);

        // Initialize arrays for all columns
        var shifts = new T[numColumns];
        var logMins = new T[numColumns];
        var logRanges = new T[numColumns];

        // Default values for columns not processed
        for (int i = 0; i < numColumns; i++)
        {
            shifts[i] = NumOps.Zero;
            logMins[i] = NumOps.Zero;
            logRanges[i] = NumOps.One;
        }

        // Compute parameters for specified columns
        foreach (var colIdx in columnsToProcess)
        {
            var column = data.GetColumn(colIdx);

            // Find min and max
            T min = column[0];
            T max = column[0];
            for (int i = 1; i < column.Length; i++)
            {
                if (NumOps.Compare(column[i], min) < 0) min = column[i];
                if (NumOps.Compare(column[i], max) > 0) max = column[i];
            }

            // Calculate shift to ensure positive values
            T shift = NumOps.Compare(min, NumOps.Zero) > 0
                ? NumOps.Zero
                : NumOps.Add(NumOps.Negate(min), NumOps.One);

            T shiftedMin = NumOps.Add(min, shift);
            T shiftedMax = NumOps.Add(max, shift);

            T logMin = NumOps.Log(shiftedMin);
            T logMax = NumOps.Log(shiftedMax);
            T logRange = NumOps.Subtract(logMax, logMin);

            // Prevent division by zero
            if (NumOps.Compare(logRange, NumOps.Zero) == 0)
            {
                logRange = NumOps.One;
            }

            shifts[colIdx] = shift;
            logMins[colIdx] = logMin;
            logRanges[colIdx] = logRange;
        }

        _shift = new Vector<T>(shifts);
        _logMin = new Vector<T>(logMins);
        _logRange = new Vector<T>(logRanges);
    }

    /// <summary>
    /// Transforms the data by applying log scaling.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The log-scaled data normalized to [0, 1].</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_shift is null || _logMin is null || _logRange is null)
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
                    T shiftedValue = NumOps.Add(value, _shift[j]);

                    // Ensure positive for log
                    if (NumOps.Compare(shiftedValue, NumOps.Zero) <= 0)
                    {
                        value = NumOps.Zero;
                    }
                    else
                    {
                        T logValue = NumOps.Log(shiftedValue);
                        value = NumOps.Divide(NumOps.Subtract(logValue, _logMin[j]), _logRange[j]);
                    }
                }

                result[i, j] = value;
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Reverses the log scaling transformation.
    /// </summary>
    /// <param name="data">The scaled data.</param>
    /// <returns>The original-scale data.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_shift is null || _logMin is null || _logRange is null)
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
                    // Reverse: exp(value * logRange + logMin) - shift
                    T logValue = NumOps.Add(NumOps.Multiply(value, _logRange[j]), _logMin[j]);
                    T expValue = NumOps.Exp(logValue);
                    value = NumOps.Subtract(expValue, _shift[j]);
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
    /// <returns>The same feature names (LogScaler doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
