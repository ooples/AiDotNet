using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Scalers;

/// <summary>
/// Applies logarithmic transformation followed by mean-variance standardization.
/// </summary>
/// <remarks>
/// <para>
/// Log-mean-variance scaling combines logarithmic transformation with z-score standardization.
/// It first applies log transformation (with shift for negative values), then standardizes
/// using mean and standard deviation. This is ideal for data spanning multiple orders of magnitude.
/// </para>
/// <para><b>For Beginners:</b> This scaler is perfect for highly skewed or exponentially distributed data:
/// - First, it takes the log of values (compressing large differences)
/// - Then, it standardizes to zero mean and unit variance
///
/// Example: [1000, 10000, 100000, 1000000] → after log: [6.9, 9.2, 11.5, 13.8] → standardized
/// This makes patterns in exponential data much easier to detect.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class LogMeanVarianceScaler<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private Vector<T>? _shift;
    private Vector<T>? _logMean;
    private Vector<T>? _logStdDev;

    private readonly T _epsilon;

    /// <summary>
    /// Gets the shift applied to each feature to ensure positive values.
    /// </summary>
    public Vector<T>? Shift => _shift;

    /// <summary>
    /// Gets the mean of log-transformed values for each feature.
    /// </summary>
    public Vector<T>? LogMean => _logMean;

    /// <summary>
    /// Gets the standard deviation of log-transformed values for each feature.
    /// </summary>
    public Vector<T>? LogStdDev => _logStdDev;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="LogMeanVarianceScaler{T}"/>.
    /// </summary>
    /// <param name="columnIndices">The column indices to scale, or null for all columns.</param>
    public LogMeanVarianceScaler(int[]? columnIndices = null)
        : base(columnIndices)
    {
        _epsilon = NumOps.FromDouble(1e-10);
    }

    /// <summary>
    /// Computes the shift, log mean, and log standard deviation for each feature.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    protected override void FitCore(Matrix<T> data)
    {
        int numColumns = data.Columns;
        var columnsToProcess = GetColumnsToProcess(numColumns);

        // Initialize arrays for all columns
        var shifts = new T[numColumns];
        var logMeans = new T[numColumns];
        var logStdDevs = new T[numColumns];

        // Default values for columns not processed
        for (int i = 0; i < numColumns; i++)
        {
            shifts[i] = NumOps.Zero;
            logMeans[i] = NumOps.Zero;
            logStdDevs[i] = NumOps.One;
        }

        // Compute parameters for specified columns
        foreach (var colIdx in columnsToProcess)
        {
            var column = data.GetColumn(colIdx);

            // Find minimum value
            T minValue = column[0];
            for (int i = 1; i < column.Length; i++)
            {
                if (NumOps.Compare(column[i], minValue) < 0)
                {
                    minValue = column[i];
                }
            }

            // Calculate shift to ensure positive values
            T shift = NumOps.Compare(minValue, NumOps.Zero) > 0
                ? NumOps.Zero
                : NumOps.Add(NumOps.Add(NumOps.Negate(minValue), NumOps.One), _epsilon);

            // Compute log-transformed values and their mean
            var logValues = new T[column.Length];
            T logSum = NumOps.Zero;
            for (int i = 0; i < column.Length; i++)
            {
                logValues[i] = NumOps.Log(NumOps.Add(column[i], shift));
                logSum = NumOps.Add(logSum, logValues[i]);
            }
            T logMean = NumOps.Divide(logSum, NumOps.FromDouble(column.Length));

            // Calculate variance and std dev of log values
            T varianceSum = NumOps.Zero;
            for (int i = 0; i < column.Length; i++)
            {
                T diff = NumOps.Subtract(logValues[i], logMean);
                varianceSum = NumOps.Add(varianceSum, NumOps.Multiply(diff, diff));
            }
            T variance = NumOps.Divide(varianceSum, NumOps.FromDouble(column.Length));
            T logStdDev = NumOps.Sqrt(NumOps.Compare(variance, _epsilon) > 0 ? variance : _epsilon);

            shifts[colIdx] = shift;
            logMeans[colIdx] = logMean;
            logStdDevs[colIdx] = logStdDev;
        }

        _shift = new Vector<T>(shifts);
        _logMean = new Vector<T>(logMeans);
        _logStdDev = new Vector<T>(logStdDevs);
    }

    /// <summary>
    /// Transforms the data by applying log transformation and standardization.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The log-standardized data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_shift is null || _logMean is null || _logStdDev is null)
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
                    // Apply shift, log, then standardize
                    T shiftedValue = NumOps.Add(value, _shift[j]);

                    // Ensure positive for log
                    if (NumOps.Compare(shiftedValue, NumOps.Zero) <= 0)
                    {
                        value = NumOps.Zero;
                    }
                    else
                    {
                        T logValue = NumOps.Log(shiftedValue);
                        value = NumOps.Divide(NumOps.Subtract(logValue, _logMean[j]), _logStdDev[j]);

                        // Handle NaN
                        if (NumOps.IsNaN(value))
                        {
                            value = NumOps.Zero;
                        }
                    }
                }

                result[i, j] = value;
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Reverses the log-mean-variance scaling transformation.
    /// </summary>
    /// <param name="data">The scaled data.</param>
    /// <returns>The original-scale data.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_shift is null || _logMean is null || _logStdDev is null)
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
                    // Reverse: un-standardize, exp, then remove shift
                    T logValue = NumOps.Add(NumOps.Multiply(value, _logStdDev[j]), _logMean[j]);
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
    /// <returns>The same feature names (LogMeanVarianceScaler doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
