using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Scalers;

/// <summary>
/// Standardizes features by removing the mean and scaling to unit variance.
/// </summary>
/// <remarks>
/// <para>
/// Standard scaling (Z-score normalization) transforms data to have a mean of 0 and a
/// standard deviation of 1. This is important for many machine learning algorithms as
/// it puts different features on comparable scales.
/// </para>
/// <para><b>For Beginners:</b> This scaler converts your data to a standard scale:
/// - The center of your data (mean) becomes 0
/// - The spread of your data (standard deviation) becomes 1
///
/// This is like converting different currencies to a common one - it makes
/// different features comparable and helps many ML algorithms work better.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class StandardScaler<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private Vector<T>? _mean;
    private Vector<T>? _stdDev;
    private readonly bool _withMean;
    private readonly bool _withStd;

    /// <summary>
    /// Gets the mean of each feature computed during fitting.
    /// </summary>
    public Vector<T>? Mean => _mean;

    /// <summary>
    /// Gets the standard deviation of each feature computed during fitting.
    /// </summary>
    public Vector<T>? StandardDeviation => _stdDev;

    /// <summary>
    /// Gets whether this scaler centers the data (subtracts mean).
    /// </summary>
    public bool WithMean => _withMean;

    /// <summary>
    /// Gets whether this scaler scales the data (divides by std).
    /// </summary>
    public bool WithStd => _withStd;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="StandardScaler{T}"/>.
    /// </summary>
    /// <param name="withMean">If true, center the data before scaling (subtract mean). Default is true.</param>
    /// <param name="withStd">If true, scale the data to unit variance. Default is true.</param>
    /// <param name="columnIndices">The column indices to scale, or null for all columns.</param>
    public StandardScaler(bool withMean = true, bool withStd = true, int[]? columnIndices = null)
        : base(columnIndices)
    {
        _withMean = withMean;
        _withStd = withStd;
    }

    /// <summary>
    /// Computes the mean and standard deviation of each feature from the training data.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    protected override void FitCore(Matrix<T> data)
    {
        int numColumns = data.Columns;
        var columnsToProcess = GetColumnsToProcess(numColumns);

        // Initialize arrays for all columns
        var means = new T[numColumns];
        var stdDevs = new T[numColumns];

        // Default values for columns not processed
        for (int i = 0; i < numColumns; i++)
        {
            means[i] = NumOps.Zero;
            stdDevs[i] = NumOps.One;
        }

        // Compute mean and std for specified columns
        foreach (var colIdx in columnsToProcess)
        {
            var column = data.GetColumn(colIdx);

            if (_withMean || _withStd)
            {
                means[colIdx] = StatisticsHelper<T>.CalculateMean(column);
            }

            if (_withStd)
            {
                T variance = StatisticsHelper<T>.CalculateVariance(column, means[colIdx]);
                T std = NumOps.Sqrt(variance);

                // Prevent division by zero - if std is zero, use 1 (no scaling)
                stdDevs[colIdx] = NumOps.Compare(std, NumOps.Zero) == 0 ? NumOps.One : std;
            }
        }

        _mean = new Vector<T>(means);
        _stdDev = new Vector<T>(stdDevs);
    }

    /// <summary>
    /// Transforms the data by applying the computed mean and standard deviation.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The standardized data.</returns>
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

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numColumns; j++)
            {
                T value = data[i, j];

                if (processSet.Contains(j))
                {
                    // Apply standardization
                    if (_withMean)
                    {
                        value = NumOps.Subtract(value, _mean[j]);
                    }

                    if (_withStd)
                    {
                        value = NumOps.Divide(value, _stdDev[j]);
                    }
                }

                result[i, j] = value;
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Reverses the standardization transformation.
    /// </summary>
    /// <param name="data">The standardized data.</param>
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

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numColumns; j++)
            {
                T value = data[i, j];

                if (processSet.Contains(j))
                {
                    // Reverse standardization: x = z * std + mean
                    if (_withStd)
                    {
                        value = NumOps.Multiply(value, _stdDev[j]);
                    }

                    if (_withMean)
                    {
                        value = NumOps.Add(value, _mean[j]);
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
    /// <returns>The same feature names (StandardScaler doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
