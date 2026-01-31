using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.OutlierHandling;

/// <summary>
/// Clips outliers based on Z-score (standard deviation) bounds.
/// </summary>
/// <remarks>
/// <para>
/// ZScoreClipper identifies outliers as values that deviate from the mean by more than
/// a specified number of standard deviations. Values beyond this threshold are clipped
/// to the boundary values.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// For each feature:
/// 1. Calculate the mean and standard deviation
/// 2. Compute bounds: [mean - threshold*std, mean + threshold*std]
/// 3. Clip values outside these bounds
/// </para>
/// <para>
/// <b>For Beginners:</b> Z-score measures how many standard deviations a value is from the mean.
/// - A Z-score of 0 means the value equals the mean
/// - A Z-score of 2 means the value is 2 standard deviations above the mean
/// - A Z-score of -3 means the value is 3 standard deviations below the mean
///
/// This clipper replaces extreme values (those with high absolute Z-scores) with
/// the boundary values, reducing the impact of outliers while preserving data size.
/// </para>
/// <para>
/// <b>Common Thresholds:</b>
/// - 2.0: Aggressive (clips ~5% of normally distributed data)
/// - 3.0: Standard (clips ~0.3% of normally distributed data)
/// - 3.5: Conservative (clips ~0.05% of normally distributed data)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class ZScoreClipper<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _threshold;

    // Fitted parameters
    private double[]? _means;
    private double[]? _stds;
    private double[]? _lowerBounds;
    private double[]? _upperBounds;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the Z-score threshold for clipping.
    /// </summary>
    public double Threshold => _threshold;

    /// <summary>
    /// Gets the computed means for each feature.
    /// </summary>
    public double[]? Means => _means;

    /// <summary>
    /// Gets the computed standard deviations for each feature.
    /// </summary>
    public double[]? StandardDeviations => _stds;

    /// <summary>
    /// Gets the computed lower bounds for each feature.
    /// </summary>
    public double[]? LowerBounds => _lowerBounds;

    /// <summary>
    /// Gets the computed upper bounds for each feature.
    /// </summary>
    public double[]? UpperBounds => _upperBounds;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="ZScoreClipper{T}"/>.
    /// </summary>
    /// <param name="threshold">The Z-score threshold. Values beyond mean +/- threshold*std are clipped. Defaults to 3.0.</param>
    /// <param name="columnIndices">The column indices to clip, or null for all columns.</param>
    public ZScoreClipper(double threshold = 3.0, int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (threshold <= 0)
        {
            throw new ArgumentException("Threshold must be positive.", nameof(threshold));
        }

        _threshold = threshold;
    }

    /// <summary>
    /// Computes the mean and standard deviation for each feature.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);
        var processSet = new HashSet<int>(columnsToProcess);

        _means = new double[_nInputFeatures];
        _stds = new double[_nInputFeatures];
        _lowerBounds = new double[_nInputFeatures];
        _upperBounds = new double[_nInputFeatures];

        for (int col = 0; col < _nInputFeatures; col++)
        {
            if (!processSet.Contains(col))
            {
                // Pass-through columns: use extreme bounds
                _means[col] = 0;
                _stds[col] = 1;
                _lowerBounds[col] = double.MinValue;
                _upperBounds[col] = double.MaxValue;
                continue;
            }

            // Calculate mean
            double sum = 0;
            for (int i = 0; i < data.Rows; i++)
            {
                sum += NumOps.ToDouble(data[i, col]);
            }
            double mean = sum / data.Rows;
            _means[col] = mean;

            // Calculate standard deviation
            double sumSquaredDiff = 0;
            for (int i = 0; i < data.Rows; i++)
            {
                double diff = NumOps.ToDouble(data[i, col]) - mean;
                sumSquaredDiff += diff * diff;
            }
            double std = Math.Sqrt(sumSquaredDiff / data.Rows);
            _stds[col] = std;

            // Calculate bounds
            if (std > 0)
            {
                _lowerBounds[col] = mean - _threshold * std;
                _upperBounds[col] = mean + _threshold * std;
            }
            else
            {
                // If std is 0, all values are the same - no clipping needed
                _lowerBounds[col] = double.MinValue;
                _upperBounds[col] = double.MaxValue;
            }
        }
    }

    /// <summary>
    /// Clips values to the computed Z-score bounds.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The data with outliers clipped.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_lowerBounds is null || _upperBounds is null)
        {
            throw new InvalidOperationException("ZScoreClipper has not been fitted.");
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
                double value = NumOps.ToDouble(data[i, j]);

                if (processSet.Contains(j))
                {
                    // Clip to bounds
                    value = Math.Max(_lowerBounds[j], Math.Min(_upperBounds[j], value));
                }

                result[i, j] = NumOps.FromDouble(value);
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ZScoreClipper does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }

    /// <summary>
    /// Calculates the Z-scores for all values in the data.
    /// </summary>
    /// <param name="data">The data to calculate Z-scores for.</param>
    /// <returns>A matrix of Z-scores.</returns>
    public Matrix<T> GetZScores(Matrix<T> data)
    {
        if (_means is null || _stds is null)
        {
            throw new InvalidOperationException("ZScoreClipper has not been fitted.");
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
                double value = NumOps.ToDouble(data[i, j]);
                double zScore;

                if (processSet.Contains(j) && _stds[j] > 0)
                {
                    zScore = (value - _means[j]) / _stds[j];
                }
                else
                {
                    zScore = 0;
                }

                result[i, j] = NumOps.FromDouble(zScore);
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Gets a boolean mask indicating which values are outliers.
    /// </summary>
    /// <param name="data">The data to check for outliers.</param>
    /// <returns>A 2D boolean array where true indicates an outlier.</returns>
    public bool[,] GetOutlierMask(Matrix<T> data)
    {
        if (_lowerBounds is null || _upperBounds is null)
        {
            throw new InvalidOperationException("ZScoreClipper has not been fitted.");
        }

        int numRows = data.Rows;
        int numColumns = data.Columns;
        var mask = new bool[numRows, numColumns];
        var columnsToProcess = GetColumnsToProcess(numColumns);
        var processSet = new HashSet<int>(columnsToProcess);

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numColumns; j++)
            {
                if (processSet.Contains(j))
                {
                    double value = NumOps.ToDouble(data[i, j]);
                    mask[i, j] = value < _lowerBounds[j] || value > _upperBounds[j];
                }
            }
        }

        return mask;
    }
}
