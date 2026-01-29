using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.OutlierHandling;

/// <summary>
/// Clips outliers based on Median Absolute Deviation (MAD) bounds.
/// </summary>
/// <remarks>
/// <para>
/// MADClipper identifies outliers using the Median Absolute Deviation, which is more robust
/// to outliers than standard deviation-based methods. Values with modified Z-scores exceeding
/// the threshold are clipped to the boundary values.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// For each feature:
/// 1. Calculate the median
/// 2. Calculate MAD = median(|x - median|)
/// 3. Calculate modified Z-score = 0.6745 * (x - median) / MAD
/// 4. Clip values where |modified Z-score| > threshold
/// </para>
/// <para>
/// <b>Why MAD is Better Than Z-Score for Outliers:</b>
/// The traditional Z-score uses mean and standard deviation, which are themselves
/// affected by outliers. MAD uses medians, which are resistant to extreme values.
/// This makes MAD better at detecting outliers when your data already contains
/// significant outliers.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of the median as the "middle value" when you sort your data.
/// MAD measures how spread out your data is from this middle value. Unlike the average,
/// the median isn't pulled toward extreme values, so MAD gives a more reliable measure
/// of spread when outliers are present.
/// </para>
/// <para>
/// <b>Common Thresholds:</b>
/// - 2.5: Aggressive (more values treated as outliers)
/// - 3.5: Standard (recommended default)
/// - 5.0: Conservative (only extreme outliers)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class MADClipper<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    /// <summary>
    /// Constant for converting MAD to standard deviation equivalent (1 / 0.6745).
    /// </summary>
    private const double MADScaleFactor = 1.4826;

    private readonly double _threshold;

    // Fitted parameters
    private double[]? _medians;
    private double[]? _mads;
    private double[]? _lowerBounds;
    private double[]? _upperBounds;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the modified Z-score threshold for clipping.
    /// </summary>
    public double Threshold => _threshold;

    /// <summary>
    /// Gets the computed medians for each feature.
    /// </summary>
    public double[]? Medians => _medians;

    /// <summary>
    /// Gets the computed MAD values for each feature.
    /// </summary>
    public double[]? MADs => _mads;

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
    /// Creates a new instance of <see cref="MADClipper{T}"/>.
    /// </summary>
    /// <param name="threshold">The modified Z-score threshold. Values beyond this are clipped. Defaults to 3.5.</param>
    /// <param name="columnIndices">The column indices to clip, or null for all columns.</param>
    public MADClipper(double threshold = 3.5, int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (threshold <= 0)
        {
            throw new ArgumentException("Threshold must be positive.", nameof(threshold));
        }

        _threshold = threshold;
    }

    /// <summary>
    /// Computes the median and MAD for each feature.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);
        var processSet = new HashSet<int>(columnsToProcess);

        _medians = new double[_nInputFeatures];
        _mads = new double[_nInputFeatures];
        _lowerBounds = new double[_nInputFeatures];
        _upperBounds = new double[_nInputFeatures];

        for (int col = 0; col < _nInputFeatures; col++)
        {
            if (!processSet.Contains(col))
            {
                // Pass-through columns: use extreme bounds
                _medians[col] = 0;
                _mads[col] = 1;
                _lowerBounds[col] = double.MinValue;
                _upperBounds[col] = double.MaxValue;
                continue;
            }

            // Collect all values for this column
            var values = new List<double>();
            for (int i = 0; i < data.Rows; i++)
            {
                values.Add(NumOps.ToDouble(data[i, col]));
            }

            // Calculate median
            double median = CalculateMedian(values);
            _medians[col] = median;

            // Calculate MAD
            var absoluteDeviations = values.Select(v => Math.Abs(v - median)).ToList();
            double mad = CalculateMedian(absoluteDeviations);
            _mads[col] = mad;

            // Calculate bounds using modified Z-score formula
            // Modified Z-score = 0.6745 * (x - median) / MAD
            // Solving for x: x = median + threshold * MAD / 0.6745
            if (mad > 0)
            {
                double scaledMad = mad * MADScaleFactor; // Equivalent to MAD / 0.6745
                _lowerBounds[col] = median - _threshold * scaledMad;
                _upperBounds[col] = median + _threshold * scaledMad;
            }
            else
            {
                // If MAD is 0, all values are the same - no clipping needed
                _lowerBounds[col] = double.MinValue;
                _upperBounds[col] = double.MaxValue;
            }
        }
    }

    private static double CalculateMedian(List<double> values)
    {
        if (values.Count == 0)
        {
            return 0;
        }

        var sorted = values.OrderBy(v => v).ToList();
        int n = sorted.Count;

        if (n % 2 == 0)
        {
            return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
        }

        return sorted[n / 2];
    }

    /// <summary>
    /// Clips values to the computed MAD bounds.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The data with outliers clipped.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_lowerBounds is null || _upperBounds is null)
        {
            throw new InvalidOperationException("MADClipper has not been fitted.");
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
        throw new NotSupportedException("MADClipper does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }

    /// <summary>
    /// Calculates the modified Z-scores for all values in the data.
    /// </summary>
    /// <param name="data">The data to calculate modified Z-scores for.</param>
    /// <returns>A matrix of modified Z-scores.</returns>
    public Matrix<T> GetModifiedZScores(Matrix<T> data)
    {
        if (_medians is null || _mads is null)
        {
            throw new InvalidOperationException("MADClipper has not been fitted.");
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
                double modifiedZScore;

                if (processSet.Contains(j) && _mads[j] > 0)
                {
                    // Modified Z-score = 0.6745 * (x - median) / MAD
                    modifiedZScore = 0.6745 * (value - _medians[j]) / _mads[j];
                }
                else
                {
                    modifiedZScore = 0;
                }

                result[i, j] = NumOps.FromDouble(modifiedZScore);
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
            throw new InvalidOperationException("MADClipper has not been fitted.");
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
