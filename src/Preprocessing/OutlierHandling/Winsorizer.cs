using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.OutlierHandling;

/// <summary>
/// Winsorizes data by replacing extreme values with percentile bounds.
/// </summary>
/// <remarks>
/// <para>
/// Winsorizer is a statistical technique that limits extreme values in the data
/// to reduce the effect of outliers. Unlike trimming (which removes outliers),
/// Winsorization replaces them with less extreme values.
/// </para>
/// <para>
/// This is equivalent to OutlierClipper but follows the traditional Winsorization
/// terminology where you specify the percentage of data to Winsorize at each tail.
/// </para>
/// <para><b>For Beginners:</b> Winsorization is named after biostatistician Charles Winsor.
/// Instead of removing outliers, it replaces them with the nearest "normal" values:
/// - If you Winsorize at 5%, the bottom 5% of values become equal to the 5th percentile
/// - The top 5% of values become equal to the 95th percentile
///
/// This preserves sample size while reducing outlier impact.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class Winsorizer<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _lowerLimit;
    private readonly double _upperLimit;
    private readonly WinsorizerLimitType _limitType;

    // Fitted parameters
    private double[]? _lowerBounds;
    private double[]? _upperBounds;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the lower limit value.
    /// </summary>
    public double LowerLimit => _lowerLimit;

    /// <summary>
    /// Gets the upper limit value.
    /// </summary>
    public double UpperLimit => _upperLimit;

    /// <summary>
    /// Gets the type of limit (percentile or IQR).
    /// </summary>
    public WinsorizerLimitType LimitType => _limitType;

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
    /// Creates a new instance of <see cref="Winsorizer{T}"/>.
    /// </summary>
    /// <param name="lowerLimit">Lower limit. For percentile type: 0-50. For IQR type: multiplier (e.g., 1.5). Defaults to 5.</param>
    /// <param name="upperLimit">Upper limit. For percentile type: 50-100. For IQR type: multiplier (e.g., 1.5). Defaults to 95.</param>
    /// <param name="limitType">Type of limits to use. Defaults to Percentile.</param>
    /// <param name="columnIndices">The column indices to Winsorize, or null for all columns.</param>
    public Winsorizer(
        double lowerLimit = 5.0,
        double upperLimit = 95.0,
        WinsorizerLimitType limitType = WinsorizerLimitType.Percentile,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _limitType = limitType;

        if (limitType == WinsorizerLimitType.Percentile)
        {
            if (lowerLimit < 0 || lowerLimit > 50)
            {
                throw new ArgumentException("Lower percentile must be between 0 and 50.", nameof(lowerLimit));
            }

            if (upperLimit < 50 || upperLimit > 100)
            {
                throw new ArgumentException("Upper percentile must be between 50 and 100.", nameof(upperLimit));
            }

            if (lowerLimit >= upperLimit)
            {
                throw new ArgumentException("Lower limit must be less than upper limit.");
            }
        }
        else // IQR type
        {
            if (lowerLimit < 0)
            {
                throw new ArgumentException("IQR multiplier must be non-negative.", nameof(lowerLimit));
            }

            if (upperLimit < 0)
            {
                throw new ArgumentException("IQR multiplier must be non-negative.", nameof(upperLimit));
            }
        }

        _lowerLimit = lowerLimit;
        _upperLimit = upperLimit;
    }

    /// <summary>
    /// Computes the Winsorization bounds for each feature.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);
        var processSet = new HashSet<int>(columnsToProcess);

        _lowerBounds = new double[_nInputFeatures];
        _upperBounds = new double[_nInputFeatures];

        for (int col = 0; col < _nInputFeatures; col++)
        {
            if (!processSet.Contains(col))
            {
                // Pass-through columns: use extreme bounds
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

            // Sort for percentile calculation
            values.Sort();

            if (_limitType == WinsorizerLimitType.Percentile)
            {
                _lowerBounds[col] = CalculatePercentile(values, _lowerLimit);
                _upperBounds[col] = CalculatePercentile(values, _upperLimit);
            }
            else // IQR type
            {
                double q1 = CalculatePercentile(values, 25);
                double q3 = CalculatePercentile(values, 75);
                double iqr = q3 - q1;

                _lowerBounds[col] = q1 - _lowerLimit * iqr;
                _upperBounds[col] = q3 + _upperLimit * iqr;
            }
        }
    }

    private static double CalculatePercentile(List<double> sortedValues, double percentile)
    {
        if (sortedValues.Count == 0)
        {
            return 0;
        }

        if (sortedValues.Count == 1)
        {
            return sortedValues[0];
        }

        double index = (percentile / 100.0) * (sortedValues.Count - 1);
        int lowerIndex = (int)Math.Floor(index);
        int upperIndex = (int)Math.Ceiling(index);

        if (lowerIndex == upperIndex)
        {
            return sortedValues[lowerIndex];
        }

        // Linear interpolation
        double fraction = index - lowerIndex;
        return sortedValues[lowerIndex] + fraction * (sortedValues[upperIndex] - sortedValues[lowerIndex]);
    }

    /// <summary>
    /// Winsorizes the data by replacing extreme values with bounds.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The Winsorized data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_lowerBounds is null || _upperBounds is null)
        {
            throw new InvalidOperationException("Winsorizer has not been fitted.");
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
                    // Winsorize to bounds
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
        throw new NotSupportedException("Winsorizer does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}

/// <summary>
/// Specifies how Winsorization limits are calculated.
/// </summary>
public enum WinsorizerLimitType
{
    /// <summary>
    /// Use percentile values directly (e.g., 5th and 95th percentile).
    /// </summary>
    Percentile,

    /// <summary>
    /// Use IQR-based limits (Q1 - k*IQR, Q3 + k*IQR where k is the limit value).
    /// </summary>
    IQR
}
