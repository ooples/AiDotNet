using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.OutlierHandling;

/// <summary>
/// Clips outliers to specified percentile bounds.
/// </summary>
/// <remarks>
/// <para>
/// OutlierClipper clips values below a lower percentile and above an upper percentile
/// to those percentile values. This reduces the impact of extreme outliers while
/// preserving the overall data distribution.
/// </para>
/// <para><b>For Beginners:</b> Outliers are extreme values that are far from most of your data.
/// They can distort your model's learning. Clipping replaces extreme values with more
/// reasonable bounds:
/// - Values below the 1st percentile → replaced with 1st percentile value
/// - Values above the 99th percentile → replaced with 99th percentile value
///
/// Example: Income data where most people earn $30K-$200K but a few billionaires
/// would be clipped to prevent the model from being skewed by extreme wealth.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class OutlierClipper<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _lowerPercentile;
    private readonly double _upperPercentile;

    // Fitted parameters
    private double[]? _lowerBounds;
    private double[]? _upperBounds;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the lower percentile (values below this are clipped).
    /// </summary>
    public double LowerPercentile => _lowerPercentile;

    /// <summary>
    /// Gets the upper percentile (values above this are clipped).
    /// </summary>
    public double UpperPercentile => _upperPercentile;

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
    /// Creates a new instance of <see cref="OutlierClipper{T}"/>.
    /// </summary>
    /// <param name="lowerPercentile">Lower percentile bound (0-100). Values below are clipped. Defaults to 1.</param>
    /// <param name="upperPercentile">Upper percentile bound (0-100). Values above are clipped. Defaults to 99.</param>
    /// <param name="columnIndices">The column indices to clip, or null for all columns.</param>
    public OutlierClipper(
        double lowerPercentile = 1.0,
        double upperPercentile = 99.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (lowerPercentile < 0 || lowerPercentile > 100)
        {
            throw new ArgumentException("Lower percentile must be between 0 and 100.", nameof(lowerPercentile));
        }

        if (upperPercentile < 0 || upperPercentile > 100)
        {
            throw new ArgumentException("Upper percentile must be between 0 and 100.", nameof(upperPercentile));
        }

        if (lowerPercentile >= upperPercentile)
        {
            throw new ArgumentException("Lower percentile must be less than upper percentile.");
        }

        _lowerPercentile = lowerPercentile;
        _upperPercentile = upperPercentile;
    }

    /// <summary>
    /// Computes the percentile bounds for each feature.
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

            // Calculate percentile indices
            _lowerBounds[col] = CalculatePercentile(values, _lowerPercentile);
            _upperBounds[col] = CalculatePercentile(values, _upperPercentile);
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
    /// Clips values to the computed percentile bounds.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The data with outliers clipped.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_lowerBounds is null || _upperBounds is null)
        {
            throw new InvalidOperationException("Clipper has not been fitted.");
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
        throw new NotSupportedException("OutlierClipper does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
