using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.OutlierHandling;

/// <summary>
/// Clips outliers using the Interquartile Range (IQR) method.
/// </summary>
/// <remarks>
/// <para>
/// IQRClipper identifies outliers using the IQR method:
/// - Lower bound = Q1 - k * IQR
/// - Upper bound = Q3 + k * IQR
/// where IQR = Q3 - Q1 and k is the multiplier (default 1.5).
/// </para>
/// <para>
/// This is the same method used in box plots for identifying outliers.
/// A multiplier of 1.5 identifies "mild" outliers, while 3.0 identifies "extreme" outliers.
/// </para>
/// <para><b>For Beginners:</b> The IQR is the range where the middle 50% of data falls.
/// Values outside 1.5× this range are considered outliers:
/// - Q1 (25th percentile): 25% of data is below this value
/// - Q3 (75th percentile): 75% of data is below this value
/// - IQR = Q3 - Q1: The spread of the middle 50%
/// - Outliers: Values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class IQRClipper<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _multiplier;

    // Fitted parameters
    private double[]? _lowerBounds;
    private double[]? _upperBounds;
    private double[]? _q1Values;
    private double[]? _q3Values;
    private double[]? _iqrValues;

    /// <summary>
    /// Gets the IQR multiplier for determining outlier boundaries.
    /// </summary>
    public double Multiplier => _multiplier;

    /// <summary>
    /// Gets the fitted lower bounds for each feature.
    /// </summary>
    public double[]? LowerBounds => _lowerBounds;

    /// <summary>
    /// Gets the fitted upper bounds for each feature.
    /// </summary>
    public double[]? UpperBounds => _upperBounds;

    /// <summary>
    /// Gets the Q1 (25th percentile) values for each feature.
    /// </summary>
    public double[]? Q1Values => _q1Values;

    /// <summary>
    /// Gets the Q3 (75th percentile) values for each feature.
    /// </summary>
    public double[]? Q3Values => _q3Values;

    /// <summary>
    /// Gets the IQR values for each feature.
    /// </summary>
    public double[]? IQRValues => _iqrValues;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="IQRClipper{T}"/>.
    /// </summary>
    /// <param name="multiplier">The IQR multiplier (default 1.5 for mild outliers, 3.0 for extreme).</param>
    /// <param name="columnIndices">The column indices to clip, or null for all columns.</param>
    public IQRClipper(
        double multiplier = 1.5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (multiplier <= 0)
        {
            throw new ArgumentException("Multiplier must be positive.", nameof(multiplier));
        }

        _multiplier = multiplier;
    }

    /// <summary>
    /// Fits the clipper by computing IQR bounds for each feature.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        int numColumns = data.Columns;
        int numRows = data.Rows;
        var columnsToProcess = GetColumnsToProcess(numColumns);

        _lowerBounds = new double[numColumns];
        _upperBounds = new double[numColumns];
        _q1Values = new double[numColumns];
        _q3Values = new double[numColumns];
        _iqrValues = new double[numColumns];

        foreach (int col in columnsToProcess)
        {
            // Extract column values
            var values = new double[numRows];
            for (int i = 0; i < numRows; i++)
            {
                values[i] = NumOps.ToDouble(data[i, col]);
            }

            // Sort for percentile calculation
            Array.Sort(values);

            // Compute Q1 (25th percentile) and Q3 (75th percentile)
            _q1Values[col] = ComputePercentile(values, 25);
            _q3Values[col] = ComputePercentile(values, 75);

            // Compute IQR
            _iqrValues[col] = _q3Values[col] - _q1Values[col];

            // Compute bounds
            _lowerBounds[col] = _q1Values[col] - _multiplier * _iqrValues[col];
            _upperBounds[col] = _q3Values[col] + _multiplier * _iqrValues[col];
        }

        // For non-processed columns, set wide bounds (no clipping)
        var processSet = new HashSet<int>(columnsToProcess);
        for (int col = 0; col < numColumns; col++)
        {
            if (!processSet.Contains(col))
            {
                _lowerBounds[col] = double.NegativeInfinity;
                _upperBounds[col] = double.PositiveInfinity;
            }
        }
    }

    private double ComputePercentile(double[] sortedValues, double percentile)
    {
        int n = sortedValues.Length;
        if (n == 0) return 0;
        if (n == 1) return sortedValues[0];

        double index = (percentile / 100.0) * (n - 1);
        int lower = (int)Math.Floor(index);
        int upper = (int)Math.Ceiling(index);

        if (lower == upper)
        {
            return sortedValues[lower];
        }

        double fraction = index - lower;
        return sortedValues[lower] * (1 - fraction) + sortedValues[upper] * fraction;
    }

    /// <summary>
    /// Transforms the data by clipping values outside IQR bounds.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The transformed data with outliers clipped.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_lowerBounds is null || _upperBounds is null)
        {
            throw new InvalidOperationException("IQRClipper has not been fitted.");
        }

        int numRows = data.Rows;
        int numColumns = data.Columns;
        var result = new T[numRows, numColumns];

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numColumns; j++)
            {
                double value = NumOps.ToDouble(data[i, j]);
                double clipped = Math.Max(_lowerBounds[j], Math.Min(_upperBounds[j], value));
                result[i, j] = NumOps.FromDouble(clipped);
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("IQRClipper does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }

    /// <summary>
    /// Gets a mask indicating which values are outliers in the input data.
    /// </summary>
    /// <param name="data">The data to check for outliers.</param>
    /// <returns>A boolean matrix where true indicates an outlier.</returns>
    public bool[,] GetOutlierMask(Matrix<T> data)
    {
        if (_lowerBounds is null || _upperBounds is null)
        {
            throw new InvalidOperationException("IQRClipper has not been fitted.");
        }

        int numRows = data.Rows;
        int numColumns = data.Columns;
        var mask = new bool[numRows, numColumns];

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numColumns; j++)
            {
                double value = NumOps.ToDouble(data[i, j]);
                mask[i, j] = value < _lowerBounds[j] || value > _upperBounds[j];
            }
        }

        return mask;
    }

    /// <summary>
    /// Counts the number of outliers per feature.
    /// </summary>
    /// <param name="data">The data to analyze.</param>
    /// <returns>Array of outlier counts per feature.</returns>
    public int[] CountOutliersPerFeature(Matrix<T> data)
    {
        var mask = GetOutlierMask(data);
        int numColumns = data.Columns;
        var counts = new int[numColumns];

        for (int j = 0; j < numColumns; j++)
        {
            for (int i = 0; i < data.Rows; i++)
            {
                if (mask[i, j]) counts[j]++;
            }
        }

        return counts;
    }
}
