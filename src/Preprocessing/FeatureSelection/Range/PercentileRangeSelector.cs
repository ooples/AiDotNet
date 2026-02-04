using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Range;

/// <summary>
/// Percentile Range based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on the range between specified percentiles,
/// allowing customizable robust measures of spread.
/// </para>
/// <para><b>For Beginners:</b> This selector measures the spread of data between
/// two percentiles you choose. For example, the 10th-90th percentile range
/// captures 80% of the data, ignoring extreme outliers at both ends.
/// </para>
/// </remarks>
public class PercentileRangeSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _lowerPercentile;
    private readonly double _upperPercentile;

    private double[]? _rangeValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double LowerPercentile => _lowerPercentile;
    public double UpperPercentile => _upperPercentile;
    public double[]? RangeValues => _rangeValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public PercentileRangeSelector(
        int nFeaturesToSelect = 10,
        double lowerPercentile = 0.10,
        double upperPercentile = 0.90,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (lowerPercentile < 0 || lowerPercentile > 1)
            throw new ArgumentException("Lower percentile must be between 0 and 1.", nameof(lowerPercentile));
        if (upperPercentile < 0 || upperPercentile > 1)
            throw new ArgumentException("Upper percentile must be between 0 and 1.", nameof(upperPercentile));
        if (lowerPercentile >= upperPercentile)
            throw new ArgumentException("Lower percentile must be less than upper percentile.");

        _nFeaturesToSelect = nFeaturesToSelect;
        _lowerPercentile = lowerPercentile;
        _upperPercentile = upperPercentile;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        _rangeValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            var sorted = col.OrderBy(v => v).ToArray();
            double lower = GetPercentile(sorted, _lowerPercentile);
            double upper = GetPercentile(sorted, _upperPercentile);
            _rangeValues[j] = upper - lower;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _rangeValues[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double GetPercentile(double[] sortedData, double percentile)
    {
        int n = sortedData.Length;
        double index = percentile * (n - 1);
        int lower = (int)Math.Floor(index);
        int upper = (int)Math.Ceiling(index);

        if (lower == upper)
            return sortedData[lower];

        double fraction = index - lower;
        return sortedData[lower] * (1 - fraction) + sortedData[upper] * fraction;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PercentileRangeSelector has not been fitted.");

        int numRows = data.Rows;
        int numCols = _selectedIndices.Length;
        var result = new T[numRows, numCols];

        for (int i = 0; i < numRows; i++)
            for (int j = 0; j < numCols; j++)
                result[i, j] = data[i, _selectedIndices[j]];

        return new Matrix<T>(result);
    }

    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("PercentileRangeSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PercentileRangeSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
