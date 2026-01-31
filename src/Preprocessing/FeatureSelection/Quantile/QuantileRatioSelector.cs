using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Quantile;

/// <summary>
/// Quantile Ratio based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on the ratio between two quantiles, measuring
/// the relative spread at different parts of the distribution.
/// </para>
/// <para><b>For Beginners:</b> Quantile ratio compares values at different
/// percentiles. For example, the 90th/10th percentile ratio shows how much
/// larger the high values are compared to low values. This can detect features
/// with skewed distributions or outliers.
/// </para>
/// </remarks>
public class QuantileRatioSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _upperQuantile;
    private readonly double _lowerQuantile;

    private double[]? _ratioValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double UpperQuantile => _upperQuantile;
    public double LowerQuantile => _lowerQuantile;
    public double[]? RatioValues => _ratioValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public QuantileRatioSelector(
        int nFeaturesToSelect = 10,
        double upperQuantile = 0.90,
        double lowerQuantile = 0.10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (upperQuantile <= 0 || upperQuantile > 1)
            throw new ArgumentException("Upper quantile must be between 0 and 1.", nameof(upperQuantile));
        if (lowerQuantile < 0 || lowerQuantile >= 1)
            throw new ArgumentException("Lower quantile must be between 0 and 1.", nameof(lowerQuantile));
        if (lowerQuantile >= upperQuantile)
            throw new ArgumentException("Lower quantile must be less than upper quantile.");

        _nFeaturesToSelect = nFeaturesToSelect;
        _upperQuantile = upperQuantile;
        _lowerQuantile = lowerQuantile;
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

        _ratioValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            var sorted = col.OrderBy(v => v).ToArray();
            double upper = GetQuantile(sorted, _upperQuantile);
            double lower = GetQuantile(sorted, _lowerQuantile);

            // Handle potential division issues
            if (Math.Abs(lower) < 1e-10)
            {
                // If lower is near zero, use difference instead of ratio
                _ratioValues[j] = upper - lower;
            }
            else if (lower < 0 && upper > 0)
            {
                // Mixed signs - use absolute ratio
                _ratioValues[j] = Math.Abs(upper) / Math.Abs(lower);
            }
            else
            {
                // Normal ratio
                _ratioValues[j] = Math.Abs(upper / lower);
            }
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _ratioValues[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double GetQuantile(double[] sortedData, double quantile)
    {
        int n = sortedData.Length;
        double index = quantile * (n - 1);
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
            throw new InvalidOperationException("QuantileRatioSelector has not been fitted.");

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
        throw new NotSupportedException("QuantileRatioSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("QuantileRatioSelector has not been fitted.");

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
