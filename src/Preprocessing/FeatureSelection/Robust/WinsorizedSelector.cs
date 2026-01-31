using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Robust;

/// <summary>
/// Winsorized Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses Winsorization to replace extreme values with less extreme values before
/// computing feature importance, providing robust feature selection.
/// </para>
/// <para><b>For Beginners:</b> Winsorizing means replacing the most extreme values
/// in your data with less extreme ones (typically at the 5th and 95th percentile).
/// This way, outliers don't disappear but are "capped" at reasonable values,
/// giving more robust feature importance measurements.
/// </para>
/// </remarks>
public class WinsorizedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _lowerPercentile;
    private readonly double _upperPercentile;

    private double[]? _winsorizedScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double LowerPercentile => _lowerPercentile;
    public double UpperPercentile => _upperPercentile;
    public double[]? WinsorizedScores => _winsorizedScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public WinsorizedSelector(
        int nFeaturesToSelect = 10,
        double lowerPercentile = 0.05,
        double upperPercentile = 0.95,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (lowerPercentile < 0 || lowerPercentile >= upperPercentile || upperPercentile > 1)
            throw new ArgumentException("Percentiles must satisfy 0 <= lower < upper <= 1.");

        _nFeaturesToSelect = nFeaturesToSelect;
        _lowerPercentile = lowerPercentile;
        _upperPercentile = upperPercentile;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "WinsorizedSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Winsorize target
        var yWinsorized = Winsorize(y);
        double yMean = yWinsorized.Average();

        _winsorizedScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            var x = new double[n];
            for (int i = 0; i < n; i++) x[i] = X[i, j];

            // Winsorize feature
            var xWinsorized = Winsorize(x);
            double xMean = xWinsorized.Average();

            // Compute correlation on Winsorized data
            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; i++)
            {
                double xd = xWinsorized[i] - xMean;
                double yd = yWinsorized[i] - yMean;
                sxy += xd * yd;
                sxx += xd * xd;
                syy += yd * yd;
            }

            _winsorizedScores[j] = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _winsorizedScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] Winsorize(double[] values)
    {
        var sorted = values.OrderBy(v => v).ToArray();
        int n = sorted.Length;

        int lowerIdx = (int)(n * _lowerPercentile);
        int upperIdx = (int)(n * _upperPercentile);
        if (upperIdx >= n) upperIdx = n - 1;

        double lowerBound = sorted[lowerIdx];
        double upperBound = sorted[upperIdx];

        return values.Select(v =>
        {
            if (v < lowerBound) return lowerBound;
            if (v > upperBound) return upperBound;
            return v;
        }).ToArray();
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("WinsorizedSelector has not been fitted.");

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
        throw new NotSupportedException("WinsorizedSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("WinsorizedSelector has not been fitted.");

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
