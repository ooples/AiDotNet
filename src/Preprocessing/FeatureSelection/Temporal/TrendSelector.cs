using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Temporal;

/// <summary>
/// Trend based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on the strength of their temporal trends,
/// identifying features with clear increasing or decreasing patterns.
/// </para>
/// <para><b>For Beginners:</b> Some features have trends - they consistently
/// go up or down over time. This selector finds features with strong trends,
/// which can be useful for forecasting applications.
/// </para>
/// </remarks>
public class TrendSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _trendStrengths;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? TrendStrengths => _trendStrengths;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public TrendSelector(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
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

        _trendStrengths = new double[p];

        // Time indices
        var t = Enumerable.Range(0, n).Select(i => (double)i).ToArray();
        double tMean = t.Average();
        double tVar = t.Select(x => (x - tMean) * (x - tMean)).Sum();

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            double yMean = col.Average();
            double yVar = col.Select(v => (v - yMean) * (v - yMean)).Sum();

            if (yVar < 1e-10 || tVar < 1e-10)
            {
                _trendStrengths[j] = 0;
                continue;
            }

            // Linear regression slope
            double covariance = 0;
            for (int i = 0; i < n; i++)
                covariance += (t[i] - tMean) * (col[i] - yMean);

            double slope = covariance / tVar;

            // Compute R-squared to measure trend strength
            double ssTot = yVar;
            double ssRes = 0;
            double intercept = yMean - slope * tMean;
            for (int i = 0; i < n; i++)
            {
                double pred = slope * t[i] + intercept;
                ssRes += (col[i] - pred) * (col[i] - pred);
            }

            double r2 = ssTot > 1e-10 ? 1 - ssRes / ssTot : 0;
            _trendStrengths[j] = Math.Max(0, r2);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _trendStrengths[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TrendSelector has not been fitted.");

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
        throw new NotSupportedException("TrendSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TrendSelector has not been fitted.");

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
