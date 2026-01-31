using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Temporal;

/// <summary>
/// Autocorrelation based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their autocorrelation properties, identifying features
/// with meaningful temporal patterns.
/// </para>
/// <para><b>For Beginners:</b> Autocorrelation measures how a signal relates to
/// delayed versions of itself. Features with strong autocorrelation have patterns
/// that repeat over time, which can be useful for prediction.
/// </para>
/// </remarks>
public class AutocorrelationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxLag;

    private double[]? _autocorrelationScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int MaxLag => _maxLag;
    public double[]? AutocorrelationScores => _autocorrelationScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public AutocorrelationSelector(
        int nFeaturesToSelect = 10,
        int maxLag = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxLag = maxLag;
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

        _autocorrelationScores = new double[p];
        int effectiveMaxLag = Math.Min(_maxLag, n / 3);

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            double mean = col.Average();
            double variance = col.Select(v => (v - mean) * (v - mean)).Sum();

            if (variance < 1e-10)
            {
                _autocorrelationScores[j] = 0;
                continue;
            }

            // Compute autocorrelation for multiple lags and take max
            double maxAC = 0;
            for (int lag = 1; lag <= effectiveMaxLag; lag++)
            {
                double autocovariance = 0;
                for (int i = 0; i < n - lag; i++)
                    autocovariance += (col[i] - mean) * (col[i + lag] - mean);

                double acf = autocovariance / variance;
                maxAC = Math.Max(maxAC, Math.Abs(acf));
            }

            _autocorrelationScores[j] = maxAC;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _autocorrelationScores[j])
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
            throw new InvalidOperationException("AutocorrelationSelector has not been fitted.");

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
        throw new NotSupportedException("AutocorrelationSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AutocorrelationSelector has not been fitted.");

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
