using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.TimeSeries;

/// <summary>
/// Autocorrelation-based feature selection for time series data.
/// </summary>
/// <remarks>
/// <para>
/// Autocorrelation measures how a feature correlates with itself at different time lags.
/// Features with strong autocorrelation patterns often contain predictable structure
/// that can be useful for time series forecasting.
/// </para>
/// <para><b>For Beginners:</b> Autocorrelation asks: "How similar is today's value
/// to yesterday's, the day before, etc.?" Features with high autocorrelation have
/// patterns that repeat or persist over time, making them potentially valuable for
/// predicting future values.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AutocorrelationFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxLag;
    private readonly double _minAutocorrelation;

    private double[]? _autocorrelationScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int MaxLag => _maxLag;
    public double[]? AutocorrelationScores => _autocorrelationScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public AutocorrelationFS(
        int nFeaturesToSelect = 10,
        int maxLag = 10,
        double minAutocorrelation = 0.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (maxLag < 1)
            throw new ArgumentException("Max lag must be at least 1.", nameof(maxLag));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxLag = maxLag;
        _minAutocorrelation = minAutocorrelation;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _autocorrelationScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Compute mean
            double mean = 0;
            for (int i = 0; i < n; i++)
                mean += NumOps.ToDouble(data[i, j]);
            mean /= n;

            // Compute variance
            double variance = 0;
            for (int i = 0; i < n; i++)
            {
                double diff = NumOps.ToDouble(data[i, j]) - mean;
                variance += diff * diff;
            }
            variance /= n;

            if (variance < 1e-10)
            {
                _autocorrelationScores[j] = 0;
                continue;
            }

            // Compute autocorrelation for each lag and take max
            double maxAutocorr = 0;
            int effectiveMaxLag = Math.Min(_maxLag, n - 1);

            for (int lag = 1; lag <= effectiveMaxLag; lag++)
            {
                double covariance = 0;
                for (int i = 0; i < n - lag; i++)
                {
                    double diff1 = NumOps.ToDouble(data[i, j]) - mean;
                    double diff2 = NumOps.ToDouble(data[i + lag, j]) - mean;
                    covariance += diff1 * diff2;
                }
                covariance /= (n - lag);

                double autocorr = Math.Abs(covariance / variance);
                maxAutocorr = Math.Max(maxAutocorr, autocorr);
            }

            _autocorrelationScores[j] = maxAutocorr;
        }

        // Filter by minimum autocorrelation and select top features
        var candidates = _autocorrelationScores
            .Select((s, idx) => (Score: s, Index: idx))
            .Where(x => x.Score >= _minAutocorrelation)
            .OrderByDescending(x => x.Score)
            .ToList();

        int numToSelect = Math.Min(_nFeaturesToSelect, candidates.Count);
        if (numToSelect == 0 && p > 0)
        {
            // If no features pass threshold, take the top feature
            _selectedIndices = _autocorrelationScores
                .Select((s, idx) => (Score: s, Index: idx))
                .OrderByDescending(x => x.Score)
                .Take(1)
                .Select(x => x.Index)
                .ToArray();
        }
        else
        {
            _selectedIndices = candidates
                .Take(numToSelect)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AutocorrelationFS has not been fitted.");

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
        throw new NotSupportedException("AutocorrelationFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AutocorrelationFS has not been fitted.");

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
