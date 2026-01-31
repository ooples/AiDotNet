using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.TimeSeries;

/// <summary>
/// Autocorrelation-based feature selection for time series data.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their autocorrelation properties. Features with
/// significant autocorrelation contain temporal patterns useful for prediction.
/// </para>
/// <para><b>For Beginners:</b> Autocorrelation measures how much a value at one
/// time relates to values at other times. Features with strong autocorrelation
/// have predictable patterns over time.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AutocorrelationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxLag;
    private readonly double _significanceThreshold;

    private double[]? _autocorrelationScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? AutocorrelationScores => _autocorrelationScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public AutocorrelationSelector(
        int nFeaturesToSelect = 10,
        int maxLag = 10,
        double significanceThreshold = 0.05,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (maxLag < 1)
            throw new ArgumentException("Max lag must be at least 1.", nameof(maxLag));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxLag = maxLag;
        _significanceThreshold = significanceThreshold;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int maxLag = Math.Min(_maxLag, n / 3);

        _autocorrelationScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Extract feature values
            var values = new double[n];
            for (int i = 0; i < n; i++)
                values[i] = NumOps.ToDouble(data[i, j]);

            // Compute mean
            double mean = values.Average();

            // Compute variance
            double variance = 0;
            for (int i = 0; i < n; i++)
                variance += Math.Pow(values[i] - mean, 2);
            variance /= n;

            if (variance < 1e-10)
            {
                _autocorrelationScores[j] = 0;
                continue;
            }

            // Compute autocorrelations for different lags
            double maxAcf = 0;
            int significantLags = 0;

            for (int lag = 1; lag <= maxLag; lag++)
            {
                double acf = 0;
                for (int i = 0; i < n - lag; i++)
                    acf += (values[i] - mean) * (values[i + lag] - mean);
                acf /= (n - lag) * variance;

                if (Math.Abs(acf) > maxAcf)
                    maxAcf = Math.Abs(acf);

                // Significance test (approximate)
                double criticalValue = 1.96 / Math.Sqrt(n);
                if (Math.Abs(acf) > criticalValue)
                    significantLags++;
            }

            // Score based on max ACF and number of significant lags
            _autocorrelationScores[j] = maxAcf * (1 + 0.1 * significantLags);
        }

        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _autocorrelationScores
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(nToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        FitCore(data);
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
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
