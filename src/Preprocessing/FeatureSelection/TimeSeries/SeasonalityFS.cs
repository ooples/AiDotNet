using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.TimeSeries;

/// <summary>
/// Seasonality-based feature selection for time series data.
/// </summary>
/// <remarks>
/// <para>
/// Seasonality feature selection identifies features with periodic patterns at specified
/// frequencies. It uses spectral analysis to detect features with strong seasonal
/// components, which can be valuable for forecasting cyclical phenomena.
/// </para>
/// <para><b>For Beginners:</b> Many real-world patterns repeat: daily traffic peaks,
/// monthly sales cycles, yearly temperature changes. This method finds features that
/// have such repeating patterns. It uses frequency analysis (like detecting musical
/// notes in sound) to identify periodic behavior in data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SeasonalityFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int[] _seasonalPeriods;

    private double[]? _seasonalityScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int[] SeasonalPeriods => _seasonalPeriods;
    public double[]? SeasonalityScores => _seasonalityScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SeasonalityFS(
        int nFeaturesToSelect = 10,
        int[]? seasonalPeriods = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _seasonalPeriods = seasonalPeriods ?? [7, 12, 24, 52]; // Common periods: weekly, monthly, hourly, yearly
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _seasonalityScores = new double[p];

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
                _seasonalityScores[j] = 0;
                continue;
            }

            // Compute spectral power at each seasonal frequency
            double maxPower = 0;
            foreach (int period in _seasonalPeriods)
            {
                if (period < 2 || period > n / 2) continue;

                double frequency = 2 * Math.PI / period;

                // Compute Fourier coefficient at this frequency
                double cosSum = 0, sinSum = 0;
                for (int i = 0; i < n; i++)
                {
                    double val = NumOps.ToDouble(data[i, j]) - mean;
                    cosSum += val * Math.Cos(frequency * i);
                    sinSum += val * Math.Sin(frequency * i);
                }

                // Spectral power at this frequency
                double power = (cosSum * cosSum + sinSum * sinSum) / (n * n);
                maxPower = Math.Max(maxPower, power);
            }

            // Normalize by variance to get proportion of variance explained
            _seasonalityScores[j] = variance > 0 ? 2 * maxPower / variance : 0;
        }

        // Select top features by seasonality score
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _seasonalityScores
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(numToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SeasonalityFS has not been fitted.");

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
        throw new NotSupportedException("SeasonalityFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SeasonalityFS has not been fitted.");

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
