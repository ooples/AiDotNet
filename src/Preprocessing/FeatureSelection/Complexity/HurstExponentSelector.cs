using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Complexity;

/// <summary>
/// Hurst Exponent based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their Hurst exponent, which measures
/// long-range dependence and self-similarity in time series.
/// </para>
/// <para><b>For Beginners:</b> The Hurst exponent H indicates trend behavior:
/// H &gt; 0.5 means trending (persistent) - ups followed by ups, downs by downs;
/// H &lt; 0.5 means mean-reverting (anti-persistent) - ups followed by downs;
/// H = 0.5 means random walk (no memory). Financial time series with H ≠ 0.5
/// may contain exploitable patterns.
/// </para>
/// </remarks>
public class HurstExponentSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly bool _preferTrending;

    private double[]? _hurstValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public bool PreferTrending => _preferTrending;
    public double[]? HurstValues => _hurstValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public HurstExponentSelector(
        int nFeaturesToSelect = 10,
        bool preferTrending = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _preferTrending = preferTrending;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        if (n < 20)
            throw new ArgumentException("Need at least 20 samples for Hurst exponent estimation.");

        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        _hurstValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            _hurstValues[j] = ComputeHurstRS(col);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        if (_preferTrending)
        {
            // Prefer features with H > 0.5 (trending/persistent)
            _selectedIndices = Enumerable.Range(0, p)
                .OrderByDescending(j => _hurstValues[j])
                .Take(numToSelect)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            // Prefer features with H furthest from 0.5 (strong memory either way)
            _selectedIndices = Enumerable.Range(0, p)
                .OrderByDescending(j => Math.Abs(_hurstValues[j] - 0.5))
                .Take(numToSelect)
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    private double ComputeHurstRS(double[] data)
    {
        int n = data.Length;

        // Use rescaled range (R/S) method
        var lagSizes = new List<int>();
        var rsValues = new List<double>();

        // Choose lag sizes from 10 to n/2
        for (int lag = 10; lag <= n / 2; lag = (int)(lag * 1.5))
        {
            var rsForLag = new List<double>();

            // Divide series into subseries
            int numSubseries = n / lag;
            for (int i = 0; i < numSubseries; i++)
            {
                int start = i * lag;
                var subseries = new double[lag];
                for (int k = 0; k < lag; k++)
                    subseries[k] = data[start + k];

                double mean = subseries.Average();

                // Cumulative deviations
                var cumDev = new double[lag];
                double sum = 0;
                for (int k = 0; k < lag; k++)
                {
                    sum += subseries[k] - mean;
                    cumDev[k] = sum;
                }

                // Range
                double range = cumDev.Max() - cumDev.Min();

                // Standard deviation
                double std = Math.Sqrt(subseries.Select(v => (v - mean) * (v - mean)).Average());

                if (std > 1e-10)
                    rsForLag.Add(range / std);
            }

            if (rsForLag.Count > 0)
            {
                lagSizes.Add(lag);
                rsValues.Add(rsForLag.Average());
            }
        }

        if (lagSizes.Count < 2)
            return 0.5; // Default to random walk

        // Linear regression of log(R/S) vs log(lag)
        var logLag = lagSizes.Select(l => Math.Log(l)).ToArray();
        var logRS = rsValues.Select(r => Math.Log(r)).ToArray();

        // Simple linear regression: H = slope
        double meanX = logLag.Average();
        double meanY = logRS.Average();

        double numerator = 0, denominator = 0;
        for (int i = 0; i < logLag.Length; i++)
        {
            numerator += (logLag[i] - meanX) * (logRS[i] - meanY);
            denominator += (logLag[i] - meanX) * (logLag[i] - meanX);
        }

        if (denominator < 1e-10)
            return 0.5;

        double hurst = numerator / denominator;

        // Clamp to valid range [0, 1]
        return Math.Max(0, Math.Min(1, hurst));
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HurstExponentSelector has not been fitted.");

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
        throw new NotSupportedException("HurstExponentSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HurstExponentSelector has not been fitted.");

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
