using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Temporal;

/// <summary>
/// Seasonality based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on the presence of seasonal (periodic) patterns,
/// identifying features with repeating cycles.
/// </para>
/// <para><b>For Beginners:</b> Seasonal features show repeating patterns (like
/// higher sales every December). This selector finds features with strong
/// periodic behavior using spectral analysis.
/// </para>
/// </remarks>
public class SeasonalitySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _minPeriod;
    private readonly int _maxPeriod;

    private double[]? _seasonalityStrengths;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int MinPeriod => _minPeriod;
    public int MaxPeriod => _maxPeriod;
    public double[]? SeasonalityStrengths => _seasonalityStrengths;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SeasonalitySelector(
        int nFeaturesToSelect = 10,
        int minPeriod = 2,
        int maxPeriod = 12,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minPeriod = minPeriod;
        _maxPeriod = maxPeriod;
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

        _seasonalityStrengths = new double[p];
        int effectiveMaxPeriod = Math.Min(_maxPeriod, n / 2);

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            // Remove mean
            double mean = col.Average();
            for (int i = 0; i < n; i++)
                col[i] -= mean;

            double totalVariance = col.Sum(v => v * v);
            if (totalVariance < 1e-10)
            {
                _seasonalityStrengths[j] = 0;
                continue;
            }

            // Find dominant periodic component using spectral analysis
            double maxPower = 0;
            for (int period = _minPeriod; period <= effectiveMaxPeriod; period++)
            {
                // Compute power at this frequency using DFT
                double freq = 1.0 / period;
                double real = 0, imag = 0;
                for (int i = 0; i < n; i++)
                {
                    double angle = 2 * Math.PI * freq * i;
                    real += col[i] * Math.Cos(angle);
                    imag -= col[i] * Math.Sin(angle);
                }
                double power = (real * real + imag * imag) / n;
                maxPower = Math.Max(maxPower, power);
            }

            // Seasonality strength as ratio of max periodic power to total variance
            _seasonalityStrengths[j] = maxPower / totalVariance;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _seasonalityStrengths[j])
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
            throw new InvalidOperationException("SeasonalitySelector has not been fitted.");

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
        throw new NotSupportedException("SeasonalitySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SeasonalitySelector has not been fitted.");

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
