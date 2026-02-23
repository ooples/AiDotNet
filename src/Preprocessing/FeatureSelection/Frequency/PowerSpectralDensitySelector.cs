using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Frequency;

/// <summary>
/// Power Spectral Density based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their power spectral density, identifying features
/// with concentrated power in specific frequency bands.
/// </para>
/// <para><b>For Beginners:</b> Power spectral density shows how signal power
/// is distributed across frequencies. Features with high spectral density
/// ratios have power concentrated in meaningful frequency ranges.
/// </para>
/// </remarks>
public class PowerSpectralDensitySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _lowFreqCutoff;
    private readonly double _highFreqCutoff;

    private double[]? _psdScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double LowFreqCutoff => _lowFreqCutoff;
    public double HighFreqCutoff => _highFreqCutoff;
    public double[]? PSDScores => _psdScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public PowerSpectralDensitySelector(
        int nFeaturesToSelect = 10,
        double lowFreqCutoff = 0.0,
        double highFreqCutoff = 0.5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _lowFreqCutoff = lowFreqCutoff;
        _highFreqCutoff = highFreqCutoff;
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

        _psdScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            // Apply Hann window
            for (int i = 0; i < n; i++)
                col[i] *= 0.5 * (1 - Math.Cos(2 * Math.PI * i / (n - 1)));

            // Compute periodogram (PSD estimate)
            double totalPower = 0;
            double bandPower = 0;
            int lowIdx = (int)(_lowFreqCutoff * n);
            int highIdx = (int)(_highFreqCutoff * n);

            for (int k = 1; k < n / 2; k++)
            {
                double real = 0, imag = 0;
                for (int i = 0; i < n; i++)
                {
                    double angle = 2 * Math.PI * k * i / n;
                    real += col[i] * Math.Cos(angle);
                    imag -= col[i] * Math.Sin(angle);
                }
                double power = (real * real + imag * imag) / n;
                totalPower += power;

                if (k >= lowIdx && k <= highIdx)
                    bandPower += power;
            }

            // Score is ratio of band power to total power
            _psdScores[j] = totalPower > 1e-10 ? bandPower / totalPower : 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _psdScores[j])
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
            throw new InvalidOperationException("PowerSpectralDensitySelector has not been fitted.");

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
        throw new NotSupportedException("PowerSpectralDensitySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PowerSpectralDensitySelector has not been fitted.");

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
