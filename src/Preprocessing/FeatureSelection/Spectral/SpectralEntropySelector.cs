using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Spectral;

/// <summary>
/// Spectral Entropy based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their spectral entropy computed from the eigenvalue
/// distribution of their covariance structure.
/// </para>
/// <para><b>For Beginners:</b> This analyzes the "frequency content" of each feature's
/// information. Features with more complex spectral patterns (higher entropy) often
/// contain more useful information for prediction.
/// </para>
/// </remarks>
public class SpectralEntropySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _windowSize;

    private double[]? _spectralEntropyScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int WindowSize => _windowSize;
    public double[]? SpectralEntropyScores => _spectralEntropyScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SpectralEntropySelector(
        int nFeaturesToSelect = 10,
        int windowSize = 32,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _windowSize = windowSize;
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

        _spectralEntropyScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            _spectralEntropyScores[j] = ComputeSpectralEntropy(col, n);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _spectralEntropyScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeSpectralEntropy(double[] signal, int n)
    {
        int effectiveWindow = Math.Min(_windowSize, n);
        int numWindows = n / effectiveWindow;
        if (numWindows == 0) numWindows = 1;

        double totalEntropy = 0;

        for (int w = 0; w < numWindows; w++)
        {
            int start = w * effectiveWindow;
            int end = Math.Min(start + effectiveWindow, n);
            int windowLen = end - start;

            // Compute power spectrum using autocorrelation approximation
            var power = new double[windowLen / 2 + 1];
            double totalPower = 0;

            for (int k = 0; k < power.Length; k++)
            {
                double re = 0, im = 0;
                for (int i = start; i < end; i++)
                {
                    double angle = 2 * Math.PI * k * (i - start) / windowLen;
                    re += signal[i] * Math.Cos(angle);
                    im -= signal[i] * Math.Sin(angle);
                }
                power[k] = re * re + im * im;
                totalPower += power[k];
            }

            // Normalize and compute entropy
            if (totalPower > 1e-10)
            {
                double entropy = 0;
                for (int k = 0; k < power.Length; k++)
                {
                    double p = power[k] / totalPower;
                    if (p > 1e-10)
                        entropy -= p * Math.Log(p) / Math.Log(2);
                }
                totalEntropy += entropy;
            }
        }

        return totalEntropy / numWindows;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SpectralEntropySelector has not been fitted.");

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
        throw new NotSupportedException("SpectralEntropySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SpectralEntropySelector has not been fitted.");

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
