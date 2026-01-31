using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Wavelet;

/// <summary>
/// Wavelet Entropy based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their wavelet entropy, which measures the complexity
/// of features in the time-frequency domain using wavelet decomposition.
/// </para>
/// <para><b>For Beginners:</b> Wavelets break down a signal into different frequency
/// components at different times. The entropy of these components tells us how
/// complex or information-rich a feature is. More complex features often capture
/// more useful patterns.
/// </para>
/// </remarks>
public class WaveletEntropySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _decompositionLevel;

    private double[]? _waveletEntropyScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int DecompositionLevel => _decompositionLevel;
    public double[]? WaveletEntropyScores => _waveletEntropyScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public WaveletEntropySelector(
        int nFeaturesToSelect = 10,
        int decompositionLevel = 3,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _decompositionLevel = decompositionLevel;
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

        _waveletEntropyScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            _waveletEntropyScores[j] = ComputeWaveletEntropy(col, n);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _waveletEntropyScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeWaveletEntropy(double[] signal, int n)
    {
        // Simplified Haar wavelet decomposition
        var current = (double[])signal.Clone();
        var energies = new List<double>();

        for (int level = 0; level < _decompositionLevel && current.Length >= 2; level++)
        {
            int len = current.Length;
            int halfLen = len / 2;
            var approx = new double[halfLen];
            var detail = new double[halfLen];

            for (int i = 0; i < halfLen; i++)
            {
                approx[i] = (current[2 * i] + current[2 * i + 1]) / Math.Sqrt(2);
                detail[i] = (current[2 * i] - current[2 * i + 1]) / Math.Sqrt(2);
            }

            // Compute energy of detail coefficients
            double energy = 0;
            for (int i = 0; i < halfLen; i++)
                energy += detail[i] * detail[i];
            energies.Add(energy);

            current = approx;
        }

        // Add final approximation energy
        double approxEnergy = 0;
        for (int i = 0; i < current.Length; i++)
            approxEnergy += current[i] * current[i];
        energies.Add(approxEnergy);

        // Compute entropy from energy distribution
        double totalEnergy = energies.Sum();
        if (totalEnergy < 1e-10) return 0;

        double entropy = 0;
        foreach (double e in energies)
        {
            double p = e / totalEnergy;
            if (p > 1e-10)
                entropy -= p * Math.Log(p) / Math.Log(2);
        }

        return entropy;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("WaveletEntropySelector has not been fitted.");

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
        throw new NotSupportedException("WaveletEntropySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("WaveletEntropySelector has not been fitted.");

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
