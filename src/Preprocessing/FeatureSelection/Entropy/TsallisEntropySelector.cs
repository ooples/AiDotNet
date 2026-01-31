using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Entropy;

/// <summary>
/// Tsallis Entropy based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their Tsallis entropy, a non-extensive generalization
/// of entropy useful for systems with long-range correlations.
/// </para>
/// <para><b>For Beginners:</b> Tsallis entropy is another generalization of entropy
/// that's particularly useful for complex systems. It doesn't assume independence
/// like Shannon entropy, making it useful when features have interactions.
/// </para>
/// </remarks>
public class TsallisEntropySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;
    private readonly double _q;

    private double[]? _tsallisEntropies;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public double Q => _q;
    public double[]? TsallisEntropies => _tsallisEntropies;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public TsallisEntropySelector(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        double q = 2.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
        _q = q;
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

        _tsallisEntropies = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            // Discretize
            double xMin = col.Min();
            double xMax = col.Max();
            double xRange = xMax - xMin;
            var bins = new int[_nBins];

            for (int i = 0; i < n; i++)
            {
                int binIdx = xRange > 1e-10
                    ? Math.Min(_nBins - 1, (int)((col[i] - xMin) / xRange * _nBins))
                    : 0;
                bins[binIdx]++;
            }

            // Compute Tsallis entropy: S_q(X) = (1 - sum(p^q)) / (q-1)
            if (Math.Abs(_q - 1) < 1e-10)
            {
                // q -> 1 gives Shannon entropy
                double entropy = 0;
                for (int b = 0; b < _nBins; b++)
                {
                    double p_b = (double)bins[b] / n;
                    if (p_b > 1e-10)
                        entropy -= p_b * Math.Log(p_b) / Math.Log(2);
                }
                _tsallisEntropies[j] = entropy;
            }
            else
            {
                double sumPowQ = 0;
                for (int b = 0; b < _nBins; b++)
                {
                    double p_b = (double)bins[b] / n;
                    if (p_b > 1e-10)
                        sumPowQ += Math.Pow(p_b, _q);
                }
                _tsallisEntropies[j] = (1 - sumPowQ) / (_q - 1);
            }
        }

        // Select features with highest entropy
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _tsallisEntropies[j])
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
            throw new InvalidOperationException("TsallisEntropySelector has not been fitted.");

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
        throw new NotSupportedException("TsallisEntropySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TsallisEntropySelector has not been fitted.");

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
