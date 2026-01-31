using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Compression;

/// <summary>
/// Compression Ratio based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their compressibility - features with higher entropy
/// (less compressible) often contain more useful information.
/// </para>
/// <para><b>For Beginners:</b> If a feature's values can be described very briefly
/// (highly compressible), it probably doesn't contain much information. This selector
/// keeps features that are harder to compress because they have more variety.
/// </para>
/// </remarks>
public class CompressionRatioSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _entropyScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public double[]? EntropyScores => _entropyScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public CompressionRatioSelector(
        int nFeaturesToSelect = 10,
        int nBins = 20,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
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

        _entropyScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Extract column
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            // Compute entropy as proxy for compressibility
            _entropyScores[j] = ComputeEntropy(col, n);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _entropyScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeEntropy(double[] values, int n)
    {
        double minVal = values.Min(), maxVal = values.Max();
        if (Math.Abs(maxVal - minVal) < 1e-10) return 0;

        double binWidth = (maxVal - minVal) / _nBins + 1e-10;
        var counts = new int[_nBins];

        for (int i = 0; i < n; i++)
        {
            int bin = Math.Min((int)((values[i] - minVal) / binWidth), _nBins - 1);
            counts[bin]++;
        }

        double entropy = 0;
        for (int b = 0; b < _nBins; b++)
        {
            if (counts[b] > 0)
            {
                double p = (double)counts[b] / n;
                entropy -= p * Math.Log(p) / Math.Log(2);
            }
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
            throw new InvalidOperationException("CompressionRatioSelector has not been fitted.");

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
        throw new NotSupportedException("CompressionRatioSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CompressionRatioSelector has not been fitted.");

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
