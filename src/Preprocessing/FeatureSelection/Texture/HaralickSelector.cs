using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Texture;

/// <summary>
/// Haralick Texture Features based Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their texture characteristics computed from
/// co-occurrence statistics, useful for image-derived or spatial data features.
/// </para>
/// <para><b>For Beginners:</b> Haralick features describe patterns like contrast,
/// homogeneity, and correlation in data. This selector identifies features
/// with strong textural patterns that often indicate meaningful structure.
/// </para>
/// </remarks>
public class HaralickSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nLevels;

    private double[]? _textureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NLevels => _nLevels;
    public double[]? TextureScores => _textureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public HaralickSelector(
        int nFeaturesToSelect = 10,
        int nLevels = 16,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nLevels = nLevels;
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

        _textureScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            // Compute Haralick-inspired statistics
            double contrast = ComputeContrast(col, n);
            double homogeneity = ComputeHomogeneity(col, n);
            double entropy = ComputeEntropy(col, n);

            // Combine into overall texture score
            _textureScores[j] = contrast + homogeneity + entropy;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _textureScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeContrast(double[] values, int n)
    {
        if (n < 2) return 0;

        // Contrast measures local variation
        double contrast = 0;
        for (int i = 1; i < n; i++)
        {
            double diff = values[i] - values[i - 1];
            contrast += diff * diff;
        }

        return contrast / (n - 1);
    }

    private double ComputeHomogeneity(double[] values, int n)
    {
        if (n < 2) return 0;

        // Homogeneity measures smoothness
        double homogeneity = 0;
        for (int i = 1; i < n; i++)
        {
            double diff = Math.Abs(values[i] - values[i - 1]);
            homogeneity += 1.0 / (1 + diff);
        }

        return homogeneity / (n - 1);
    }

    private double ComputeEntropy(double[] values, int n)
    {
        double minVal = values.Min(), maxVal = values.Max();
        if (Math.Abs(maxVal - minVal) < 1e-10) return 0;

        double binWidth = (maxVal - minVal) / _nLevels + 1e-10;
        var counts = new int[_nLevels];

        for (int i = 0; i < n; i++)
        {
            int bin = Math.Min((int)((values[i] - minVal) / binWidth), _nLevels - 1);
            counts[bin]++;
        }

        double entropy = 0;
        for (int b = 0; b < _nLevels; b++)
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
            throw new InvalidOperationException("HaralickSelector has not been fitted.");

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
        throw new NotSupportedException("HaralickSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HaralickSelector has not been fitted.");

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
