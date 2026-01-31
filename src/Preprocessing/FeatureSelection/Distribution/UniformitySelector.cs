using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Distribution;

/// <summary>
/// Uniformity based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on how uniformly distributed their values are,
/// using entropy as a measure of uniformity.
/// </para>
/// <para><b>For Beginners:</b> A uniform distribution has equal probability for
/// all values in a range. This selector measures how close each feature is to
/// being uniformly distributed. High uniformity means values are evenly spread out,
/// while low uniformity means some values are much more common than others.
/// </para>
/// </remarks>
public class UniformitySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;
    private readonly bool _preferUniform;

    private double[]? _uniformityScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public bool PreferUniform => _preferUniform;
    public double[]? UniformityScores => _uniformityScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public UniformitySelector(
        int nFeaturesToSelect = 10,
        int nBins = 20,
        bool preferUniform = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
        _preferUniform = preferUniform;
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

        _uniformityScores = new double[p];

        // Maximum entropy for uniform distribution with _nBins bins
        double maxEntropy = Math.Log(_nBins) / Math.Log(2);

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            double min = col.Min();
            double max = col.Max();
            double range = max - min;

            if (range < 1e-10)
            {
                // Constant feature - zero uniformity
                _uniformityScores[j] = 0;
                continue;
            }

            // Create histogram
            var binCounts = new int[_nBins];
            foreach (var val in col)
            {
                int bin = (int)((val - min) / range * (_nBins - 1));
                bin = Math.Min(bin, _nBins - 1);
                binCounts[bin]++;
            }

            // Compute entropy
            double entropy = 0;
            foreach (var count in binCounts)
            {
                if (count > 0)
                {
                    double p_i = (double)count / n;
                    entropy -= p_i * Math.Log(p_i) / Math.Log(2);
                }
            }

            // Uniformity score = entropy / max_entropy (normalized)
            _uniformityScores[j] = maxEntropy > 0 ? entropy / maxEntropy : 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        if (_preferUniform)
        {
            // Prefer features with high uniformity
            _selectedIndices = Enumerable.Range(0, p)
                .OrderByDescending(j => _uniformityScores[j])
                .Take(numToSelect)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            // Prefer features with low uniformity (more concentrated)
            _selectedIndices = Enumerable.Range(0, p)
                .Where(j => _uniformityScores[j] > 0)
                .OrderBy(j => _uniformityScores[j])
                .Take(numToSelect)
                .OrderBy(x => x)
                .ToArray();
        }

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
            throw new InvalidOperationException("UniformitySelector has not been fitted.");

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
        throw new NotSupportedException("UniformitySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("UniformitySelector has not been fitted.");

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
