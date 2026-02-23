using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Texture;

/// <summary>
/// Local Binary Pattern inspired Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their local pattern characteristics computed
/// from neighboring value comparisons.
/// </para>
/// <para><b>For Beginners:</b> This looks at how each value compares to its neighbors,
/// creating a "pattern" at each point. Features with more interesting local patterns
/// often carry more discriminative information.
/// </para>
/// </remarks>
public class LocalBinaryPatternSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _neighborhoodSize;

    private double[]? _patternScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NeighborhoodSize => _neighborhoodSize;
    public double[]? PatternScores => _patternScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public LocalBinaryPatternSelector(
        int nFeaturesToSelect = 10,
        int neighborhoodSize = 3,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _neighborhoodSize = neighborhoodSize;
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

        _patternScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            _patternScores[j] = ComputePatternDiversity(col, n);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _patternScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputePatternDiversity(double[] values, int n)
    {
        int halfNeighbor = _neighborhoodSize / 2;
        var patternCounts = new Dictionary<int, int>();

        for (int i = halfNeighbor; i < n - halfNeighbor; i++)
        {
            int pattern = 0;
            int bit = 0;

            // Compare with neighbors
            for (int k = -halfNeighbor; k <= halfNeighbor; k++)
            {
                if (k == 0) continue;

                if (values[i + k] >= values[i])
                    pattern |= (1 << bit);
                bit++;
            }

            if (!patternCounts.ContainsKey(pattern))
                patternCounts[pattern] = 0;
            patternCounts[pattern]++;
        }

        // Compute entropy of pattern distribution
        int total = n - 2 * halfNeighbor;
        if (total <= 0) return 0;

        double entropy = 0;
        foreach (var count in patternCounts.Values)
        {
            double p = (double)count / total;
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
            throw new InvalidOperationException("LocalBinaryPatternSelector has not been fitted.");

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
        throw new NotSupportedException("LocalBinaryPatternSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LocalBinaryPatternSelector has not been fitted.");

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
