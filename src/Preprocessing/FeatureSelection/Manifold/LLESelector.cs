using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Manifold;

/// <summary>
/// Locally Linear Embedding (LLE) inspired Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their importance in preserving local linear
/// relationships in the data manifold.
/// </para>
/// <para><b>For Beginners:</b> LLE assumes each point can be described as a
/// linear combination of its neighbors. This selector finds features important
/// for maintaining those local linear relationships.
/// </para>
/// </remarks>
public class LLESelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;

    private double[]? _localLinearityScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NNeighbors => _nNeighbors;
    public double[]? LocalLinearityScores => _localLinearityScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public LLESelector(
        int nFeaturesToSelect = 10,
        int nNeighbors = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nNeighbors = nNeighbors;
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

        _localLinearityScores = new double[p];

        // For each feature, measure how well it supports local linearity
        for (int j = 0; j < p; j++)
        {
            double totalScore = 0;
            int count = 0;

            for (int i = 0; i < n; i++)
            {
                // Find k nearest neighbors for this point
                var neighbors = GetNearestNeighbors(X, i, n, p);

                if (neighbors.Count < 2) continue;

                // Compute reconstruction error for this feature
                double centerVal = X[i, j];
                double avgNeighborVal = neighbors.Average(idx => X[idx, j]);
                double error = Math.Abs(centerVal - avgNeighborVal);

                // Features with low reconstruction error are more "locally linear"
                totalScore += 1.0 / (1 + error);
                count++;
            }

            _localLinearityScores[j] = count > 0 ? totalScore / count : 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _localLinearityScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private List<int> GetNearestNeighbors(double[,] X, int pointIdx, int n, int p)
    {
        var distances = new List<(int idx, double dist)>();

        for (int i = 0; i < n; i++)
        {
            if (i == pointIdx) continue;

            double dist = 0;
            for (int j = 0; j < p; j++)
            {
                double diff = X[pointIdx, j] - X[i, j];
                dist += diff * diff;
            }

            distances.Add((i, Math.Sqrt(dist)));
        }

        return distances
            .OrderBy(d => d.dist)
            .Take(_nNeighbors)
            .Select(d => d.idx)
            .ToList();
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LLESelector has not been fitted.");

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
        throw new NotSupportedException("LLESelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LLESelector has not been fitted.");

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
