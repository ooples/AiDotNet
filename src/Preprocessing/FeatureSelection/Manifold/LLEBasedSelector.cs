using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Manifold;

/// <summary>
/// Locally Linear Embedding (LLE) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses Locally Linear Embedding to identify features that best preserve
/// the local neighborhood structure of the data manifold.
/// </para>
/// <para><b>For Beginners:</b> LLE assumes data lies on a curved surface
/// (manifold) in high-dimensional space. It finds how to reconstruct each
/// point from its neighbors. Features that are most important for these
/// local reconstructions are selected.
/// </para>
/// </remarks>
public class LLEBasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public LLEBasedSelector(
        int nFeaturesToSelect = 10,
        int nNeighbors = 10,
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

        int k = Math.Min(_nNeighbors, n - 1);

        // Find k-nearest neighbors for each point
        var neighbors = new int[n, k];
        for (int i = 0; i < n; i++)
        {
            var distances = new List<(int idx, double dist)>();
            for (int j = 0; j < n; j++)
            {
                if (i == j) continue;
                double dist = 0;
                for (int d = 0; d < p; d++)
                    dist += (X[i, d] - X[j, d]) * (X[i, d] - X[j, d]);
                distances.Add((j, dist));
            }
            var sorted = distances.OrderBy(x => x.dist).Take(k).ToList();
            for (int j = 0; j < k; j++)
                neighbors[i, j] = sorted[j].idx;
        }

        // Compute reconstruction weights
        var W = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            // Build local covariance matrix
            var C = new double[k, k];
            for (int j1 = 0; j1 < k; j1++)
            {
                for (int j2 = 0; j2 < k; j2++)
                {
                    int idx1 = neighbors[i, j1];
                    int idx2 = neighbors[i, j2];
                    for (int d = 0; d < p; d++)
                        C[j1, j2] += (X[i, d] - X[idx1, d]) * (X[i, d] - X[idx2, d]);
                }
                C[j1, j1] += 1e-6; // Regularization
            }

            // Solve for weights (C * w = 1)
            var ones = new double[k];
            for (int j = 0; j < k; j++) ones[j] = 1;
            var w = SolveSystem(C, ones, k);

            // Normalize
            double sum = w.Sum();
            if (Math.Abs(sum) > 1e-10)
                for (int j = 0; j < k; j++)
                    w[j] /= sum;

            for (int j = 0; j < k; j++)
                W[i, neighbors[i, j]] = w[j];
        }

        // Compute feature scores based on reconstruction quality
        _featureScores = new double[p];
        for (int d = 0; d < p; d++)
        {
            double score = 0;
            for (int i = 0; i < n; i++)
            {
                // Reconstruction error for this feature
                double reconstructed = 0;
                for (int j = 0; j < n; j++)
                    reconstructed += W[i, j] * X[j, d];
                double error = (X[i, d] - reconstructed) * (X[i, d] - reconstructed);
                score += 1.0 / (error + 1e-10);
            }
            _featureScores[d] = score / n;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _featureScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] SolveSystem(double[,] A, double[] b, int n)
    {
        var aug = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                aug[i, j] = A[i, j];
            aug[i, n] = b[i];
        }

        for (int col = 0; col < n; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
                if (Math.Abs(aug[row, col]) > Math.Abs(aug[maxRow, col]))
                    maxRow = row;

            for (int j = 0; j <= n; j++)
                (aug[col, j], aug[maxRow, j]) = (aug[maxRow, j], aug[col, j]);

            if (Math.Abs(aug[col, col]) < 1e-10) continue;

            for (int row = col + 1; row < n; row++)
            {
                double factor = aug[row, col] / aug[col, col];
                for (int j = col; j <= n; j++)
                    aug[row, j] -= factor * aug[col, j];
            }
        }

        var x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            x[i] = aug[i, n];
            for (int j = i + 1; j < n; j++)
                x[i] -= aug[i, j] * x[j];
            x[i] /= (Math.Abs(aug[i, i]) > 1e-10 ? aug[i, i] : 1);
        }

        return x;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LLEBasedSelector has not been fitted.");

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
        throw new NotSupportedException("LLEBasedSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LLEBasedSelector has not been fitted.");

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
