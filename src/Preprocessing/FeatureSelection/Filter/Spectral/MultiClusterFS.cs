using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Spectral;

/// <summary>
/// Multi-Cluster Feature Selection (MCFS) using spectral analysis for unsupervised feature selection.
/// </summary>
/// <remarks>
/// <para>
/// MCFS uses spectral clustering to discover the underlying cluster structure of the data,
/// then selects features that best preserve this structure. It combines spectral embedding
/// with sparse regression to select informative features.
/// </para>
/// <para><b>For Beginners:</b> MCFS first discovers natural groups (clusters) in your data
/// without knowing the labels. Then it picks features that best explain these groupings.
/// If a feature helps distinguish between clusters, it's selected. This is useful when
/// you have no labels but want to find discriminative features.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MultiClusterFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nClusters;
    private readonly int _nNeighbors;
    private readonly double _sigma;
    private readonly int? _randomState;

    private double[]? _mcfsScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NClusters => _nClusters;
    public double[]? McfsScores => _mcfsScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MultiClusterFS(
        int nFeaturesToSelect = 10,
        int nClusters = 5,
        int nNeighbors = 5,
        double sigma = 1.0,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nClusters = nClusters;
        _nNeighbors = nNeighbors;
        _sigma = sigma;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Build affinity matrix
        var W = BuildAffinityMatrix(data, n, p);

        // Compute normalized Laplacian
        var D = new double[n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                D[i] += W[i, j];
        }

        // Compute eigenvectors
        int k = Math.Min(_nClusters, n - 1);
        var eigenvectors = ComputeBottomEigenvectors(W, D, n, k, random);

        // For each feature, compute MCFS score using L1-regularized regression
        _mcfsScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Extract feature column
            var f = new double[n];
            for (int i = 0; i < n; i++)
                f[i] = NumOps.ToDouble(data[i, j]);

            // Normalize feature
            double mean = f.Average();
            double std = Math.Sqrt(f.Select(x => (x - mean) * (x - mean)).Sum() / n);
            if (std > 1e-10)
            {
                for (int i = 0; i < n; i++)
                    f[i] = (f[i] - mean) / std;
            }

            // Compute maximum absolute correlation with eigenvectors
            double maxCorr = 0;
            foreach (var ev in eigenvectors)
            {
                double corr = 0;
                for (int i = 0; i < n; i++)
                    corr += f[i] * ev[i];
                maxCorr = Math.Max(maxCorr, Math.Abs(corr / n));
            }

            _mcfsScores[j] = maxCorr;
        }

        // Select features with highest MCFS scores
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _mcfsScores
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(numToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[,] BuildAffinityMatrix(Matrix<T> data, int n, int p)
    {
        var W = new double[n, n];

        // Compute pairwise distances
        var distances = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double dist = 0;
                for (int f = 0; f < p; f++)
                {
                    double diff = NumOps.ToDouble(data[i, f]) - NumOps.ToDouble(data[j, f]);
                    dist += diff * diff;
                }
                distances[i, j] = Math.Sqrt(dist);
                distances[j, i] = distances[i, j];
            }
        }

        // Build k-NN graph with Gaussian kernel
        for (int i = 0; i < n; i++)
        {
            var neighbors = new List<(int Index, double Distance)>();
            for (int j = 0; j < n; j++)
            {
                if (j != i)
                    neighbors.Add((j, distances[i, j]));
            }

            var kNearest = neighbors
                .OrderBy(x => x.Distance)
                .Take(_nNeighbors)
                .ToList();

            foreach (var neighbor in kNearest)
            {
                double weight = Math.Exp(-distances[i, neighbor.Index] * distances[i, neighbor.Index] /
                    (2 * _sigma * _sigma));
                W[i, neighbor.Index] = weight;
                W[neighbor.Index, i] = weight;
            }
        }

        return W;
    }

    private List<double[]> ComputeBottomEigenvectors(double[,] W, double[] D, int n, int k, Random random)
    {
        // Compute normalized Laplacian L = D^(-1/2) * W * D^(-1/2)
        // We want bottom k eigenvectors (smallest eigenvalues except 0)
        var L = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (D[i] > 1e-10 && D[j] > 1e-10)
                    L[i, j] = W[i, j] / Math.Sqrt(D[i] * D[j]);
            }
        }

        var eigenvectors = new List<double[]>();

        // Use power iteration on (I - L) to get smallest eigenvectors of L
        for (int ev = 0; ev < k; ev++)
        {
            var v = new double[n];
            for (int i = 0; i < n; i++)
                v[i] = random.NextDouble() - 0.5;

            // Power iteration
            for (int iter = 0; iter < 50; iter++)
            {
                var Lv = new double[n];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        Lv[i] += L[i, j] * v[j];

                // Orthogonalize against previous eigenvectors
                foreach (var prev in eigenvectors)
                {
                    double dot = 0;
                    for (int i = 0; i < n; i++)
                        dot += Lv[i] * prev[i];
                    for (int i = 0; i < n; i++)
                        Lv[i] -= dot * prev[i];
                }

                // Normalize
                double norm = Math.Sqrt(Lv.Sum(x => x * x));
                if (norm > 1e-10)
                {
                    for (int i = 0; i < n; i++)
                        v[i] = Lv[i] / norm;
                }
            }

            eigenvectors.Add((double[])v.Clone());
        }

        return eigenvectors;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MultiClusterFS has not been fitted.");

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
        throw new NotSupportedException("MultiClusterFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MultiClusterFS has not been fitted.");

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
