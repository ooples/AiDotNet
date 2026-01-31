using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Spectral;

/// <summary>
/// Multi-Cluster Feature Selection using spectral graph analysis.
/// </summary>
/// <remarks>
/// <para>
/// MCFS uses spectral analysis to find features that preserve the multi-cluster
/// structure in data. It computes cluster indicators via spectral clustering
/// and then selects features using sparse regression with L1 regularization.
/// </para>
/// <para><b>For Beginners:</b> MCFS finds features that help preserve the natural
/// groupings (clusters) in your data. It first discovers hidden cluster structures
/// using spectral methods, then picks features that best capture those clusters.
/// Great when you don't have labels but want features that preserve data structure.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MCFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nClusters;
    private readonly int _nNeighbors;
    private readonly double _alpha;
    private readonly int? _randomState;

    private double[]? _mcfsScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? MCFSScores => _mcfsScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MCFS(
        int nFeaturesToSelect = 10,
        int nClusters = 5,
        int nNeighbors = 5,
        double alpha = 1.0,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nClusters < 2)
            throw new ArgumentException("Number of clusters must be at least 2.", nameof(nClusters));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nClusters = nClusters;
        _nNeighbors = nNeighbors;
        _alpha = alpha;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Construct k-NN affinity graph
        var affinityMatrix = ConstructAffinityMatrix(data, n, p);

        // Compute normalized Laplacian
        var laplacian = ComputeNormalizedLaplacian(affinityMatrix, n);

        // Get bottom k eigenvectors (cluster indicators)
        var clusterIndicators = ComputeBottomEigenvectors(laplacian, n, _nClusters);

        // Sparse regression to find features that predict cluster indicators
        _mcfsScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            double score = 0;
            for (int k = 0; k < _nClusters; k++)
            {
                // Compute correlation-based regression coefficient
                double featureMean = 0, indicatorMean = 0;
                for (int i = 0; i < n; i++)
                {
                    featureMean += NumOps.ToDouble(data[i, j]);
                    indicatorMean += clusterIndicators[i, k];
                }
                featureMean /= n;
                indicatorMean /= n;

                double covariance = 0, featureVar = 0;
                for (int i = 0; i < n; i++)
                {
                    double xDiff = NumOps.ToDouble(data[i, j]) - featureMean;
                    double yDiff = clusterIndicators[i, k] - indicatorMean;
                    covariance += xDiff * yDiff;
                    featureVar += xDiff * xDiff;
                }

                if (featureVar > 1e-10)
                {
                    double coef = Math.Abs(covariance / featureVar);
                    // Apply L1-style sparsity
                    double sparseCoef = Math.Max(0, coef - _alpha / n);
                    score += sparseCoef * sparseCoef;
                }
            }
            _mcfsScores[j] = Math.Sqrt(score);
        }

        // Select top features
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _mcfsScores
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(nToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[,] ConstructAffinityMatrix(Matrix<T> data, int n, int p)
    {
        var distances = new double[n, n];
        var affinity = new double[n, n];

        // Compute pairwise distances
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

        // Build k-NN graph with heat kernel weights
        for (int i = 0; i < n; i++)
        {
            // Find k nearest neighbors
            var neighbors = Enumerable.Range(0, n)
                .Where(j => j != i)
                .OrderBy(j => distances[i, j])
                .Take(_nNeighbors)
                .ToList();

            // Compute local bandwidth (average distance to k neighbors)
            double sigma = neighbors.Average(j => distances[i, j]);
            if (sigma < 1e-10) sigma = 1;

            foreach (int j in neighbors)
            {
                double weight = Math.Exp(-distances[i, j] * distances[i, j] / (2 * sigma * sigma));
                affinity[i, j] = Math.Max(affinity[i, j], weight);
                affinity[j, i] = affinity[i, j]; // Symmetrize
            }
        }

        return affinity;
    }

    private double[,] ComputeNormalizedLaplacian(double[,] affinity, int n)
    {
        var degree = new double[n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                degree[i] += affinity[i, j];

        var laplacian = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            double di = Math.Sqrt(degree[i] + 1e-10);
            for (int j = 0; j < n; j++)
            {
                double dj = Math.Sqrt(degree[j] + 1e-10);
                if (i == j)
                    laplacian[i, j] = 1;
                else
                    laplacian[i, j] = -affinity[i, j] / (di * dj);
            }
        }

        return laplacian;
    }

    private double[,] ComputeBottomEigenvectors(double[,] laplacian, int n, int k)
    {
        // Power iteration with deflation to get bottom k eigenvectors
        var eigenvectors = new double[n, k];
        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var previousVectors = new List<double[]>();

        for (int ev = 0; ev < k; ev++)
        {
            var v = new double[n];
            for (int i = 0; i < n; i++)
                v[i] = rand.NextDouble() - 0.5;

            // Power iteration for smallest eigenvalue (inverse iteration concept)
            // Use (I - L) instead of L to get largest eigenvalue which corresponds to smallest of L
            for (int iter = 0; iter < 100; iter++)
            {
                var newV = new double[n];
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        double val = (i == j ? 1 : 0) - laplacian[i, j];
                        newV[i] += val * v[j];
                    }
                }

                // Deflate against previous eigenvectors
                foreach (var prev in previousVectors)
                {
                    double dot = 0;
                    for (int i = 0; i < n; i++)
                        dot += newV[i] * prev[i];
                    for (int i = 0; i < n; i++)
                        newV[i] -= dot * prev[i];
                }

                // Normalize
                double norm = 0;
                for (int i = 0; i < n; i++)
                    norm += newV[i] * newV[i];
                norm = Math.Sqrt(norm);
                if (norm > 1e-10)
                    for (int i = 0; i < n; i++)
                        newV[i] /= norm;

                v = newV;
            }

            previousVectors.Add(v);
            for (int i = 0; i < n; i++)
                eigenvectors[i, ev] = v[i];
        }

        return eigenvectors;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        FitCore(data);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MCFS has not been fitted.");

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
        throw new NotSupportedException("MCFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MCFS has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
