using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Graph;

/// <summary>
/// Spectral Clustering-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses spectral graph theory to cluster features and select representative
/// features from each cluster based on eigenvector analysis.
/// </para>
/// <para><b>For Beginners:</b> Spectral clustering looks at features as points
/// in a graph and uses special math (eigenvectors) to find natural groups.
/// We then pick the most representative feature from each group.
/// </para>
/// </remarks>
public class SpectralClusteringFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nClusters;
    private readonly double _sigma;

    private int[]? _clusterAssignments;
    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NClusters => _nClusters;
    public int[]? ClusterAssignments => _clusterAssignments;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SpectralClusteringFS(
        int nClusters = 10,
        double sigma = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nClusters < 1)
            throw new ArgumentException("Number of clusters must be at least 1.", nameof(nClusters));

        _nClusters = nClusters;
        _sigma = sigma;
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

        // Build similarity matrix between features
        var W = new double[p, p];
        for (int j1 = 0; j1 < p; j1++)
        {
            for (int j2 = j1; j2 < p; j2++)
            {
                double dist = 0;
                for (int i = 0; i < n; i++)
                {
                    double diff = X[i, j1] - X[i, j2];
                    dist += diff * diff;
                }
                double sim = Math.Exp(-dist / (2 * _sigma * _sigma * n));
                W[j1, j2] = sim;
                W[j2, j1] = sim;
            }
        }

        // Compute degree matrix and Laplacian
        var D = new double[p];
        for (int j = 0; j < p; j++)
            for (int kk = 0; kk < p; kk++)
                D[j] += W[j, kk];

        // Normalized Laplacian: L = I - D^(-1/2) W D^(-1/2)
        var L = new double[p, p];
        for (int j1 = 0; j1 < p; j1++)
        {
            for (int j2 = 0; j2 < p; j2++)
            {
                if (j1 == j2)
                    L[j1, j2] = 1 - W[j1, j2] / (Math.Sqrt(D[j1] * D[j2]) + 1e-10);
                else
                    L[j1, j2] = -W[j1, j2] / (Math.Sqrt(D[j1] * D[j2]) + 1e-10);
            }
        }

        // Get bottom k eigenvectors
        int k = Math.Min(_nClusters, p);
        var eigenvectors = PowerIterationBottomK(L, p, k, 100);

        // K-means on eigenvector embedding
        _clusterAssignments = KMeansClustering(eigenvectors, p, k);

        // Compute feature scores (variance)
        _featureScores = new double[p];
        for (int j = 0; j < p; j++)
        {
            double mean = 0;
            for (int i = 0; i < n; i++) mean += X[i, j];
            mean /= n;
            for (int i = 0; i < n; i++) _featureScores[j] += (X[i, j] - mean) * (X[i, j] - mean);
            _featureScores[j] /= (n - 1);
        }

        // Select best feature from each cluster
        var selectedList = new List<int>();
        for (int c = 0; c < k; c++)
        {
            var clusterFeatures = Enumerable.Range(0, p).Where(j => _clusterAssignments[j] == c).ToList();
            if (clusterFeatures.Count > 0)
                selectedList.Add(clusterFeatures.OrderByDescending(j => _featureScores[j]).First());
        }

        _selectedIndices = selectedList.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private double[,] PowerIterationBottomK(double[,] L, int p, int k, int maxIter)
    {
        var result = new double[p, k];
        var rand = RandomHelper.CreateSecureRandom();

        // Compute max eigenvalue for shift
        double maxEig = 0;
        for (int j = 0; j < p; j++) maxEig = Math.Max(maxEig, L[j, j]);
        maxEig *= 2;

        // Shift: L' = maxEig * I - L (converts smallest to largest)
        var Lp = new double[p, p];
        for (int j1 = 0; j1 < p; j1++)
            for (int j2 = 0; j2 < p; j2++)
                Lp[j1, j2] = (j1 == j2 ? maxEig : 0) - L[j1, j2];

        for (int ki = 0; ki < k; ki++)
        {
            var v = new double[p];
            for (int j = 0; j < p; j++) v[j] = rand.NextDouble();

            for (int iter = 0; iter < maxIter; iter++)
            {
                // Orthogonalize against previous
                for (int prev = 0; prev < ki; prev++)
                {
                    double dot = 0;
                    for (int j = 0; j < p; j++) dot += v[j] * result[j, prev];
                    for (int j = 0; j < p; j++) v[j] -= dot * result[j, prev];
                }

                var newV = new double[p];
                for (int j1 = 0; j1 < p; j1++)
                    for (int j2 = 0; j2 < p; j2++)
                        newV[j1] += Lp[j1, j2] * v[j2];

                double norm = Math.Sqrt(newV.Sum(x => x * x)) + 1e-10;
                for (int j = 0; j < p; j++) v[j] = newV[j] / norm;
            }

            for (int j = 0; j < p; j++) result[j, ki] = v[j];
        }

        return result;
    }

    private int[] KMeansClustering(double[,] data, int p, int k)
    {
        var assignments = new int[p];
        var centroids = new double[k, k];
        var rand = RandomHelper.CreateSecureRandom();

        // Initialize centroids
        var initIdx = Enumerable.Range(0, p).OrderBy(_ => rand.Next()).Take(k).ToList();
        for (int c = 0; c < k; c++)
            for (int d = 0; d < k; d++)
                centroids[c, d] = data[initIdx[c], d];

        for (int iter = 0; iter < 50; iter++)
        {
            // Assign
            for (int j = 0; j < p; j++)
            {
                double minDist = double.MaxValue;
                for (int c = 0; c < k; c++)
                {
                    double dist = 0;
                    for (int d = 0; d < k; d++)
                        dist += (data[j, d] - centroids[c, d]) * (data[j, d] - centroids[c, d]);
                    if (dist < minDist) { minDist = dist; assignments[j] = c; }
                }
            }

            // Update
            var counts = new int[k];
            var newCentroids = new double[k, k];
            for (int j = 0; j < p; j++)
            {
                int c = assignments[j];
                counts[c]++;
                for (int d = 0; d < k; d++)
                    newCentroids[c, d] += data[j, d];
            }
            for (int c = 0; c < k; c++)
                if (counts[c] > 0)
                    for (int d = 0; d < k; d++)
                        centroids[c, d] = newCentroids[c, d] / counts[c];
        }

        return assignments;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SpectralClusteringFS has not been fitted.");

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
        throw new NotSupportedException("SpectralClusteringFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SpectralClusteringFS has not been fitted.");

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
