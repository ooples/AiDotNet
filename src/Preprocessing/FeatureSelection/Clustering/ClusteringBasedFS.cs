using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Clustering;

/// <summary>
/// Clustering-based Feature Selection using feature grouping.
/// </summary>
/// <remarks>
/// <para>
/// Clusters similar features together and selects representative features
/// from each cluster. This reduces redundancy by keeping diverse features
/// that capture different aspects of the data.
/// </para>
/// <para><b>For Beginners:</b> Sometimes features are very similar to each other
/// (like height in inches vs height in cm). This method groups similar features
/// together and picks one from each group. This way, you get diverse features
/// without redundant information.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ClusteringBasedFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nClusters;
    private readonly int _maxIterations;

    private int[]? _clusterAssignments;
    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NClusters => _nClusters;
    public int[]? ClusterAssignments => _clusterAssignments;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ClusteringBasedFS(
        int nClusters = 10,
        int maxIterations = 100,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nClusters < 1)
            throw new ArgumentException("Number of clusters must be at least 1.", nameof(nClusters));

        _nClusters = nClusters;
        _maxIterations = maxIterations;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Convert to arrays
        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        // Compute feature correlation matrix
        var correlations = ComputeFeatureCorrelations(X, n, p);

        // Cluster features using k-means on correlation profiles
        int nClusters = Math.Min(_nClusters, p);
        _clusterAssignments = ClusterFeatures(correlations, p, nClusters);

        // Select best feature from each cluster based on variance
        _featureScores = new double[p];
        for (int j = 0; j < p; j++)
        {
            double mean = 0;
            for (int i = 0; i < n; i++)
                mean += X[i, j];
            mean /= n;

            double variance = 0;
            for (int i = 0; i < n; i++)
            {
                double diff = X[i, j] - mean;
                variance += diff * diff;
            }
            _featureScores[j] = variance / (n - 1);
        }

        // Select one feature per cluster with highest variance
        var selectedList = new List<int>();
        for (int c = 0; c < nClusters; c++)
        {
            var clusterFeatures = Enumerable.Range(0, p)
                .Where(j => _clusterAssignments[j] == c)
                .ToList();

            if (clusterFeatures.Count > 0)
            {
                int bestFeature = clusterFeatures
                    .OrderByDescending(j => _featureScores[j])
                    .First();
                selectedList.Add(bestFeature);
            }
        }

        _selectedIndices = selectedList.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private double[,] ComputeFeatureCorrelations(double[,] X, int n, int p)
    {
        var corr = new double[p, p];
        var means = new double[p];
        var stds = new double[p];

        // Compute means
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                means[j] += X[i, j];
            means[j] /= n;
        }

        // Compute standard deviations
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
            {
                double diff = X[i, j] - means[j];
                stds[j] += diff * diff;
            }
            stds[j] = Math.Sqrt(stds[j] / (n - 1)) + 1e-10;
        }

        // Compute correlations
        for (int j1 = 0; j1 < p; j1++)
        {
            corr[j1, j1] = 1.0;
            for (int j2 = j1 + 1; j2 < p; j2++)
            {
                double sum = 0;
                for (int i = 0; i < n; i++)
                    sum += (X[i, j1] - means[j1]) * (X[i, j2] - means[j2]);
                corr[j1, j2] = sum / ((n - 1) * stds[j1] * stds[j2]);
                corr[j2, j1] = corr[j1, j2];
            }
        }

        return corr;
    }

    private int[] ClusterFeatures(double[,] correlations, int p, int nClusters)
    {
        var assignments = new int[p];
        var centroids = new double[nClusters, p];

        // Initialize centroids randomly
        var rand = RandomHelper.CreateSecureRandom();
        var indices = Enumerable.Range(0, p).OrderBy(_ => rand.Next()).Take(nClusters).ToList();
        for (int c = 0; c < nClusters; c++)
        {
            for (int j = 0; j < p; j++)
                centroids[c, j] = correlations[indices[c], j];
        }

        // K-means iterations
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            bool changed = false;

            // Assign features to nearest centroid
            for (int j = 0; j < p; j++)
            {
                double minDist = double.MaxValue;
                int bestCluster = 0;

                for (int c = 0; c < nClusters; c++)
                {
                    double dist = 0;
                    for (int j2 = 0; j2 < p; j2++)
                    {
                        double diff = correlations[j, j2] - centroids[c, j2];
                        dist += diff * diff;
                    }

                    if (dist < minDist)
                    {
                        minDist = dist;
                        bestCluster = c;
                    }
                }

                if (assignments[j] != bestCluster)
                {
                    assignments[j] = bestCluster;
                    changed = true;
                }
            }

            if (!changed)
                break;

            // Update centroids
            for (int c = 0; c < nClusters; c++)
            {
                var clusterMembers = Enumerable.Range(0, p)
                    .Where(j => assignments[j] == c)
                    .ToList();

                if (clusterMembers.Count == 0) continue;

                for (int j = 0; j < p; j++)
                {
                    centroids[c, j] = 0;
                    foreach (int member in clusterMembers)
                        centroids[c, j] += correlations[member, j];
                    centroids[c, j] /= clusterMembers.Count;
                }
            }
        }

        return assignments;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ClusteringBasedFS has not been fitted.");

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
        throw new NotSupportedException("ClusteringBasedFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ClusteringBasedFS has not been fitted.");

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
