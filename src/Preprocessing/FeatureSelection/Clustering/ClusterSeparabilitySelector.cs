using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Clustering;

/// <summary>
/// Cluster Separability based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on how well they separate natural clusters in the data,
/// measuring the ratio of between-cluster to within-cluster variance.
/// </para>
/// <para><b>For Beginners:</b> Good features should make clusters easy to tell apart.
/// This selector finds features where different groups of data are far from each
/// other but points within each group are close together.
/// </para>
/// </remarks>
public class ClusterSeparabilitySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nClusters;

    private double[]? _separabilityScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NClusters => _nClusters;
    public double[]? SeparabilityScores => _separabilityScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ClusterSeparabilitySelector(
        int nFeaturesToSelect = 10,
        int nClusters = 3,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nClusters = nClusters;
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

        // Simple k-means clustering to find natural clusters
        var clusterLabels = SimpleKMeans(X, n, p);

        _separabilityScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            _separabilityScores[j] = ComputeSeparability(X, clusterLabels, j, n);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _separabilityScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private int[] SimpleKMeans(double[,] X, int n, int p)
    {
        var labels = new int[n];
        var centroids = new double[_nClusters, p];

        // Initialize centroids by spreading evenly
        for (int k = 0; k < _nClusters; k++)
        {
            int idx = (k * n) / _nClusters;
            for (int j = 0; j < p; j++)
                centroids[k, j] = X[idx, j];
        }

        // Run a few iterations of k-means
        for (int iter = 0; iter < 10; iter++)
        {
            // Assign points to nearest centroid
            for (int i = 0; i < n; i++)
            {
                double minDist = double.MaxValue;
                for (int k = 0; k < _nClusters; k++)
                {
                    double dist = 0;
                    for (int j = 0; j < p; j++)
                    {
                        double diff = X[i, j] - centroids[k, j];
                        dist += diff * diff;
                    }
                    if (dist < minDist)
                    {
                        minDist = dist;
                        labels[i] = k;
                    }
                }
            }

            // Update centroids
            var counts = new int[_nClusters];
            for (int k = 0; k < _nClusters; k++)
                for (int j = 0; j < p; j++)
                    centroids[k, j] = 0;

            for (int i = 0; i < n; i++)
            {
                int k = labels[i];
                counts[k]++;
                for (int j = 0; j < p; j++)
                    centroids[k, j] += X[i, j];
            }

            for (int k = 0; k < _nClusters; k++)
            {
                if (counts[k] > 0)
                {
                    for (int j = 0; j < p; j++)
                        centroids[k, j] /= counts[k];
                }
            }
        }

        return labels;
    }

    private double ComputeSeparability(double[,] X, int[] labels, int featureIdx, int n)
    {
        // Compute cluster means
        var clusterMeans = new double[_nClusters];
        var clusterCounts = new int[_nClusters];

        for (int i = 0; i < n; i++)
        {
            clusterMeans[labels[i]] += X[i, featureIdx];
            clusterCounts[labels[i]]++;
        }

        for (int k = 0; k < _nClusters; k++)
        {
            if (clusterCounts[k] > 0)
                clusterMeans[k] /= clusterCounts[k];
        }

        // Global mean
        double globalMean = 0;
        for (int i = 0; i < n; i++)
            globalMean += X[i, featureIdx];
        globalMean /= n;

        // Between-cluster variance
        double ssb = 0;
        for (int k = 0; k < _nClusters; k++)
        {
            ssb += clusterCounts[k] * (clusterMeans[k] - globalMean) * (clusterMeans[k] - globalMean);
        }

        // Within-cluster variance
        double ssw = 0;
        for (int i = 0; i < n; i++)
        {
            double diff = X[i, featureIdx] - clusterMeans[labels[i]];
            ssw += diff * diff;
        }

        return ssw > 1e-10 ? ssb / ssw : 0;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ClusterSeparabilitySelector has not been fitted.");

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
        throw new NotSupportedException("ClusterSeparabilitySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ClusterSeparabilitySelector has not been fitted.");

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
