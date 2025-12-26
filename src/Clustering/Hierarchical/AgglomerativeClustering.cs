using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Clustering.Hierarchical;

/// <summary>
/// Agglomerative Hierarchical Clustering implementation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Agglomerative clustering builds a hierarchy of clusters bottom-up. Starting with
/// each sample as its own cluster, it iteratively merges the closest pair of clusters
/// until the desired number of clusters is reached.
/// </para>
/// <para>
/// Time complexity: O(n³) for naive implementation, O(n² log n) with efficient data structures.
/// Space complexity: O(n²) for distance matrix.
/// </para>
/// <para><b>For Beginners:</b> Hierarchical clustering builds a "family tree" of your data.
///
/// Imagine sorting photos of animals:
/// 1. Start with each photo as its own group
/// 2. Find the two most similar photos and group them
/// 3. Keep grouping until you have the number of groups you want
///
/// The result can be shown as a dendrogram (tree diagram) where:
/// - Bottom: Each individual item
/// - Top: All items merged into one group
/// - Height: Shows how different merged groups are
///
/// Use Ward linkage for most cases - it creates nice, balanced clusters.
/// </para>
/// </remarks>
public class AgglomerativeClustering<T> : ClusteringBase<T>
{
    private readonly HierarchicalOptions<T> _options;
    private List<(int Cluster1, int Cluster2, double Distance, int Size)>? _dendrogram;

    /// <summary>
    /// Initializes a new AgglomerativeClustering instance.
    /// </summary>
    /// <param name="options">The clustering options.</param>
    public AgglomerativeClustering(HierarchicalOptions<T>? options = null)
        : base(options ?? new HierarchicalOptions<T>())
    {
        _options = options ?? new HierarchicalOptions<T>();

        if (_options.DistanceMetric is null)
        {
            _options.DistanceMetric = new EuclideanDistance<T>();
        }

        NumClusters = _options.NumClusters;
    }

    /// <summary>
    /// Gets the dendrogram (merge history).
    /// </summary>
    public List<(int Cluster1, int Cluster2, double Distance, int Size)>? Dendrogram => _dendrogram;

    /// <summary>
    /// Gets the linkage method used.
    /// </summary>
    public LinkageMethod Linkage => _options.Linkage;

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.Clustering;

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new AgglomerativeClustering<T>(new HierarchicalOptions<T>
        {
            NumClusters = _options.NumClusters,
            Linkage = _options.Linkage,
            DistanceThreshold = _options.DistanceThreshold,
            ComputeFullTree = _options.ComputeFullTree,
            DistanceMetric = _options.DistanceMetric
        });
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (AgglomerativeClustering<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;

        if (n < 2)
        {
            Labels = new Vector<T>(n);
            if (n == 1)
            {
                Labels[0] = NumOps.Zero;
            }
            NumClusters = Math.Min(n, _options.NumClusters);
            IsTrained = true;
            return;
        }

        // Compute initial pairwise distance matrix
        var distMatrix = ComputeDistanceMatrix(x);

        // Initialize cluster assignments
        var clusterIds = new int[n];
        for (int i = 0; i < n; i++)
        {
            clusterIds[i] = i;
        }

        // Track cluster sizes
        var clusterSizes = new int[2 * n - 1];
        for (int i = 0; i < n; i++)
        {
            clusterSizes[i] = 1;
        }

        // Track cluster centroids for Ward's method
        var centroids = new double[2 * n - 1, x.Columns];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                centroids[i, j] = NumOps.ToDouble(x[i, j]);
            }
        }

        // Dendrogram storage
        _dendrogram = new List<(int, int, double, int)>();

        // Active clusters (using UnionFind-like structure)
        var active = new HashSet<int>();
        for (int i = 0; i < n; i++)
        {
            active.Add(i);
        }

        int nextClusterId = n;

        // Main loop: merge until desired number of clusters
        int targetClusters = _options.DistanceThreshold.HasValue ? 1 : _options.NumClusters;

        while (active.Count > targetClusters)
        {
            // Find closest pair
            var (minI, minJ, minDist) = FindClosestPair(distMatrix, active, clusterSizes, centroids, x.Columns);

            if (minI < 0 || minJ < 0)
            {
                break; // No more pairs to merge
            }

            // Check distance threshold
            if (_options.DistanceThreshold.HasValue && minDist > _options.DistanceThreshold.Value)
            {
                break;
            }

            // Record merge
            int newSize = clusterSizes[minI] + clusterSizes[minJ];
            _dendrogram.Add((minI, minJ, minDist, newSize));

            // Update centroid
            for (int d = 0; d < x.Columns; d++)
            {
                centroids[nextClusterId, d] = (centroids[minI, d] * clusterSizes[minI] +
                                                centroids[minJ, d] * clusterSizes[minJ]) / newSize;
            }
            clusterSizes[nextClusterId] = newSize;

            // Update distance matrix with Lance-Williams formula
            UpdateDistanceMatrix(distMatrix, active, minI, minJ, nextClusterId, clusterSizes);

            // Update active clusters
            active.Remove(minI);
            active.Remove(minJ);
            active.Add(nextClusterId);

            nextClusterId++;
        }

        // Assign final cluster labels
        Labels = AssignLabels(n, active);
        NumClusters = active.Count;

        // Compute cluster centers
        ComputeClusterCenters(x);

        IsTrained = true;
    }

    private double[,] ComputeDistanceMatrix(Matrix<T> x)
    {
        int n = x.Rows;
        var distMatrix = new double[2 * n - 1, 2 * n - 1];
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        for (int i = 0; i < n; i++)
        {
            var rowI = GetRow(x, i);
            distMatrix[i, i] = 0;

            for (int j = i + 1; j < n; j++)
            {
                var rowJ = GetRow(x, j);
                double dist = NumOps.ToDouble(metric.Compute(rowI, rowJ));

                // For Ward's method, use squared distance
                if (_options.Linkage == LinkageMethod.Ward)
                {
                    dist = dist * dist;
                }

                distMatrix[i, j] = dist;
                distMatrix[j, i] = dist;
            }
        }

        // Initialize remaining entries to infinity
        for (int i = n; i < 2 * n - 1; i++)
        {
            for (int j = 0; j < 2 * n - 1; j++)
            {
                distMatrix[i, j] = double.MaxValue;
                distMatrix[j, i] = double.MaxValue;
            }
        }

        return distMatrix;
    }

    private (int I, int J, double Dist) FindClosestPair(
        double[,] distMatrix,
        HashSet<int> active,
        int[] sizes,
        double[,] centroids,
        int dims)
    {
        int minI = -1, minJ = -1;
        double minDist = double.MaxValue;

        var activeList = active.ToList();

        for (int ai = 0; ai < activeList.Count; ai++)
        {
            int i = activeList[ai];
            for (int aj = ai + 1; aj < activeList.Count; aj++)
            {
                int j = activeList[aj];

                double dist = distMatrix[i, j];

                if (dist < minDist)
                {
                    minDist = dist;
                    minI = i;
                    minJ = j;
                }
            }
        }

        return (minI, minJ, minDist);
    }

    private void UpdateDistanceMatrix(
        double[,] distMatrix,
        HashSet<int> active,
        int mergedI,
        int mergedJ,
        int newCluster,
        int[] sizes)
    {
        double sizeI = sizes[mergedI];
        double sizeJ = sizes[mergedJ];
        double sizeNew = sizes[newCluster];

        foreach (int k in active)
        {
            if (k == mergedI || k == mergedJ)
            {
                continue;
            }

            double dIK = distMatrix[mergedI, k];
            double dJK = distMatrix[mergedJ, k];
            double dIJ = distMatrix[mergedI, mergedJ];
            double sizeK = sizes[k];

            double newDist = ComputeLinkageDistance(dIK, dJK, dIJ, sizeI, sizeJ, sizeK);

            distMatrix[newCluster, k] = newDist;
            distMatrix[k, newCluster] = newDist;
        }
    }

    private double ComputeLinkageDistance(
        double dIK, double dJK, double dIJ,
        double sizeI, double sizeJ, double sizeK)
    {
        return _options.Linkage switch
        {
            LinkageMethod.Single => Math.Min(dIK, dJK),

            LinkageMethod.Complete => Math.Max(dIK, dJK),

            LinkageMethod.Average => (sizeI * dIK + sizeJ * dJK) / (sizeI + sizeJ),

            LinkageMethod.Weighted => (dIK + dJK) / 2.0,

            LinkageMethod.Ward =>
                ((sizeI + sizeK) * dIK + (sizeJ + sizeK) * dJK - sizeK * dIJ) / (sizeI + sizeJ + sizeK),

            LinkageMethod.Centroid =>
                (sizeI * dIK + sizeJ * dJK) / (sizeI + sizeJ) - (sizeI * sizeJ * dIJ) / ((sizeI + sizeJ) * (sizeI + sizeJ)),

            LinkageMethod.Median =>
                (dIK + dJK) / 2.0 - dIJ / 4.0,

            _ => Math.Min(dIK, dJK)
        };
    }

    private Vector<T> AssignLabels(int n, HashSet<int> finalClusters)
    {
        var labels = new int[n];
        for (int i = 0; i < n; i++)
        {
            labels[i] = -1;
        }

        // Build a mapping from original points to their final cluster
        var clusterMap = new int[2 * n - 1];
        for (int i = 0; i < clusterMap.Length; i++)
        {
            clusterMap[i] = -1;
        }

        int clusterIdx = 0;
        foreach (int cluster in finalClusters)
        {
            clusterMap[cluster] = clusterIdx++;
        }

        // Trace each merge to find which final cluster each point belongs to
        var parent = new int[2 * n - 1];
        for (int i = 0; i < parent.Length; i++)
        {
            parent[i] = i;
        }

        int nextId = n;
        foreach (var (c1, c2, dist, size) in _dendrogram!)
        {
            parent[c1] = nextId;
            parent[c2] = nextId;
            nextId++;
        }

        // Find root for each original point
        for (int i = 0; i < n; i++)
        {
            int current = i;
            while (parent[current] != current && clusterMap[current] < 0)
            {
                current = parent[current];
            }

            if (clusterMap[current] >= 0)
            {
                labels[i] = clusterMap[current];
            }
            else
            {
                // Point is in its own cluster (no merges)
                labels[i] = clusterMap[i] >= 0 ? clusterMap[i] : 0;
            }
        }

        var result = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            result[i] = NumOps.FromDouble(labels[i]);
        }

        return result;
    }

    private void ComputeClusterCenters(Matrix<T> x)
    {
        if (Labels is null || NumClusters <= 0)
        {
            return;
        }

        ClusterCenters = new Matrix<T>(NumClusters, x.Columns);
        var counts = new int[NumClusters];

        for (int i = 0; i < x.Rows; i++)
        {
            int cluster = (int)NumOps.ToDouble(Labels[i]);
            if (cluster >= 0 && cluster < NumClusters)
            {
                counts[cluster]++;
                for (int j = 0; j < x.Columns; j++)
                {
                    ClusterCenters[cluster, j] = NumOps.Add(ClusterCenters[cluster, j], x[i, j]);
                }
            }
        }

        for (int k = 0; k < NumClusters; k++)
        {
            if (counts[k] > 0)
            {
                T count = NumOps.FromDouble(counts[k]);
                for (int j = 0; j < x.Columns; j++)
                {
                    ClusterCenters[k, j] = NumOps.Divide(ClusterCenters[k, j], count);
                }
            }
        }
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();

        var labels = new Vector<T>(x.Rows);
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        for (int i = 0; i < x.Rows; i++)
        {
            var point = GetRow(x, i);
            double minDist = double.MaxValue;
            int nearest = 0;

            if (ClusterCenters is not null)
            {
                for (int k = 0; k < NumClusters; k++)
                {
                    var center = GetRow(ClusterCenters, k);
                    double dist = NumOps.ToDouble(metric.Compute(point, center));

                    if (dist < minDist)
                    {
                        minDist = dist;
                        nearest = k;
                    }
                }
            }

            labels[i] = NumOps.FromDouble(nearest);
        }

        return labels;
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels!;
    }

    /// <summary>
    /// Gets labels for a given number of clusters from the dendrogram.
    /// </summary>
    /// <param name="numClusters">The desired number of clusters.</param>
    /// <returns>Cluster labels for the specified number of clusters.</returns>
    public Vector<T> GetLabelsForNClusters(int numClusters)
    {
        if (_dendrogram is null || Labels is null)
        {
            throw new InvalidOperationException("Model must be trained first.");
        }

        int n = Labels.Length;
        if (numClusters <= 0 || numClusters > n)
        {
            throw new ArgumentException($"Number of clusters must be between 1 and {n}.");
        }

        // Cut the dendrogram at the appropriate level
        int numMerges = n - numClusters;

        var parent = new int[2 * n - 1];
        for (int i = 0; i < parent.Length; i++)
        {
            parent[i] = i;
        }

        int nextId = n;
        for (int m = 0; m < numMerges && m < _dendrogram.Count; m++)
        {
            var (c1, c2, dist, size) = _dendrogram[m];
            parent[c1] = nextId;
            parent[c2] = nextId;
            nextId++;
        }

        // Find unique clusters
        var rootToCluster = new Dictionary<int, int>();
        var labels = new int[n];

        for (int i = 0; i < n; i++)
        {
            int root = FindRoot(parent, i);
            if (!rootToCluster.ContainsKey(root))
            {
                rootToCluster[root] = rootToCluster.Count;
            }
            labels[i] = rootToCluster[root];
        }

        var result = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            result[i] = NumOps.FromDouble(labels[i]);
        }

        return result;
    }

    private int FindRoot(int[] parent, int i)
    {
        while (parent[i] != i)
        {
            parent[i] = parent[parent[i]]; // Path compression
            i = parent[i];
        }
        return i;
    }
}
