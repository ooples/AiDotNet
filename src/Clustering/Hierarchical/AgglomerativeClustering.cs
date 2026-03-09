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

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private List<(int Cluster1, int Cluster2, T Distance, int Size)>? _dendrogram;

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
    public List<(int Cluster1, int Cluster2, T Distance, int Size)>? Dendrogram => _dendrogram;

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
        var centroids = new T[2 * n - 1, x.Columns];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                centroids[i, j] = x[i, j];
            }
        }

        // Dendrogram storage
        _dendrogram = new List<(int, int, T, int)>();

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
            if (_options.DistanceThreshold.HasValue &&
                NumOps.GreaterThan(minDist, NumOps.FromDouble(_options.DistanceThreshold.Value)))
            {
                break;
            }

            // Record merge
            int newSize = clusterSizes[minI] + clusterSizes[minJ];
            _dendrogram.Add((minI, minJ, minDist, newSize));

            // Update centroid
            T sizeIT = NumOps.FromDouble(clusterSizes[minI]);
            T sizeJT = NumOps.FromDouble(clusterSizes[minJ]);
            T newSizeT = NumOps.FromDouble(newSize);
            for (int d = 0; d < x.Columns; d++)
            {
                centroids[nextClusterId, d] = NumOps.Divide(
                    NumOps.Add(
                        NumOps.Multiply(centroids[minI, d], sizeIT),
                        NumOps.Multiply(centroids[minJ, d], sizeJT)),
                    newSizeT);
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

    private T[,] ComputeDistanceMatrix(Matrix<T> x)
    {
        int n = x.Rows;
        var distMatrix = new T[2 * n - 1, 2 * n - 1];
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        for (int i = 0; i < n; i++)
        {
            var rowI = GetRow(x, i);
            distMatrix[i, i] = NumOps.Zero;

            for (int j = i + 1; j < n; j++)
            {
                var rowJ = GetRow(x, j);
                T dist = metric.Compute(rowI, rowJ);

                // For Ward's method, use squared distance
                if (_options.Linkage == LinkageMethod.Ward)
                {
                    dist = NumOps.Multiply(dist, dist);
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
                distMatrix[i, j] = NumOps.MaxValue;
                distMatrix[j, i] = NumOps.MaxValue;
            }
        }

        return distMatrix;
    }

    private (int I, int J, T Dist) FindClosestPair(
        T[,] distMatrix,
        HashSet<int> active,
        int[] sizes,
        T[,] centroids,
        int dims)
    {
        int minI = -1, minJ = -1;
        T minDist = NumOps.MaxValue;

        var activeList = active.ToList();

        for (int ai = 0; ai < activeList.Count; ai++)
        {
            int i = activeList[ai];
            for (int aj = ai + 1; aj < activeList.Count; aj++)
            {
                int j = activeList[aj];

                T dist = distMatrix[i, j];

                if (NumOps.LessThan(dist, minDist))
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
        T[,] distMatrix,
        HashSet<int> active,
        int mergedI,
        int mergedJ,
        int newCluster,
        int[] sizes)
    {
        T sizeI = NumOps.FromDouble(sizes[mergedI]);
        T sizeJ = NumOps.FromDouble(sizes[mergedJ]);

        foreach (int k in active)
        {
            if (k == mergedI || k == mergedJ)
            {
                continue;
            }

            T dIK = distMatrix[mergedI, k];
            T dJK = distMatrix[mergedJ, k];
            T dIJ = distMatrix[mergedI, mergedJ];
            T sizeK = NumOps.FromDouble(sizes[k]);

            T newDist = ComputeLinkageDistance(dIK, dJK, dIJ, sizeI, sizeJ, sizeK);

            distMatrix[newCluster, k] = newDist;
            distMatrix[k, newCluster] = newDist;
        }
    }

    private T ComputeLinkageDistance(
        T dIK, T dJK, T dIJ,
        T sizeI, T sizeJ, T sizeK)
    {
        T two = NumOps.FromDouble(2.0);
        T four = NumOps.FromDouble(4.0);
        T sizeIJ = NumOps.Add(sizeI, sizeJ);

        return _options.Linkage switch
        {
            LinkageMethod.Single =>
                NumOps.LessThan(dIK, dJK) ? dIK : dJK,

            LinkageMethod.Complete =>
                NumOps.GreaterThan(dIK, dJK) ? dIK : dJK,

            LinkageMethod.Average =>
                NumOps.Divide(NumOps.Add(NumOps.Multiply(sizeI, dIK), NumOps.Multiply(sizeJ, dJK)), sizeIJ),

            LinkageMethod.Weighted =>
                NumOps.Divide(NumOps.Add(dIK, dJK), two),

            LinkageMethod.Ward =>
                NumOps.Divide(
                    NumOps.Subtract(
                        NumOps.Add(
                            NumOps.Multiply(NumOps.Add(sizeI, sizeK), dIK),
                            NumOps.Multiply(NumOps.Add(sizeJ, sizeK), dJK)),
                        NumOps.Multiply(sizeK, dIJ)),
                    NumOps.Add(sizeIJ, sizeK)),

            LinkageMethod.Centroid =>
                NumOps.Subtract(
                    NumOps.Divide(NumOps.Add(NumOps.Multiply(sizeI, dIK), NumOps.Multiply(sizeJ, dJK)), sizeIJ),
                    NumOps.Divide(NumOps.Multiply(NumOps.Multiply(sizeI, sizeJ), dIJ), NumOps.Multiply(sizeIJ, sizeIJ))),

            LinkageMethod.Median =>
                NumOps.Subtract(NumOps.Divide(NumOps.Add(dIK, dJK), two), NumOps.Divide(dIJ, four)),

            _ => NumOps.LessThan(dIK, dJK) ? dIK : dJK
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
            T minDist = NumOps.MaxValue;
            int nearest = 0;

            if (ClusterCenters is not null)
            {
                for (int k = 0; k < NumClusters; k++)
                {
                    var center = GetRow(ClusterCenters, k);
                    T dist = metric.Compute(point, center);

                    if (NumOps.LessThan(dist, minDist))
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
        return Labels ?? new Vector<T>(0);
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
