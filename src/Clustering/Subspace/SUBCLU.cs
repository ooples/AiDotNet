using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Clustering.Subspace;

/// <summary>
/// SUBCLU (SUBspace CLUstering) density-connected subspace clustering algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SUBCLU is a subspace clustering algorithm based on the DBSCAN density concept.
/// It exploits the monotonicity property: if a cluster exists in a k-dimensional
/// subspace, it must exist in all (k-1)-dimensional projections of that subspace.
/// </para>
/// <para>
/// Algorithm steps:
/// 1. Apply DBSCAN to each 1-D subspace
/// 2. Generate candidate 2-D subspaces from 1-D clusters
/// 3. Apply DBSCAN to candidate subspaces
/// 4. Use monotonicity to prune: no cluster in 2-D means skip higher dimensions
/// 5. Continue until no more candidates or max dimension reached
/// </para>
/// <para><b>For Beginners:</b> SUBCLU efficiently finds density-based clusters:
///
/// The key insight is "downward closure":
/// - If points are NOT a cluster in 2D, they can't be a cluster in 3D or higher
/// - So we can skip many subspace combinations!
///
/// Example:
/// - Check features 1 and 2: no cluster found
/// - Skip checking features 1, 2, and 3 (waste of time)
/// - Only check subspaces that might have clusters
///
/// This makes SUBCLU much faster than brute-force subspace search.
/// </para>
/// </remarks>
public class SUBCLU<T> : ClusteringBase<T>
{
    private readonly SUBCLUOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private List<SubspaceClusterInfo>? _subspaceClusterInfos;
    private Matrix<T>? _trainingData;

    /// <summary>
    /// Initializes a new SUBCLU instance.
    /// </summary>
    /// <param name="options">The SUBCLU configuration options.</param>
    public SUBCLU(SUBCLUOptions<T>? options = null)
        : base(options ?? new SUBCLUOptions<T>())
    {
        _options = options ?? new SUBCLUOptions<T>();
    }

    /// <summary>
    /// Gets the discovered subspace clusters.
    /// </summary>
    public IReadOnlyList<SubspaceCluster>? SubspaceClusters =>
        _subspaceClusterInfos?.Select(c => new SubspaceCluster
        {
            ClusterId = c.ClusterId,
            Dimensions = c.Dimensions,
            Points = c.Points,
            NumUnits = 1
        }).ToList().AsReadOnly();

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.Clustering;

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new SUBCLU<T>(new SUBCLUOptions<T>
        {
            MaxIterations = _options.MaxIterations,
            Tolerance = _options.Tolerance,
            RandomState = _options.RandomState,
            Epsilon = _options.Epsilon,
            MinPoints = _options.MinPoints,
            MaxSubspaceDimensions = _options.MaxSubspaceDimensions,
            MinClusterSize = _options.MinClusterSize
        });
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (SUBCLU<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidateInputData(x);

        int n = x.Rows;
        int d = x.Columns;
        NumFeatures = d;

        // Store training data for prediction
        _trainingData = new Matrix<T>(n, d);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                _trainingData[i, j] = x[i, j];
            }
        }

        _subspaceClusterInfos = new List<SubspaceClusterInfo>();

        // Find 1-D dense clusters
        var oneDClusters = new Dictionary<int, List<HashSet<int>>>();

        for (int dim = 0; dim < d; dim++)
        {
            var clusters = RunDBSCAN1D(x, dim);
            if (clusters.Count > 0)
            {
                oneDClusters[dim] = clusters;
            }
        }

        // Add 1-D clusters to results
        foreach (var kvp in oneDClusters)
        {
            int dim = kvp.Key;
            foreach (var cluster in kvp.Value)
            {
                _subspaceClusterInfos.Add(new SubspaceClusterInfo
                {
                    ClusterId = _subspaceClusterInfos.Count,
                    Dimensions = new[] { dim },
                    Points = cluster,
                    CorePoints = IdentifyCorePoints(x, cluster, new[] { dim })
                });
            }
        }

        // Generate candidates for higher dimensions
        int maxDim = _options.MaxSubspaceDimensions > 0
            ? Math.Min(_options.MaxSubspaceDimensions, d)
            : d;

        var currentDimClusters = oneDClusters.ToDictionary(
            kvp => new HashSet<int> { kvp.Key },
            kvp => kvp.Value,
            HashSet<int>.CreateSetComparer()
        );

        for (int k = 2; k <= maxDim && currentDimClusters.Count > 0; k++)
        {
            var nextDimClusters = new Dictionary<HashSet<int>, List<HashSet<int>>>(HashSet<int>.CreateSetComparer());

            // Generate candidates by combining k-1 dimensional subspaces
            var subspaces = currentDimClusters.Keys.ToList();

            for (int i = 0; i < subspaces.Count; i++)
            {
                for (int j = i + 1; j < subspaces.Count; j++)
                {
                    var merged = new HashSet<int>(subspaces[i]);
                    merged.UnionWith(subspaces[j]);

                    if (merged.Count == k)
                    {
                        // Check if all (k-1) subsets have clusters (Apriori pruning)
                        bool allSubsetsHaveClusters = true;
                        var dimList = merged.ToList();

                        for (int skip = 0; skip < k && allSubsetsHaveClusters; skip++)
                        {
                            var subset = new HashSet<int>(dimList.Where((_, idx) => idx != skip));
                            if (!currentDimClusters.ContainsKey(subset))
                            {
                                allSubsetsHaveClusters = false;
                            }
                        }

                        if (allSubsetsHaveClusters && !nextDimClusters.ContainsKey(merged))
                        {
                            // Run DBSCAN on this subspace
                            var clusters = RunDBSCAN(x, merged.ToArray());
                            if (clusters.Count > 0)
                            {
                                nextDimClusters[merged] = clusters;

                                // Add to results
                                var dimsArray = merged.OrderBy(dd => dd).ToArray();
                                foreach (var cluster in clusters)
                                {
                                    _subspaceClusterInfos.Add(new SubspaceClusterInfo
                                    {
                                        ClusterId = _subspaceClusterInfos.Count,
                                        Dimensions = dimsArray,
                                        Points = cluster,
                                        CorePoints = IdentifyCorePoints(x, cluster, dimsArray)
                                    });
                                }
                            }
                        }
                    }
                }
            }

            currentDimClusters = nextDimClusters;
        }

        // Assign labels based on highest-dimensional cluster membership
        Labels = AssignLabels(n, _subspaceClusterInfos);

        // Count unique clusters
        var uniqueLabels = new HashSet<int>();
        for (int i = 0; i < n; i++)
        {
            int label = (int)NumOps.ToDouble(Labels[i]);
            if (label >= 0)
            {
                uniqueLabels.Add(label);
            }
        }
        NumClusters = uniqueLabels.Count;

        // Compute cluster centers
        ComputeClusterCenters(x);

        IsTrained = true;
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();
        ValidatePredictInput(x);

        if (_subspaceClusterInfos is null || _subspaceClusterInfos.Count == 0 || _trainingData is null)
        {
            var noise = new Vector<T>(x.Rows);
            for (int i = 0; i < x.Rows; i++)
            {
                noise[i] = NumOps.FromDouble(-1);
            }
            return noise;
        }

        return AssignLabelsForNewData(x);
    }

    private HashSet<int> IdentifyCorePoints(Matrix<T> x, HashSet<int> clusterPoints, int[] dims)
    {
        var corePoints = new HashSet<int>();

        foreach (int pointIdx in clusterPoints)
        {
            int neighborCount = 0;
            foreach (int otherIdx in clusterPoints)
            {
                double dist = ComputeSubspaceDistance(x, pointIdx, otherIdx, dims);
                if (dist <= _options.Epsilon)
                {
                    neighborCount++;
                }
            }

            if (neighborCount >= _options.MinPoints)
            {
                corePoints.Add(pointIdx);
            }
        }

        return corePoints;
    }

    private double ComputeSubspaceDistance(Matrix<T> x, int idx1, int idx2, int[] dims)
    {
        double distSq = 0;
        foreach (int dim in dims)
        {
            double diff = NumOps.ToDouble(x[idx1, dim]) - NumOps.ToDouble(x[idx2, dim]);
            distSq += diff * diff;
        }
        return Math.Sqrt(distSq);
    }

    private double ComputeSubspaceDistanceToPoint(Matrix<T> trainingData, int trainIdx, Matrix<T> newData, int newIdx, int[] dims)
    {
        double distSq = 0;
        foreach (int dim in dims)
        {
            double diff = NumOps.ToDouble(trainingData[trainIdx, dim]) - NumOps.ToDouble(newData[newIdx, dim]);
            distSq += diff * diff;
        }
        return Math.Sqrt(distSq);
    }

    private List<HashSet<int>> RunDBSCAN1D(Matrix<T> x, int dim)
    {
        int n = x.Rows;
        var values = new List<(int Index, double Value)>();

        for (int i = 0; i < n; i++)
        {
            values.Add((i, NumOps.ToDouble(x[i, dim])));
        }

        // Sort by value for efficient neighbor search
        values.Sort((a, b) => a.Value.CompareTo(b.Value));

        var visited = new bool[n];
        var clusters = new List<HashSet<int>>();

        foreach (var (idx, val) in values)
        {
            if (visited[idx]) continue;

            // Find neighbors within epsilon
            var neighbors = new List<int>();
            for (int i = 0; i < n; i++)
            {
                double dist = Math.Abs(NumOps.ToDouble(x[i, dim]) - val);
                if (dist <= _options.Epsilon)
                {
                    neighbors.Add(i);
                }
            }

            if (neighbors.Count >= _options.MinPoints)
            {
                // Expand cluster
                var cluster = new HashSet<int>();
                var queue = new Queue<int>(neighbors);

                foreach (int n2 in neighbors)
                {
                    visited[n2] = true;
                    cluster.Add(n2);
                }

                while (queue.Count > 0)
                {
                    int current = queue.Dequeue();
                    double currentVal = NumOps.ToDouble(x[current, dim]);

                    var currentNeighbors = new List<int>();
                    for (int i = 0; i < n; i++)
                    {
                        double dist = Math.Abs(NumOps.ToDouble(x[i, dim]) - currentVal);
                        if (dist <= _options.Epsilon)
                        {
                            currentNeighbors.Add(i);
                        }
                    }

                    if (currentNeighbors.Count >= _options.MinPoints)
                    {
                        foreach (int neighbor in currentNeighbors)
                        {
                            if (!visited[neighbor])
                            {
                                visited[neighbor] = true;
                                cluster.Add(neighbor);
                                queue.Enqueue(neighbor);
                            }
                            else if (!cluster.Contains(neighbor))
                            {
                                cluster.Add(neighbor);
                            }
                        }
                    }
                }

                if (cluster.Count >= _options.MinClusterSize)
                {
                    clusters.Add(cluster);
                }
            }
        }

        return clusters;
    }

    private List<HashSet<int>> RunDBSCAN(Matrix<T> x, int[] dims)
    {
        int n = x.Rows;
        var visited = new bool[n];
        var clusters = new List<HashSet<int>>();

        for (int i = 0; i < n; i++)
        {
            if (visited[i]) continue;

            var neighbors = GetNeighbors(x, i, dims);

            if (neighbors.Count >= _options.MinPoints)
            {
                var cluster = new HashSet<int>();
                var queue = new Queue<int>(neighbors);

                visited[i] = true;
                cluster.Add(i);
                foreach (int nb in neighbors)
                {
                    if (!visited[nb])
                    {
                        visited[nb] = true;
                        cluster.Add(nb);
                    }
                }

                while (queue.Count > 0)
                {
                    int current = queue.Dequeue();
                    var currentNeighbors = GetNeighbors(x, current, dims);

                    if (currentNeighbors.Count >= _options.MinPoints)
                    {
                        foreach (int nb in currentNeighbors)
                        {
                            if (!visited[nb])
                            {
                                visited[nb] = true;
                                cluster.Add(nb);
                                queue.Enqueue(nb);
                            }
                            else if (!cluster.Contains(nb))
                            {
                                cluster.Add(nb);
                            }
                        }
                    }
                }

                if (cluster.Count >= _options.MinClusterSize)
                {
                    clusters.Add(cluster);
                }
            }
        }

        return clusters;
    }

    private List<int> GetNeighbors(Matrix<T> x, int pointIdx, int[] dims)
    {
        int n = x.Rows;
        var neighbors = new List<int>();

        for (int i = 0; i < n; i++)
        {
            double dist = ComputeSubspaceDistance(x, pointIdx, i, dims);
            if (dist <= _options.Epsilon)
            {
                neighbors.Add(i);
            }
        }

        return neighbors;
    }

    private Vector<T> AssignLabels(int n, List<SubspaceClusterInfo> clusters)
    {
        var labels = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            labels[i] = NumOps.FromDouble(-1);
        }

        if (clusters.Count == 0)
        {
            return labels;
        }

        // Sort clusters by dimensionality (prefer higher dimensional)
        var sortedClusters = clusters.OrderByDescending(c => c.Dimensions.Length).ToList();

        for (int i = 0; i < n; i++)
        {
            foreach (var cluster in sortedClusters)
            {
                if (cluster.Points.Contains(i))
                {
                    labels[i] = NumOps.FromDouble(cluster.ClusterId);
                    break;
                }
            }
        }

        return labels;
    }

    private Vector<T> AssignLabelsForNewData(Matrix<T> x)
    {
        int n = x.Rows;
        var labels = new Vector<T>(n);

        // Initialize all as noise
        for (int i = 0; i < n; i++)
        {
            labels[i] = NumOps.FromDouble(-1);
        }

        // Sort clusters by dimensionality (prefer higher dimensional matches)
        var sortedClusters = _subspaceClusterInfos!
            .OrderByDescending(c => c.Dimensions.Length)
            .ToList();

        for (int i = 0; i < n; i++)
        {
            // For each point, check if it's density-reachable from any cluster
            // A new point is assigned to a cluster if it's within epsilon of any core point
            // in that cluster's subspace
            foreach (var cluster in sortedClusters)
            {
                bool isReachable = false;

                // Check distance to core points in this cluster's subspace
                foreach (int corePointIdx in cluster.CorePoints)
                {
                    double dist = ComputeSubspaceDistanceToPoint(
                        _trainingData!, corePointIdx, x, i, cluster.Dimensions);

                    if (dist <= _options.Epsilon)
                    {
                        isReachable = true;
                        break;
                    }
                }

                if (isReachable)
                {
                    labels[i] = NumOps.FromDouble(cluster.ClusterId);
                    break; // Assigned to highest-dimensional reachable cluster
                }
            }
        }

        return labels;
    }

    private void ComputeClusterCenters(Matrix<T> x)
    {
        if (NumClusters == 0)
        {
            ClusterCenters = null;
            return;
        }

        int d = x.Columns;
        int n = x.Rows;
        ClusterCenters = new Matrix<T>(NumClusters, d);
        var counts = new int[NumClusters];

        for (int i = 0; i < n; i++)
        {
            int label = (int)NumOps.ToDouble(Labels![i]);
            if (label >= 0 && label < NumClusters)
            {
                counts[label]++;
                for (int j = 0; j < d; j++)
                {
                    ClusterCenters[label, j] = NumOps.Add(ClusterCenters[label, j], x[i, j]);
                }
            }
        }

        for (int k = 0; k < NumClusters; k++)
        {
            if (counts[k] > 0)
            {
                for (int j = 0; j < d; j++)
                {
                    ClusterCenters[k, j] = NumOps.Divide(ClusterCenters[k, j], NumOps.FromDouble(counts[k]));
                }
            }
        }
    }

    private void ValidateInputData(Matrix<T> x)
    {
        if (x.Rows == 0 || x.Columns == 0)
        {
            throw new ArgumentException("Input data cannot be empty.");
        }
    }

    private void ValidatePredictInput(Matrix<T> x)
    {
        if (x.Columns != NumFeatures)
        {
            throw new ArgumentException($"Expected {NumFeatures} features, got {x.Columns}.");
        }
    }

    /// <summary>
    /// Internal cluster info including core points for prediction.
    /// </summary>
    private class SubspaceClusterInfo
    {
        public int ClusterId { get; set; }
        public int[] Dimensions { get; set; } = Array.Empty<int>();
        public HashSet<int> Points { get; set; } = new HashSet<int>();
        public HashSet<int> CorePoints { get; set; } = new HashSet<int>();
    }
}
