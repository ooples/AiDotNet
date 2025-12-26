using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Clustering.Hierarchical;

/// <summary>
/// Bisecting K-Means clustering algorithm implementation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Bisecting K-Means is a divisive hierarchical clustering algorithm that combines
/// the efficiency of K-Means with a top-down hierarchical approach. It starts with
/// all data in one cluster and recursively bisects until k clusters are achieved.
/// </para>
/// <para>
/// Algorithm steps:
/// 1. Start with all points in a single cluster
/// 2. Select a cluster to split (usually the largest or highest inertia)
/// 3. Bisect the selected cluster using K-Means with k=2
/// 4. Replace the original cluster with the two new clusters
/// 5. Repeat steps 2-4 until k clusters are formed
/// </para>
/// <para><b>For Beginners:</b> Bisecting K-Means is like chopping a log:
/// - Start with one big piece (all your data)
/// - Find the biggest piece and split it in half
/// - Keep splitting the biggest pieces
/// - Stop when you have k pieces
///
/// Why use this instead of regular K-Means?
/// - More consistent results (less affected by random starting points)
/// - Often finds better clusters for complex data
/// - Naturally creates a hierarchy (like a family tree of clusters)
/// - Each split is a small K-Means problem, so it's efficient
/// </para>
/// </remarks>
public class BisectingKMeans<T> : ClusteringBase<T>
{
    private readonly BisectingKMeansOptions<T> _options;
    private readonly IDistanceMetric<T> _distanceMetric;
    private Random _random;
    private List<BisectionNode>? _hierarchy;

    /// <summary>
    /// Initializes a new BisectingKMeans instance with the specified options.
    /// </summary>
    /// <param name="options">The Bisecting K-Means configuration options.</param>
    public BisectingKMeans(BisectingKMeansOptions<T>? options = null)
        : base(options ?? new BisectingKMeansOptions<T>())
    {
        _options = options ?? new BisectingKMeansOptions<T>();
        _distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        _random = _options.RandomState.HasValue
            ? RandomHelper.CreateSeededRandom(_options.RandomState.Value)
            : RandomHelper.CreateSeededRandom(42);
        NumClusters = _options.NumClusters;
    }

    /// <summary>
    /// Gets the bisection hierarchy if BuildHierarchy was enabled.
    /// </summary>
    public IReadOnlyList<BisectionNode>? Hierarchy => _hierarchy?.AsReadOnly();

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.Clustering;

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new BisectingKMeans<T>(new BisectingKMeansOptions<T>
        {
            NumClusters = _options.NumClusters,
            MaxIterations = _options.MaxIterations,
            Tolerance = _options.Tolerance,
            RandomState = _options.RandomState,
            NumBisectionTrials = _options.NumBisectionTrials,
            ClusterSelection = _options.ClusterSelection,
            MinClusterSizeForBisection = _options.MinClusterSizeForBisection,
            BuildHierarchy = _options.BuildHierarchy,
            DistanceMetric = _options.DistanceMetric
        });
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (BisectingKMeans<T>)CreateNewInstance();
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

        if (_options.NumClusters <= 0)
        {
            throw new ArgumentException("Number of clusters must be positive.");
        }

        if (_options.NumClusters >= n)
        {
            // Each point is its own cluster
            Labels = new Vector<T>(n);
            ClusterCenters = new Matrix<T>(n, d);
            for (int i = 0; i < n; i++)
            {
                Labels[i] = NumOps.FromDouble(i);
                for (int j = 0; j < d; j++)
                {
                    ClusterCenters[i, j] = x[i, j];
                }
            }
            Inertia = NumOps.Zero;
            IsTrained = true;
            return;
        }

        // Initialize with all points in cluster 0
        var clusterAssignments = new int[n];
        var clusterInfos = new List<ClusterInfo>
        {
            CreateClusterInfo(x, clusterAssignments, 0, Enumerable.Range(0, n).ToList())
        };

        if (_options.BuildHierarchy)
        {
            var firstClusterInertia = clusterInfos[0].Inertia;
            double firstInertiaValue = firstClusterInertia is not null ? NumOps.ToDouble(firstClusterInertia) : 0;
            _hierarchy = new List<BisectionNode>
            {
                new BisectionNode
                {
                    ClusterId = 0,
                    ParentId = -1,
                    Size = n,
                    Inertia = firstInertiaValue
                }
            };
        }

        int currentNumClusters = 1;
        int nextClusterId = 1;

        // Bisect until we have the desired number of clusters
        while (currentNumClusters < _options.NumClusters)
        {
            // Select cluster to bisect
            int clusterToSplit = SelectClusterToBisect(clusterInfos);

            if (clusterToSplit < 0 || clusterInfos[clusterToSplit].Indices.Count < _options.MinClusterSizeForBisection)
            {
                // No cluster can be split further
                break;
            }

            // Get the points in this cluster
            var indices = clusterInfos[clusterToSplit].Indices;
            var subData = ExtractSubMatrix(x, indices);

            // Bisect the cluster
            var (leftIndices, rightIndices, leftCenter, rightCenter) = BisectCluster(subData);

            if (leftIndices.Count == 0 || rightIndices.Count == 0)
            {
                // Bisection failed, mark cluster as unsplittable
                clusterInfos[clusterToSplit].CanSplit = false;
                continue;
            }

            // Map back to original indices
            var originalLeftIndices = leftIndices.Select(i => indices[i]).ToList();
            var originalRightIndices = rightIndices.Select(i => indices[i]).ToList();

            // Update cluster assignments
            int leftClusterId = clusterInfos[clusterToSplit].ClusterId;
            int rightClusterId = nextClusterId++;

            foreach (int idx in originalLeftIndices)
            {
                clusterAssignments[idx] = leftClusterId;
            }
            foreach (int idx in originalRightIndices)
            {
                clusterAssignments[idx] = rightClusterId;
            }

            // Update cluster info
            clusterInfos[clusterToSplit] = CreateClusterInfo(x, clusterAssignments, leftClusterId, originalLeftIndices);
            clusterInfos[clusterToSplit].Center = leftCenter;

            clusterInfos.Add(CreateClusterInfo(x, clusterAssignments, rightClusterId, originalRightIndices));
            clusterInfos[clusterInfos.Count - 1].Center = rightCenter;

            // Record hierarchy
            if (_options.BuildHierarchy && _hierarchy is not null)
            {
                var lastClusterInertia = clusterInfos[clusterInfos.Count - 1].Inertia;
                double lastInertiaValue = lastClusterInertia is not null ? NumOps.ToDouble(lastClusterInertia) : 0;
                _hierarchy.Add(new BisectionNode
                {
                    ClusterId = rightClusterId,
                    ParentId = leftClusterId,
                    Size = originalRightIndices.Count,
                    Inertia = lastInertiaValue
                });
            }

            currentNumClusters++;
        }

        // Renumber clusters to be 0, 1, 2, ... k-1
        var clusterMapping = new Dictionary<int, int>();
        int newClusterId = 0;
        for (int i = 0; i < n; i++)
        {
            if (!clusterMapping.ContainsKey(clusterAssignments[i]))
            {
                clusterMapping[clusterAssignments[i]] = newClusterId++;
            }
        }

        Labels = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            Labels[i] = NumOps.FromDouble(clusterMapping[clusterAssignments[i]]);
        }

        // Build final cluster centers
        NumClusters = clusterMapping.Count;
        ClusterCenters = new Matrix<T>(NumClusters, d);
        var centerCounts = new int[NumClusters];

        for (int i = 0; i < n; i++)
        {
            int cluster = clusterMapping[clusterAssignments[i]];
            centerCounts[cluster]++;
            for (int j = 0; j < d; j++)
            {
                ClusterCenters[cluster, j] = NumOps.Add(ClusterCenters[cluster, j], x[i, j]);
            }
        }

        for (int k = 0; k < NumClusters; k++)
        {
            if (centerCounts[k] > 0)
            {
                for (int j = 0; j < d; j++)
                {
                    ClusterCenters[k, j] = NumOps.Divide(ClusterCenters[k, j], NumOps.FromDouble(centerCounts[k]));
                }
            }
        }

        // Compute final inertia
        Inertia = ComputeInertia(x, Labels, ClusterCenters);
        IsTrained = true;
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();
        ValidatePredictInput(x);

        var labels = new Vector<T>(x.Rows);

        for (int i = 0; i < x.Rows; i++)
        {
            double minDist = double.MaxValue;
            int nearestCluster = 0;

            for (int k = 0; k < NumClusters; k++)
            {
                var point = GetRow(x, i);
                var center = GetRow(ClusterCenters!, k);
                double dist = NumOps.ToDouble(_distanceMetric.Compute(point, center));

                if (dist < minDist)
                {
                    minDist = dist;
                    nearestCluster = k;
                }
            }

            labels[i] = NumOps.FromDouble(nearestCluster);
        }

        return labels;
    }

    private int SelectClusterToBisect(List<ClusterInfo> clusterInfos)
    {
        int bestIdx = -1;
        double bestScore = double.MinValue;

        for (int i = 0; i < clusterInfos.Count; i++)
        {
            var info = clusterInfos[i];
            if (!info.CanSplit || info.Indices.Count < _options.MinClusterSizeForBisection)
            {
                continue;
            }

            var inertiaForScore = info.Inertia;
            double inertiaValue = inertiaForScore is not null ? NumOps.ToDouble(inertiaForScore) : 0;
            double score = _options.ClusterSelection switch
            {
                BisectionClusterSelection.Largest => info.Indices.Count,
                BisectionClusterSelection.HighestInertia => inertiaValue,
                BisectionClusterSelection.LargestDiameter => info.Diameter,
                _ => info.Indices.Count
            };

            if (score > bestScore)
            {
                bestScore = score;
                bestIdx = i;
            }
        }

        return bestIdx;
    }

    private (List<int> LeftIndices, List<int> RightIndices, Vector<T> LeftCenter, Vector<T> RightCenter)
        BisectCluster(Matrix<T> subData)
    {
        if (subData.Rows < 2)
        {
            return (new List<int>(), new List<int>(), new Vector<T>(subData.Columns), new Vector<T>(subData.Columns));
        }

        int n = subData.Rows;
        int d = subData.Columns;

        List<int> bestLeftIndices = new List<int>();
        List<int> bestRightIndices = new List<int>();
        Vector<T> bestLeftCenter = new Vector<T>(d);
        Vector<T> bestRightCenter = new Vector<T>(d);
        double bestInertia = double.MaxValue;

        // Try multiple bisection attempts
        for (int trial = 0; trial < _options.NumBisectionTrials; trial++)
        {
            // Initialize two centers randomly
            var center1 = new Vector<T>(d);
            var center2 = new Vector<T>(d);

            int idx1 = _random.Next(n);
            int idx2;
            do
            {
                idx2 = _random.Next(n);
            } while (idx2 == idx1 && n > 1);

            for (int j = 0; j < d; j++)
            {
                center1[j] = subData[idx1, j];
                center2[j] = subData[idx2, j];
            }

            // Run K-Means with k=2
            var (leftIndices, rightIndices, finalCenter1, finalCenter2, inertia) =
                RunBinaryKMeans(subData, center1, center2);

            if (inertia < bestInertia && leftIndices.Count > 0 && rightIndices.Count > 0)
            {
                bestInertia = inertia;
                bestLeftIndices = leftIndices;
                bestRightIndices = rightIndices;
                bestLeftCenter = finalCenter1;
                bestRightCenter = finalCenter2;
            }
        }

        return (bestLeftIndices, bestRightIndices, bestLeftCenter, bestRightCenter);
    }

    private (List<int> LeftIndices, List<int> RightIndices, Vector<T> Center1, Vector<T> Center2, double Inertia)
        RunBinaryKMeans(Matrix<T> data, Vector<T> center1, Vector<T> center2)
    {
        int n = data.Rows;
        int d = data.Columns;

        var assignments = new int[n];
        var prevAssignments = new int[n];
        for (int i = 0; i < n; i++)
        {
            prevAssignments[i] = -1;
        }

        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            // Assignment step
            bool changed = false;
            for (int i = 0; i < n; i++)
            {
                var point = GetRow(data, i);
                double dist1 = NumOps.ToDouble(_distanceMetric.Compute(point, center1));
                double dist2 = NumOps.ToDouble(_distanceMetric.Compute(point, center2));

                assignments[i] = dist1 <= dist2 ? 0 : 1;

                if (assignments[i] != prevAssignments[i])
                {
                    changed = true;
                }
            }

            if (!changed)
            {
                break;
            }

            Array.Copy(assignments, prevAssignments, n);

            // Update step
            var newCenter1 = new Vector<T>(d);
            var newCenter2 = new Vector<T>(d);
            int count1 = 0, count2 = 0;

            for (int i = 0; i < n; i++)
            {
                if (assignments[i] == 0)
                {
                    count1++;
                    for (int j = 0; j < d; j++)
                    {
                        newCenter1[j] = NumOps.Add(newCenter1[j], data[i, j]);
                    }
                }
                else
                {
                    count2++;
                    for (int j = 0; j < d; j++)
                    {
                        newCenter2[j] = NumOps.Add(newCenter2[j], data[i, j]);
                    }
                }
            }

            if (count1 > 0)
            {
                for (int j = 0; j < d; j++)
                {
                    center1[j] = NumOps.Divide(newCenter1[j], NumOps.FromDouble(count1));
                }
            }

            if (count2 > 0)
            {
                for (int j = 0; j < d; j++)
                {
                    center2[j] = NumOps.Divide(newCenter2[j], NumOps.FromDouble(count2));
                }
            }
        }

        // Build result lists
        var leftIndices = new List<int>();
        var rightIndices = new List<int>();
        double totalInertia = 0;

        for (int i = 0; i < n; i++)
        {
            var point = GetRow(data, i);
            if (assignments[i] == 0)
            {
                leftIndices.Add(i);
                double dist = NumOps.ToDouble(_distanceMetric.Compute(point, center1));
                totalInertia += dist * dist;
            }
            else
            {
                rightIndices.Add(i);
                double dist = NumOps.ToDouble(_distanceMetric.Compute(point, center2));
                totalInertia += dist * dist;
            }
        }

        return (leftIndices, rightIndices, center1, center2, totalInertia);
    }

    private ClusterInfo CreateClusterInfo(Matrix<T> x, int[] assignments, int clusterId, List<int> indices)
    {
        int d = x.Columns;
        var center = new Vector<T>(d);

        // Compute center
        foreach (int idx in indices)
        {
            for (int j = 0; j < d; j++)
            {
                center[j] = NumOps.Add(center[j], x[idx, j]);
            }
        }

        if (indices.Count > 0)
        {
            for (int j = 0; j < d; j++)
            {
                center[j] = NumOps.Divide(center[j], NumOps.FromDouble(indices.Count));
            }
        }

        // Compute inertia and diameter
        T inertia = NumOps.Zero;
        double diameter = 0;

        foreach (int idx in indices)
        {
            var point = GetRow(x, idx);
            double dist = NumOps.ToDouble(_distanceMetric.Compute(point, center));
            inertia = NumOps.Add(inertia, NumOps.FromDouble(dist * dist));
        }

        // Approximate diameter using center + max distance from center * 2
        if (indices.Count > 1)
        {
            double maxDist = 0;
            foreach (int idx in indices)
            {
                var point = GetRow(x, idx);
                double dist = NumOps.ToDouble(_distanceMetric.Compute(point, center));
                maxDist = Math.Max(maxDist, dist);
            }
            diameter = maxDist * 2;
        }

        return new ClusterInfo
        {
            ClusterId = clusterId,
            Indices = indices,
            Center = center,
            Inertia = inertia,
            Diameter = diameter,
            CanSplit = indices.Count >= _options.MinClusterSizeForBisection
        };
    }

    private Matrix<T> ExtractSubMatrix(Matrix<T> x, List<int> indices)
    {
        var subMatrix = new Matrix<T>(indices.Count, x.Columns);
        for (int i = 0; i < indices.Count; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                subMatrix[i, j] = x[indices[i], j];
            }
        }
        return subMatrix;
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

    private class ClusterInfo
    {
        public int ClusterId { get; set; }
        public List<int> Indices { get; set; } = new List<int>();
        public Vector<T>? Center { get; set; }
        public T? Inertia { get; set; }
        public double Diameter { get; set; }
        public bool CanSplit { get; set; } = true;
    }
}

/// <summary>
/// Represents a node in the bisection hierarchy tree.
/// </summary>
public class BisectionNode
{
    /// <summary>
    /// The cluster ID of this node.
    /// </summary>
    public int ClusterId { get; set; }

    /// <summary>
    /// The parent cluster ID (-1 for root).
    /// </summary>
    public int ParentId { get; set; }

    /// <summary>
    /// Number of points in this cluster.
    /// </summary>
    public int Size { get; set; }

    /// <summary>
    /// Inertia (sum of squared distances to center) of this cluster.
    /// </summary>
    public double Inertia { get; set; }
}
