using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Clustering.Density;

/// <summary>
/// HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// HDBSCAN extends DBSCAN by finding clusters at all density levels and selecting
/// the most stable clusters. It uses mutual reachability distance and minimum
/// spanning tree construction.
/// </para>
/// <para>
/// Algorithm steps:
/// 1. Compute core distances (distance to k-th nearest neighbor)
/// 2. Build mutual reachability graph
/// 3. Construct minimum spanning tree
/// 4. Build cluster hierarchy (condensed tree)
/// 5. Extract clusters using stability
/// </para>
/// <para><b>For Beginners:</b> HDBSCAN automatically finds the right density level.
///
/// The key insight:
/// - Instead of picking ONE epsilon (like DBSCAN)
/// - Look at how clusters form at ALL epsilon values
/// - Pick clusters that "live longest" in the hierarchy
///
/// Core distance: How far to reach k nearest neighbors
/// Mutual reachability: Max of core distances and actual distance
/// This makes sparse points "push away" from dense clusters.
///
/// Benefits:
/// - No epsilon to tune
/// - Finds varying-density clusters
/// - Robust noise detection
/// - Provides cluster hierarchy
/// </para>
/// </remarks>
public class HDBSCAN<T> : ClusteringBase<T>
{
    private readonly HDBSCANOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private double[]? _outlierScores;
    private double[]? _probabilities;
    private List<CondensedTreeNode>? _condensedTree;

    /// <summary>
    /// Initializes a new HDBSCAN instance.
    /// </summary>
    /// <param name="options">The HDBSCAN options.</param>
    public HDBSCAN(HDBSCANOptions<T>? options = null)
        : base(options ?? new HDBSCANOptions<T>())
    {
        _options = options ?? new HDBSCANOptions<T>();
    }

    /// <summary>
    /// Gets the outlier scores for each point.
    /// </summary>
    public double[]? OutlierScores => _outlierScores;

    /// <summary>
    /// Gets the cluster membership probabilities.
    /// </summary>
    public double[]? Probabilities => _probabilities;

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.Clustering;

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new HDBSCAN<T>(new HDBSCANOptions<T>
        {
            MinClusterSize = _options.MinClusterSize,
            MinSamples = _options.MinSamples,
            ClusterSelection = _options.ClusterSelection,
            AllowSingleCluster = _options.AllowSingleCluster,
            ClusterSelectionEpsilon = _options.ClusterSelectionEpsilon,
            Alpha = _options.Alpha,
            DistanceMetric = _options.DistanceMetric
        });
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (HDBSCAN<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int d = x.Columns;
        NumFeatures = d;

        int minSamples = _options.MinSamples ?? _options.MinClusterSize;
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        // Step 1: Compute distance matrix
        var distMatrix = ComputeDistanceMatrix(x, n, metric);

        // Step 2: Compute core distances
        var coreDistances = ComputeCoreDistances(distMatrix, n, minSamples);

        // Step 3: Compute mutual reachability distances
        var mrdMatrix = ComputeMutualReachabilityDistances(distMatrix, coreDistances, n);

        // Step 4: Build minimum spanning tree
        var mst = BuildMinimumSpanningTree(mrdMatrix, n);

        // Step 5: Build condensed tree
        _condensedTree = BuildCondensedTree(mst, n, _options.MinClusterSize);

        // Step 6: Extract clusters
        var clusterLabels = ExtractClusters(_condensedTree, n, _options.ClusterSelection);

        // Compute probabilities and outlier scores
        ComputeProbabilitiesAndOutlierScores(clusterLabels, _condensedTree, n);

        // Set labels
        Labels = new Vector<T>(n);
        int maxLabel = -1;
        for (int i = 0; i < n; i++)
        {
            Labels[i] = NumOps.FromDouble(clusterLabels[i]);
            if (clusterLabels[i] > maxLabel) maxLabel = clusterLabels[i];
        }

        NumClusters = maxLabel + 1;

        // Compute cluster centers (mean of each cluster)
        if (NumClusters > 0)
        {
            ClusterCenters = new Matrix<T>(NumClusters, d);
            var clusterCounts = new int[NumClusters];
            var clusterSums = new double[NumClusters, d];

            for (int i = 0; i < n; i++)
            {
                int label = clusterLabels[i];
                if (label >= 0)
                {
                    clusterCounts[label]++;
                    for (int j = 0; j < d; j++)
                    {
                        clusterSums[label, j] += NumOps.ToDouble(x[i, j]);
                    }
                }
            }

            for (int k = 0; k < NumClusters; k++)
            {
                if (clusterCounts[k] > 0)
                {
                    for (int j = 0; j < d; j++)
                    {
                        ClusterCenters[k, j] = NumOps.FromDouble(clusterSums[k, j] / clusterCounts[k]);
                    }
                }
            }
        }
        else
        {
            ClusterCenters = new Matrix<T>(0, d);
        }

        IsTrained = true;
    }

    private double[,] ComputeDistanceMatrix(Matrix<T> x, int n, IDistanceMetric<T> metric)
    {
        var distMatrix = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            var pointI = GetRow(x, i);
            for (int j = i + 1; j < n; j++)
            {
                var pointJ = GetRow(x, j);
                double dist = NumOps.ToDouble(metric.Compute(pointI, pointJ));
                distMatrix[i, j] = dist;
                distMatrix[j, i] = dist;
            }
        }

        return distMatrix;
    }

    private double[] ComputeCoreDistances(double[,] distMatrix, int n, int minSamples)
    {
        var coreDistances = new double[n];
        int k = Math.Min(minSamples, n - 1);

        for (int i = 0; i < n; i++)
        {
            // Get distances to all other points
            var distances = new List<double>(n);
            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    distances.Add(distMatrix[i, j]);
                }
            }

            distances.Sort();

            // Core distance is distance to k-th nearest neighbor
            coreDistances[i] = k > 0 && k <= distances.Count ? distances[k - 1] : 0;
        }

        return coreDistances;
    }

    private double[,] ComputeMutualReachabilityDistances(double[,] distMatrix, double[] coreDistances, int n)
    {
        var mrdMatrix = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                // Mutual reachability distance = max(core_dist[i], core_dist[j], dist[i,j])
                double mrd = Math.Max(coreDistances[i], Math.Max(coreDistances[j], distMatrix[i, j]));
                mrdMatrix[i, j] = mrd;
                mrdMatrix[j, i] = mrd;
            }
        }

        return mrdMatrix;
    }

    private List<MSTEdge> BuildMinimumSpanningTree(double[,] mrdMatrix, int n)
    {
        // Prim's algorithm for MST
        var mst = new List<MSTEdge>();
        var inTree = new bool[n];
        var minDist = new double[n];
        var minFrom = new int[n];

        for (int i = 0; i < n; i++)
        {
            minDist[i] = double.MaxValue;
            minFrom[i] = -1;
        }

        // Start from node 0
        minDist[0] = 0;

        for (int count = 0; count < n; count++)
        {
            // Find minimum distance node not in tree
            int u = -1;
            double minVal = double.MaxValue;
            for (int i = 0; i < n; i++)
            {
                if (!inTree[i] && minDist[i] < minVal)
                {
                    minVal = minDist[i];
                    u = i;
                }
            }

            if (u == -1) break;

            inTree[u] = true;

            if (minFrom[u] >= 0)
            {
                mst.Add(new MSTEdge(minFrom[u], u, minDist[u]));
            }

            // Update distances
            for (int v = 0; v < n; v++)
            {
                if (!inTree[v] && mrdMatrix[u, v] < minDist[v])
                {
                    minDist[v] = mrdMatrix[u, v];
                    minFrom[v] = u;
                }
            }
        }

        // Sort edges by weight
        mst.Sort((a, b) => a.Weight.CompareTo(b.Weight));

        return mst;
    }

    private List<CondensedTreeNode> BuildCondensedTree(List<MSTEdge> mst, int n, int minClusterSize)
    {
        var condensedTree = new List<CondensedTreeNode>();

        // Use Union-Find to track cluster membership
        var parent = new int[n];
        var size = new int[n];
        var lambda = new double[n]; // Birth lambda for each point

        for (int i = 0; i < n; i++)
        {
            parent[i] = i;
            size[i] = 1;
            lambda[i] = 0;
        }

        int nextCluster = n;

        // Process MST edges in order (smallest to largest weight)
        foreach (var edge in mst)
        {
            int root1 = Find(parent, edge.U);
            int root2 = Find(parent, edge.V);

            if (root1 == root2) continue;

            double edgeLambda = edge.Weight > 0 ? 1.0 / edge.Weight : double.MaxValue;

            int size1 = size[root1];
            int size2 = size[root2];

            bool split1 = size1 < minClusterSize;
            bool split2 = size2 < minClusterSize;

            if (split1 && split2)
            {
                // Both too small, just merge
                Union(parent, size, root1, root2);
            }
            else if (split1 || split2)
            {
                // One is too small - it falls out as noise
                int smallRoot = split1 ? root1 : root2;
                int bigRoot = split1 ? root2 : root1;

                // Add noise points to tree
                for (int i = 0; i < n; i++)
                {
                    if (Find(parent, i) == smallRoot)
                    {
                        condensedTree.Add(new CondensedTreeNode(
                            bigRoot >= n ? bigRoot : nextCluster,
                            i,
                            edgeLambda,
                            1));
                    }
                }

                Union(parent, size, root1, root2);
            }
            else
            {
                // Both are large enough - create new parent cluster
                int newCluster = nextCluster++;

                // Add children to condensed tree
                condensedTree.Add(new CondensedTreeNode(newCluster, root1, edgeLambda, size1));
                condensedTree.Add(new CondensedTreeNode(newCluster, root2, edgeLambda, size2));

                // Union under new cluster ID
                parent[root1] = newCluster;
                parent[root2] = newCluster;
                if (nextCluster > parent.Length)
                {
                    // Extend parent array
                    var newParent = new int[nextCluster + 1];
                    var newSize = new int[nextCluster + 1];
                    Array.Copy(parent, newParent, parent.Length);
                    Array.Copy(size, newSize, size.Length);
                    parent = newParent;
                    size = newSize;
                }
                parent[newCluster] = newCluster;
                size[newCluster] = size1 + size2;
            }
        }

        return condensedTree;
    }

    private int Find(int[] parent, int i)
    {
        if (i >= parent.Length) return i;
        if (parent[i] != i)
        {
            parent[i] = Find(parent, parent[i]);
        }
        return parent[i];
    }

    private void Union(int[] parent, int[] size, int i, int j)
    {
        int rootI = Find(parent, i);
        int rootJ = Find(parent, j);

        if (rootI != rootJ)
        {
            if (size[rootI] < size[rootJ])
            {
                parent[rootI] = rootJ;
                size[rootJ] += size[rootI];
            }
            else
            {
                parent[rootJ] = rootI;
                size[rootI] += size[rootJ];
            }
        }
    }

    private int[] ExtractClusters(List<CondensedTreeNode> condensedTree, int n, HDBSCANClusterSelection method)
    {
        var labels = new int[n];
        for (int i = 0; i < n; i++)
        {
            labels[i] = -1; // Noise by default
        }

        if (condensedTree.Count == 0)
        {
            return labels;
        }

        // Find all cluster nodes
        var clusterNodes = new HashSet<int>();
        foreach (var node in condensedTree)
        {
            if (node.Child >= n)
            {
                clusterNodes.Add(node.Child);
            }
            clusterNodes.Add(node.Parent);
        }

        if (method == HDBSCANClusterSelection.EOM)
        {
            // Compute stability for each cluster
            var stability = new Dictionary<int, double>();
            var birthLambda = new Dictionary<int, double>();
            var children = new Dictionary<int, List<int>>();

            foreach (int cluster in clusterNodes)
            {
                stability[cluster] = 0;
                birthLambda[cluster] = double.MaxValue;
                children[cluster] = new List<int>();
            }

            foreach (var node in condensedTree)
            {
                if (birthLambda.ContainsKey(node.Child))
                {
                    birthLambda[node.Child] = Math.Min(birthLambda[node.Child], node.Lambda);
                }
                if (children.ContainsKey(node.Parent))
                {
                    children[node.Parent].Add(node.Child);
                }

                // Add stability contribution
                if (stability.ContainsKey(node.Parent))
                {
                    double deathLambda = node.Lambda;
                    double birth = birthLambda.ContainsKey(node.Parent) ? birthLambda[node.Parent] : 0;
                    stability[node.Parent] += (deathLambda - birth) * node.Size;
                }
            }

            // Select clusters with max stability
            var selectedClusters = new HashSet<int>();
            var clusterList = clusterNodes.Where(c => c >= n).OrderByDescending(c => c).ToList();

            foreach (int cluster in clusterList)
            {
                double childStability = children.ContainsKey(cluster)
                    ? children[cluster].Where(c => c >= n).Sum(c => stability.ContainsKey(c) ? stability[c] : 0)
                    : 0;

                if (stability.ContainsKey(cluster) && stability[cluster] > childStability)
                {
                    selectedClusters.Add(cluster);
                    // Remove descendants
                    if (children.ContainsKey(cluster))
                    {
                        foreach (var child in children[cluster])
                        {
                            selectedClusters.Remove(child);
                        }
                    }
                }
                else if (childStability > 0)
                {
                    // Propagate child stability
                    stability[cluster] = childStability;
                }
            }

            // Assign labels based on selected clusters
            int labelNum = 0;
            var clusterToLabel = new Dictionary<int, int>();

            foreach (int cluster in selectedClusters.OrderBy(c => c))
            {
                clusterToLabel[cluster] = labelNum++;
            }

            // Assign points to clusters
            foreach (var node in condensedTree)
            {
                if (node.Child < n)
                {
                    // Find which selected cluster this point belongs to
                    int current = node.Parent;
                    while (current >= n && !selectedClusters.Contains(current))
                    {
                        var parentNode = condensedTree.FirstOrDefault(x => x.Child == current);
                        if (parentNode.Parent == 0 && parentNode.Child == 0) break;
                        current = parentNode.Parent;
                    }

                    if (clusterToLabel.ContainsKey(current))
                    {
                        labels[node.Child] = clusterToLabel[current];
                    }
                }
            }
        }
        else // Leaf selection
        {
            // Find leaf clusters (clusters with no child clusters)
            var hasChildCluster = new HashSet<int>();
            foreach (var node in condensedTree)
            {
                if (node.Child >= n)
                {
                    hasChildCluster.Add(node.Parent);
                }
            }

            var leafClusters = clusterNodes.Where(c => c >= n && !hasChildCluster.Contains(c)).ToList();

            int labelNum = 0;
            var clusterToLabel = new Dictionary<int, int>();

            foreach (int cluster in leafClusters.OrderBy(c => c))
            {
                clusterToLabel[cluster] = labelNum++;
            }

            // Assign points
            foreach (var node in condensedTree)
            {
                if (node.Child < n && clusterToLabel.ContainsKey(node.Parent))
                {
                    labels[node.Child] = clusterToLabel[node.Parent];
                }
            }
        }

        return labels;
    }

    private void ComputeProbabilitiesAndOutlierScores(int[] labels, List<CondensedTreeNode> condensedTree, int n)
    {
        _probabilities = new double[n];
        _outlierScores = new double[n];

        // For each point, probability is based on lambda at which it joined its cluster
        foreach (var node in condensedTree)
        {
            if (node.Child < n)
            {
                _probabilities[node.Child] = labels[node.Child] >= 0 ? 1.0 : 0.0;
                _outlierScores[node.Child] = labels[node.Child] >= 0 ? 0.0 : 1.0;
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
            int nearestCluster = -1;

            if (ClusterCenters is not null && NumClusters > 0)
            {
                for (int k = 0; k < NumClusters; k++)
                {
                    var center = GetRow(ClusterCenters, k);
                    double dist = NumOps.ToDouble(metric.Compute(point, center));

                    if (dist < minDist)
                    {
                        minDist = dist;
                        nearestCluster = k;
                    }
                }
            }

            labels[i] = NumOps.FromDouble(nearestCluster);
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
    /// Represents an edge in the minimum spanning tree.
    /// </summary>
    private struct MSTEdge
    {
        public int U { get; }
        public int V { get; }
        public double Weight { get; }

        public MSTEdge(int u, int v, double weight)
        {
            U = u;
            V = v;
            Weight = weight;
        }
    }

    /// <summary>
    /// Represents a node in the condensed tree.
    /// </summary>
    public struct CondensedTreeNode
    {
        /// <summary>Parent cluster ID.</summary>
        public int Parent { get; }

        /// <summary>Child cluster or point ID.</summary>
        public int Child { get; }

        /// <summary>Lambda value (1/distance) at which this split occurred.</summary>
        public double Lambda { get; }

        /// <summary>Size of the child.</summary>
        public int Size { get; }

        /// <summary>
        /// Initializes a new CondensedTreeNode.
        /// </summary>
        public CondensedTreeNode(int parent, int child, double lambda, int size)
        {
            Parent = parent;
            Child = child;
            Lambda = lambda;
            Size = size;
        }
    }
}
