using AiDotNet.Attributes;
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
/// <example>
/// <code>
/// // Use AiModelBuilder facade for HDBSCAN clustering
/// var builder = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new HDBSCAN&lt;double&gt;(new HDBSCANOptions&lt;double&gt;()));
///
/// var result = builder.Build(dataMatrix, labels);
/// var predictions = result.Predict(newData);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Hierarchical Density Estimates for Data Clustering, Visualization, and Outlier Detection", "https://doi.org/10.1145/2733381", Year = 2015, Authors = "Ricardo J. G. B. Campello, Davoud Moulavi, Arthur Zimek, Jorg Sander")]
public class HDBSCAN<T> : ClusteringBase<T>
{
    private readonly HDBSCANOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private T[]? _outlierScores;
    private T[]? _probabilities;
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
    public T[]? OutlierScores => _outlierScores;

    /// <summary>
    /// Gets the cluster membership probabilities.
    /// </summary>
    public T[]? Probabilities => _probabilities;

    /// <inheritdoc />

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
    public override bool SupportsParameterInitialization => false;

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = (HDBSCAN<T>)CreateNewInstance();
        clone._outlierScores = _outlierScores?.ToArray();
        clone._probabilities = _probabilities?.ToArray();
        clone._condensedTree = _condensedTree?.ToList();
        clone.NumClusters = NumClusters;
        clone.NumFeatures = NumFeatures;
        clone.IsTrained = IsTrained;

        if (Labels is not null)
        {
            clone.Labels = new Vector<T>(Labels.Length);
            for (int i = 0; i < Labels.Length; i++)
                clone.Labels[i] = Labels[i];
        }

        if (ClusterCenters is not null)
        {
            clone.ClusterCenters = new Matrix<T>(ClusterCenters.Rows, ClusterCenters.Columns);
            for (int i = 0; i < ClusterCenters.Rows; i++)
                for (int j = 0; j < ClusterCenters.Columns; j++)
                    clone.ClusterCenters[i, j] = ClusterCenters[i, j];
        }

        return clone;
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

        // Step 5: Build condensed tree (also returns final union-find for point→cluster mapping)
        int[] finalParent;
        (_condensedTree, finalParent) = BuildCondensedTree(mst, n, _options.MinClusterSize);

        // Step 6: Extract clusters
        var clusterLabels = ExtractClusters(_condensedTree, n, _options.ClusterSelection, _options.AllowSingleCluster, finalParent);

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
            var clusterSums = new T[NumClusters, d];
            for (int k = 0; k < NumClusters; k++)
                for (int j = 0; j < d; j++)
                    clusterSums[k, j] = NumOps.Zero;

            for (int i = 0; i < n; i++)
            {
                int label = clusterLabels[i];
                if (label >= 0)
                {
                    clusterCounts[label]++;
                    for (int j = 0; j < d; j++)
                    {
                        clusterSums[label, j] = NumOps.Add(clusterSums[label, j], x[i, j]);
                    }
                }
            }

            for (int k = 0; k < NumClusters; k++)
            {
                if (clusterCounts[k] > 0)
                {
                    T countT = NumOps.FromDouble(clusterCounts[k]);
                    for (int j = 0; j < d; j++)
                    {
                        ClusterCenters[k, j] = NumOps.Divide(clusterSums[k, j], countT);
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

    private T[,] ComputeDistanceMatrix(Matrix<T> x, int n, IDistanceMetric<T> metric)
    {
        int d = x.Columns;
        var distMatrix = new T[n, n];

        // Cache rows as arrays for allocation-free distance
        var rowArrays = new T[n][];
        for (int i = 0; i < n; i++)
        {
            rowArrays[i] = new T[d];
            for (int c = 0; c < d; c++)
                rowArrays[i][c] = x[i, c];
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                T dist = metric.ComputeInline(rowArrays[i], rowArrays[j], d);
                distMatrix[i, j] = dist;
                distMatrix[j, i] = dist;
            }
        }

        return distMatrix;
    }

    /// <summary>
    /// BFS deselect all descendants of a cluster per scikit-learn reference.
    /// </summary>
    private static void MarkDescendantsNotCluster(int cluster, Dictionary<int, List<int>> children,
        HashSet<int> clusterNodes, Dictionary<int, bool> isCluster)
    {
        // Iterative traversal to avoid stack overflow on deep condensed trees
        var stack = new Stack<int>();
        stack.Push(cluster);
        while (stack.Count > 0)
        {
            int current = stack.Pop();
            if (!children.TryGetValue(current, out var kids)) continue;
            foreach (int child in kids)
            {
                if (clusterNodes.Contains(child))
                {
                    isCluster[child] = false;
                    stack.Push(child);
                }
            }
        }
    }

    private T[] ComputeCoreDistances(T[,] distMatrix, int n, int minSamples)
    {
        var coreDistances = new T[n];
        int k = Math.Min(minSamples, n - 1);

        for (int i = 0; i < n; i++)
        {
            // Get distances to all other points
            var distances = new List<T>(n);
            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    distances.Add(distMatrix[i, j]);
                }
            }

            distances.Sort((a, b) => NumOps.Compare(a, b));

            // Core distance is distance to k-th nearest neighbor
            coreDistances[i] = k > 0 && k <= distances.Count ? distances[k - 1] : NumOps.Zero;
        }

        return coreDistances;
    }

    private T[,] ComputeMutualReachabilityDistances(T[,] distMatrix, T[] coreDistances, int n)
    {
        var mrdMatrix = new T[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                // Mutual reachability distance = max(core_dist[i], core_dist[j], dist[i,j])
                T mrd = coreDistances[i];
                if (NumOps.GreaterThan(coreDistances[j], mrd)) mrd = coreDistances[j];
                if (NumOps.GreaterThan(distMatrix[i, j], mrd)) mrd = distMatrix[i, j];
                mrdMatrix[i, j] = mrd;
                mrdMatrix[j, i] = mrd;
            }
        }

        return mrdMatrix;
    }

    private List<MSTEdge> BuildMinimumSpanningTree(T[,] mrdMatrix, int n)
    {
        // Prim's algorithm for MST
        var mst = new List<MSTEdge>();
        var inTree = new bool[n];
        var minDist = new T[n];
        var minFrom = new int[n];

        for (int i = 0; i < n; i++)
        {
            minDist[i] = NumOps.MaxValue;
            minFrom[i] = -1;
        }

        // Start from node 0
        minDist[0] = NumOps.Zero;

        for (int count = 0; count < n; count++)
        {
            // Find minimum distance node not in tree
            int u = -1;
            T minVal = NumOps.MaxValue;
            for (int i = 0; i < n; i++)
            {
                if (!inTree[i] && NumOps.LessThan(minDist[i], minVal))
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
                if (!inTree[v] && NumOps.LessThan(mrdMatrix[u, v], minDist[v]))
                {
                    minDist[v] = mrdMatrix[u, v];
                    minFrom[v] = u;
                }
            }
        }

        // Sort edges by weight
        mst.Sort((a, b) => NumOps.Compare(a.Weight, b.Weight));

        return mst;
    }

    private (List<CondensedTreeNode> tree, int[] finalParent) BuildCondensedTree(List<MSTEdge> mst, int n, int minClusterSize)
    {
        // Per Campello et al. 2013 and the scikit-learn reference implementation:
        // Build the condensed tree by processing MST edges (bottom-up).
        // Track which union-find roots are "active clusters" (have a cluster ID >= n).
        // Only record entries in the condensed tree for:
        //   1. Point noise: when a small group merges into an active cluster
        //   2. Cluster split: when two active clusters merge → new parent
        //   3. Cluster birth: when a group first reaches minClusterSize

        var condensedTree = new List<CondensedTreeNode>();

        int maxId = n + mst.Count + 1;
        var parent = new int[maxId];
        var size = new int[maxId];
        var clusterLabel = new int[maxId]; // Maps UF root → cluster ID (-1 = not yet a cluster)

        for (int i = 0; i < n; i++)
        {
            parent[i] = i;
            size[i] = 1;
            clusterLabel[i] = -1;
        }
        for (int i = n; i < maxId; i++)
        {
            parent[i] = i;
            size[i] = 0;
            clusterLabel[i] = -1;
        }

        int nextCluster = n;

        foreach (var edge in mst)
        {
            int root1 = Find(parent, edge.U);
            int root2 = Find(parent, edge.V);

            if (root1 == root2) continue;

            T edgeLambda = NumOps.GreaterThan(edge.Weight, NumOps.Zero)
                ? NumOps.Divide(NumOps.One, edge.Weight)
                : NumOps.MaxValue;

            int size1 = size[root1];
            int size2 = size[root2];

            bool isCluster1 = clusterLabel[root1] >= 0;
            bool isCluster2 = clusterLabel[root2] >= 0;
            bool big1 = size1 >= minClusterSize;
            bool big2 = size2 >= minClusterSize;

            if (!big1 && !big2)
            {
                // Both too small — just merge, no condensed tree entry
                Union(parent, size, root1, root2);
                int newRoot = Find(parent, root1);
                // Propagate cluster label if either had one
                if (isCluster1) clusterLabel[newRoot] = clusterLabel[root1];
                else if (isCluster2) clusterLabel[newRoot] = clusterLabel[root2];
            }
            else if (big1 && big2 && isCluster1 && isCluster2)
            {
                // Two active clusters merge → real split point
                int newClusterId = nextCluster++;
                int cid1 = clusterLabel[root1];
                int cid2 = clusterLabel[root2];

                condensedTree.Add(new CondensedTreeNode(newClusterId, cid1, edgeLambda, size1));
                condensedTree.Add(new CondensedTreeNode(newClusterId, cid2, edgeLambda, size2));

                Union(parent, size, root1, root2);
                int newRoot = Find(parent, root1);
                if (newRoot >= parent.Length)
                {
                    Array.Resize(ref parent, newRoot + 2);
                    Array.Resize(ref size, newRoot + 2);
                    Array.Resize(ref clusterLabel, newRoot + 2);
                    parent[newRoot] = newRoot;
                }
                clusterLabel[newRoot] = newClusterId;
            }
            else
            {
                // At least one side is big enough. Determine the surviving cluster
                // and the "runt" side. If the big side doesn't have a cluster ID yet,
                // assign one (first time this group reached minClusterSize).
                int bigRoot, smallRoot;
                if (big1 && (!big2 || size1 >= size2))
                {
                    bigRoot = root1;
                    smallRoot = root2;
                }
                else
                {
                    bigRoot = root2;
                    smallRoot = root1;
                }

                // Ensure the surviving side has a cluster ID
                if (clusterLabel[bigRoot] < 0)
                {
                    clusterLabel[bigRoot] = nextCluster++;
                }
                int clusterId = clusterLabel[bigRoot];

                // Runt side: add individual points as noise under the surviving cluster
                for (int i = 0; i < n; i++)
                {
                    if (Find(parent, i) == smallRoot)
                    {
                        condensedTree.Add(new CondensedTreeNode(
                            clusterId, i, edgeLambda, 1));
                    }
                }

                Union(parent, size, smallRoot, bigRoot);
                int mergedRoot = Find(parent, bigRoot);
                clusterLabel[mergedRoot] = clusterId;
            }
        }

        return (condensedTree, parent);
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

    private int[] ExtractClusters(List<CondensedTreeNode> condensedTree, int n, HDBSCANClusterSelection method, bool allowSingleCluster = false, int[]? ufParent = null)
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

        // Find all cluster nodes. A cluster node is any entry with Size > 1
        // (sub-cluster) or any parent. Note: union-find root IDs can be < n
        // even for entries representing clusters, so check Size, not just ID >= n.
        var clusterNodes = new HashSet<int>();
        foreach (var node in condensedTree)
        {
            if (node.Child >= n || node.Size > 1)
            {
                clusterNodes.Add(node.Child);
            }
            clusterNodes.Add(node.Parent);
        }

        if (method == HDBSCANClusterSelection.EOM)
        {
            // Compute stability for each cluster
            var stability = new Dictionary<int, T>();
            var birthLambda = new Dictionary<int, T>();
            var children = new Dictionary<int, List<int>>();

            foreach (int cluster in clusterNodes)
            {
                stability[cluster] = NumOps.Zero;
                birthLambda[cluster] = NumOps.MaxValue;
                children[cluster] = new List<int>();
            }

            // Pass 1: Compute birth lambda and parent-child relationships.
            // Per Campello et al. 2013, birth lambda of a cluster = the lambda at which
            // it first appears in the condensed tree (minimum lambda across all its entries).
            foreach (var node in condensedTree)
            {
                // Birth lambda of a cluster = minimum lambda at which it appears as a child
                if (birthLambda.ContainsKey(node.Child))
                {
                    T current = birthLambda[node.Child];
                    if (NumOps.LessThan(node.Lambda, current))
                        birthLambda[node.Child] = node.Lambda;
                }
                if (children.ContainsKey(node.Parent))
                {
                    children[node.Parent].Add(node.Child);
                }
            }

            // Per scikit-learn reference: clusters that never appeared as a child
            // (i.e., the root cluster) have birth lambda = 0. Without this, the root's
            // stability is always 0, causing systematic over-segmentation.
            foreach (int cluster in clusterNodes)
            {
                if (NumOps.Equals(birthLambda[cluster], NumOps.MaxValue))
                    birthLambda[cluster] = NumOps.Zero;
            }

            // Pass 2: Compute stability per Campello et al. 2013 / scikit-learn.
            // Stability of cluster C = Σ (lambda_p - lambda_birth(C)) * size_p
            // for ALL entries in the condensed tree under parent C.
            foreach (var node in condensedTree)
            {
                if (stability.ContainsKey(node.Parent))
                {
                    T birth = birthLambda[node.Parent];
                    T lambda = node.Lambda;
                    // Clamp: if birth > lambda (shouldn't happen with correct tree),
                    // treat as zero contribution rather than negative
                    if (NumOps.GreaterThan(birth, lambda))
                        lambda = birth;
                    T contribution = NumOps.Multiply(
                        NumOps.Subtract(lambda, birth),
                        NumOps.FromDouble(node.Size));
                    stability[node.Parent] = NumOps.Add(stability[node.Parent], contribution);
                }
            }

            // Select clusters with max stability
            var selectedClusters = new HashSet<int>();
            var clusterList = clusterNodes.OrderByDescending(c => c).ToList();


            // Per scikit-learn reference: use isCluster flags.
            // Start with all clusters selected, then sweep bottom-up.
            // When parent stability > sum of children stability: keep parent, deselect children.
            // When children stability >= parent: propagate children stability to parent.
            var isCluster = new Dictionary<int, bool>();
            foreach (int c in clusterList)
                isCluster[c] = true;

            foreach (int cluster in clusterList)
            {
                T childStability = NumOps.Zero;
                bool hasClusterChildren = false;
                if (children.ContainsKey(cluster))
                {
                    foreach (int c in children[cluster].Where(c => clusterNodes.Contains(c)))
                    {
                        hasClusterChildren = true;
                        childStability = NumOps.Add(childStability, stability[c]);
                    }
                }

                if (!hasClusterChildren)
                    continue; // leaf cluster — stays selected

                if (NumOps.GreaterThan(stability[cluster], childStability))
                {
                    // Parent wins — deselect ALL descendants
                    isCluster[cluster] = true;
                    MarkDescendantsNotCluster(cluster, children, clusterNodes, isCluster);
                }
                else
                {
                    // Children win — propagate their stability up, deselect parent
                    stability[cluster] = childStability;
                    isCluster[cluster] = false;
                }
            }

            foreach (var kv in isCluster)
            {
                if (kv.Value)
                    selectedClusters.Add(kv.Key);
            }

            // Per scikit-learn: when allowSingleCluster and the result is 0 clusters,
            // select the root cluster (the one with no parent in condensed tree)
            if (allowSingleCluster && selectedClusters.Count == 0 && clusterList.Count > 0)
            {
                var childClusters = new HashSet<int>(
                    condensedTree.Where(node => node.Child >= n).Select(node => node.Child));
                int root = clusterList.Where(c => !childClusters.Contains(c)).DefaultIfEmpty(clusterList[0]).First();
                selectedClusters.Add(root);
            }

            int labelNum = 0;
            var clusterToLabel = new Dictionary<int, int>();

            foreach (int cluster in selectedClusters.OrderBy(c => c))
            {
                clusterToLabel[cluster] = labelNum++;
            }

            // Build parent lookup for efficient tree walking
            var parentLookup = new Dictionary<int, int>();
            foreach (var node in condensedTree)
            {
                if (!parentLookup.ContainsKey(node.Child))
                    parentLookup[node.Child] = node.Parent;
            }

            // Assign points to clusters
            foreach (var node in condensedTree)
            {
                if (node.Child < n)
                {
                    // Find which selected cluster this point belongs to by walking up the tree
                    int current = node.Parent;
                    while (current >= n && !selectedClusters.Contains(current))
                    {
                        if (!parentLookup.TryGetValue(current, out int parent) || parent == current)
                            break;
                        current = parent;
                    }

                    if (clusterToLabel.ContainsKey(current))
                    {
                        labels[node.Child] = clusterToLabel[current];
                    }
                }
            }

            // Assign points not found in the condensed tree using the union-find.
            // These are "core" points that stayed in their cluster throughout and
            // were never individually recorded as noise or point-level children.
            if (ufParent != null)
            {
                for (int i = 0; i < n; i++)
                {
                    if (labels[i] >= 0) continue; // already assigned

                    // Walk the union-find to find this point's cluster root
                    int root = Find(ufParent, i);
                    // Walk from the root up the condensed tree to find a selected cluster
                    int current = root;
                    while (current >= n && !selectedClusters.Contains(current))
                    {
                        if (!parentLookup.TryGetValue(current, out int par) || par == current)
                            break;
                        current = par;
                    }
                    if (clusterToLabel.ContainsKey(current))
                    {
                        labels[i] = clusterToLabel[current];
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

            // Assign points directly attached to leaf clusters
            foreach (var node in condensedTree)
            {
                if (node.Child < n && clusterToLabel.ContainsKey(node.Parent))
                {
                    labels[node.Child] = clusterToLabel[node.Parent];
                }
            }

            // Fallback for unlabeled points: walk up condensed tree to find a leaf cluster
            if (ufParent is not null)
            {
                var parentLookup = new Dictionary<int, int>();
                foreach (var node in condensedTree)
                {
                    if (node.Child >= n && !parentLookup.ContainsKey(node.Child))
                        parentLookup[node.Child] = node.Parent;
                }

                var leafSet = new HashSet<int>(leafClusters);
                for (int i = 0; i < n; i++)
                {
                    if (labels[i] >= 0) continue;

                    int root = Find(ufParent, i);
                    int current = root;
                    while (current >= n && !leafSet.Contains(current))
                    {
                        if (!parentLookup.TryGetValue(current, out int par) || par == current)
                            break;
                        current = par;
                    }
                    if (clusterToLabel.TryGetValue(current, out int lbl))
                    {
                        labels[i] = lbl;
                    }
                }
            }
        }

        return labels;
    }

    private void ComputeProbabilitiesAndOutlierScores(int[] labels, List<CondensedTreeNode> condensedTree, int n)
    {
        _probabilities = new T[n];
        _outlierScores = new T[n];

        // Initialize every point from its final label so that points recovered via the
        // union-find fallback (which never appear as a direct child in the condensed tree)
        // still receive proper probability/outlier metadata. Points labeled to a cluster
        // get probability=1, outlier=0; noise points (label = -1) get probability=0, outlier=1.
        for (int i = 0; i < n; i++)
        {
            bool inCluster = labels[i] >= 0;
            _probabilities[i] = inCluster ? NumOps.One : NumOps.Zero;
            _outlierScores[i] = inCluster ? NumOps.Zero : NumOps.One;
        }

        // Refine from the condensed tree for points that were directly recorded there.
        // (Currently uses the same {0,1} scheme as the initial pass; future work can
        // derive richer per-point probability from the lambda at which the point left
        // its cluster, but that refinement still benefits from the all-points init above.)
        foreach (var node in condensedTree)
        {
            if (node.Child < n)
            {
                _probabilities[node.Child] = labels[node.Child] >= 0 ? NumOps.One : NumOps.Zero;
                _outlierScores[node.Child] = labels[node.Child] >= 0 ? NumOps.Zero : NumOps.One;
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
            int nearestCluster = -1;

            if (ClusterCenters is not null && NumClusters > 0)
            {
                for (int k = 0; k < NumClusters; k++)
                {
                    var center = GetRow(ClusterCenters, k);
                    T dist = metric.Compute(point, center);

                    if (NumOps.LessThan(dist, minDist))
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
        return Labels ?? new Vector<T>(0);
    }

    /// <summary>
    /// Represents an edge in the minimum spanning tree.
    /// </summary>
    private struct MSTEdge
    {
        public int U { get; }
        public int V { get; }
        public T Weight { get; }

        public MSTEdge(int u, int v, T weight)
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
        public T Lambda { get; }

        /// <summary>Size of the child.</summary>
        public int Size { get; }

        /// <summary>
        /// Initializes a new CondensedTreeNode.
        /// </summary>
        public CondensedTreeNode(int parent, int child, T lambda, int size)
        {
            Parent = parent;
            Child = child;
            Lambda = lambda;
            Size = size;
        }
    }
}
