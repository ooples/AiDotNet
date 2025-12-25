using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Clustering.Hierarchical;

/// <summary>
/// BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) clustering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// BIRCH is designed for very large datasets. It builds a CF (Clustering Feature)
/// tree that summarizes the data, then optionally applies a global clustering
/// algorithm to the leaf entries.
/// </para>
/// <para>
/// Each node in the CF-tree stores:
/// - N: Number of points
/// - LS: Linear Sum (vector sum of points)
/// - SS: Squared Sum (sum of squared norms)
/// </para>
/// <para><b>For Beginners:</b> BIRCH creates a compressed summary of your data.
///
/// The process:
/// 1. Build a tree where each node summarizes nearby points
/// 2. The tree automatically adjusts to fit memory constraints
/// 3. Use the tree for fast approximate clustering
///
/// CF (Clustering Feature) = (N, LS, SS) where:
/// - N = count of points
/// - LS = sum of all point coordinates
/// - SS = sum of squared coordinates
///
/// From these, you can compute:
/// - Centroid = LS / N
/// - Radius = sqrt((SS/N) - (LS/N)²)
/// </para>
/// </remarks>
public class BIRCH<T> : ClusteringBase<T>
{
    private readonly BIRCHOptions<T> _options;
    private CFNode? _root;
    private List<CFEntry>? _leafEntries;

    /// <summary>
    /// Initializes a new BIRCH instance.
    /// </summary>
    /// <param name="options">The BIRCH options.</param>
    public BIRCH(BIRCHOptions<T>? options = null)
        : base(options ?? new BIRCHOptions<T>())
    {
        _options = options ?? new BIRCHOptions<T>();
    }

    /// <summary>
    /// Gets the leaf entries from the CF-tree.
    /// </summary>
    public IReadOnlyList<CFEntry>? LeafEntries => _leafEntries?.AsReadOnly();

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.Clustering;

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new BIRCH<T>(new BIRCHOptions<T>
        {
            Threshold = _options.Threshold,
            BranchingFactor = _options.BranchingFactor,
            NumClusters = _options.NumClusters,
            ComputeLabels = _options.ComputeLabels,
            DistanceMetric = _options.DistanceMetric
        });
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (BIRCH<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int d = x.Columns;
        NumFeatures = d;

        // Initialize root
        _root = new CFNode(_options.BranchingFactor, _options.Threshold, true);

        // Insert each point into the CF-tree
        for (int i = 0; i < n; i++)
        {
            var point = new double[d];
            for (int j = 0; j < d; j++)
            {
                point[j] = NumOps.ToDouble(x[i, j]);
            }

            var entry = CFEntry.FromPoint(point);
            InsertEntry(_root, entry);
        }

        // Collect leaf entries
        _leafEntries = new List<CFEntry>();
        CollectLeafEntries(_root, _leafEntries);

        // Determine clusters
        if (_options.NumClusters.HasValue && _options.NumClusters.Value < _leafEntries.Count)
        {
            // Apply global clustering to leaf entries
            ApplyGlobalClustering(_leafEntries, _options.NumClusters.Value, d);
        }
        else
        {
            // Each leaf entry is a cluster
            NumClusters = _leafEntries.Count;

            ClusterCenters = new Matrix<T>(NumClusters, d);
            for (int k = 0; k < NumClusters; k++)
            {
                var centroid = _leafEntries[k].Centroid;
                for (int j = 0; j < d; j++)
                {
                    ClusterCenters[k, j] = NumOps.FromDouble(centroid[j]);
                }
            }
        }

        // Assign labels if requested
        if (_options.ComputeLabels)
        {
            Labels = Predict(x);
        }
        else
        {
            Labels = new Vector<T>(n);
        }

        IsTrained = true;
    }

    private void InsertEntry(CFNode node, CFEntry entry)
    {
        if (node.IsLeaf)
        {
            // Try to find closest entry to merge with
            int closestIdx = -1;
            double closestDist = double.MaxValue;

            for (int i = 0; i < node.Entries.Count; i++)
            {
                double dist = CFEntry.Distance(node.Entries[i], entry);
                if (dist < closestDist)
                {
                    closestDist = dist;
                    closestIdx = i;
                }
            }

            if (closestIdx >= 0)
            {
                // Try to merge
                var merged = CFEntry.Merge(node.Entries[closestIdx], entry);
                if (merged.Radius <= _options.Threshold)
                {
                    node.Entries[closestIdx] = merged;
                    return;
                }
            }

            // Cannot merge, add as new entry
            node.Entries.Add(entry);

            // Split if necessary
            if (node.Entries.Count > _options.BranchingFactor)
            {
                SplitNode(node);
            }
        }
        else
        {
            // Find closest child
            int closestIdx = 0;
            double closestDist = double.MaxValue;

            for (int i = 0; i < node.Children.Count; i++)
            {
                var childCF = ComputeNodeCF(node.Children[i]);
                double dist = CFEntry.Distance(childCF, entry);
                if (dist < closestDist)
                {
                    closestDist = dist;
                    closestIdx = i;
                }
            }

            // Recursively insert
            InsertEntry(node.Children[closestIdx], entry);

            // Update node's entry for this child
            if (closestIdx < node.Entries.Count)
            {
                node.Entries[closestIdx] = ComputeNodeCF(node.Children[closestIdx]);
            }
        }
    }

    private void SplitNode(CFNode node)
    {
        // Find two farthest entries as seeds
        int seed1 = 0, seed2 = 1;
        double maxDist = 0;

        for (int i = 0; i < node.Entries.Count; i++)
        {
            for (int j = i + 1; j < node.Entries.Count; j++)
            {
                double dist = CFEntry.Distance(node.Entries[i], node.Entries[j]);
                if (dist > maxDist)
                {
                    maxDist = dist;
                    seed1 = i;
                    seed2 = j;
                }
            }
        }

        // Create two new nodes
        var newNode1 = new CFNode(_options.BranchingFactor, _options.Threshold, node.IsLeaf);
        var newNode2 = new CFNode(_options.BranchingFactor, _options.Threshold, node.IsLeaf);

        // Distribute entries
        for (int i = 0; i < node.Entries.Count; i++)
        {
            double dist1 = CFEntry.Distance(node.Entries[i], node.Entries[seed1]);
            double dist2 = CFEntry.Distance(node.Entries[i], node.Entries[seed2]);

            if (dist1 <= dist2)
            {
                newNode1.Entries.Add(node.Entries[i]);
                if (!node.IsLeaf && i < node.Children.Count)
                {
                    newNode1.Children.Add(node.Children[i]);
                }
            }
            else
            {
                newNode2.Entries.Add(node.Entries[i]);
                if (!node.IsLeaf && i < node.Children.Count)
                {
                    newNode2.Children.Add(node.Children[i]);
                }
            }
        }

        // Replace current node content with newNode1
        node.Entries.Clear();
        node.Children.Clear();

        foreach (var e in newNode1.Entries) node.Entries.Add(e);
        foreach (var c in newNode1.Children) node.Children.Add(c);

        // If this is root, create new root
        if (node == _root)
        {
            var oldRoot = new CFNode(_options.BranchingFactor, _options.Threshold, node.IsLeaf);
            foreach (var e in node.Entries) oldRoot.Entries.Add(e);
            foreach (var c in node.Children) oldRoot.Children.Add(c);

            node.Entries.Clear();
            node.Children.Clear();
            node.IsLeaf = false;

            node.Entries.Add(ComputeNodeCF(oldRoot));
            node.Entries.Add(ComputeNodeCF(newNode2));
            node.Children.Add(oldRoot);
            node.Children.Add(newNode2);
        }
    }

    private CFEntry ComputeNodeCF(CFNode node)
    {
        if (node.Entries.Count == 0)
        {
            return new CFEntry(0, Array.Empty<double>(), 0);
        }

        var merged = node.Entries[0];
        for (int i = 1; i < node.Entries.Count; i++)
        {
            merged = CFEntry.Merge(merged, node.Entries[i]);
        }

        return merged;
    }

    private void CollectLeafEntries(CFNode node, List<CFEntry> entries)
    {
        if (node.IsLeaf)
        {
            entries.AddRange(node.Entries);
        }
        else
        {
            foreach (var child in node.Children)
            {
                CollectLeafEntries(child, entries);
            }
        }
    }

    private void ApplyGlobalClustering(List<CFEntry> entries, int numClusters, int d)
    {
        // Use agglomerative clustering on centroids
        int n = entries.Count;
        NumClusters = numClusters;

        // Initialize: each entry is its own cluster
        var clusterAssignments = new int[n];
        for (int i = 0; i < n; i++)
        {
            clusterAssignments[i] = i;
        }

        // Compute distance matrix
        var distances = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                distances[i, j] = CFEntry.Distance(entries[i], entries[j]);
                distances[j, i] = distances[i, j];
            }
        }

        // Agglomerative merging until numClusters remain
        var activeClusters = new HashSet<int>(Enumerable.Range(0, n));

        while (activeClusters.Count > numClusters)
        {
            // Find closest pair
            double minDist = double.MaxValue;
            int merge1 = -1, merge2 = -1;

            var activeList = activeClusters.ToList();
            for (int i = 0; i < activeList.Count; i++)
            {
                for (int j = i + 1; j < activeList.Count; j++)
                {
                    int c1 = activeList[i];
                    int c2 = activeList[j];

                    if (distances[c1, c2] < minDist)
                    {
                        minDist = distances[c1, c2];
                        merge1 = c1;
                        merge2 = c2;
                    }
                }
            }

            if (merge1 < 0) break;

            // Merge cluster2 into cluster1
            for (int i = 0; i < n; i++)
            {
                if (clusterAssignments[i] == merge2)
                {
                    clusterAssignments[i] = merge1;
                }
            }

            // Update distances (average linkage)
            foreach (int c in activeClusters)
            {
                if (c != merge1 && c != merge2)
                {
                    distances[merge1, c] = (distances[merge1, c] + distances[merge2, c]) / 2;
                    distances[c, merge1] = distances[merge1, c];
                }
            }

            activeClusters.Remove(merge2);
        }

        // Renumber clusters
        var clusterMap = new Dictionary<int, int>();
        int clusterNum = 0;
        foreach (int c in activeClusters)
        {
            clusterMap[c] = clusterNum++;
        }

        for (int i = 0; i < n; i++)
        {
            // Find the active cluster this point belongs to
            int originalCluster = clusterAssignments[i];
            while (!activeClusters.Contains(originalCluster))
            {
                // Find which cluster it merged into
                for (int j = 0; j < n; j++)
                {
                    if (clusterAssignments[j] == originalCluster && j != i)
                    {
                        originalCluster = clusterAssignments[j];
                        break;
                    }
                }
                break; // Prevent infinite loop
            }

            if (clusterMap.ContainsKey(originalCluster))
            {
                clusterAssignments[i] = clusterMap[originalCluster];
            }
        }

        // Compute cluster centers from entries
        var clusterCentroids = new double[numClusters][];
        var clusterCounts = new int[numClusters];

        for (int k = 0; k < numClusters; k++)
        {
            clusterCentroids[k] = new double[d];
        }

        for (int i = 0; i < n; i++)
        {
            int cluster = clusterAssignments[i];
            if (cluster >= 0 && cluster < numClusters)
            {
                var centroid = entries[i].Centroid;
                int weight = entries[i].N;

                for (int j = 0; j < d; j++)
                {
                    clusterCentroids[cluster][j] += centroid[j] * weight;
                }
                clusterCounts[cluster] += weight;
            }
        }

        ClusterCenters = new Matrix<T>(numClusters, d);
        for (int k = 0; k < numClusters; k++)
        {
            if (clusterCounts[k] > 0)
            {
                for (int j = 0; j < d; j++)
                {
                    ClusterCenters[k, j] = NumOps.FromDouble(clusterCentroids[k][j] / clusterCounts[k]);
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
            int nearestCluster = 0;

            if (ClusterCenters is not null)
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
    /// Represents a Clustering Feature (CF) entry.
    /// </summary>
    public class CFEntry
    {
        /// <summary>
        /// Number of points in the cluster.
        /// </summary>
        public int N { get; }

        /// <summary>
        /// Linear sum of points.
        /// </summary>
        public double[] LS { get; }

        /// <summary>
        /// Sum of squared norms.
        /// </summary>
        public double SS { get; }

        /// <summary>
        /// Initializes a new CFEntry.
        /// </summary>
        public CFEntry(int n, double[] ls, double ss)
        {
            N = n;
            LS = ls;
            SS = ss;
        }

        /// <summary>
        /// Creates a CFEntry from a single point.
        /// </summary>
        public static CFEntry FromPoint(double[] point)
        {
            double ss = 0;
            for (int i = 0; i < point.Length; i++)
            {
                ss += point[i] * point[i];
            }

            return new CFEntry(1, (double[])point.Clone(), ss);
        }

        /// <summary>
        /// Merges two CFEntries.
        /// </summary>
        public static CFEntry Merge(CFEntry a, CFEntry b)
        {
            int d = Math.Max(a.LS.Length, b.LS.Length);
            var ls = new double[d];

            for (int i = 0; i < a.LS.Length; i++)
            {
                ls[i] += a.LS[i];
            }
            for (int i = 0; i < b.LS.Length; i++)
            {
                ls[i] += b.LS[i];
            }

            return new CFEntry(a.N + b.N, ls, a.SS + b.SS);
        }

        /// <summary>
        /// Computes distance between two CFEntries.
        /// </summary>
        public static double Distance(CFEntry a, CFEntry b)
        {
            // Use centroid distance
            var ca = a.Centroid;
            var cb = b.Centroid;

            int d = Math.Max(ca.Length, cb.Length);
            double sum = 0;

            for (int i = 0; i < d; i++)
            {
                double va = i < ca.Length ? ca[i] : 0;
                double vb = i < cb.Length ? cb[i] : 0;
                double diff = va - vb;
                sum += diff * diff;
            }

            return Math.Sqrt(sum);
        }

        /// <summary>
        /// Gets the centroid of this entry.
        /// </summary>
        public double[] Centroid
        {
            get
            {
                if (N == 0) return LS;
                var result = new double[LS.Length];
                for (int i = 0; i < LS.Length; i++)
                {
                    result[i] = LS[i] / N;
                }
                return result;
            }
        }

        /// <summary>
        /// Gets the radius of this entry.
        /// </summary>
        public double Radius
        {
            get
            {
                if (N <= 1) return 0;

                // Radius = sqrt((SS/N) - ||LS/N||²)
                double centroidNormSq = 0;
                var c = Centroid;
                for (int i = 0; i < c.Length; i++)
                {
                    centroidNormSq += c[i] * c[i];
                }

                double variance = SS / N - centroidNormSq;
                return variance > 0 ? Math.Sqrt(variance) : 0;
            }
        }
    }

    /// <summary>
    /// Represents a node in the CF-tree.
    /// </summary>
    public class CFNode
    {
        /// <summary>
        /// The entries in this node.
        /// </summary>
        public List<CFEntry> Entries { get; } = new List<CFEntry>();

        /// <summary>
        /// Child nodes (for non-leaf nodes).
        /// </summary>
        public List<CFNode> Children { get; } = new List<CFNode>();

        /// <summary>
        /// Whether this is a leaf node.
        /// </summary>
        public bool IsLeaf { get; set; }

        private readonly int _branchingFactor;
        private readonly double _threshold;

        /// <summary>
        /// Initializes a new CFNode.
        /// </summary>
        public CFNode(int branchingFactor, double threshold, bool isLeaf)
        {
            _branchingFactor = branchingFactor;
            _threshold = threshold;
            IsLeaf = isLeaf;
        }
    }
}
