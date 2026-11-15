using AiDotNet.Tensors.Engines;
using AiDotNet.Attributes;
using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Helpers;
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
/// <example>
/// <code>
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 1.5, 1.8 }, { 5.0, 8.0 }, { 8.0, 8.0 }, { 1.0, 0.6 }, { 9.0, 11.0 } });
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new BIRCH&lt;double&gt;(new BIRCHOptions&lt;double&gt;()))
///     .Build(dataMatrix);
///
/// // one cluster index per row
/// var assignments = result.Predict(dataMatrix);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("BIRCH: An Efficient Data Clustering Method for Very Large Databases", "https://doi.org/10.1145/233269.233324", Year = 1996, Authors = "Tian Zhang, Raghu Ramakrishnan, Miron Livny")]
public partial class BIRCH<T> : ClusteringBase<T>
{
    private readonly BIRCHOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private CFNode? _root;
    private List<CFEntry>? _leafEntries;

    /// <summary>
    /// Initializes a new BIRCH instance.
    /// </summary>
    /// <param name="options">The BIRCH options.</param>
    public BIRCH(BIRCHOptions<T>? options = null)
        : base(options ??= new BIRCHOptions<T>())
    {
        _options = options;
    }

    /// <summary>
    /// Gets the leaf entries from the CF-tree.
    /// </summary>
    public IReadOnlyList<CFEntry>? LeafEntries => _leafEntries?.AsReadOnly();

    /// <inheritdoc />

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
        T threshold = NumOps.FromDouble(_options.Threshold);
        _root = new CFNode(_options.BranchingFactor, threshold, true);

        // Insert each point into the CF-tree
        for (int i = 0; i < n; i++)
        {
            var entry = CFEntry.FromPoint(x.GetRow(i).ToArray());
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
            ClusterCenters = StackCentroids(_leafEntries, d);
        }

        IsTrained = true;

        // Assign labels if requested
        if (_options.ComputeLabels)
        {
            Labels = Predict(x);
        }
        else
        {
            Labels = new Vector<T>(n);
        }
    }

    /// <summary>Centroids of the entries as rows of a [count, d] matrix (shorter centroids are zero-padded).</summary>
    private static Matrix<T> StackCentroids(List<CFEntry> entries, int d)
    {
        var result = new Matrix<T>(entries.Count, d);
        for (int i = 0; i < entries.Count; i++)
        {
            var centroid = entries[i].Centroid;
            if (centroid.Length > 0)
            {
                result.SetSubMatrix(i, 0, Tensor<T>.FromVector(Vector<T>.FromArray(centroid), new[] { 1, centroid.Length }).ToMatrix());
            }
        }

        return result;
    }

    /// <summary>Index of the entry whose centroid is closest to the point (first on ties), or -1 when none is below MaxValue.</summary>
    private int ClosestEntry(List<CFEntry> entries, CFEntry entry)
    {
        if (entries.Count == 0)
        {
            return -1;
        }

        var centroid = entry.Centroid;
        int d = Math.Max(centroid.Length, entries.Max(e => e.Centroid.Length));
        var distances = ClusteringTensorOps.DistancesToPoint(Pad(centroid, d), StackCentroids(entries, d));
        int best = ClusteringTensorOps.ArgMin(distances);
        return NumOps.LessThan(distances[best], NumOps.MaxValue) ? best : -1;
    }

    /// <summary>The values as a length-d vector (zero-padded).</summary>
    private static Vector<T> Pad(T[] values, int d)
        => values.Length == d
            ? Vector<T>.FromArray(values)
            : Vector<T>.Concatenate(new[] { Vector<T>.FromArray(values), new Vector<T>(d - values.Length) });

    private void InsertEntry(CFNode node, CFEntry entry)
    {
        T threshold = NumOps.FromDouble(_options.Threshold);

        if (node.IsLeaf)
        {
            // Try to find closest entry to merge with
            int closestIdx = ClosestEntry(node.Entries, entry);

            if (closestIdx >= 0)
            {
                // Try to merge
                var merged = CFEntry.Merge(node.Entries[closestIdx], entry);
                if (!NumOps.GreaterThan(merged.Radius, threshold))
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
            int closestIdx = Math.Max(0, ClosestEntry(node.Children.Select(ComputeNodeCF).ToList(), entry));

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
        T threshold = NumOps.FromDouble(_options.Threshold);

        // Centroid distances between all entries
        int count = node.Entries.Count;
        int d = node.Entries.Max(e => e.Centroid.Length);
        var centroids = StackCentroids(node.Entries, d);
        var distances = Tensor<T>.FromMatrix(ClusteringTensorOps.Distances(centroids, centroids));

        // Find two farthest entries as seeds (first strictly largest pair i < j; defaults to 0 and 1)
        var upper = Engine.TensorTriu(Engine.TensorOnesLike(distances), 1);
        var pairDistances = Engine.Reshape(Engine.TensorMultiply(distances, upper), new[] { count * count }).ToVector();
        int farthest = ClusteringTensorOps.ArgMax(pairDistances);
        bool separated = NumOps.GreaterThan(pairDistances[farthest], NumOps.Zero);
        int seed1 = separated ? farthest / count : 0;
        int seed2 = separated ? farthest % count : 1;

        // Each entry goes to the nearer seed (the first seed on ties)
        var distanceMatrix = distances.ToMatrix();
        var toFirst = ClusteringTensorOps.LessOrEqualFlags(
            Engine.Subtract(distanceMatrix.GetRow(seed1), distanceMatrix.GetRow(seed2)), NumOps.Zero);

        // Create two new nodes
        var newNode1 = new CFNode(_options.BranchingFactor, threshold, node.IsLeaf);
        var newNode2 = new CFNode(_options.BranchingFactor, threshold, node.IsLeaf);

        // Distribute entries
        for (int i = 0; i < count; i++)
        {
            var target = toFirst[i] ? newNode1 : newNode2;
            target.Entries.Add(node.Entries[i]);
            if (!node.IsLeaf && i < node.Children.Count)
            {
                target.Children.Add(node.Children[i]);
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
            var oldRoot = new CFNode(_options.BranchingFactor, threshold, node.IsLeaf);
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
            return new CFEntry(0, Array.Empty<T>(), NumOps.Zero);
        }

        // CF additivity: counts, linear sums and squared sums add up
        int d = node.Entries.Max(e => e.LS.Length);
        var sums = new Matrix<T>(node.Entries.Count, d);
        for (int i = 0; i < node.Entries.Count; i++)
        {
            var ls = node.Entries[i].LS;
            if (ls.Length > 0)
            {
                sums.SetSubMatrix(i, 0, Tensor<T>.FromVector(Vector<T>.FromArray(ls), new[] { 1, ls.Length }).ToMatrix());
            }
        }

        var linearSum = Engine.ReduceSum(Tensor<T>.FromMatrix(sums), new[] { 0 }, false).ToArray();
        var squaredSum = Engine.TensorSum(Vector<T>.FromArray(node.Entries.Select(e => e.SS).ToArray()).AsTensor());
        return new CFEntry(node.Entries.Sum(e => e.N), linearSum, squaredSum);
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

        // Compute distance matrix (zero diagonal)
        var centroids = StackCentroids(entries, d);
        var distances = ClusteringTensorOps.Distances(centroids, centroids);
        distances = Engine.TensorWhere(
            Engine.TensorEye<T>(n), Tensor<T>.CreateZeros(new[] { n, n }), Tensor<T>.FromMatrix(distances)).ToMatrix();

        // Agglomerative merging until numClusters remain
        var activeClusters = new HashSet<int>(Enumerable.Range(0, n));

        // Track merge history: when cluster B merges into A, record mergedInto[B] = A
        var mergedInto = new Dictionary<int, int>();

        while (activeClusters.Count > numClusters)
        {
            // Find closest pair
            var (merge1, merge2) = ClosestActivePair(distances, activeClusters);

            if (merge1 < 0) break;

            // Track the merge before updating assignments
            mergedInto[merge2] = merge1;

            // Merge cluster2 into cluster1
            for (int i = 0; i < n; i++)
            {
                if (clusterAssignments[i] == merge2)
                {
                    clusterAssignments[i] = merge1;
                }
            }

            // Update distances (average linkage) to the other active clusters
            AverageLinkage(distances, activeClusters, merge1, merge2);

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
            int cluster = clusterAssignments[i];

            // Follow the merge chain to find the final active cluster
            while (!activeClusters.Contains(cluster) && mergedInto.ContainsKey(cluster))
            {
                cluster = mergedInto[cluster];
            }

            // Fallback if still not in active clusters (should not happen normally)
            if (!activeClusters.Contains(cluster))
            {
                cluster = activeClusters.First();
            }

            clusterAssignments[i] = clusterMap.ContainsKey(cluster) ? clusterMap[cluster] : 0;
        }

        // Compute cluster centers from entries: sum_i N_i c_i / sum_i N_i over each cluster's entries
        var sizes = ClusteringTensorOps.FromIntArray<T>(entries.Select(e => e.N).ToArray());
        var weighted = Engine.TensorMultiply(
            Tensor<T>.FromMatrix(centroids), Engine.TensorBroadcastTo(Tensor<T>.FromVector(sizes, new[] { n, 1 }), new[] { n, d }));
        var members = Engine.TensorTranspose(ClusteringTensorOps.OneHotMask(ClusteringTensorOps.FromIntArray<T>(clusterAssignments), numClusters));
        var sums = Engine.TensorMatMul(members, weighted).ToMatrix();
        var counts = Engine.TensorMatMul(members, Tensor<T>.FromVector(sizes, new[] { n, 1 })).ToVector();
        var means = ClusteringTensorOps.DivideRows(sums, counts);
        ClusterCenters = ClusteringTensorOps.RowsWherePositive(counts, means, new Matrix<T>(numClusters, d));
    }

    /// <summary>The closest pair of active clusters in active-set scan order (first strictly smallest), or (-1, -1).</summary>
    private (int, int) ClosestActivePair(Matrix<T> distances, HashSet<int> activeClusters)
    {
        var activeList = activeClusters.ToArray();
        int a = activeList.Length;
        if (a < 2)
        {
            return (-1, -1);
        }

        var sub = ClusteringTensorOps.SelectColumns(Tensor<T>.FromMatrix(ClusteringTensorOps.SelectRows(distances, activeList)), activeList);
        var upper = Engine.TensorTriu(Engine.TensorOnesLike(sub), 1);
        var candidates = Engine.Reshape(Engine.TensorWhere(upper, sub, Engine.TensorFullLike(sub, NumOps.MaxValue)), new[] { a * a }).ToVector();
        int best = ClusteringTensorOps.ArgMin(candidates);
        return NumOps.LessThan(candidates[best], NumOps.MaxValue) ? (activeList[best / a], activeList[best % a]) : (-1, -1);
    }

    /// <summary>d(merge1, c) = (d(merge1, c) + d(merge2, c)) / 2 for every other active cluster c (kept symmetric).</summary>
    private void AverageLinkage(Matrix<T> distances, HashSet<int> activeClusters, int merge1, int merge2)
    {
        var others = new int[distances.Rows];
        foreach (int c in activeClusters)
        {
            if (c != merge1 && c != merge2)
            {
                others[c] = 1;
            }
        }

        var averaged = Engine.Multiply(Engine.Add(distances.GetRow(merge1), distances.GetRow(merge2)), NumOps.FromDouble(0.5));
        var mask = ClusteringTensorOps.FromIntArray<T>(others).AsTensor();
        var row = Engine.TensorWhere(mask, averaged.AsTensor(), distances.GetRow(merge1).AsTensor()).ToVector();
        Engine.SetRow(distances, merge1, row);
        var column = Engine.TensorWhere(mask, averaged.AsTensor(), distances.GetColumn(merge1).AsTensor()).ToVector();
        Engine.SetColumn(distances, merge1, column);
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();

        if (ClusterCenters is null || NumClusters == 0)
        {
            return new Vector<T>(x.Rows);
        }

        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        var centers = ClusterCenters.Rows == NumClusters
            ? ClusterCenters
            : ClusterCenters.GetSubMatrix(0, 0, NumClusters, ClusterCenters.Columns);
        return ClusteringTensorOps.FromIntArray<T>(ClusteringTensorOps.NearestCenters(metric, x, centers));
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels ?? new Vector<T>(0);
    }

    /// <summary>
    /// Represents a Clustering Feature (CF) entry.
    /// </summary>
    public class CFEntry
    {
        private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

        /// <summary>
        /// Number of points in the cluster.
        /// </summary>
        public int N { get; }

        /// <summary>
        /// Linear sum of points.
        /// </summary>
        public T[] LS { get; }

        /// <summary>
        /// Sum of squared norms.
        /// </summary>
        public T SS { get; }

        /// <summary>
        /// Initializes a new CFEntry.
        /// </summary>
        public CFEntry(int n, T[] ls, T ss)
        {
            N = n;
            LS = ls;
            SS = ss;
        }

        /// <summary>
        /// Creates a CFEntry from a single point.
        /// </summary>
        public static CFEntry FromPoint(T[] point)
        {
            var values = Vector<T>.FromArray(point);
            T ss = point.Length == 0 ? NumOps.Zero : CfEngine.DotProduct(values, values);
            return new CFEntry(1, (T[])point.Clone(), ss);
        }

        /// <summary>
        /// Merges two CFEntries.
        /// </summary>
        public static CFEntry Merge(CFEntry a, CFEntry b)
        {
            int d = Math.Max(a.LS.Length, b.LS.Length);
            var ls = d == 0 ? Array.Empty<T>() : CfEngine.Add(PadTo(a.LS, d), PadTo(b.LS, d)).ToArray();
            return new CFEntry(a.N + b.N, ls, NumOps.Add(a.SS, b.SS));
        }

        private static IEngine CfEngine => AiDotNetEngine.Current;

        /// <summary>The values as a length-d vector (zero-padded).</summary>
        private static Vector<T> PadTo(T[] values, int d)
            => values.Length == d
                ? Vector<T>.FromArray(values)
                : Vector<T>.Concatenate(new[] { Vector<T>.FromArray(values), new Vector<T>(d - values.Length) });

        /// <summary>
        /// Computes distance between two CFEntries.
        /// </summary>
        public static T Distance(CFEntry a, CFEntry b)
        {
            // Use centroid distance (shorter centroids are zero-padded)
            var ca = a.Centroid;
            var cb = b.Centroid;

            int d = Math.Max(ca.Length, cb.Length);
            if (d == 0)
            {
                return NumOps.Zero;
            }

            var diff = CfEngine.Subtract(PadTo(ca, d), PadTo(cb, d));
            return NumOps.Sqrt(CfEngine.DotProduct(diff, diff));
        }

        /// <summary>
        /// Gets the centroid of this entry.
        /// </summary>
        public T[] Centroid
        {
            get
            {
                if (N == 0 || LS.Length == 0) return LS;
                return CfEngine.Divide(Vector<T>.FromArray(LS), NumOps.FromDouble(N)).ToArray();
            }
        }

        /// <summary>
        /// Gets the radius of this entry.
        /// </summary>
        public T Radius
        {
            get
            {
                if (N <= 1) return NumOps.Zero;

                // Radius = sqrt((SS/N) - ||LS/N||²)
                var c = Vector<T>.FromArray(Centroid);
                T centroidNormSq = c.Length == 0 ? NumOps.Zero : CfEngine.DotProduct(c, c);

                T nT = NumOps.FromDouble(N);
                T variance = NumOps.Subtract(NumOps.Divide(SS, nT), centroidNormSq);
                return NumOps.GreaterThan(variance, NumOps.Zero) ? NumOps.Sqrt(variance) : NumOps.Zero;
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
        private readonly T _threshold;

        /// <summary>
        /// Initializes a new CFNode.
        /// </summary>
        public CFNode(int branchingFactor, T threshold, bool isLeaf)
        {
            _branchingFactor = branchingFactor;
            _threshold = threshold;
            IsLeaf = isLeaf;
        }
    }
}
