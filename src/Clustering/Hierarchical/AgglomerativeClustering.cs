using AiDotNet.Attributes;
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
/// <example>
/// <code>
/// var labels = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
/// var newData = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// // Use AiModelBuilder facade for agglomerative clustering
/// var builder = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new AgglomerativeClustering&lt;double&gt;(
///         new HierarchicalOptions&lt;double&gt;()));
///
/// var result = builder.Build(dataMatrix, labels);
/// var predictions = result.Predict(newData);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Hierarchical Grouping to Optimize an Objective Function", "https://doi.org/10.1080/01621459.1963.10500845")]
public partial class AgglomerativeClustering<T> : ClusteringBase<T>
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
        : base(options ??= new HierarchicalOptions<T>())
    {
        _options = options;

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

        // Track cluster sizes
        var clusterSizes = new int[2 * n - 1];
        for (int i = 0; i < n; i++)
        {
            clusterSizes[i] = 1;
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
            var (minI, minJ, minDist) = FindClosestPair(distMatrix, active);

            if (minI < 0 || minJ < 0)
            {
                break; // No more pairs to merge
            }

            // Check distance threshold
            if (ExceedsDistanceThreshold(minDist))
            {
                break;
            }

            // Record merge
            int newSize = clusterSizes[minI] + clusterSizes[minJ];
            _dendrogram.Add((minI, minJ, minDist, newSize));
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

        // Merge degenerate clusters (handles single natural cluster split into K)
        MergeDegenerateClusters(x);

        IsTrained = true;
    }

    /// <summary>True when a distance threshold is configured and the merge distance exceeds it.</summary>
    private bool ExceedsDistanceThreshold(T distance)
        => _options.DistanceThreshold.HasValue &&
           NumOps.GreaterThan(distance, NumOps.FromDouble(_options.DistanceThreshold.Value));

    private Matrix<T> ComputeDistanceMatrix(Matrix<T> x)
    {
        int n = x.Rows;
        int size = 2 * n - 1;
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        // Pairwise distances (squared for Ward) between the points; entries for merged clusters start at infinity
        var pairwise = Tensor<T>.FromMatrix(metric.ComputePairwise(x));
        if (_options.Linkage == LinkageMethod.Ward)
        {
            pairwise = Engine.TensorMultiply(pairwise, pairwise);
        }

        var distMatrix = Engine.TensorFullLike(Tensor<T>.CreateZeros(new[] { size, size }), NumOps.MaxValue).ToMatrix();
        distMatrix.SetSubMatrix(0, 0, pairwise.ToMatrix());
        return distMatrix;
    }

    private (int I, int J, T Dist) FindClosestPair(Matrix<T> distMatrix, HashSet<int> active)
    {
        // Scan pairs (ai < aj) in active-set order; the first strictly smallest distance wins
        var activeList = active.ToArray();
        int a = activeList.Length;
        if (a < 2)
        {
            return (-1, -1, NumOps.MaxValue);
        }

        var sub = ClusteringTensorOps.SelectColumns(Tensor<T>.FromMatrix(ClusteringTensorOps.SelectRows(distMatrix, activeList)), activeList);
        var upper = Engine.TensorTriu(Engine.TensorOnesLike(sub), 1);
        var candidates = Engine.Reshape(Engine.TensorWhere(upper, sub, Engine.TensorFullLike(sub, NumOps.MaxValue)), new[] { a * a }).ToVector();
        int best = ClusteringTensorOps.ArgMin(candidates);
        T minDist = candidates[best];
        if (!NumOps.LessThan(minDist, NumOps.MaxValue))
        {
            return (-1, -1, NumOps.MaxValue);
        }

        return (activeList[best / a], activeList[best % a], minDist);
    }

    private void UpdateDistanceMatrix(
        Matrix<T> distMatrix,
        HashSet<int> active,
        int mergedI,
        int mergedJ,
        int newCluster,
        int[] sizes)
    {
        int size = distMatrix.Rows;

        // Lance-Williams update against every other active cluster k
        var others = new int[size];
        foreach (int k in active)
        {
            if (k != mergedI && k != mergedJ)
            {
                others[k] = 1;
            }
        }

        var newRow = ComputeLinkageDistances(
            distMatrix.GetRow(mergedI).AsTensor(),
            distMatrix.GetRow(mergedJ).AsTensor(),
            distMatrix[mergedI, mergedJ],
            NumOps.FromDouble(sizes[mergedI]),
            NumOps.FromDouble(sizes[mergedJ]),
            ClusteringTensorOps.FromIntArray<T>(sizes).AsTensor());

        var mask = ClusteringTensorOps.FromIntArray<T>(others).AsTensor();
        var updated = Engine.TensorWhere(mask, newRow, distMatrix.GetRow(newCluster).AsTensor()).ToVector();
        Engine.SetRow(distMatrix, newCluster, updated);

        var column = Engine.TensorWhere(mask, newRow, distMatrix.GetColumn(newCluster).AsTensor()).ToVector();
        Engine.SetColumn(distMatrix, newCluster, column);
    }

    private Tensor<T> ComputeLinkageDistances(
        Tensor<T> dIK, Tensor<T> dJK, T dIJ,
        T sizeI, T sizeJ, Tensor<T> sizeK)
    {
        T sizeIJ = NumOps.Add(sizeI, sizeJ);
        Tensor<T> SizeWeighted() => Engine.TensorDivideScalar(
            Engine.TensorAdd(Engine.TensorMultiplyScalar(dIK, sizeI), Engine.TensorMultiplyScalar(dJK, sizeJ)), sizeIJ);
        Tensor<T> HalfSum() => Engine.TensorMultiplyScalar(Engine.TensorAdd(dIK, dJK), NumOps.FromDouble(0.5));

        switch (_options.Linkage)
        {
            case LinkageMethod.Complete:
                return Engine.TensorMax(dIK, dJK);

            case LinkageMethod.Average:
                return SizeWeighted();

            case LinkageMethod.Weighted:
                return HalfSum();

            case LinkageMethod.Ward:
            {
                // ((sI + sK) dIK + (sJ + sK) dJK - sK dIJ) / (sI + sJ + sK)
                var numerator = Engine.TensorSubtract(
                    Engine.TensorAdd(
                        Engine.TensorMultiply(Engine.TensorAddScalar(sizeK, sizeI), dIK),
                        Engine.TensorMultiply(Engine.TensorAddScalar(sizeK, sizeJ), dJK)),
                    Engine.TensorMultiplyScalar(sizeK, dIJ));
                return Engine.TensorDivide(numerator, Engine.TensorAddScalar(sizeK, sizeIJ));
            }

            case LinkageMethod.Centroid:
                return Engine.TensorSubtractScalar(
                    SizeWeighted(),
                    NumOps.Divide(NumOps.Multiply(NumOps.Multiply(sizeI, sizeJ), dIJ), NumOps.Multiply(sizeIJ, sizeIJ)));

            case LinkageMethod.Median:
                return Engine.TensorSubtractScalar(HalfSum(), NumOps.Divide(dIJ, NumOps.FromDouble(4.0)));

            default:
                // Single linkage
                return Engine.TensorMin(dIK, dJK);
        }
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
        foreach (var (c1, c2, dist, size) in (_dendrogram ?? throw new InvalidOperationException("_dendrogram is null where it is required.")))
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

        return ClusteringTensorOps.FromIntArray<T>(labels);
    }

    private void ComputeClusterCenters(Matrix<T> x)
    {
        if (Labels is null || NumClusters <= 0)
        {
            return;
        }

        // Mean of each cluster's members (empty clusters stay at zero)
        ClusterCenters = ClusteringTensorOps.ClusterMeans(x, Labels, NumClusters, out _);
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

        return ClusteringTensorOps.FromIntArray<T>(labels);
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
