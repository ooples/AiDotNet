using AiDotNet.Attributes;
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
/// <example>
/// <code>
/// var labels = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
/// var newData = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// // Use AiModelBuilder facade for subspace clustering
/// var builder = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new SUBCLU&lt;double&gt;(new SUBCLUOptions&lt;double&gt;()));
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
[ResearchPaper("Density-Connected Subspace Clustering for High-Dimensional Data", "https://doi.org/10.1137/1.9781611972740.23", Year = 2004, Authors = "Karin Kailing, Hans-Peter Kriegel, Peer Kroger")]
public partial class SUBCLU<T> : ClusteringBase<T>
{
    private readonly SUBCLUOptions<T> _options;
    private Vector<T>? _normMeans;
    private Vector<T>? _normStds;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private List<SubspaceClusterInfo>? _subspaceClusterInfos;
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T>? _trainingData;

    /// <summary>
    /// Initializes a new SUBCLU instance.
    /// </summary>
    /// <param name="options">The SUBCLU configuration options.</param>
    public SUBCLU(SUBCLUOptions<T>? options = null)
        : base(options ??= new SUBCLUOptions<T>())
    {
        _options = options;
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

        // Normalize features to zero-mean unit-variance for scale-invariant epsilon.
        // Without normalization, epsilon is scale-dependent.
        var xNorm = ClusteringTensorOps.StandardizeColumns(x, out var featureMeans, out var featureStds);

        // Store normalization parameters for prediction consistency
        _normMeans = featureMeans;
        _normStds = featureStds;
        x = xNorm;

        // Store the normalized training data so Predict distance comparisons
        // use the same scale as the clustered data
        _trainingData = xNorm.Clone();

        _subspaceClusterInfos = new List<SubspaceClusterInfo>();

        // Find 1-D dense clusters (with their core points)
        var oneDClusters = new Dictionary<int, List<HashSet<int>>>();
        var oneDCorePoints = new Dictionary<int, List<HashSet<int>>>();

        for (int dim = 0; dim < d; dim++)
        {
            var within = WithinEpsilon(x, new[] { dim });
            var clusters = RunDBSCAN1D(x, dim, within);
            if (clusters.Count > 0)
            {
                oneDClusters[dim] = clusters;
                oneDCorePoints[dim] = clusters.Select(cluster => IdentifyCorePoints(within, n, cluster)).ToList();
            }
        }

        // Add 1-D clusters to results
        foreach (var kvp in oneDClusters)
        {
            int dim = kvp.Key;
            var cores = oneDCorePoints[dim];
            for (int c = 0; c < kvp.Value.Count; c++)
            {
                _subspaceClusterInfos.Add(new SubspaceClusterInfo
                {
                    ClusterId = _subspaceClusterInfos.Count,
                    Dimensions = new[] { dim },
                    Points = kvp.Value[c],
                    CorePoints = cores[c]
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
                            var dimsArray = merged.OrderBy(dd => dd).ToArray();
                            var within = WithinEpsilon(x, dimsArray);
                            var clusters = RunDBSCAN(within, n);
                            if (clusters.Count > 0)
                            {
                                nextDimClusters[merged] = clusters;

                                // Add to results
                                foreach (var cluster in clusters)
                                {
                                    _subspaceClusterInfos.Add(new SubspaceClusterInfo
                                    {
                                        ClusterId = _subspaceClusterInfos.Count,
                                        Dimensions = dimsArray,
                                        Points = cluster,
                                        CorePoints = IdentifyCorePoints(within, n, cluster)
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
        var labelIds = AssignLabels(n, _subspaceClusterInfos);
        Labels = ClusteringTensorOps.FromIntArray<T>(labelIds);

        // Count unique clusters
        NumClusters = labelIds.Where(label => label >= 0).Distinct().Count();

        // Compute cluster centers
        ComputeClusterCenters(x);

        IsTrained = true;
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();
        ValidatePredictInput(x);

        // Normalize prediction input using the same parameters as training
        if (_normMeans is not null && _normStds is not null && !ReferenceEquals(x, TrainingDataRef))
        {
            x = ClusteringTensorOps.Standardize(x, _normMeans, _normStds);
        }

        if (_subspaceClusterInfos is null || _subspaceClusterInfos.Count == 0 || _trainingData is null)
        {
            return Engine.Fill(x.Rows, NumOps.FromDouble(-1));
        }

        return AssignLabelsForNewData(x);
    }

    private HashSet<int> IdentifyCorePoints(bool[] within, int n, HashSet<int> clusterPoints)
    {
        var corePoints = new HashSet<int>();

        foreach (int pointIdx in clusterPoints)
        {
            int neighborCount = 0;
            foreach (int otherIdx in clusterPoints)
            {
                if (within[pointIdx * n + otherIdx])
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

    /// <summary>
    /// Row-major [n * n] flags: true when the Euclidean distance between two points, measured over the given
    /// dimensions only, is at most epsilon.
    /// </summary>
    private bool[] WithinEpsilon(Matrix<T> x, int[] dims)
    {
        var subspace = ClusteringTensorOps.SelectColumns(Tensor<T>.FromMatrix(x), dims).ToMatrix();
        var distances = ClusteringTensorOps.Distances(subspace, subspace);
        return ClusteringTensorOps.LessOrEqualFlags(ClusteringTensorOps.Flatten(distances), NumOps.FromDouble(_options.Epsilon));
    }

    private List<HashSet<int>> RunDBSCAN1D(Matrix<T> x, int dim, bool[] within)
    {
        int n = x.Rows;

        // Visit points in order of value
        var order = Engine.TensorArgsort(Tensor<T>.FromVector(x.GetColumn(dim), new[] { n }), 0, false).ToArray();

        var visited = new bool[n];
        var clusters = new List<HashSet<int>>();

        foreach (int idx in order)
        {
            if (visited[idx]) continue;

            // Find neighbors within epsilon
            var neighbors = GetNeighbors(within, n, idx);

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
                    var currentNeighbors = GetNeighbors(within, n, current);

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

    private List<HashSet<int>> RunDBSCAN(bool[] within, int n)
    {
        var visited = new bool[n];
        var clusters = new List<HashSet<int>>();

        for (int i = 0; i < n; i++)
        {
            if (visited[i]) continue;

            var neighbors = GetNeighbors(within, n, i);

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
                    var currentNeighbors = GetNeighbors(within, n, current);

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

    private static List<int> GetNeighbors(bool[] within, int n, int pointIdx)
    {
        var neighbors = new List<int>();
        int offset = pointIdx * n;

        for (int i = 0; i < n; i++)
        {
            if (within[offset + i])
            {
                neighbors.Add(i);
            }
        }

        return neighbors;
    }

    private int[] AssignLabels(int n, List<SubspaceClusterInfo> clusters)
    {
        var labels = new int[n];

        for (int i = 0; i < n; i++)
        {
            labels[i] = -1;
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
                    labels[i] = cluster.ClusterId;
                    break;
                }
            }
        }

        return labels;
    }

    private Vector<T> AssignLabelsForNewData(Matrix<T> x)
    {
        int n = x.Rows;
        var trainingData = _trainingData ?? throw new InvalidOperationException("_trainingData is null where it is required.");

        // Initialize all as noise
        var labels = new int[n];
        for (int i = 0; i < n; i++)
        {
            labels[i] = -1;
        }

        // Sort clusters by dimensionality (prefer higher dimensional matches)
        var clusterInfos = _subspaceClusterInfos ?? throw new InvalidOperationException("_subspaceClusterInfos has not been initialized.");
        var sortedClusters = clusterInfos
            .OrderByDescending(c => c.Dimensions.Length)
            .ToList();

        // A new point is assigned to the first (highest-dimensional) cluster with a core point
        // within epsilon in that cluster's subspace
        foreach (var cluster in sortedClusters)
        {
            if (cluster.CorePoints.Count == 0)
            {
                continue;
            }

            var reachable = ReachableFromCorePoints(x, trainingData, cluster);
            for (int i = 0; i < n; i++)
            {
                if (labels[i] < 0 && reachable[i])
                {
                    labels[i] = cluster.ClusterId;
                }
            }
        }

        return ClusteringTensorOps.FromIntArray<T>(labels);
    }

    /// <summary>For each row of x, whether some core point of the cluster lies within epsilon in its subspace.</summary>
    private bool[] ReachableFromCorePoints(Matrix<T> x, Matrix<T> trainingData, SubspaceClusterInfo cluster)
    {
        var points = ClusteringTensorOps.SelectColumns(Tensor<T>.FromMatrix(x), cluster.Dimensions).ToMatrix();
        var cores = ClusteringTensorOps.SelectColumns(
            Tensor<T>.FromMatrix(ClusteringTensorOps.SelectRows(trainingData, cluster.CorePoints.ToArray())), cluster.Dimensions).ToMatrix();
        var nearest = ClusteringTensorOps.MinRows(ClusteringTensorOps.Distances(points, cores));
        return ClusteringTensorOps.LessOrEqualFlags(nearest, NumOps.FromDouble(_options.Epsilon));
    }

    private void ComputeClusterCenters(Matrix<T> x)
    {
        if (NumClusters == 0)
        {
            ClusterCenters = null;
            return;
        }

        // Mean of each cluster's members (labels outside [0, NumClusters) are ignored, empty clusters stay zero)
        ClusterCenters = ClusteringTensorOps.ClusterMeans(
            x, Labels ?? throw new InvalidOperationException("Labels has not been initialized."), NumClusters, out _);
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
