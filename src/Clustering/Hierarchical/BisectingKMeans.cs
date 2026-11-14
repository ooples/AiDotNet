using AiDotNet.Attributes;
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
/// <example>
/// <code>
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 1.5, 1.8 }, { 5.0, 8.0 }, { 8.0, 8.0 }, { 1.0, 0.6 }, { 9.0, 11.0 } });
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new BisectingKMeans&lt;double&gt;(new BisectingKMeansOptions&lt;double&gt;()))
///     .Build(dataMatrix);
///
/// // the cluster each training row was assigned to (Predict places rows the model has not seen)
/// var labels = result.GetClusterLabels();
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("A Comparison of Document Clustering Techniques", "https://www.cs.cmu.edu/~dunja/KDDpapers/Steinbach_IR.pdf")]
public partial class BisectingKMeans<T> : ClusteringBase<T>
{
    private readonly BisectingKMeansOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private readonly IDistanceMetric<T> _distanceMetric;
    private Random _random;
    private List<BisectionNode>? _hierarchy;

    /// <summary>
    /// Initializes a new BisectingKMeans instance with the specified options.
    /// </summary>
    /// <param name="options">The Bisecting K-Means configuration options.</param>
    public BisectingKMeans(BisectingKMeansOptions<T>? options = null)
        : base(options ??= new BisectingKMeansOptions<T>())
    {
        _options = options;
        _distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSeededRandom(42);
        NumClusters = _options.NumClusters;
    }

    /// <summary>
    /// Gets the bisection hierarchy if BuildHierarchy was enabled.
    /// </summary>
    public IReadOnlyList<BisectionNode>? Hierarchy => _hierarchy?.AsReadOnly();

    /// <inheritdoc />

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
            Labels = Engine.TensorArange<T>(0, n, 1).ToVector();
            ClusterCenters = x.Clone();
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
            _hierarchy = new List<BisectionNode>
            {
                new BisectionNode
                {
                    ClusterId = 0,
                    ParentId = -1,
                    Size = n,
                    Inertia = InertiaValue(clusterInfos[0].Inertia)
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
                _hierarchy.Add(new BisectionNode
                {
                    ClusterId = rightClusterId,
                    ParentId = leftClusterId,
                    Size = originalRightIndices.Count,
                    Inertia = InertiaValue(clusterInfos[clusterInfos.Count - 1].Inertia)
                });
            }

            currentNumClusters++;
        }

        // Renumber clusters to be 0, 1, 2, ... k-1 (in order of first appearance)
        var clusterMapping = new Dictionary<int, int>();
        var finalLabels = new int[n];
        for (int i = 0; i < n; i++)
        {
            if (!clusterMapping.TryGetValue(clusterAssignments[i], out int mapped))
            {
                mapped = clusterMapping.Count;
                clusterMapping[clusterAssignments[i]] = mapped;
            }

            finalLabels[i] = mapped;
        }

        Labels = ClusteringTensorOps.FromIntArray<T>(finalLabels);

        // Build final cluster centers (every renumbered cluster has members)
        NumClusters = clusterMapping.Count;
        ClusterCenters = ClusteringTensorOps.ClusterMeans(x, Labels, NumClusters, out _);

        // Compute final inertia
        Inertia = ComputeInertia(x, Labels, ClusterCenters);

        // Merge degenerate clusters (single natural cluster split into K)
        MergeDegenerateClusters(x);

        IsTrained = true;
    }

    /// <summary>Inertia as the double stored on hierarchy nodes (0 when unknown).</summary>
    private double InertiaValue(T? inertia) => inertia is T value ? NumOps.ToDouble(value) : 0;

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();
        ValidatePredictInput(x);

        if (ClusterCenters is null || NumClusters == 0)
        {
            return new Vector<T>(x.Rows);
        }

        var centers = ClusterCenters.Rows == NumClusters
            ? ClusterCenters
            : ClusterCenters.GetSubMatrix(0, 0, NumClusters, ClusterCenters.Columns);
        return ClusteringTensorOps.FromIntArray<T>(ClusteringTensorOps.NearestCenters(_distanceMetric, x, centers));
    }

    private int SelectClusterToBisect(List<ClusterInfo> clusterInfos)
    {
        int bestIdx = -1;
        T bestScore = NumOps.Negate(NumOps.MaxValue);

        for (int i = 0; i < clusterInfos.Count; i++)
        {
            var info = clusterInfos[i];
            if (!info.CanSplit || info.Indices.Count < _options.MinClusterSizeForBisection)
            {
                continue;
            }

            T score = SelectionScore(info);
            if (IsHigher(score, bestScore))
            {
                bestScore = score;
                bestIdx = i;
            }
        }

        return bestIdx;
    }

    /// <summary>The bisection priority of a cluster under the configured selection rule.</summary>
    private T SelectionScore(ClusterInfo info) => _options.ClusterSelection switch
    {
        BisectionClusterSelection.Largest => NumOps.FromDouble(info.Indices.Count),
        BisectionClusterSelection.HighestInertia => info.Inertia ?? NumOps.Zero,
        BisectionClusterSelection.LargestDiameter => info.Diameter,
        _ => NumOps.FromDouble(info.Indices.Count)
    };

    /// <summary>Scalar comparison of two scores.</summary>
    private bool IsHigher(T candidate, T reference) => NumOps.GreaterThan(candidate, reference);

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
        T bestInertia = NumOps.MaxValue;

        // Try multiple bisection attempts
        for (int trial = 0; trial < _options.NumBisectionTrials; trial++)
        {
            // Initialize two centers randomly
            int idx1 = _random.Next(n);
            int idx2;
            do
            {
                idx2 = _random.Next(n);
            } while (idx2 == idx1 && n > 1);

            // Run K-Means with k=2
            var (leftIndices, rightIndices, finalCenter1, finalCenter2, inertia) =
                RunBinaryKMeans(subData, subData.GetRow(idx1), subData.GetRow(idx2));

            if (IsLower(inertia, bestInertia) && leftIndices.Count > 0 && rightIndices.Count > 0)
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

    /// <summary>Scalar comparison of two inertias.</summary>
    private bool IsLower(T candidate, T reference) => NumOps.LessThan(candidate, reference);

    private (List<int> LeftIndices, List<int> RightIndices, Vector<T> Center1, Vector<T> Center2, T Inertia)
    RunBinaryKMeans(Matrix<T> data, Vector<T> center1, Vector<T> center2)
    {
        int n = data.Rows;

        var centers = Matrix<T>.FromRows(new IEnumerable<T>[] { center1, center2 });
        var assignments = new int[n];
        var prevAssignments = new int[n];
        for (int i = 0; i < n; i++)
        {
            prevAssignments[i] = -1;
        }

        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            // Assignment step (ties go to the first center)
            assignments = ClusteringTensorOps.NearestCenters(_distanceMetric, data, centers);
            bool changed = false;
            for (int i = 0; i < n; i++)
            {
                if (assignments[i] != prevAssignments[i])
                {
                    changed = true;
                    break;
                }
            }

            if (!changed)
            {
                break;
            }

            Array.Copy(assignments, prevAssignments, n);

            // Update step (an empty side keeps its center)
            var means = ClusteringTensorOps.ClusterMeans(data, ClusteringTensorOps.FromIntArray<T>(assignments), 2, out var counts);
            centers = ClusteringTensorOps.RowsWherePositive(counts, means, centers);
        }

        // Build result lists
        var leftIndices = new List<int>();
        var rightIndices = new List<int>();
        for (int i = 0; i < n; i++)
        {
            if (assignments[i] == 0)
            {
                leftIndices.Add(i);
            }
            else
            {
                rightIndices.Add(i);
            }
        }

        var distances = ClusteringTensorOps.AssignedDistances(
            _distanceMetric, data, ClusteringTensorOps.FromIntArray<T>(assignments), centers);
        T totalInertia = Engine.DotProduct(distances, distances);

        return (leftIndices, rightIndices, centers.GetRow(0), centers.GetRow(1), totalInertia);
    }

    private ClusterInfo CreateClusterInfo(Matrix<T> x, int[] assignments, int clusterId, List<int> indices)
    {
        int d = x.Columns;
        var center = new Vector<T>(d);

        // Compute inertia and diameter
        T inertia = NumOps.Zero;
        T diameter = NumOps.Zero;

        if (indices.Count > 0)
        {
            var members = ExtractSubMatrix(x, indices);
            center = ClusteringTensorOps.ColumnMeans(members);
            var distances = ClusteringTensorOps.DistancesToPoint(_distanceMetric, members, center);
            inertia = Engine.DotProduct(distances, distances);

            // Approximate diameter using center + max distance from center * 2
            if (indices.Count > 1)
            {
                diameter = NumOps.Multiply(ClusteringTensorOps.MaxOrZero(distances), NumOps.FromDouble(2.0));
            }
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
        return indices.Count == 0
            ? new Matrix<T>(0, x.Columns)
            : ClusteringTensorOps.SelectRows(x, indices.ToArray());
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
        public T Diameter { get; set; } = MathHelper.GetNumericOperations<T>().Zero;
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
