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
/// CURE (Clustering Using REpresentatives) hierarchical clustering algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CURE is an agglomerative hierarchical clustering algorithm that uses multiple
/// representative points per cluster to better capture non-spherical cluster shapes.
/// Representatives are shrunk toward the cluster center to reduce sensitivity to outliers.
/// </para>
/// <para>
/// Algorithm steps:
/// 1. Start with each point as its own cluster
/// 2. Select representative points for each cluster (well-scattered)
/// 3. Shrink representatives toward cluster center
/// 4. Find and merge the two clusters with closest representatives
/// 5. Repeat until desired number of clusters is reached
/// </para>
/// <para><b>For Beginners:</b> CURE finds clusters that aren't round:
///
/// Traditional clustering (like K-Means) assumes round clusters.
/// But real data often has:
/// - Banana-shaped clusters
/// - Spiral clusters
/// - Elongated clusters
///
/// CURE solves this by:
/// 1. Using multiple "marker" points per cluster, not just one center
/// 2. Placing these markers throughout the cluster
/// 3. Measuring cluster similarity by comparing markers
///
/// This way, two banana-shaped clusters can be recognized as separate,
/// even if their centers are close together!
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 1.5, 1.8 }, { 5.0, 8.0 }, { 8.0, 8.0 }, { 1.0, 0.6 }, { 9.0, 11.0 } });
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new CURE&lt;double&gt;(new CUREOptions&lt;double&gt;()))
///     .Build(dataMatrix);
///
/// // one cluster index per row
/// var assignments = result.Predict(dataMatrix);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("CURE: An Efficient Clustering Algorithm for Large Databases", "https://doi.org/10.1145/276304.276312", Year = 1998, Authors = "Sudipto Guha, Rajeev Rastogi, Kyuseok Shim")]
public partial class CURE<T> : ClusteringBase<T>
{
    private readonly CUREOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private readonly IDistanceMetric<T> _distanceMetric;
    private Random _random;
    private List<CureCluster>? _clusters;

    /// <summary>
    /// Initializes a new CURE instance.
    /// </summary>
    /// <param name="options">The CURE configuration options.</param>
    public CURE(CUREOptions<T>? options = null)
        : base(options ?? new CUREOptions<T>())
    {
        // Use the options passed to base constructor to avoid double instantiation
        _options = (CUREOptions<T>)Options;
        _distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSeededRandom(42);
        NumClusters = _options.NumClusters;
    }

    /// <inheritdoc />

    /// <inheritdoc />
    public override bool SupportsParameterInitialization => false;

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (CURE<T>)CreateNewInstance();
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

        // Sample data if needed
        Matrix<T> data;
        int[] sampleIndices;

        if (_options.SampleFraction < 1.0 && n > 100)
        {
            int sampleSize = Math.Max(_options.NumClusters, (int)(n * _options.SampleFraction));
            sampleIndices = Enumerable.Range(0, n).OrderBy(_ => _random.Next()).Take(sampleSize).ToArray();
            data = ExtractSubMatrix(x, sampleIndices);
        }
        else
        {
            sampleIndices = Enumerable.Range(0, n).ToArray();
            data = x;
        }

        int sampleN = data.Rows;

        // Validate NumClusters is within valid range
        if (_options.NumClusters < 1 || _options.NumClusters > sampleN)
        {
            throw new ArgumentException(
                $"NumClusters must be between 1 and {sampleN} (number of data points), got {_options.NumClusters}.");
        }

        // Initialize: each point is its own cluster
        _clusters = new List<CureCluster>();
        for (int i = 0; i < sampleN; i++)
        {
            _clusters.Add(new CureCluster
            {
                Points = new List<int> { i },
                Center = data.GetRow(i),
                Representatives = data.GetSubMatrix(i, 0, 1, d)
            });
        }

        // Pairwise cluster distances (minimum representative distance), kept in cluster-list order
        var clusterDist = _distanceMetric.ComputePairwise(data);

        // Agglomerative clustering with cached distances
        while (_clusters.Count > _options.NumClusters)
        {
            // Find closest pair (first in list order on ties)
            var (bestI, bestJ) = FindClosestClusters(clusterDist);

            if (bestI < 0 || bestJ < 0) break;

            // Merge
            var mergedCluster = MergeClusters(_clusters[bestI], _clusters[bestJ], data);

            // Remove old clusters (higher index first)
            int hi = Math.Max(bestI, bestJ), lo = Math.Min(bestI, bestJ);
            _clusters.RemoveAt(hi);
            _clusters.RemoveAt(lo);

            // Distances between the remaining clusters are unchanged; only the merged cluster needs new ones
            var remaining = Enumerable.Range(0, clusterDist.Rows).Where(i => i != lo && i != hi).ToArray();
            var toMerged = DistancesToCluster(mergedCluster, _clusters);
            _clusters.Add(mergedCluster);
            clusterDist = AppendCluster(clusterDist, remaining, toMerged);
        }

        NumClusters = _clusters.Count;

        // Assign labels to all original points
        var labels = new int[n];
        for (int i = 0; i < n; i++)
        {
            labels[i] = -1;
        }

        // Assign sampled points based on their cluster
        for (int clusterIdx = 0; clusterIdx < _clusters.Count; clusterIdx++)
        {
            foreach (int sampleIdx in _clusters[clusterIdx].Points)
            {
                labels[sampleIndices[sampleIdx]] = clusterIdx;
            }
        }

        // Assign non-sampled points to nearest cluster
        if (sampleIndices.Length < n)
        {
            var sampledSet = new HashSet<int>(sampleIndices);
            var unsampled = Enumerable.Range(0, n).Where(i => !sampledSet.Contains(i)).ToArray();
            var nearest = FindNearestClusters(ExtractSubMatrix(x, unsampled));
            for (int u = 0; u < unsampled.Length; u++)
            {
                labels[unsampled[u]] = nearest[u];
            }
        }

        Labels = ClusteringTensorOps.FromIntArray<T>(labels);

        // Compute cluster centers
        ComputeClusterCenters(x);

        MergeDegenerateClusters(x);
        // Rebuild _clusters to match post-merge state (Labels/NumClusters/ClusterCenters updated by merge)
        RebuildClustersFromLabels(x);
        IsTrained = true;
    }

    /// <summary>
    /// Distances in list order after removing two clusters and appending a new one: the rows/columns
    /// <paramref name="remaining"/> are kept and the new cluster's distances become the last row and column.
    /// </summary>
    private Matrix<T> AppendCluster(Matrix<T> clusterDist, int[] remaining, Vector<T> toNew)
    {
        int k = remaining.Length;
        var result = new Matrix<T>(k + 1, k + 1);
        if (k > 0)
        {
            var kept = ClusteringTensorOps.SelectColumns(
                Tensor<T>.FromMatrix(ClusteringTensorOps.SelectRows(clusterDist, remaining)), remaining).ToMatrix();
            result.SetSubMatrix(0, 0, kept);
        }

        var edge = Vector<T>.Concatenate(new[] { toNew, Engine.Fill(1, NumOps.MaxValue) });
        Engine.SetRow(result, k, edge);
        Engine.SetColumn(result, k, edge);
        return result;
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();
        ValidatePredictInput(x);

        // Return stored labels for in-sample prediction (preserves merge results)
        if (Labels is not null && ReferenceEquals(x, TrainingDataRef))
            return new Vector<T>(Labels);

        int n = x.Rows;

        if (_clusters is null || _clusters.Count == 0)
        {
            var unassigned = new Vector<T>(n);
            return n == 0 ? unassigned : Engine.TensorFullLike(unassigned.AsTensor(), NumOps.FromDouble(-1)).ToVector();
        }

        return ClusteringTensorOps.FromIntArray<T>(FindNearestClusters(x));
    }

    private (int, int) FindClosestClusters(Matrix<T> clusterDist)
    {
        // Pairs (i < j) in row-major order; the first strictly smallest distance below MaxValue wins
        int k = clusterDist.Rows;
        if (k < 2)
        {
            return (-1, -1);
        }

        var dist = Tensor<T>.FromMatrix(clusterDist);
        var upper = Engine.TensorTriu(Engine.TensorOnesLike(dist), 1);
        var candidates = Engine.Reshape(Engine.TensorWhere(upper, dist, Engine.TensorFullLike(dist, NumOps.MaxValue)), new[] { k * k }).ToVector();
        int best = ClusteringTensorOps.ArgMin(candidates);
        return NumOps.LessThan(candidates[best], NumOps.MaxValue) ? (best / k, best % k) : (-1, -1);
    }

    /// <summary>Stacks the representatives of the clusters and records the owning cluster of every row.</summary>
    private static Matrix<T> StackRepresentatives(List<CureCluster> clusters, int d, out int[] owners)
    {
        var ownerList = new List<int>();
        var blocks = new List<Matrix<T>>();
        for (int c = 0; c < clusters.Count; c++)
        {
            var reps = clusters[c].Representatives;
            blocks.Add(reps);
            ownerList.AddRange(Enumerable.Repeat(c, reps.Rows));
        }

        owners = ownerList.ToArray();
        var stacked = new Matrix<T>(owners.Length, d);
        int row = 0;
        foreach (var block in blocks)
        {
            if (block.Rows > 0)
            {
                stacked.SetSubMatrix(row, 0, block);
                row += block.Rows;
            }
        }

        return stacked;
    }

    /// <summary>
    /// Minimum representative distance from <paramref name="cluster"/> to each of <paramref name="others"/>
    /// (MaxValue when either side has no representatives).
    /// </summary>
    private Vector<T> DistancesToCluster(CureCluster cluster, List<CureCluster> others)
    {
        int k = others.Count;
        int d = cluster.Representatives.Columns;
        var stacked = StackRepresentatives(others, d, out var owners);
        if (k == 0 || owners.Length == 0 || cluster.Representatives.Rows == 0)
        {
            return k == 0 ? new Vector<T>(0) : Engine.Fill(k, NumOps.MaxValue);
        }

        // Closest representative of the cluster for every stacked representative, then the minimum per owner
        var pairwise = Tensor<T>.FromMatrix(_distanceMetric.ComputePairwise(cluster.Representatives, stacked));
        var closest = Engine.TensorAmin(pairwise, new[] { 0 });
        var ownership = Engine.TensorTranspose(ClusteringTensorOps.OneHotMask(ClusteringTensorOps.FromIntArray<T>(owners), k));
        var perOwner = Engine.TensorWhere(
            ownership,
            Engine.TensorBroadcastTo(Engine.Reshape(closest, new[] { 1, owners.Length }), new[] { k, owners.Length }),
            Engine.TensorFullLike(ownership, NumOps.MaxValue));
        return Engine.TensorAmin(perOwner, new[] { 1 }).ToVector();
    }

    private CureCluster MergeClusters(CureCluster c1, CureCluster c2, Matrix<T> data)
    {
        int d = data.Columns;

        // Combine points
        var mergedPoints = new List<int>(c1.Points);
        mergedPoints.AddRange(c2.Points);

        // Compute new center
        var members = ExtractSubMatrix(data, mergedPoints.ToArray());
        var center = ClusteringTensorOps.ColumnMeans(members);

        // Select representatives using farthest-point heuristic
        var representatives = SelectRepresentatives(members, center);

        // Shrink representatives toward center: rep + s (center - rep)
        if (representatives.Rows > 0)
        {
            T shrinkFactor = NumOps.FromDouble(_options.ShrinkFactor);
            var reps = Tensor<T>.FromMatrix(representatives);
            var toCenter = Engine.TensorSubtract(ClusteringTensorOps.TileRow(center, representatives.Rows), reps);
            representatives = Engine.TensorAdd(reps, Engine.TensorMultiplyScalar(toCenter, shrinkFactor)).ToMatrix();
        }

        return new CureCluster
        {
            Points = mergedPoints,
            Center = center,
            Representatives = representatives
        };
    }

    private Matrix<T> SelectRepresentatives(Matrix<T> members, Vector<T> center)
    {
        int m = members.Rows;
        int numReps = Math.Min(_options.NumRepresentatives, m);
        if (numReps <= 0)
        {
            return new Matrix<T>(0, members.Columns);
        }

        // Start with point farthest from center (first on ties)
        var chosen = new List<int> { ClusteringTensorOps.ArgMax(ClusteringTensorOps.DistancesToPoint(_distanceMetric, members, center)) };

        // Add more representatives using farthest-point heuristic: the point farthest from its nearest
        // representative, skipping points that coincide (closer than 1e-10) with a representative
        T epsilon = NumOps.FromDouble(1e-10);
        var minToReps = ClusteringTensorOps.DistancesToPoint(_distanceMetric, members, members.GetRow(chosen[0])).AsTensor();
        var isRep = Engine.TensorLessThan(minToReps, epsilon);
        T minusOne = NumOps.Negate(NumOps.One);
        while (chosen.Count < numReps)
        {
            var candidates = Engine.TensorWhere(isRep, Engine.TensorFullLike(minToReps, minusOne), minToReps).ToVector();
            int next = ClusteringTensorOps.ArgMax(candidates);
            if (!IsCandidate(candidates, next))
            {
                break; // No more points to add
            }

            chosen.Add(next);
            var toNext = ClusteringTensorOps.DistancesToPoint(_distanceMetric, members, members.GetRow(next)).AsTensor();
            minToReps = Engine.TensorMin(minToReps, toNext);
            isRep = Engine.TensorMax(isRep, Engine.TensorLessThan(toNext, epsilon));
        }

        return ClusteringTensorOps.SelectRows(members, chosen.ToArray());
    }

    /// <summary>True when the selected entry is a real candidate (non-representatives are marked by a distance above -1).</summary>
    private bool IsCandidate(Vector<T> candidates, int index) => NumOps.GreaterThan(candidates[index], NumOps.Negate(NumOps.One));

    /// <summary>
    /// Rebuilds the internal _clusters list from the current Labels after degenerate merge.
    /// </summary>
    private void RebuildClustersFromLabels(Matrix<T> x)
    {
        if (Labels is null || _clusters is null) return;

        int d = x.Columns;
        var labelIds = ClusteringTensorOps.ToIntArray(Labels);
        var centroids = NumClusters > 0
            ? ClusteringTensorOps.ClusterMeans(x, Labels, NumClusters, out _)
            : new Matrix<T>(0, d);
        var newClusters = new List<CureCluster>();

        for (int c = 0; c < NumClusters; c++)
        {
            var cluster = new CureCluster
            {
                Points = new List<int>(),
                Representatives = new Matrix<T>(0, d)
            };

            for (int i = 0; i < labelIds.Length; i++)
            {
                if (labelIds[i] == c)
                {
                    cluster.Points.Add(i);
                }
            }

            if (cluster.Points.Count > 0)
            {
                // Centroid as representative point
                cluster.Center = centroids.GetRow(c);
                cluster.Representatives = centroids.GetSubMatrix(c, 0, 1, d);
            }

            newClusters.Add(cluster);
        }

        _clusters = newClusters;
    }

    private int[] FindNearestClusters(Matrix<T> points)
    {
        // Cluster owning the closest representative (first in cluster/representative order); 0 when there is none
        var clusters = _clusters ?? throw new InvalidOperationException("CURE: Clusters not initialized.");
        var stacked = StackRepresentatives(clusters, points.Columns, out var owners);
        var nearest = new int[points.Rows];
        if (owners.Length == 0 || points.Rows == 0)
        {
            return nearest;
        }

        var closestRep = ClusteringTensorOps.ArgMinRows(_distanceMetric.ComputePairwise(points, stacked));
        for (int i = 0; i < nearest.Length; i++)
        {
            nearest[i] = owners[closestRep[i]];
        }

        return nearest;
    }

    private void ComputeClusterCenters(Matrix<T> x)
    {
        if (NumClusters == 0)
        {
            ClusterCenters = null;
            return;
        }

        // Mean of each cluster's members (unassigned points are ignored; empty clusters stay at zero)
        var labels = Labels ?? throw new InvalidOperationException("Labels have not been computed.");
        ClusterCenters = ClusteringTensorOps.ClusterMeans(x, labels, NumClusters, out _);
    }

    private Matrix<T> ExtractSubMatrix(Matrix<T> data, int[] indices) => ClusteringTensorOps.SelectRows(data, indices);

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

    private class CureCluster
    {
        public List<int> Points { get; set; } = new List<int>();
        public Vector<T> Center { get; set; } = new Vector<T>(0);
        public Matrix<T> Representatives { get; set; } = new Matrix<T>(0, 0);
    }
}
