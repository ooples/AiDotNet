using AiDotNet.Tensors.Engines;
using AiDotNet.Clustering.Base;
using AiDotNet.Attributes;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Enums;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Clustering.Evaluation;

/// <summary>
/// Silhouette Score for evaluating cluster quality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Silhouette Score measures how similar a point is to its own cluster
/// compared to other clusters. Values range from -1 to 1:
/// - Near +1: Points are well-matched to their cluster
/// - Near 0: Points are on cluster boundaries
/// - Near -1: Points may be assigned to wrong clusters
/// </para>
/// <para>
/// For each point i:
/// - a(i) = mean distance to other points in same cluster
/// - b(i) = mean distance to points in nearest other cluster
/// - s(i) = (b(i) - a(i)) / max(a(i), b(i))
/// </para>
/// <para><b>For Beginners:</b> Silhouette Score asks "Is each point in the right cluster?"
///
/// For each point, we measure:
/// 1. How close it is to its cluster-mates (a)
/// 2. How far it is from the nearest other cluster (b)
///
/// If b >> a: Point is clearly in the right place (+1)
/// If b ≈ a: Point is between clusters (0)
/// If b &lt;&lt; a: Point might be in wrong cluster (-1)
///
/// The overall score is the average of all points.
/// Higher is better!
/// </para>
/// </remarks>
[ComponentType(ComponentType.Evaluator)]
[PipelineStage(PipelineStage.Evaluation)]
public class SilhouetteScore<T> : IClusterMetric<T>
{
    private readonly IDistanceMetric<T>? _distanceMetric;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new SilhouetteScore instance.
    /// </summary>
    /// <param name="distanceMetric">Distance metric to use, or null for Euclidean.</param>
    public SilhouetteScore(IDistanceMetric<T>? distanceMetric = null)
    {
        _distanceMetric = distanceMetric;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc />
    public string Name => "Silhouette Score";

    /// <inheritdoc />
    public double Compute(Matrix<T> data, Vector<T> labels)
    {
        int n = data.Rows;

        if (n < 2)
        {
            return 0;
        }

        var metric = _distanceMetric ?? new EuclideanDistance<T>();

        // Get unique labels (excluding -1 for noise)
        var uniqueLabels = new HashSet<int>();
        var labelsIds = ClusteringTensorOps.ToIntArray(labels);
        for (int i = 0; i < n; i++)
        {
            int label = labelsIds[i];
            if (label >= 0)
            {
                uniqueLabels.Add(label);
            }
        }

        if (uniqueLabels.Count < 2)
        {
            return 0; // Need at least 2 clusters
        }

        // Per-sample silhouette scores (zero for noise points and singleton clusters)
        var compact = ClusteringTensorOps.CompactLabels<T>(labelsIds, out var slotLabels);
        var silhouetteScores = SampleScores(data, compact, slotLabels.Length, metric);

        // Average over non-noise points
        int validPoints = labelsIds.Count(label => label >= 0);
        double sum = _numOps.ToDouble(AiDotNetEngine.Current.TensorSum(silhouetteScores.AsTensor()));

        return validPoints > 0 ? sum / validPoints : 0;
    }

    /// <summary>
    /// Computes the silhouette score for every sample.
    /// </summary>
    /// <param name="data">The data matrix.</param>
    /// <param name="labels">The cluster labels.</param>
    /// <returns>The per-sample silhouette scores (zero for noise points and singleton clusters).</returns>
    public Vector<T> ComputeSampleScores(Matrix<T> data, Vector<T> labels)
    {
        int n = data.Rows;
        var metric = _distanceMetric ?? new EuclideanDistance<T>();

        var uniqueLabels = new HashSet<int>();
        var labelsIds = ClusteringTensorOps.ToIntArray(labels);
        for (int i = 0; i < n; i++)
        {
            int label = labelsIds[i];
            if (label >= 0)
            {
                uniqueLabels.Add(label);
            }
        }

        if (uniqueLabels.Count < 2)
        {
            return new Vector<T>(n);
        }

        var compact = ClusteringTensorOps.CompactLabels<T>(labelsIds, out var slotLabels);
        return SampleScores(data, compact, slotLabels.Length, metric);
    }

    /// <summary>Upper bound on the elements of one [rows, n] or [rows, clusters] block.</summary>
    private const int BlockElements = 1 << 22;

    /// <summary>
    /// s(i) = (b - a) / max(a, b) where a is the mean distance to the other members of the cluster of i and b the
    /// smallest mean distance to another non-empty cluster; zero for noise, singletons and max(a, b) = 0.
    /// </summary>
    /// <remarks>
    /// Rows are scored in blocks so the working set is [rows, n] rather than several n x n tensors: the
    /// score of one sample only needs its own row of distances.
    /// </remarks>
    private Vector<T> SampleScores(Matrix<T> data, Vector<T> labels, int numSlots, IDistanceMetric<T> metric)
    {
        var engine = AiDotNetEngine.Current;
        int n = data.Rows;
        var mask = ClusteringTensorOps.OneHotMask(labels, numSlots);
        var counts = engine.Reshape(engine.ReduceSum(mask, new[] { 0 }, false), new[] { 1, numSlots });

        int rowsPerBlock = Math.Max(1, Math.Min(n, BlockElements / Math.Max(n, numSlots)));
        if (rowsPerBlock >= n)
        {
            return BlockScores(data, data, mask, mask, counts, 0, numSlots, metric).ToVector();
        }

        var dataTensor = Tensor<T>.FromMatrix(data);
        var scores = Tensor<T>.CreateZeros(new[] { n });
        for (int start = 0; start < n; start += rowsPerBlock)
        {
            int count = Math.Min(rowsPerBlock, n - start);
            var rows = engine.TensorSlice(dataTensor, new[] { start, 0 }, new[] { count, data.Columns }).ToMatrix();
            var rowMask = engine.TensorSlice(mask, new[] { start, 0 }, new[] { count, numSlots });
            scores = engine.TensorSetSlice(
                scores, BlockScores(rows, data, mask, rowMask, counts, start, numSlots, metric), new[] { start });
        }

        return scores.ToVector();
    }

    /// <summary>
    /// Silhouette scores of <paramref name="rows"/>, which are rows start..start+rows.Rows-1 of
    /// <paramref name="data"/>; <paramref name="rowMask"/> is their slice of the membership <paramref name="mask"/>.
    /// </summary>
    private Tensor<T> BlockScores(
        Matrix<T> rows, Matrix<T> data, Tensor<T> mask, Tensor<T> rowMask, Tensor<T> counts, int start, int numSlots,
        IDistanceMetric<T> metric)
    {
        var engine = AiDotNetEngine.Current;
        int n = data.Rows;
        int count = rows.Rows;
        var zero = _numOps.Zero;
        var one = _numOps.One;

        // Distances to every other point (self excluded) and per-cluster sums of them
        var rowIds = engine.TensorBroadcastTo(
            engine.Reshape(engine.TensorArange<T>(start, start + count, 1), new[] { count, 1 }), new[] { count, n });
        var columnIds = engine.TensorBroadcastTo(
            engine.Reshape(engine.TensorArange<T>(0, n, 1), new[] { 1, n }), new[] { count, n });
        var offDiagonal = engine.ScalarMinusTensor(one, engine.TensorEquals(rowIds, columnIds));
        var distances = engine.TensorMultiply(Tensor<T>.FromMatrix(metric.ComputePairwise(rows, data)), offDiagonal);
        var clusterSums = engine.TensorMatMul(distances, mask);

        // Other members of each cluster as seen from each point (own cluster excludes the point itself)
        var others = engine.TensorSubtract(engine.TensorBroadcastTo(counts, new[] { count, numSlots }), rowMask);
        var meanDistances = engine.TensorDivide(clusterSums, engine.TensorMax(others, one));

        // a(i) and the own-cluster size (excluding i)
        var a = engine.ReduceSum(engine.TensorMultiply(meanDistances, rowMask), new[] { 1 }, false);
        var ownOthers = engine.ReduceSum(engine.TensorMultiply(others, rowMask), new[] { 1 }, false);

        // b(i): smallest mean distance over the other non-empty clusters (0 when there is none)
        var available = engine.TensorMultiply(
            engine.TensorGreaterThan(others, zero), engine.ScalarMinusTensor(one, rowMask));
        var ceiling = _numOps.Add(engine.TensorMaxValue(meanDistances), one);
        var candidates = engine.TensorWhere(available, meanDistances, engine.TensorFullLike(meanDistances, ceiling));
        var hasOther = engine.ReduceMax(available, new[] { 1 }, false);
        var b = engine.TensorMultiply(engine.TensorAmin(candidates, new[] { 1 }, false), hasOther);

        // s(i) = (b - a) / max(a, b), zero for singletons, noise and max(a, b) = 0
        var maxAB = engine.TensorMax(a, b);
        var valid = engine.TensorMultiply(engine.TensorGreaterThan(maxAB, zero), engine.TensorGreaterThan(ownOthers, zero));
        var safeMax = engine.TensorWhere(valid, maxAB, engine.TensorOnesLike(maxAB));
        return engine.TensorMultiply(engine.TensorDivide(engine.TensorSubtract(b, a), safeMax), valid);
    }
}
