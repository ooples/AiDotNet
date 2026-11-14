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
/// Davies-Bouldin Index for evaluating cluster quality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Davies-Bouldin Index measures the average similarity between
/// each cluster and its most similar cluster. Lower values indicate
/// better clustering (more compact and well-separated clusters).
/// </para>
/// <para>
/// For each cluster i:
/// - S(i) = average distance from points to cluster centroid
/// - d(i,j) = distance between cluster centroids
/// - R(i,j) = (S(i) + S(j)) / d(i,j)
/// - DB = (1/k) * sum(max_j(R(i,j)))
/// </para>
/// <para><b>For Beginners:</b> Davies-Bouldin measures cluster separation.
///
/// The idea:
/// - Good clusters are compact (small S)
/// - Good clusters are well-separated (large d)
/// - For each cluster, find the worst overlap with another
/// - Average these worst cases
///
/// Lower score = Better clustering!
/// (Unlike Silhouette where higher is better)
///
/// A score of 0 would be perfect separation.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Evaluator)]
[PipelineStage(PipelineStage.Evaluation)]
public class DaviesBouldinIndex<T> : IClusterMetric<T>
{
    private readonly IDistanceMetric<T>? _distanceMetric;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new DaviesBouldinIndex instance.
    /// </summary>
    /// <param name="distanceMetric">Distance metric to use, or null for Euclidean.</param>
    public DaviesBouldinIndex(IDistanceMetric<T>? distanceMetric = null)
    {
        _distanceMetric = distanceMetric;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc />
    public string Name => "Davies-Bouldin Index";

    /// <inheritdoc />
    public double Compute(Matrix<T> data, Vector<T> labels)
    {
        int n = data.Rows;
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

        int k = uniqueLabels.Count;
        if (k < 2)
        {
            return 0; // Need at least 2 clusters
        }

        var labelList = uniqueLabels.ToList();
        var engine = AiDotNetEngine.Current;

        // Centroids over label slots 0..max; only the slots in labelList are used below
        int numSlots = labelList.Max() + 1;
        var mask = ClusteringTensorOps.OneHotMask(labels, numSlots);
        var counts = engine.ReduceSum(mask, new[] { 0 }, false);
        var sums = engine.TensorMatMul(engine.TensorTranspose(mask), Tensor<T>.FromMatrix(data)).ToMatrix();
        var allCentroids = ClusteringTensorOps.DivideRows(sums, counts.ToVector());

        // Scatter: average distance (configured metric) from each member to its centroid
        var pointToCentroid = Tensor<T>.FromMatrix(metric.ComputePairwise(data, allCentroids));
        var ownDistance = engine.ReduceSum(engine.TensorMultiply(pointToCentroid, mask), new[] { 1 }, false);
        var scatterSums = engine.TensorMatMul(
            engine.TensorTranspose(mask), engine.Reshape(ownDistance, new[] { n, 1 }));
        var allScatter = engine.TensorDivide(
            engine.Reshape(scatterSums, new[] { numSlots }), engine.TensorMax(counts, _numOps.One));

        // Restrict to the clusters that are present
        var present = labelList.ToArray();
        var centroids = ClusteringTensorOps.SelectRows(allCentroids, present);
        var scatter = ClusteringTensorOps.SelectElements(allScatter.ToVector(), present);

        // R_ij = (s_i + s_j) / d(c_i, c_j) for i != j with d > 0; DB = mean_i max_j R_ij (0 when no such j)
        var centroidDist = Tensor<T>.FromMatrix(metric.ComputePairwise(centroids, centroids));
        var scatterRow = engine.TensorBroadcastTo(Tensor<T>.FromVector(scatter, new[] { k, 1 }), new[] { k, k });
        var scatterCol = engine.TensorBroadcastTo(Tensor<T>.FromVector(scatter, new[] { 1, k }), new[] { k, k });
        var offDiagonal = engine.ScalarMinusTensor(_numOps.One, engine.TensorEye<T>(k));
        var usable = engine.TensorMultiply(engine.TensorGreaterThan(centroidDist, _numOps.Zero), offDiagonal);
        var safeDist = engine.TensorWhere(usable, centroidDist, engine.TensorOnesLike(centroidDist));
        var ratios = engine.TensorMultiply(
            engine.TensorDivide(engine.TensorAdd(scatterRow, scatterCol), safeDist), usable);
        var maxR = engine.ReduceMax(ratios, new[] { 1 }, false);
        double dbSum = _numOps.ToDouble(engine.TensorSum(maxR));

        return dbSum / k;
    }
}
