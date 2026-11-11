using AiDotNet.Tensors.Engines;
using AiDotNet.Clustering.Base;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Clustering.Evaluation;

/// <summary>
/// Calinski-Harabasz Index (Variance Ratio Criterion) for evaluating cluster quality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Calinski-Harabasz Index is the ratio of between-cluster variance
/// to within-cluster variance. Higher values indicate better clustering.
/// </para>
/// <para>
/// CH = (BGS / (k-1)) / (WGS / (n-k))
/// Where:
/// - BGS = Between-Group Sum of Squares (cluster separation)
/// - WGS = Within-Group Sum of Squares (cluster compactness)
/// - k = number of clusters
/// - n = number of points
/// </para>
/// <para><b>For Beginners:</b> Calinski-Harabasz measures variance ratio.
///
/// Good clustering has:
/// - High variance BETWEEN clusters (clusters are different)
/// - Low variance WITHIN clusters (clusters are tight)
///
/// The index is the ratio of these variances.
/// Higher score = Better clustering!
///
/// Think of it like:
/// - Numerator: How spread apart are the cluster centers?
/// - Denominator: How tight are points around their centers?
/// - We want big numerator, small denominator.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Evaluator)]
[PipelineStage(PipelineStage.Evaluation)]
public class CalinskiHarabaszIndex<T> : IClusterMetric<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new CalinskiHarabaszIndex instance.
    /// </summary>
    public CalinskiHarabaszIndex()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc />
    public string Name => "Calinski-Harabasz Index";

    /// <inheritdoc />
    public double Compute(Matrix<T> data, Vector<T> labels)
    {
        int n = data.Rows;
        int d = data.Columns;

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

        int validPoints = labelsIds.Count(label => label >= 0);
        if (validPoints == 0) return 0;

        // Cluster sums/counts over label slots 0..max (unused slots have zero members and contribute nothing)
        var engine = AiDotNetEngine.Current;
        int numSlots = labelsIds.Max() + 1;
        var sums = ClusteringTensorOps.ClusterSums(data, labels, numSlots, out var counts);
        var centroids = ClusteringTensorOps.DivideRows(sums, counts);
        var overallCentroid = engine.Divide(
            engine.ReduceSum(Tensor<T>.FromMatrix(sums), new[] { 0 }, false).ToVector(), _numOps.FromDouble(validPoints));

        // Between-Group Sum of Squares (BGS): sum_c n_c * ||centroid_c - overall||^2
        var centroidOffsets = ClusteringTensorOps.SquaredDistancesToPoint(overallCentroid, centroids);
        double bgs = _numOps.ToDouble(engine.DotProduct(counts, centroidOffsets));

        // Within-Group Sum of Squares (WGS)
        double wgs = _numOps.ToDouble(ClusteringTensorOps.Inertia(data, labels, centroids));
        // Compute Calinski-Harabasz index
        if (wgs == 0 || validPoints <= k)
        {
            return 0;
        }

        return (bgs / (k - 1)) / (wgs / (validPoints - k));
    }
}
