using AiDotNet.Tensors.Engines;
using AiDotNet.Clustering.Base;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Clustering.Evaluation;

/// <summary>
/// Within-Cluster Sum of Squares (WCSS) metric for evaluating cluster compactness.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// WCSS measures the total squared distance of each point to its cluster centroid.
/// Lower values indicate more compact clusters.
/// </para>
/// <para>
/// WCSS = sum over all clusters k of sum over all points i in k of ||x_i - c_k||^2
/// </para>
/// <para><b>For Beginners:</b> WCSS measures how "tight" your clusters are.
///
/// For each point:
/// - Find the center of its cluster
/// - Measure the distance squared
/// - Add up all these distances
///
/// Lower WCSS = Better clustering (tighter clusters)
///
/// Also called "inertia" in scikit-learn.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Evaluator)]
[PipelineStage(PipelineStage.Evaluation)]
public class WCSS<T> : IClusterMetric<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new WCSS instance.
    /// </summary>
    public WCSS()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc />
    public string Name => "Within-Cluster Sum of Squares";

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

        if (uniqueLabels.Count == 0)
        {
            return 0;
        }

        // WCSS = sum of squared distances of the non-noise samples to their cluster centroid
        var centroids = ClusteringTensorOps.ClusterMeans(data, labels, labelsIds.Max() + 1, out _);
        return _numOps.ToDouble(ClusteringTensorOps.Inertia(data, labels, centroids));
    }

    /// <summary>
    /// Computes WCSS per cluster.
    /// </summary>
    /// <param name="data">The data matrix.</param>
    /// <param name="labels">The cluster assignments.</param>
    /// <returns>Dictionary mapping cluster labels to their WCSS values.</returns>
    public Dictionary<int, T> ComputePerCluster(Matrix<T> data, Vector<T> labels)
    {
        int n = data.Rows;
        int d = data.Columns;

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

        // Squared distance of every non-noise sample to its cluster centroid, summed per cluster
        int numSlots = uniqueLabels.Count == 0 ? 0 : labelsIds.Max() + 1;
        var result = new Dictionary<int, T>();
        if (numSlots == 0)
        {
            return result;
        }

        var engine = AiDotNetEngine.Current;
        var centroids = ClusteringTensorOps.ClusterMeans(data, labels, numSlots, out _);
        var sampleDistances = ClusteringTensorOps.AssignedSquaredDistances(data, labels, centroids);
        var mask = ClusteringTensorOps.OneHotMask(labels, numSlots);
        var perSlot = engine.TensorMatMul(
            engine.TensorTranspose(mask), Tensor<T>.FromVector(sampleDistances, new[] { n, 1 })).ToArray();
        foreach (int label in uniqueLabels)
        {
            result[label] = perSlot[label];
        }

        return result;
    }
}

/// <summary>
/// Between-Cluster Sum of Squares (BCSS) metric for evaluating cluster separation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// BCSS measures the weighted sum of squared distances between cluster centroids
/// and the overall data centroid. Higher values indicate better separated clusters.
/// </para>
/// <para>
/// BCSS = sum over all clusters k of n_k * ||c_k - c_global||^2
/// </para>
/// <para><b>For Beginners:</b> BCSS measures how "spread apart" your cluster centers are.
///
/// - Find the overall center of all data
/// - For each cluster, measure distance from cluster center to overall center
/// - Weight by cluster size
///
/// Higher BCSS = Better clustering (clusters are more separated)
///
/// Total variance = WCSS + BCSS
/// Good clustering has low WCSS (tight) and high BCSS (separated).
/// </para>
/// </remarks>
public class BCSS<T> : IClusterMetric<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new BCSS instance.
    /// </summary>
    public BCSS()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc />
    public string Name => "Between-Cluster Sum of Squares";

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

        if (uniqueLabels.Count < 2)
        {
            return 0; // Need at least 2 clusters
        }

        int validPoints = labelsIds.Count(label => label >= 0);
        if (validPoints == 0) return 0;

        // Cluster centroids over label slots 0..max and the centroid of all non-noise points
        var engine = AiDotNetEngine.Current;
        var sums = ClusteringTensorOps.ClusterSums(data, labels, labelsIds.Max() + 1, out var counts);
        var centroids = ClusteringTensorOps.DivideRows(sums, counts);
        var globalCentroid = engine.Divide(
            engine.ReduceSum(Tensor<T>.FromMatrix(sums), new[] { 0 }, false).ToVector(), _numOps.FromDouble(validPoints));

        // BCSS = sum_c n_c * ||centroid_c - global||^2
        var offsets = ClusteringTensorOps.SquaredDistancesToPoint(globalCentroid, centroids);
        return _numOps.ToDouble(engine.DotProduct(counts, offsets));
    }
}
