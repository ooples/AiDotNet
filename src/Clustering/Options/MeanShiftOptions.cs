using AiDotNet.Clustering.Interfaces;

namespace AiDotNet.Clustering.Options;

/// <summary>
/// Configuration options for Mean Shift clustering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Mean Shift is a non-parametric clustering algorithm that doesn't require
/// specifying the number of clusters. It finds clusters by iteratively shifting
/// points toward the mode (densest area) of the local density.
/// </para>
/// <para><b>For Beginners:</b> Mean Shift finds the "peaks" in your data.
///
/// Imagine your data as a landscape with hills:
/// - Each point starts at its location
/// - Points "roll uphill" toward the nearest peak
/// - Points that end up at the same peak form a cluster
///
/// Key features:
/// - Automatically determines number of clusters
/// - Works well for finding dense regions
/// - Good for image segmentation
///
/// Main parameter: bandwidth (how wide the "hills" are)
/// - Small bandwidth: Many small clusters
/// - Large bandwidth: Few large clusters
/// </para>
/// </remarks>
public class MeanShiftOptions<T> : ClusteringOptions<T>
{
    /// <summary>
    /// Gets or sets the bandwidth parameter.
    /// </summary>
    /// <value>Kernel bandwidth, or null for automatic estimation.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Bandwidth controls how far each point "looks"
    /// for neighbors. Larger values mean smoother density and fewer clusters.
    /// If null, bandwidth is estimated automatically from data.
    /// </para>
    /// </remarks>
    public double? Bandwidth { get; set; }

    /// <summary>
    /// Gets or sets the quantile for automatic bandwidth estimation.
    /// </summary>
    /// <value>Quantile (0 to 1). Default is 0.3.</value>
    public double BandwidthQuantile { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets the minimum distance between cluster centers.
    /// </summary>
    /// <value>Minimum distance to merge centers. Default is computed from bandwidth.</value>
    public double? ClusterMergeThreshold { get; set; }

    /// <summary>
    /// Gets or sets whether to bin the seeds to speed up computation.
    /// </summary>
    /// <value>True to use binning. Default is true.</value>
    public bool BinSeeding { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to cluster all points or just the seeds.
    /// </summary>
    /// <value>True to cluster all points. Default is true.</value>
    public bool ClusterAll { get; set; } = true;

    /// <summary>
    /// Gets or sets the minimum number of points to consider a cluster.
    /// </summary>
    /// <value>Minimum cluster size. Default is 1.</value>
    public int MinBinFrequency { get; set; } = 1;

    /// <summary>
    /// Gets or sets the neighbor finding algorithm.
    /// </summary>
    /// <value>Algorithm type. Default is Auto.</value>
    public NeighborAlgorithm Algorithm { get; set; } = NeighborAlgorithm.Auto;

    /// <summary>
    /// Gets or sets the leaf size for tree algorithms.
    /// </summary>
    /// <value>Leaf size. Default is 30.</value>
    public int LeafSize { get; set; } = 30;

    /// <summary>
    /// Gets or sets the distance metric.
    /// </summary>
    /// <value>Distance metric, or null for Euclidean.</value>
    public new IDistanceMetric<T>? DistanceMetric { get; set; }

    /// <summary>
    /// Gets or sets the number of parallel jobs.
    /// </summary>
    /// <value>Number of jobs. Default is 1.</value>
    public int NumJobs { get; set; } = 1;
}
