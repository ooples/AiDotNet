namespace AiDotNet.Clustering.Options;

/// <summary>
/// Configuration options for CURE clustering algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CURE (Clustering Using REpresentatives) is a hierarchical clustering algorithm
/// that represents each cluster by a set of well-scattered representative points,
/// which are then shrunk toward the cluster center.
/// </para>
/// <para><b>For Beginners:</b> CURE is designed to find non-spherical clusters:
/// - Uses multiple "representative" points per cluster instead of just a center
/// - These points are pulled slightly toward the center (shrinking)
/// - This helps find oddly-shaped clusters (like bananas or spirals)
///
/// It's like describing a cluster by several key locations within it,
/// rather than just one center point.
/// </para>
/// </remarks>
public class CUREOptions<T> : ClusteringOptions<T>
{
    /// <summary>
    /// Gets or sets the number of clusters to find.
    /// </summary>
    /// <value>The number of clusters. Default is 8.</value>
    public int NumClusters { get; set; } = 8;

    /// <summary>
    /// Gets or sets the number of representative points per cluster.
    /// </summary>
    /// <value>
    /// Number of representatives. Default is 5.
    /// More representatives = better capture of cluster shape but slower.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Representatives are scattered points
    /// that describe the cluster's shape. More representatives can
    /// capture complex shapes but increase computation time.
    /// </para>
    /// </remarks>
    public int NumRepresentatives { get; set; } = 5;

    /// <summary>
    /// Gets or sets the shrink factor for representative points.
    /// </summary>
    /// <value>
    /// Shrink factor between 0 and 1. Default is 0.3.
    /// 0 = no shrinking, 1 = all representatives at center.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The shrink factor controls how much
    /// representatives are pulled toward the center:
    /// - Low value (0.1): Representatives stay near edges (good for chains)
    /// - High value (0.5): Representatives move more toward center
    /// - Too high: Loses cluster shape information
    /// </para>
    /// </remarks>
    public double ShrinkFactor { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets the sample fraction for large datasets.
    /// </summary>
    /// <value>
    /// Fraction of data to sample. Default is 1.0 (use all data).
    /// For very large datasets, try 0.1-0.5.
    /// </value>
    public double SampleFraction { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether to use random partitioning.
    /// </summary>
    /// <value>
    /// When true, randomly partition data before clustering. Default is false.
    /// </value>
    /// <remarks>
    /// <para>
    /// Random partitioning can speed up clustering of large datasets
    /// by processing partitions independently, then merging.
    /// </para>
    /// </remarks>
    public bool UsePartitioning { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of partitions for parallel processing.
    /// </summary>
    /// <value>Number of partitions. Default is 4.</value>
    public int NumPartitions { get; set; } = 4;
}
