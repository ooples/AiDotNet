using AiDotNet.Clustering.Interfaces;

namespace AiDotNet.Clustering.Options;

/// <summary>
/// Configuration options for BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// BIRCH incrementally builds a CF (Clustering Feature) tree to summarize the data,
/// then applies a clustering algorithm to the leaf entries. It's designed for
/// very large datasets that don't fit in memory.
/// </para>
/// <para><b>For Beginners:</b> BIRCH is like creating a summary tree of your data.
///
/// Imagine organizing a library:
/// 1. First, create sections (CF-tree nodes)
/// 2. Group similar books together (clustering features)
/// 3. Each node summarizes: count, sum, sum of squares
/// 4. Finally, cluster the summaries
///
/// Benefits:
/// - Handles very large datasets efficiently
/// - Incremental: can add data without rebuilding
/// - Single pass through data (mostly)
/// - Memory efficient with controllable tree size
///
/// Key parameters:
/// - Threshold: How similar points must be to join a cluster
/// - BranchingFactor: Max children per node (controls tree width)
/// </para>
/// </remarks>
public class BIRCHOptions<T> : ClusteringOptions<T>
{
    /// <summary>
    /// Initializes BIRCHOptions with appropriate defaults.
    /// </summary>
    public BIRCHOptions()
    {
        MaxIterations = 1; // Usually single pass
    }

    /// <summary>
    /// Gets or sets the threshold for cluster radius.
    /// </summary>
    /// <value>Threshold for merging. Default is 0.5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This controls the maximum radius of a cluster.
    /// Smaller threshold creates more clusters (tighter groups).
    /// Larger threshold creates fewer clusters (looser groups).
    /// </para>
    /// </remarks>
    public double Threshold { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the maximum branching factor.
    /// </summary>
    /// <value>Max children per node. Default is 50.</value>
    /// <remarks>
    /// <para>
    /// Controls the width of the CF-tree. Higher values mean
    /// fewer levels but more comparisons per node.
    /// </para>
    /// </remarks>
    public int BranchingFactor { get; set; } = 50;

    /// <summary>
    /// Gets or sets the target number of clusters.
    /// </summary>
    /// <value>Number of clusters, or null for automatic.</value>
    /// <remarks>
    /// <para>
    /// If specified, a global clustering step is performed on the
    /// leaf entries. If null, each leaf entry becomes a cluster.
    /// </para>
    /// </remarks>
    public int? NumClusters { get; set; }

    /// <summary>
    /// Gets or sets whether to compute cluster labels.
    /// </summary>
    /// <value>True to compute labels. Default is true.</value>
    public bool ComputeLabels { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to copy the input data.
    /// </summary>
    /// <value>True to copy data. Default is true.</value>
    public bool Copy { get; set; } = true;

    /// <summary>
    /// Gets or sets the distance metric.
    /// </summary>
    /// <value>Distance metric, or null for Euclidean.</value>
    public new IDistanceMetric<T>? DistanceMetric { get; set; }
}
