using AiDotNet.Clustering.Interfaces;

namespace AiDotNet.Clustering.Options;

/// <summary>
/// Configuration options for HDBSCAN (Hierarchical DBSCAN).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// HDBSCAN extends DBSCAN by constructing a hierarchy of clusters at different
/// density levels, then extracting flat clusters using a stability-based method.
/// It handles clusters of varying densities better than DBSCAN.
/// </para>
/// <para><b>For Beginners:</b> HDBSCAN finds clusters at all density levels.
///
/// Problems with DBSCAN:
/// - Need to pick epsilon (cluster radius) - hard to choose!
/// - One epsilon can't handle both dense and sparse areas
///
/// HDBSCAN solution:
/// - Try ALL possible epsilon values automatically
/// - Build a tree of how clusters merge as density changes
/// - Pick the most "stable" clusters from this tree
///
/// Benefits:
/// - No epsilon parameter needed
/// - Finds clusters of varying densities
/// - Only need minClusterSize (intuitive parameter)
/// - Robust noise detection
/// </para>
/// </remarks>
public class HDBSCANOptions<T> : ClusteringOptions<T>
{
    /// <summary>
    /// Gets or sets the minimum cluster size.
    /// </summary>
    /// <value>Minimum points for a cluster. Default is 5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The smallest group you'd consider a cluster.
    /// Smaller values find more clusters; larger values find fewer, bigger clusters.
    /// </para>
    /// </remarks>
    public int MinClusterSize { get; set; } = 5;

    /// <summary>
    /// Gets or sets the minimum samples for core points.
    /// </summary>
    /// <value>Min samples, or null to use MinClusterSize.</value>
    /// <remarks>
    /// <para>
    /// Controls how conservative cluster formation is.
    /// Higher values make the algorithm more conservative.
    /// </para>
    /// </remarks>
    public int? MinSamples { get; set; }

    /// <summary>
    /// Gets or sets the cluster selection method.
    /// </summary>
    /// <value>Selection method. Default is EOM (Excess of Mass).</value>
    public HDBSCANClusterSelection ClusterSelection { get; set; } = HDBSCANClusterSelection.EOM;

    /// <summary>
    /// Gets or sets whether to allow single-cluster result.
    /// </summary>
    /// <value>True to allow. Default is false.</value>
    public bool AllowSingleCluster { get; set; } = false;

    /// <summary>
    /// Gets or sets the cluster selection epsilon.
    /// </summary>
    /// <value>Epsilon for cluster selection. Default is 0.</value>
    /// <remarks>
    /// <para>
    /// A distance threshold below which clusters are merged.
    /// Only used with EOM cluster selection.
    /// </para>
    /// </remarks>
    public double ClusterSelectionEpsilon { get; set; } = 0;

    /// <summary>
    /// Gets or sets the alpha value for metric.
    /// </summary>
    /// <value>Alpha parameter. Default is 1.0.</value>
    public double Alpha { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the distance metric.
    /// </summary>
    /// <value>Distance metric, or null for Euclidean.</value>
    public new IDistanceMetric<T>? DistanceMetric { get; set; }
}

/// <summary>
/// Cluster selection methods for HDBSCAN.
/// </summary>
public enum HDBSCANClusterSelection
{
    /// <summary>
    /// Excess of Mass: Maximizes cluster stability.
    /// </summary>
    EOM,

    /// <summary>
    /// Leaf: Select all leaf clusters in the tree.
    /// </summary>
    Leaf
}
