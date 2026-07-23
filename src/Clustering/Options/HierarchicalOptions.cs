using AiDotNet.Clustering.Interfaces;

namespace AiDotNet.Clustering.Options;

/// <summary>
/// Configuration options for Hierarchical (Agglomerative) clustering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Hierarchical clustering builds a tree (dendrogram) of clusters by successively
/// merging or splitting clusters based on distance. Agglomerative clustering
/// starts with each point as its own cluster and merges them bottom-up.
/// </para>
/// <para><b>For Beginners:</b> Hierarchical clustering creates a family tree of data.
///
/// How it works:
/// 1. Start with each point as its own cluster
/// 2. Find the two closest clusters
/// 3. Merge them into one
/// 4. Repeat until desired number of clusters
///
/// The result is a dendrogram (tree diagram) showing how clusters merge.
/// You can cut the tree at different levels to get different numbers of clusters.
///
/// Linkage methods determine "closest":
/// - Single: Nearest points in clusters (chain-like clusters)
/// - Complete: Farthest points in clusters (compact clusters)
/// - Average: Average of all pairwise distances
/// - Ward: Minimizes within-cluster variance (most popular)
/// </para>
/// </remarks>
public class HierarchicalOptions<T> : ClusteringOptions<T>
{
    /// <summary>
    /// Gets or sets the number of clusters to find.
    /// </summary>
    /// <value>The number of clusters. Default is 2.</value>
    /// <remarks>
    /// <para>
    /// If null, the full dendrogram is computed without cutting.
    /// Use GetDendrogram() to access the full hierarchy.
    /// </para>
    /// </remarks>
    public int NumClusters { get; set; } = 2;

    /// <summary>
    /// Gets or sets the linkage criterion for merging clusters.
    /// </summary>
    /// <value>The linkage method. Default is Ward.</value>
    public LinkageMethod Linkage { get; set; } = LinkageMethod.Ward;

    /// <summary>
    /// Gets or sets the distance threshold for cluster formation.
    /// </summary>
    /// <value>
    /// If set, forms clusters at the specified distance threshold.
    /// Overrides NumClusters if both are specified.
    /// </value>
    public double? DistanceThreshold { get; set; }

    /// <summary>
    /// Gets or sets whether to compute the full dendrogram.
    /// </summary>
    /// <value>When true, stores the full merge history. Default is false.</value>
    public bool ComputeFullTree { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to compute distances between clusters.
    /// </summary>
    /// <value>When true, stores inter-cluster distances. Default is false.</value>
    public bool ComputeDistances { get; set; } = false;

    /// <summary>
    /// Gets or sets the distance metric to use.
    /// </summary>
    /// <value>The distance metric, or null for Euclidean.</value>
    public new IDistanceMetric<T>? DistanceMetric { get; set; }

    /// <summary>
    /// Gets or sets the connectivity constraint matrix.
    /// </summary>
    /// <value>
    /// A sparse matrix defining which samples can be merged together.
    /// Useful for spatial constraints or image segmentation.
    /// </value>
    public Matrix<T>? Connectivity { get; set; }
}

/// <summary>
/// Linkage methods for hierarchical clustering.
/// </summary>
/// <remarks>
/// <para>
/// The linkage method determines how the distance between two clusters is computed.
/// Different methods lead to different cluster shapes and behaviors.
/// </para>
/// </remarks>
public enum LinkageMethod
{
    /// <summary>
    /// Ward's minimum variance method.
    /// Minimizes within-cluster variance. Creates compact, spherical clusters.
    /// This is the most commonly used method and works well for most cases.
    /// </summary>
    Ward,

    /// <summary>
    /// Complete linkage (farthest neighbor).
    /// Uses maximum distance between points in different clusters.
    /// Creates compact clusters, but sensitive to outliers.
    /// </summary>
    Complete,

    /// <summary>
    /// Average linkage (UPGMA).
    /// Uses average distance between all pairs of points in different clusters.
    /// Good balance between single and complete linkage.
    /// </summary>
    Average,

    /// <summary>
    /// Single linkage (nearest neighbor).
    /// Uses minimum distance between points in different clusters.
    /// Can create "chaining" effect with elongated clusters.
    /// </summary>
    Single,

    /// <summary>
    /// Weighted average linkage (WPGMA).
    /// Similar to average but weights by cluster size.
    /// </summary>
    Weighted,

    /// <summary>
    /// Centroid linkage (UPGMC).
    /// Uses distance between cluster centroids.
    /// Can produce inversions in the dendrogram.
    /// </summary>
    Centroid,

    /// <summary>
    /// Median linkage (WPGMC).
    /// Uses distance between weighted centroids.
    /// Less sensitive to outliers than centroid.
    /// </summary>
    Median
}
