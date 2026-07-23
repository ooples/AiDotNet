namespace AiDotNet.Clustering.Options;

/// <summary>
/// Configuration options for Bisecting K-Means clustering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Bisecting K-Means is a divisive hierarchical clustering algorithm that
/// starts with all points in one cluster and recursively bisects clusters
/// until the desired number of clusters is reached.
/// </para>
/// <para><b>For Beginners:</b> Bisecting K-Means works by:
/// 1. Starting with all data in one big cluster
/// 2. Splitting the largest cluster into two using K-Means
/// 3. Repeating step 2 until you have k clusters
///
/// Advantages over regular K-Means:
/// - Often produces better clusters (more balanced)
/// - Less sensitive to initialization
/// - Builds a cluster hierarchy as a side effect
/// </para>
/// </remarks>
public class BisectingKMeansOptions<T> : ClusteringOptions<T>
{
    /// <summary>
    /// Gets or sets the number of clusters to find.
    /// </summary>
    /// <value>The number of clusters. Must be positive. Default is 8.</value>
    public int NumClusters { get; set; } = 8;

    /// <summary>
    /// Gets or sets the number of bisection trials at each split.
    /// </summary>
    /// <value>
    /// Number of times to try bisecting a cluster, keeping the best result.
    /// Higher values give better quality but are slower. Default is 5.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When splitting a cluster, we try multiple times
    /// and keep the best split. This helps avoid bad splits caused by random initialization.
    /// </para>
    /// </remarks>
    public int NumBisectionTrials { get; set; } = 5;

    /// <summary>
    /// Gets or sets the cluster selection method for bisection.
    /// </summary>
    /// <value>The selection method. Default is Largest.</value>
    public BisectionClusterSelection ClusterSelection { get; set; } = BisectionClusterSelection.Largest;

    /// <summary>
    /// Gets or sets the minimum cluster size that can be bisected.
    /// </summary>
    /// <value>Clusters smaller than this won't be split. Default is 2.</value>
    public int MinClusterSizeForBisection { get; set; } = 2;

    /// <summary>
    /// Gets or sets whether to build a hierarchy tree during clustering.
    /// </summary>
    /// <value>When true, records the bisection history. Default is false.</value>
    public bool BuildHierarchy { get; set; } = false;
}

/// <summary>
/// Methods for selecting which cluster to bisect next.
/// </summary>
public enum BisectionClusterSelection
{
    /// <summary>
    /// Always bisect the cluster with the most points.
    /// This tends to produce more balanced cluster sizes.
    /// </summary>
    Largest,

    /// <summary>
    /// Bisect the cluster with the highest inertia (sum of squared distances).
    /// This tends to break up the "worst" cluster first.
    /// </summary>
    HighestInertia,

    /// <summary>
    /// Bisect the cluster with the largest diameter (max pairwise distance).
    /// This tends to split the most spread out clusters.
    /// </summary>
    LargestDiameter
}
