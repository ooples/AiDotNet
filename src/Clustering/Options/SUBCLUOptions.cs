namespace AiDotNet.Clustering.Options;

/// <summary>
/// Configuration options for SUBCLU subspace clustering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SUBCLU (SUBspace CLUstering) is a density-connected subspace clustering algorithm
/// that uses DBSCAN as its base clustering method and the monotonicity of density-connected
/// clusters to efficiently prune the subspace search.
/// </para>
/// <para><b>For Beginners:</b> SUBCLU finds clusters hiding in feature subsets:
/// - Uses DBSCAN's concept of density-connected clusters
/// - Exploits the fact that if there's no cluster in 2D, there can't be one in 3D+
/// - This pruning makes it much faster than testing all possible subspaces
///
/// Great for datasets with many features where clusters only exist in some dimensions.
/// </para>
/// </remarks>
public class SUBCLUOptions<T> : ClusteringOptions<T>
{
    /// <summary>
    /// Gets or sets the epsilon radius for DBSCAN density computation.
    /// </summary>
    /// <value>The neighborhood radius. Default is 0.5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This defines what "close" means:
    /// - Smaller epsilon: Only very nearby points are neighbors (tighter clusters)
    /// - Larger epsilon: Points farther apart are still neighbors (looser clusters)
    /// </para>
    /// </remarks>
    public double Epsilon { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the minimum points required for a dense region.
    /// </summary>
    /// <value>Minimum neighborhood size. Default is 5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many neighbors make a "core" point:
    /// - Higher MinPoints: Needs more points to form clusters (less noise sensitive)
    /// - Lower MinPoints: Easier to form clusters (more noise sensitive)
    ///
    /// Rule of thumb: Set to dimensionality + 1 or higher.
    /// </para>
    /// </remarks>
    public int MinPoints { get; set; } = 5;

    /// <summary>
    /// Gets or sets the maximum subspace dimensionality to explore.
    /// </summary>
    /// <value>
    /// Maximum dimensions. Default is 0 (explore all possible subspaces).
    /// </value>
    public int MaxSubspaceDimensions { get; set; } = 0;

    /// <summary>
    /// Gets or sets the minimum cluster size to keep.
    /// </summary>
    /// <value>Minimum points in a cluster. Default is 2.</value>
    public int MinClusterSize { get; set; } = 2;
}
