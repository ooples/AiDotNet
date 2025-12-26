using AiDotNet.Clustering.Interfaces;

namespace AiDotNet.Clustering.Options;

/// <summary>
/// Configuration options for DBSCAN clustering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DBSCAN (Density-Based Spatial Clustering of Applications with Noise) finds clusters
/// based on density. Unlike K-Means, it doesn't require specifying the number of clusters
/// and can discover clusters of arbitrary shape.
/// </para>
/// <para><b>For Beginners:</b> DBSCAN works by finding dense regions in your data.
///
/// Key concepts:
/// - Epsilon (eps): The maximum distance between two points to be neighbors
/// - MinPoints: Minimum neighbors needed to form a dense region
/// - Core point: Has at least MinPoints neighbors within epsilon
/// - Border point: Near a core point but not enough neighbors to be core
/// - Noise: Points that don't belong to any cluster
///
/// Advantages over K-Means:
/// - No need to specify number of clusters
/// - Can find arbitrarily shaped clusters
/// - Robust to outliers (marks them as noise)
/// - Doesn't assume spherical clusters
///
/// Choosing parameters:
/// - eps: Use a k-distance graph (elbow method)
/// - MinPoints: 2 × dimensions is a good starting point
/// </para>
/// </remarks>
public class DBSCANOptions<T> : ClusteringOptions<T>
{
    /// <summary>
    /// Gets or sets the epsilon radius for neighborhood queries.
    /// </summary>
    /// <value>The maximum distance between points to be considered neighbors. Default is 0.5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Epsilon defines what "close" means.
    /// Two points are neighbors if their distance is less than epsilon.
    ///
    /// - Too small: Many points become noise, clusters fragment
    /// - Too large: Separate clusters merge into one
    ///
    /// Tip: Plot a k-distance graph and look for the "elbow".
    /// </para>
    /// </remarks>
    public double Epsilon { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the minimum number of points to form a dense region.
    /// </summary>
    /// <value>Minimum number of neighbors (including the point itself). Default is 5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> MinPoints defines how many neighbors make a "dense" area.
    /// A point needs at least MinPoints neighbors (including itself) to be a core point.
    ///
    /// - Rule of thumb: Use MinPoints >= dimensions + 1
    /// - Common default: MinPoints = 2 × dimensions
    /// - Higher values: More robust to noise, fewer small clusters
    /// - Lower values: More sensitive, may create many tiny clusters
    /// </para>
    /// </remarks>
    public int MinPoints { get; set; } = 5;

    /// <summary>
    /// Gets or sets the algorithm for computing core sample neighborhoods.
    /// </summary>
    /// <value>The algorithm to use. Default is Auto.</value>
    public NeighborAlgorithm Algorithm { get; set; } = NeighborAlgorithm.Auto;

    /// <summary>
    /// Gets or sets the leaf size for BallTree or KDTree.
    /// </summary>
    /// <value>The leaf size for tree algorithms. Default is 30.</value>
    public int LeafSize { get; set; } = 30;

    /// <summary>
    /// Gets or sets the power parameter for Minkowski distance.
    /// </summary>
    /// <value>The Minkowski p parameter. Default is 2 (Euclidean).</value>
    public double P { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the metric for distance calculations.
    /// </summary>
    /// <value>The distance metric to use, or null for default (Euclidean).</value>
    public new IDistanceMetric<T>? DistanceMetric { get; set; }

    /// <summary>
    /// Gets or sets the number of parallel jobs (-1 for all cores).
    /// </summary>
    /// <value>Number of parallel jobs. Default is 1 (no parallelism).</value>
    public int NumJobs { get; set; } = 1;
}

/// <summary>
/// Algorithms for computing nearest neighbors.
/// </summary>
public enum NeighborAlgorithm
{
    /// <summary>
    /// Automatically select based on data characteristics.
    /// </summary>
    Auto,

    /// <summary>
    /// Use KD-Tree for efficient queries (best for low dimensions).
    /// </summary>
    KDTree,

    /// <summary>
    /// Use Ball Tree (better for high dimensions or non-Euclidean metrics).
    /// </summary>
    BallTree,

    /// <summary>
    /// Brute force computation (guaranteed correct, O(n²) complexity).
    /// </summary>
    BruteForce
}
