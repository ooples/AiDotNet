using AiDotNet.Clustering.Interfaces;

namespace AiDotNet.Clustering.Options;

/// <summary>
/// Configuration options for CLARANS (Clustering Large Applications based on Randomized Search).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CLARANS is a medoid-based algorithm that uses randomized sampling to efficiently
/// search for good cluster medoids. It's more scalable than PAM while maintaining
/// the benefit of using actual data points as cluster centers.
/// </para>
/// <para><b>For Beginners:</b> CLARANS randomly explores different cluster configurations.
///
/// Think of it like:
/// 1. Start with random cluster representatives (medoids)
/// 2. Try swapping representatives with random other points
/// 3. Keep changes that improve clustering quality
/// 4. Repeat until satisfied
///
/// Benefits over K-Means:
/// - Cluster centers are actual data points (medoids)
/// - More robust to outliers
/// - Works with any distance function
///
/// Trade-offs:
/// - Slower than K-Means for large datasets
/// - Randomized: may find different solutions each run
/// </para>
/// </remarks>
public class CLARANSOptions<T> : ClusteringOptions<T>
{
    /// <summary>
    /// Initializes CLARANSOptions with appropriate defaults.
    /// </summary>
    public CLARANSOptions()
    {
        MaxIterations = 2; // Number of restarts
    }

    /// <summary>
    /// Gets or sets the number of clusters.
    /// </summary>
    /// <value>Number of clusters. Default is 5.</value>
    public int NumClusters { get; set; } = 5;

    /// <summary>
    /// Gets or sets the number of local search iterations.
    /// </summary>
    /// <value>Max neighbors to explore. Default is max(250, 1.25% of n*k).</value>
    /// <remarks>
    /// <para>
    /// Controls how many neighbor solutions are explored before accepting
    /// a local minimum. Higher values find better solutions but take longer.
    /// </para>
    /// </remarks>
    public int? MaxNeighbor { get; set; }

    /// <summary>
    /// Gets or sets the number of local minima to find.
    /// </summary>
    /// <value>Number of restarts. Default is 2.</value>
    /// <remarks>
    /// <para>
    /// The algorithm is run multiple times with different starting points,
    /// and the best result is kept. More restarts = better results but slower.
    /// </para>
    /// </remarks>
    public int NumLocal { get; set; } = 2;

    /// <summary>
    /// Gets or sets the distance metric.
    /// </summary>
    /// <value>Distance metric, or null for Euclidean.</value>
    public new IDistanceMetric<T>? DistanceMetric { get; set; }
}
