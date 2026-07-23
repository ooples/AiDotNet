using AiDotNet.Clustering.Interfaces;

namespace AiDotNet.Clustering.Options;

/// <summary>
/// Configuration options for COP-KMeans (Constrained K-Means).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// COP-KMeans extends K-Means with pairwise constraints:
/// - Must-link: Two points must be in the same cluster
/// - Cannot-link: Two points must be in different clusters
/// </para>
/// <para><b>For Beginners:</b> Sometimes you have partial knowledge about your data.
///
/// You might know:
/// - "These two customers bought the same product" (must-link)
/// - "These two users are different people" (cannot-link)
///
/// COP-KMeans uses this information to guide clustering:
/// - It won't separate must-linked pairs
/// - It won't group cannot-linked pairs
///
/// This is "semi-supervised" because you have some labels but not all.
/// </para>
/// </remarks>
public class COPKMeansOptions<T> : ClusteringOptions<T>
{
    /// <summary>
    /// Initializes COPKMeansOptions with appropriate defaults.
    /// </summary>
    public COPKMeansOptions()
    {
        MaxIterations = 100;
    }

    /// <summary>
    /// Gets or sets the number of clusters.
    /// </summary>
    /// <value>Number of clusters. Default is 3.</value>
    public int NumClusters { get; set; } = 3;

    /// <summary>
    /// Gets or sets the must-link constraints.
    /// </summary>
    /// <value>List of (i, j) pairs that must be in the same cluster.</value>
    public List<(int, int)>? MustLink { get; set; }

    /// <summary>
    /// Gets or sets the cannot-link constraints.
    /// </summary>
    /// <value>List of (i, j) pairs that must be in different clusters.</value>
    public List<(int, int)>? CannotLink { get; set; }

    /// <summary>
    /// Gets or sets whether to use transitive closure for must-link constraints.
    /// </summary>
    /// <value>True to compute transitive closure. Default is true.</value>
    /// <remarks>
    /// <para>
    /// If A must-link B and B must-link C, then A must-link C.
    /// Enabling this computes all implied must-links automatically.
    /// </para>
    /// </remarks>
    public bool UseTransitiveClosure { get; set; } = true;

    /// <summary>
    /// Gets or sets the distance metric.
    /// </summary>
    /// <value>Distance metric, or null for Euclidean.</value>
    public new IDistanceMetric<T>? DistanceMetric { get; set; }
}

/// <summary>
/// Configuration options for Seeded K-Means.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Seeded K-Means uses pre-labeled data points as initial cluster seeds.
/// The algorithm starts with these seeds and then proceeds like regular K-Means.
/// </para>
/// <para><b>For Beginners:</b> Instead of random initialization,
/// use data points you already know the labels for.
///
/// Example:
/// - You have 1000 customer profiles
/// - You've manually labeled 50 as "budget", "premium", or "enterprise"
/// - Seeded K-Means uses those 50 to initialize 3 clusters
/// - Then it clusters the remaining 950 automatically
///
/// This often produces better results than random initialization.
/// </para>
/// </remarks>
public class SeededKMeansOptions<T> : ClusteringOptions<T>
{
    /// <summary>
    /// Initializes SeededKMeansOptions with appropriate defaults.
    /// </summary>
    public SeededKMeansOptions()
    {
        MaxIterations = 100;
    }

    /// <summary>
    /// Gets or sets the number of clusters (inferred from seeds if not set).
    /// </summary>
    /// <value>Number of clusters, or -1 to infer from seeds. Default is -1.</value>
    public int NumClusters { get; set; } = -1;

    /// <summary>
    /// Gets or sets the labeled seed points.
    /// </summary>
    /// <value>Dictionary mapping data point index to cluster label.</value>
    public Dictionary<int, int>? Seeds { get; set; }

    /// <summary>
    /// Gets or sets whether seeded points are constrained to their original cluster.
    /// </summary>
    /// <value>True to keep seeds in original clusters. Default is false.</value>
    /// <remarks>
    /// <para>
    /// If true, seeded points cannot be reassigned during clustering.
    /// This is useful when seed labels are certain.
    /// </para>
    /// </remarks>
    public bool ConstrainSeeds { get; set; }

    /// <summary>
    /// Gets or sets the distance metric.
    /// </summary>
    /// <value>Distance metric, or null for Euclidean.</value>
    public new IDistanceMetric<T>? DistanceMetric { get; set; }
}
