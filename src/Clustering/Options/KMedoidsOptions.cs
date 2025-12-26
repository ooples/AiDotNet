using AiDotNet.Clustering.Interfaces;

namespace AiDotNet.Clustering.Options;

/// <summary>
/// Configuration options for K-Medoids (PAM) clustering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// K-Medoids is similar to K-Means but uses actual data points (medoids) as
/// cluster centers instead of means. This makes it more robust to outliers
/// and allows use of any distance metric.
/// </para>
/// <para><b>For Beginners:</b> K-Medoids uses real data points as cluster centers.
///
/// K-Means vs K-Medoids:
/// - K-Means: Centers are averages (may not be real points)
/// - K-Medoids: Centers are actual data points (medoids)
///
/// When to use K-Medoids:
/// - Data has outliers (medoids are more robust)
/// - Need interpretable cluster representatives
/// - Distance metric isn't Euclidean
/// - Categorical or mixed data
///
/// PAM (Partitioning Around Medoids) is the classic algorithm:
/// 1. Initialize with random medoids
/// 2. Assign points to nearest medoid
/// 3. Try swapping medoids with non-medoids
/// 4. Keep swaps that reduce total cost
/// 5. Repeat until no improvement
/// </para>
/// </remarks>
public class KMedoidsOptions<T> : ClusteringOptions<T>
{
    /// <summary>
    /// Initializes KMedoidsOptions with appropriate defaults.
    /// </summary>
    public KMedoidsOptions()
    {
        MaxIterations = 300;
    }

    /// <summary>
    /// Gets or sets the number of clusters.
    /// </summary>
    /// <value>Number of clusters. Default is 8.</value>
    public int NumClusters { get; set; } = 8;

    /// <summary>
    /// Gets or sets the initialization method.
    /// </summary>
    /// <value>Initialization method. Default is KMedoidsPlusPlus.</value>
    public KMedoidsInit Init { get; set; } = KMedoidsInit.KMedoidsPlusPlus;

    /// <summary>
    /// Gets or sets the algorithm variant.
    /// </summary>
    /// <value>Algorithm variant. Default is PAM.</value>
    public KMedoidsAlgorithm Algorithm { get; set; } = KMedoidsAlgorithm.PAM;

    /// <summary>
    /// Gets or sets the distance metric.
    /// </summary>
    /// <value>Distance metric, or null for Euclidean.</value>
    public new IDistanceMetric<T>? DistanceMetric { get; set; }
}

/// <summary>
/// Initialization methods for K-Medoids.
/// </summary>
public enum KMedoidsInit
{
    /// <summary>
    /// Random initialization.
    /// </summary>
    Random,

    /// <summary>
    /// K-Medoids++ initialization (like K-Means++).
    /// </summary>
    KMedoidsPlusPlus,

    /// <summary>
    /// BUILD heuristic from original PAM paper.
    /// </summary>
    Build
}

/// <summary>
/// Algorithm variants for K-Medoids.
/// </summary>
public enum KMedoidsAlgorithm
{
    /// <summary>
    /// PAM: Partitioning Around Medoids (exact).
    /// </summary>
    PAM,

    /// <summary>
    /// FastPAM: Faster variant with same quality.
    /// </summary>
    FastPAM,

    /// <summary>
    /// Alternate: Simple alternating optimization.
    /// </summary>
    Alternate
}
