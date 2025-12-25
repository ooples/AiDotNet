using AiDotNet.Clustering.Interfaces;

namespace AiDotNet.Clustering.Options;

/// <summary>
/// Configuration options for Fuzzy C-Means clustering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Fuzzy C-Means (FCM) is a soft clustering algorithm where each point can belong
/// to multiple clusters with varying degrees of membership. Unlike K-Means which
/// assigns each point to exactly one cluster, FCM assigns membership probabilities.
/// </para>
/// <para><b>For Beginners:</b> Fuzzy C-Means allows points to be "partially" in clusters.
///
/// Regular K-Means: "This point belongs to Cluster A"
/// Fuzzy C-Means: "This point is 70% Cluster A, 25% Cluster B, 5% Cluster C"
///
/// This is useful when:
/// - Cluster boundaries are unclear
/// - Points naturally belong to multiple categories
/// - You want uncertainty information
///
/// The fuzziness parameter (m) controls how soft the clustering is:
/// - m close to 1: Hard clustering (like K-Means)
/// - m = 2: Standard fuzzy clustering (recommended)
/// - m > 2: Very soft, overlapping clusters
/// </para>
/// </remarks>
public class FuzzyCMeansOptions<T> : ClusteringOptions<T>
{
    /// <summary>
    /// Initializes FuzzyCMeansOptions with appropriate defaults.
    /// </summary>
    public FuzzyCMeansOptions()
    {
        MaxIterations = 300;
    }

    /// <summary>
    /// Gets or sets the number of clusters.
    /// </summary>
    /// <value>Number of clusters. Default is 3.</value>
    public int NumClusters { get; set; } = 3;

    /// <summary>
    /// Gets or sets the fuzziness parameter (m).
    /// </summary>
    /// <value>Fuzziness, must be > 1. Default is 2.0.</value>
    /// <remarks>
    /// <para>
    /// Controls the degree of cluster overlap:
    /// - Values close to 1: Sharp boundaries (like K-Means)
    /// - Value of 2: Standard fuzzy clustering
    /// - Values > 2: Very soft, overlapping clusters
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown if value is not greater than 1.</exception>
    private double _fuzziness = 2.0;
    public double Fuzziness
    {
        get => _fuzziness;
        set
        {
            if (value <= 1.0)
            {
                throw new ArgumentOutOfRangeException(nameof(value), "Fuzziness must be greater than 1.");
            }
            _fuzziness = value;
        }
    }

    /// <summary>
    /// Gets or sets the distance metric.
    /// </summary>
    /// <value>Distance metric, or null for Euclidean.</value>
    public new IDistanceMetric<T>? DistanceMetric { get; set; }
}
