using AiDotNet.Clustering.Interfaces;

namespace AiDotNet.Clustering.Options;

/// <summary>
/// Configuration options for G-Means clustering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// G-Means extends K-Means by testing whether data in each cluster follows
/// a Gaussian distribution. If not, the cluster is split. It uses the
/// Anderson-Darling test for normality.
/// </para>
/// <para><b>For Beginners:</b> G-Means splits clusters that aren't "bell-shaped".
///
/// The assumption: Good clusters should look like Gaussian (bell curve) distributions.
///
/// How it works:
/// 1. Start with a few clusters (K-Means)
/// 2. For each cluster, test if points form a bell curve
/// 3. If not Gaussian, split the cluster into two
/// 4. Repeat until all clusters pass the Gaussian test
///
/// The significance level controls sensitivity:
/// - Higher: More likely to split (more clusters)
/// - Lower: Less likely to split (fewer clusters)
/// </para>
/// </remarks>
public class GMeansOptions<T> : ClusteringOptions<T>
{
    /// <summary>
    /// Initializes GMeansOptions with appropriate defaults.
    /// </summary>
    public GMeansOptions()
    {
        MaxIterations = 100;
    }

    /// <summary>
    /// Gets or sets the minimum number of clusters.
    /// </summary>
    /// <value>Minimum clusters. Default is 1.</value>
    public int MinClusters { get; set; } = 1;

    /// <summary>
    /// Gets or sets the maximum number of clusters.
    /// </summary>
    /// <value>Maximum clusters. Default is 20.</value>
    public int MaxClusters { get; set; } = 20;

    /// <summary>
    /// Gets or sets the significance level for the Anderson-Darling test.
    /// </summary>
    /// <value>Significance level. Default is 0.0001.</value>
    /// <remarks>
    /// <para>
    /// Lower values make the test more stringent (fewer splits).
    /// Higher values make it more lenient (more splits).
    /// Common values: 0.05, 0.01, 0.001, 0.0001
    /// </para>
    /// </remarks>
    public double SignificanceLevel { get; set; } = 0.0001;

    /// <summary>
    /// Gets or sets the distance metric.
    /// </summary>
    /// <value>Distance metric, or null for Euclidean.</value>
    public new IDistanceMetric<T>? DistanceMetric { get; set; }
}
