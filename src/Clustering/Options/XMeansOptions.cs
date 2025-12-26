using AiDotNet.Clustering.Interfaces;

namespace AiDotNet.Clustering.Options;

/// <summary>
/// Configuration options for X-Means clustering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// X-Means extends K-Means by automatically determining the optimal number
/// of clusters using the Bayesian Information Criterion (BIC). It starts
/// with a minimum number of clusters and iteratively splits clusters until
/// BIC stops improving.
/// </para>
/// <para><b>For Beginners:</b> X-Means finds K automatically.
///
/// The problem with K-Means: You must choose K (number of clusters).
/// X-Means solves this by:
/// 1. Start with a small K
/// 2. Try splitting each cluster into two
/// 3. Keep the split if it improves the model (using BIC)
/// 4. Stop when no splits improve the model
///
/// BIC (Bayesian Information Criterion) balances:
/// - Model fit (how well clusters explain data)
/// - Model complexity (penalizes too many clusters)
/// </para>
/// </remarks>
public class XMeansOptions<T> : ClusteringOptions<T>
{
    /// <summary>
    /// Initializes XMeansOptions with appropriate defaults.
    /// </summary>
    public XMeansOptions()
    {
        MaxIterations = 100;
    }

    /// <summary>
    /// Gets or sets the minimum number of clusters.
    /// </summary>
    /// <value>Minimum clusters. Default is 2.</value>
    public int MinClusters { get; set; } = 2;

    /// <summary>
    /// Gets or sets the maximum number of clusters.
    /// </summary>
    /// <value>Maximum clusters. Default is 20.</value>
    public int MaxClusters { get; set; } = 20;

    /// <summary>
    /// Gets or sets the information criterion to use.
    /// </summary>
    /// <value>Criterion type. Default is BIC.</value>
    public InformationCriterion Criterion { get; set; } = InformationCriterion.BIC;

    /// <summary>
    /// Gets or sets the distance metric.
    /// </summary>
    /// <value>Distance metric, or null for Euclidean.</value>
    public new IDistanceMetric<T>? DistanceMetric { get; set; }
}

/// <summary>
/// Information criteria for model selection.
/// </summary>
public enum InformationCriterion
{
    /// <summary>
    /// Bayesian Information Criterion.
    /// </summary>
    BIC,

    /// <summary>
    /// Akaike Information Criterion.
    /// </summary>
    AIC
}
