namespace AiDotNet.Models.Results;

/// <summary>
/// Represents clustering quality metrics for evaluating the performance of clustering algorithms.
/// </summary>
/// <remarks>
/// <para>
/// This class encapsulates various metrics used to assess the quality of clustering results.
/// These metrics help determine how well data points are grouped into clusters and whether
/// the clustering algorithm has produced meaningful, well-separated groups.
/// </para>
/// <para><b>For Beginners:</b> This class stores measurements that tell you how good your clustering is.
///
/// When you group data into clusters (like organizing customers into segments or grouping similar documents),
/// you need to know if the grouping makes sense. This class provides several scores that help answer questions like:
/// - Are items in the same cluster similar to each other?
/// - Are different clusters well-separated from each other?
/// - How does your clustering compare to known "ground truth" groupings?
///
/// The metrics included are:
/// - **Silhouette Score**: Measures how well each item fits in its cluster (-1 to 1, higher is better)
/// - **Calinski-Harabasz Index**: Measures cluster separation (higher is better)
/// - **Davies-Bouldin Index**: Measures cluster compactness and separation (lower is better)
/// - **Adjusted Rand Index**: Compares clustering to ground truth labels (-1 to 1, higher is better)
///
/// These metrics are automatically calculated during cross-validation when your model produces cluster labels.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for metric values, typically float or double.</typeparam>
public class ClusteringMetrics<T>
{
    /// <summary>
    /// Gets or sets the Silhouette Score for the clustering.
    /// </summary>
    /// <value>The Silhouette Score value, or null if not calculated.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This score measures how well each data point fits into its assigned cluster.
    /// - Range: -1 to 1
    /// - 1 = perfect clustering (points are very similar to their cluster and very different from other clusters)
    /// - 0 = clusters are overlapping or ambiguous
    /// - -1 = points might be in the wrong clusters
    /// </para>
    /// </remarks>
    public T? SilhouetteScore { get; set; }

    /// <summary>
    /// Gets or sets the Calinski-Harabasz Index for the clustering.
    /// </summary>
    /// <value>The Calinski-Harabasz Index value, or null if not calculated.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This index measures how well-separated and compact your clusters are.
    /// - Higher values are better
    /// - No fixed maximum - higher means more distinct, well-separated clusters
    /// - Best used to compare different numbers of clusters on the same data
    /// </para>
    /// </remarks>
    public T? CalinskiHarabaszIndex { get; set; }

    /// <summary>
    /// Gets or sets the Davies-Bouldin Index for the clustering.
    /// </summary>
    /// <value>The Davies-Bouldin Index value, or null if not calculated.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This index measures the average similarity between each cluster and its most similar neighbor.
    /// - Lower values are better
    /// - 0 = perfect clustering
    /// - Higher values mean clusters are less distinct
    /// </para>
    /// </remarks>
    public T? DaviesBouldinIndex { get; set; }

    /// <summary>
    /// Gets or sets the Adjusted Rand Index comparing clustering to ground truth labels.
    /// </summary>
    /// <value>The Adjusted Rand Index value, or null if ground truth labels are not available.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This index compares your clustering to known "correct" labels, accounting for chance.
    /// - Range: -1 to 1 (can be slightly negative)
    /// - 1 = perfect match to ground truth
    /// - 0 = random clustering
    /// - Negative = worse than random
    ///
    /// This metric requires ground truth labels to compare against. If your dataset has known categories
    /// (like product types, customer segments, etc.), this tells you how well your clustering recovered those categories.
    /// </para>
    /// </remarks>
    public T? AdjustedRandIndex { get; set; }

    /// <summary>
    /// Initializes a new instance of the ClusteringMetrics class with default values (all metrics null).
    /// </summary>
    public ClusteringMetrics()
    {
    }

    /// <summary>
    /// Initializes a new instance of the ClusteringMetrics class with specified metric values.
    /// </summary>
    /// <param name="silhouetteScore">The Silhouette Score value.</param>
    /// <param name="calinskiHarabaszIndex">The Calinski-Harabasz Index value.</param>
    /// <param name="daviesBouldinIndex">The Davies-Bouldin Index value.</param>
    /// <param name="adjustedRandIndex">The Adjusted Rand Index value.</param>
    public ClusteringMetrics(T? silhouetteScore = default, T? calinskiHarabaszIndex = default,
        T? daviesBouldinIndex = default, T? adjustedRandIndex = default)
    {
        SilhouetteScore = silhouetteScore;
        CalinskiHarabaszIndex = calinskiHarabaszIndex;
        DaviesBouldinIndex = daviesBouldinIndex;
        AdjustedRandIndex = adjustedRandIndex;
    }
}
