namespace AiDotNet.Clustering.Evaluation;

/// <summary>
/// Interface for cluster evaluation metrics.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Cluster metrics assess the quality of clustering results.
/// They can be internal (using only the data) or external
/// (comparing to ground truth labels).
/// </para>
/// <para><b>For Beginners:</b> Cluster metrics answer "How good is this clustering?"
///
/// Two main types:
/// 1. Internal: Use data only (no "correct" answer needed)
///    - Silhouette Score: How similar are points to their own cluster?
///    - Davies-Bouldin: Are clusters compact and well-separated?
///    - Calinski-Harabasz: Ratio of between-cluster to within-cluster variance
///
/// 2. External: Compare to known labels
///    - Adjusted Rand Index: Agreement with ground truth
///    - Normalized Mutual Information: Information shared with ground truth
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("ClusterMetric")]
public interface IClusterMetric<T>
{
    /// <summary>
    /// Gets the name of this metric.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Computes the metric value.
    /// </summary>
    /// <param name="data">The data matrix (n_samples x n_features).</param>
    /// <param name="labels">The cluster assignments.</param>
    /// <returns>The metric value.</returns>
    double Compute(Matrix<T> data, Vector<T> labels);
}

/// <summary>
/// Interface for external cluster metrics that require ground truth.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("ExternalClusterMetric")]
public interface IExternalClusterMetric<T>
{
    /// <summary>
    /// Computes the metric comparing predicted labels to true labels.
    /// </summary>
    /// <param name="trueLabels">The ground truth labels.</param>
    /// <param name="predictedLabels">The predicted cluster assignments.</param>
    /// <returns>The metric value.</returns>
    double Compute(Vector<T> trueLabels, Vector<T> predictedLabels);
}
