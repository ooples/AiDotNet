using AiDotNet.Clustering.Interfaces;

namespace AiDotNet.Clustering.Evaluation;

/// <summary>
/// Convenience class for computing multiple cluster evaluation metrics at once.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This class provides a simple interface to compute multiple clustering
/// metrics with a single call. It returns a ClusteringScores object
/// containing all computed values.
/// </para>
/// <para><b>For Beginners:</b> Use this class to evaluate your clustering results.
///
/// Example usage:
/// <code>
/// var metrics = new ClusterMetrics&lt;double&gt;();
/// var scores = metrics.Evaluate(data, predictedLabels);
/// Console.WriteLine($"Silhouette: {scores.Silhouette}");
/// Console.WriteLine($"Davies-Bouldin: {scores.DaviesBouldin}");
/// </code>
///
/// If you have ground truth labels:
/// <code>
/// var scores = metrics.Evaluate(data, predictedLabels, trueLabels);
/// Console.WriteLine($"ARI: {scores.AdjustedRandIndex}");
/// </code>
/// </para>
/// </remarks>
public class ClusterMetrics<T>
{
    private readonly IDistanceMetric<T>? _distanceMetric;

    /// <summary>
    /// Initializes a new ClusterMetrics instance.
    /// </summary>
    /// <param name="distanceMetric">Distance metric to use, or null for Euclidean.</param>
    public ClusterMetrics(IDistanceMetric<T>? distanceMetric = null)
    {
        _distanceMetric = distanceMetric;
    }

    /// <summary>
    /// Evaluates clustering using internal metrics only.
    /// </summary>
    /// <param name="data">The data matrix.</param>
    /// <param name="labels">The cluster assignments.</param>
    /// <returns>Clustering scores.</returns>
    public ClusteringScores Evaluate(Matrix<T> data, Vector<T> labels)
    {
        var silhouette = new SilhouetteScore<T>(_distanceMetric);
        var daviesBouldin = new DaviesBouldinIndex<T>(_distanceMetric);
        var calinskiHarabasz = new CalinskiHarabaszIndex<T>();

        return new ClusteringScores
        {
            Silhouette = silhouette.Compute(data, labels),
            DaviesBouldin = daviesBouldin.Compute(data, labels),
            CalinskiHarabasz = calinskiHarabasz.Compute(data, labels)
        };
    }

    /// <summary>
    /// Evaluates clustering using both internal and external metrics.
    /// </summary>
    /// <param name="data">The data matrix.</param>
    /// <param name="predictedLabels">The predicted cluster assignments.</param>
    /// <param name="trueLabels">The ground truth labels.</param>
    /// <returns>Clustering scores including external metrics.</returns>
    public ClusteringScores Evaluate(Matrix<T> data, Vector<T> predictedLabels, Vector<T> trueLabels)
    {
        var scores = Evaluate(data, predictedLabels);

        var ari = new AdjustedRandIndex<T>();
        var nmi = new NormalizedMutualInformation<T>();

        scores.AdjustedRandIndex = ari.Compute(trueLabels, predictedLabels);
        scores.NormalizedMutualInformation = nmi.Compute(trueLabels, predictedLabels);
        scores.HasExternalMetrics = true;

        return scores;
    }

    /// <summary>
    /// Computes only external metrics (when data is not available).
    /// </summary>
    /// <param name="predictedLabels">The predicted cluster assignments.</param>
    /// <param name="trueLabels">The ground truth labels.</param>
    /// <returns>Clustering scores with external metrics only.</returns>
    public ClusteringScores EvaluateExternal(Vector<T> predictedLabels, Vector<T> trueLabels)
    {
        var ari = new AdjustedRandIndex<T>();
        var nmi = new NormalizedMutualInformation<T>();

        return new ClusteringScores
        {
            AdjustedRandIndex = ari.Compute(trueLabels, predictedLabels),
            NormalizedMutualInformation = nmi.Compute(trueLabels, predictedLabels),
            HasExternalMetrics = true
        };
    }
}

/// <summary>
/// Contains the results of cluster evaluation metrics.
/// </summary>
public class ClusteringScores
{
    /// <summary>
    /// Silhouette Score (-1 to 1, higher is better).
    /// </summary>
    public double Silhouette { get; set; }

    /// <summary>
    /// Davies-Bouldin Index (lower is better).
    /// </summary>
    public double DaviesBouldin { get; set; }

    /// <summary>
    /// Calinski-Harabasz Index (higher is better).
    /// </summary>
    public double CalinskiHarabasz { get; set; }

    /// <summary>
    /// Adjusted Rand Index (-1 to 1, higher is better).
    /// </summary>
    public double? AdjustedRandIndex { get; set; }

    /// <summary>
    /// Normalized Mutual Information (0 to 1, higher is better).
    /// </summary>
    public double? NormalizedMutualInformation { get; set; }

    /// <summary>
    /// Whether external metrics (ARI, NMI) are available.
    /// </summary>
    public bool HasExternalMetrics { get; set; }

    /// <summary>
    /// Returns a summary string of all metrics.
    /// </summary>
    public override string ToString()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("Clustering Evaluation Metrics:");
        sb.AppendLine($"  Silhouette Score:        {Silhouette:F4} (higher is better, range: -1 to 1)");
        sb.AppendLine($"  Davies-Bouldin Index:    {DaviesBouldin:F4} (lower is better)");
        sb.AppendLine($"  Calinski-Harabasz Index: {CalinskiHarabasz:F4} (higher is better)");

        if (HasExternalMetrics)
        {
            sb.AppendLine($"  Adjusted Rand Index:     {AdjustedRandIndex:F4} (higher is better, range: -1 to 1)");
            sb.AppendLine($"  Normalized Mutual Info:  {NormalizedMutualInformation:F4} (higher is better, range: 0 to 1)");
        }

        return sb.ToString();
    }
}
