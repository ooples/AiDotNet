using System.ComponentModel;

namespace AiDotNet.Enums;

/// <summary>
/// Defines the types of metrics used to evaluate clustering quality.
/// </summary>
public enum ClusteringMetricType
{
    /// <summary>
    /// Silhouette Score (-1 to 1, higher is better).
    /// Measures how similar each point is to its own cluster compared to other clusters.
    /// </summary>
    [Description("Silhouette Score")]
    [ClusteringMetricInfo(higherIsBetter: true)]
    SilhouetteScore,

    /// <summary>
    /// Davies-Bouldin Index (lower is better).
    /// Measures average similarity between each cluster and its most similar cluster.
    /// </summary>
    [Description("Davies-Bouldin Index")]
    [ClusteringMetricInfo(higherIsBetter: false)]
    DaviesBouldinIndex,

    /// <summary>
    /// Calinski-Harabasz Index (higher is better).
    /// Ratio of between-cluster dispersion to within-cluster dispersion.
    /// </summary>
    [Description("Calinski-Harabasz Index")]
    [ClusteringMetricInfo(higherIsBetter: true)]
    CalinskiHarabaszIndex,

    /// <summary>
    /// Adjusted Rand Index (-1 to 1, higher is better).
    /// Measures agreement between predicted and true labels (requires ground truth).
    /// </summary>
    [Description("Adjusted Rand Index")]
    [ClusteringMetricInfo(higherIsBetter: true)]
    AdjustedRandIndex,

    /// <summary>
    /// Normalized Mutual Information (0 to 1, higher is better).
    /// Measures mutual dependence between predicted and true labels (requires ground truth).
    /// </summary>
    [Description("Normalized Mutual Information")]
    [ClusteringMetricInfo(higherIsBetter: true)]
    NormalizedMutualInformation
}

/// <summary>
/// Attribute to annotate clustering metric enum values with their optimization direction.
/// </summary>
[AttributeUsage(AttributeTargets.Field)]
public sealed class ClusteringMetricInfoAttribute : Attribute
{
    /// <summary>
    /// Gets whether higher values of this metric indicate better clustering.
    /// </summary>
    public bool HigherIsBetter { get; }

    /// <summary>
    /// Creates a new ClusteringMetricInfoAttribute.
    /// </summary>
    /// <param name="higherIsBetter">Whether higher values are better.</param>
    public ClusteringMetricInfoAttribute(bool higherIsBetter)
    {
        HigherIsBetter = higherIsBetter;
    }
}

/// <summary>
/// Extension methods for <see cref="ClusteringMetricType"/>.
/// </summary>
public static class ClusteringMetricTypeExtensions
{
    /// <summary>
    /// Gets the display name for a clustering metric type from its Description attribute.
    /// </summary>
    public static string GetDisplayName(this ClusteringMetricType metricType)
    {
        var field = typeof(ClusteringMetricType).GetField(metricType.ToString());
        if (field is null) return metricType.ToString();

        var attr = (DescriptionAttribute?)Attribute.GetCustomAttribute(field, typeof(DescriptionAttribute));
        return attr?.Description ?? metricType.ToString();
    }

    /// <summary>
    /// Gets whether higher values of this metric indicate better clustering.
    /// </summary>
    public static bool IsHigherBetter(this ClusteringMetricType metricType)
    {
        var field = typeof(ClusteringMetricType).GetField(metricType.ToString());
        if (field is null) return true;

        var attr = (ClusteringMetricInfoAttribute?)Attribute.GetCustomAttribute(field, typeof(ClusteringMetricInfoAttribute));
        return attr?.HigherIsBetter ?? true;
    }
}
