using AiDotNet.Clustering.Interfaces;

namespace AiDotNet.Clustering.Options;

/// <summary>
/// Configuration options for Affinity Propagation clustering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Affinity Propagation clusters data by passing messages between pairs of samples
/// until a high-quality set of exemplars (cluster centers) emerges.
/// </para>
/// <para><b>For Beginners:</b> Affinity Propagation lets data points "vote" for leaders.
///
/// Imagine a group trying to choose team captains:
/// - Everyone can propose themselves as captain
/// - People vote for who should represent them
/// - The algorithm finds natural leaders (exemplars)
///
/// Key features:
/// - Doesn't need to know number of clusters beforehand
/// - Each cluster has a real data point as its center (exemplar)
/// - Works well when you have a good similarity measure
///
/// Main parameter: preference
/// - Higher preference: More clusters (everyone wants to be a leader)
/// - Lower preference: Fewer clusters (fewer people volunteer)
/// </para>
/// </remarks>
public class AffinityPropagationOptions<T> : ClusteringOptions<T>
{
    /// <summary>
    /// Initializes AffinityPropagationOptions with appropriate defaults.
    /// </summary>
    public AffinityPropagationOptions()
    {
        MaxIterations = 200;
    }

    /// <summary>
    /// Gets or sets the damping factor.
    /// </summary>
    /// <value>Damping between 0.5 and 1. Default is 0.5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Damping controls how quickly opinions change.
    /// Higher damping means opinions change more slowly, which can help
    /// the algorithm converge when it otherwise would oscillate.
    /// </para>
    /// </remarks>
    public double Damping { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the preference value.
    /// </summary>
    /// <value>Preference, or null for median of similarities.</value>
    /// <remarks>
    /// <para>
    /// The preference determines how likely each point is to become an exemplar.
    /// Higher values create more clusters, lower values create fewer.
    /// If null, uses the median of the similarity matrix.
    /// </para>
    /// </remarks>
    public double? Preference { get; set; }

    /// <summary>
    /// Gets or sets the number of iterations with no change before stopping.
    /// </summary>
    /// <value>Convergence iterations. Default is 15.</value>
    public int ConvergenceIterations { get; set; } = 15;

    /// <summary>
    /// Gets or sets whether to copy the affinity matrix.
    /// </summary>
    /// <value>True to copy. Default is true.</value>
    public bool Copy { get; set; } = true;

    /// <summary>
    /// Gets or sets the affinity type.
    /// </summary>
    /// <value>Affinity type. Default is Euclidean.</value>
    public AffinityPropagationAffinityType AffinityType { get; set; } = AffinityPropagationAffinityType.Euclidean;

    /// <summary>
    /// Gets or sets the distance metric for Euclidean affinity.
    /// </summary>
    /// <value>Distance metric, or null for default.</value>
    public new IDistanceMetric<T>? DistanceMetric { get; set; }
}

/// <summary>
/// Types of affinity computation for Affinity Propagation.
/// </summary>
public enum AffinityPropagationAffinityType
{
    /// <summary>
    /// Negative squared Euclidean distance.
    /// </summary>
    Euclidean,

    /// <summary>
    /// Precomputed similarity matrix.
    /// </summary>
    Precomputed
}
