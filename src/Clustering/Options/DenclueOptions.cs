using AiDotNet.Clustering.Interfaces;

namespace AiDotNet.Clustering.Options;

/// <summary>
/// Configuration options for DENCLUE clustering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DENCLUE (DENsity-based CLUstEring) uses kernel density estimation
/// to find clusters. Points are attracted to density maxima using
/// gradient ascent, and clusters are formed by grouping points that
/// converge to the same attractor.
/// </para>
/// <para><b>For Beginners:</b> DENCLUE sees data as a "landscape" where
/// dense areas form "mountains" (attractors).
///
/// Imagine pouring water on a terrain:
/// - Water flows downhill to valleys
/// - In DENCLUE, points "flow uphill" to peaks
/// - Points reaching the same peak form a cluster
///
/// Key parameters:
/// - Bandwidth (h): Controls how wide the "mountains" are
///   - Small h: Many narrow peaks (more clusters)
///   - Large h: Few wide peaks (fewer clusters)
/// - MinDensity: Minimum density for a peak to be a cluster center
/// </para>
/// </remarks>
public class DenclueOptions<T> : ClusteringOptions<T>
{
    /// <summary>
    /// Initializes DenclueOptions with appropriate defaults.
    /// </summary>
    public DenclueOptions()
    {
        MaxIterations = 100;
    }

    /// <summary>
    /// Gets or sets the bandwidth parameter for the Gaussian kernel.
    /// </summary>
    /// <value>Bandwidth (h). Default is 1.0.</value>
    /// <remarks>
    /// <para>
    /// Controls the width of the Gaussian kernel used for density estimation.
    /// Smaller values make the density function more peaked; larger values
    /// smooth it out.
    /// </para>
    /// </remarks>
    public double Bandwidth { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the minimum density threshold for cluster attractors.
    /// </summary>
    /// <value>Minimum density. Default is 0.01.</value>
    /// <remarks>
    /// <para>
    /// Points converging to attractors with density below this threshold
    /// are considered noise and not assigned to any cluster.
    /// </para>
    /// </remarks>
    public double MinDensity { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the convergence threshold for hill climbing.
    /// </summary>
    /// <value>Convergence threshold. Default is 1e-6.</value>
    public double ConvergenceThreshold { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the distance threshold for merging attractors.
    /// </summary>
    /// <value>Attractor merge threshold. Default is 0.1.</value>
    /// <remarks>
    /// <para>
    /// Attractors closer than this threshold are merged into a single cluster.
    /// This helps avoid creating many tiny clusters from nearly identical attractors.
    /// </para>
    /// </remarks>
    public double AttractorMergeThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the distance metric.
    /// </summary>
    /// <value>Distance metric, or null for Euclidean.</value>
    public new IDistanceMetric<T>? DistanceMetric { get; set; }
}
