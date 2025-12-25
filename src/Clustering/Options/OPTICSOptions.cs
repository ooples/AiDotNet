using AiDotNet.Clustering.Interfaces;

namespace AiDotNet.Clustering.Options;

/// <summary>
/// Configuration options for OPTICS clustering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// OPTICS (Ordering Points To Identify the Clustering Structure) is a density-based
/// algorithm similar to DBSCAN but doesn't require a specific epsilon value upfront.
/// Instead, it creates an ordering of points and their reachability distances.
/// </para>
/// <para><b>For Beginners:</b> OPTICS is like DBSCAN but smarter about finding clusters.
///
/// The problem with DBSCAN:
/// - You must choose epsilon carefully
/// - One epsilon might not work for all clusters
///
/// OPTICS solves this by:
/// - Computing "reachability distances" for all points
/// - Creating an ordering that reveals cluster structure
/// - Allowing you to extract clusters at different density levels
///
/// Think of it like hiking through mountains:
/// - Valleys are dense clusters
/// - Peaks are boundaries between clusters
/// - You can see the structure at different altitudes
/// </para>
/// </remarks>
public class OPTICSOptions<T> : ClusteringOptions<T>
{
    /// <summary>
    /// Gets or sets the minimum number of samples in a neighborhood.
    /// </summary>
    /// <value>Minimum samples. Default is 5.</value>
    public int MinSamples { get; set; } = 5;

    /// <summary>
    /// Gets or sets the maximum epsilon distance.
    /// </summary>
    /// <value>Maximum distance for neighbors. Default is infinity.</value>
    /// <remarks>
    /// Unlike DBSCAN, this is just an upper bound for efficiency.
    /// Setting a reasonable maximum improves performance.
    /// </remarks>
    public double MaxEpsilon { get; set; } = double.PositiveInfinity;

    /// <summary>
    /// Gets or sets the cluster extraction method.
    /// </summary>
    /// <value>Extraction method. Default is Xi.</value>
    public OPTICSExtractionMethod ExtractionMethod { get; set; } = OPTICSExtractionMethod.Xi;

    /// <summary>
    /// Gets or sets the xi parameter for Xi extraction method.
    /// </summary>
    /// <value>Xi value (0 to 1). Default is 0.05.</value>
    /// <remarks>
    /// <para>
    /// Controls cluster steepness. Higher values create fewer, larger clusters.
    /// Lower values create more, smaller clusters.
    /// </para>
    /// </remarks>
    public double Xi { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets the epsilon for DBSCAN-style extraction.
    /// </summary>
    /// <value>Epsilon for cluster extraction, or null to use Xi method.</value>
    public double? ClusterEpsilon { get; set; }

    /// <summary>
    /// Gets or sets the minimum cluster size as fraction of total points.
    /// </summary>
    /// <value>Minimum cluster size fraction. Default is 0.01.</value>
    public double MinClusterSizeFraction { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the neighbor finding algorithm.
    /// </summary>
    /// <value>The algorithm to use. Default is Auto.</value>
    public NeighborAlgorithm Algorithm { get; set; } = NeighborAlgorithm.Auto;

    /// <summary>
    /// Gets or sets the leaf size for tree algorithms.
    /// </summary>
    /// <value>Leaf size. Default is 30.</value>
    public int LeafSize { get; set; } = 30;

    /// <summary>
    /// Gets or sets the distance metric.
    /// </summary>
    /// <value>Distance metric, or null for Euclidean.</value>
    public new IDistanceMetric<T>? DistanceMetric { get; set; }

    /// <summary>
    /// Gets or sets the predecessor correction setting.
    /// </summary>
    /// <value>True to apply correction. Default is true.</value>
    public bool PredecessorCorrection { get; set; } = true;
}

/// <summary>
/// Methods for extracting clusters from OPTICS output.
/// </summary>
public enum OPTICSExtractionMethod
{
    /// <summary>
    /// Xi method: Finds steep areas in the reachability plot.
    /// Best for hierarchical cluster structures.
    /// </summary>
    Xi,

    /// <summary>
    /// DBSCAN-style: Cuts the reachability plot at a fixed epsilon.
    /// Produces flat clusters like DBSCAN.
    /// </summary>
    DbscanStyle
}
