namespace AiDotNet.Clustering.Options;

/// <summary>
/// Configuration options for CLIQUE subspace clustering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CLIQUE (CLustering In QUEst) is a grid-based, density-based algorithm
/// for identifying clusters in subspaces of high-dimensional data.
/// </para>
/// <para><b>For Beginners:</b> CLIQUE finds clusters that may only exist in some dimensions:
/// - Divides each dimension into bins (like a grid)
/// - Finds dense cells (many points in a small region)
/// - Connects adjacent dense cells into clusters
/// - Works bottom-up: starts with 1D, extends to 2D, 3D, etc.
///
/// Key insight: Sometimes clusters only appear when looking at a few features,
/// not all of them. CLIQUE finds these hidden patterns.
/// </para>
/// </remarks>
public class CLIQUEOptions<T> : ClusteringOptions<T>
{
    /// <summary>
    /// Gets or sets the number of intervals (bins) per dimension.
    /// </summary>
    /// <value>Number of grid cells per dimension. Default is 10.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> More intervals = finer grid = can find smaller clusters,
    /// but also more computation. Fewer intervals = faster but may miss small patterns.
    /// </para>
    /// </remarks>
    public int NumIntervals { get; set; } = 10;

    /// <summary>
    /// Gets or sets the density threshold as a fraction of total points.
    /// </summary>
    /// <value>
    /// Minimum fraction of points for a cell to be considered dense.
    /// Default is 0.1 (10% of points).
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This controls what counts as "dense":
    /// - Higher threshold = only very crowded regions become clusters
    /// - Lower threshold = more regions qualify as clusters
    /// Start with 0.1 and adjust based on results.
    /// </para>
    /// </remarks>
    public double DensityThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the minimum number of points for a dense unit.
    /// </summary>
    /// <value>
    /// Minimum point count for a cell to be dense. Default is 0 (use fraction instead).
    /// </value>
    /// <remarks>
    /// <para>
    /// If set to a positive value, this overrides DensityThreshold.
    /// Useful when you want an absolute minimum rather than a fraction.
    /// </para>
    /// </remarks>
    public int MinPoints { get; set; } = 0;

    /// <summary>
    /// Gets or sets the maximum subspace dimensionality to explore.
    /// </summary>
    /// <value>
    /// Maximum number of dimensions to combine. Default is 0 (all dimensions).
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Limits how many features can be combined:
    /// - Setting to 2 only finds clusters in pairs of features
    /// - Setting to 3 finds clusters in up to 3 features at once
    /// - 0 means no limit (but exponentially slower with more features)
    /// </para>
    /// </remarks>
    public int MaxSubspaceDimensions { get; set; } = 0;

    /// <summary>
    /// Gets or sets whether to prune using the Apriori principle.
    /// </summary>
    /// <value>When true, uses Apriori pruning for efficiency. Default is true.</value>
    /// <remarks>
    /// <para>
    /// The Apriori principle: If a cell is not dense in k dimensions,
    /// it cannot be dense in any (k+1) dimension superset.
    /// This dramatically speeds up the algorithm.
    /// </para>
    /// </remarks>
    public bool UseAprioriPruning { get; set; } = true;
}
