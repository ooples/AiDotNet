namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for robust aggregation strategies in federated learning.
/// </summary>
/// <remarks>
/// Robust aggregation helps defend against outliers and Byzantine (malicious or faulty) clients.
///
/// <b>For Beginners:</b> If some clients send "bad" updates (because of bugs, corrupted data, or attacks),
/// robust aggregation tries to reduce their impact so the global model stays stable.
///
/// Common strategies:
/// - Trimmed mean: drops extreme values before averaging.
/// - Median: takes the middle value per parameter (resistant to outliers).
/// - Krum/Multi-Krum: chooses the most "central" client updates by distance.
/// - Bulyan: combines Multi-Krum selection with trimming for stronger robustness.
/// </remarks>
public class RobustAggregationOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the trimming fraction for trimmed-mean based aggregations (0.0 to &lt; 0.5).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A trim fraction of 0.2 means "ignore the largest 20% and smallest 20% of values"
    /// when computing the mean for each parameter.
    /// </remarks>
    public double TrimFraction { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets the assumed number of Byzantine clients (f) for Krum/Multi-Krum/Bulyan.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> If you think up to 2 clients per round might be malicious, set this to 2.
    /// These algorithms require enough participating clients for the math to be valid.
    /// </remarks>
    public int ByzantineClientCount { get; set; } = 1;

    /// <summary>
    /// Gets or sets how many client updates (m) Multi-Krum keeps before averaging.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Multi-Krum first picks the "best" m client updates, then averages them.
    /// If set to 0 or less, a conservative default is chosen based on the number of clients and f.
    /// </remarks>
    public int MultiKrumSelectionCount { get; set; } = 0;

    /// <summary>
    /// Gets or sets whether robust strategies should use clientWeights when averaging selected updates.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Using clientWeights (often sample counts) can improve accuracy, but it may reduce
    /// robustness if a malicious client can manipulate its weight. Default is false (unweighted).
    /// </remarks>
    public bool UseClientWeightsWhenAveragingSelectedUpdates { get; set; } = false;

    /// <summary>
    /// Gets or sets the maximum number of iterations for geometric-median/RFA aggregation.
    /// </summary>
    public int GeometricMedianMaxIterations { get; set; } = 10;

    /// <summary>
    /// Gets or sets the convergence tolerance for geometric-median/RFA aggregation.
    /// </summary>
    public double GeometricMedianTolerance { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets a small epsilon used to avoid division-by-zero in RFA (Weiszfeld) updates.
    /// </summary>
    public double GeometricMedianEpsilon { get; set; } = 1e-12;
}
