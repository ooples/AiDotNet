using AiDotNet.Evaluation.Enums;

namespace AiDotNet.Evaluation.Options;

/// <summary>
/// Configuration options for subgroup (slice-based) analysis.
/// </summary>
/// <remarks>
/// <para>
/// Subgroup analysis computes metrics for different slices of your data, helping identify
/// where the model performs well or poorly. This is essential for understanding model behavior.
/// </para>
/// <para>
/// <b>For Beginners:</b> Your model might have 90% accuracy overall, but:
/// <list type="bullet">
/// <item>95% accuracy on common cases, 50% on rare cases</item>
/// <item>92% on young users, 75% on elderly users</item>
/// <item>98% on clean data, 60% on noisy data</item>
/// </list>
/// Subgroup analysis reveals these hidden variations that overall metrics miss.
/// </para>
/// </remarks>
public class SubgroupAnalysisOptions
{
    /// <summary>
    /// Feature indices to slice by. Default: null (auto-detect categorical features).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Specify which features to use for slicing. For example,
    /// if feature 3 is "age_group", slicing by it shows metrics for each age group.</para>
    /// </remarks>
    public int[]? SliceFeatureIndices { get; set; }

    /// <summary>
    /// Feature names for slicing (alternative to indices). Default: null.
    /// </summary>
    public string[]? SliceFeatureNames { get; set; }

    /// <summary>
    /// Whether to auto-detect categorical features for slicing. Default: true.
    /// </summary>
    public bool? AutoDetectCategoricalFeatures { get; set; }

    /// <summary>
    /// Maximum unique values for auto-detected categorical features. Default: 20.
    /// </summary>
    public int? MaxCategoricalUniques { get; set; }

    /// <summary>
    /// Whether to bin continuous features for slicing. Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> For continuous features like "income", create bins
    /// like "low", "medium", "high" to slice by.</para>
    /// </remarks>
    public bool? BinContinuousFeatures { get; set; }

    /// <summary>
    /// Number of bins for continuous features. Default: 5.
    /// </summary>
    public int? ContinuousBins { get; set; }

    /// <summary>
    /// Binning strategy for continuous features. Default: Quantile.
    /// </summary>
    public ContinuousBinningStrategy? BinningStrategy { get; set; }

    /// <summary>
    /// Custom bin edges for specific features. Default: null.
    /// </summary>
    public Dictionary<int, double[]>? CustomBinEdges { get; set; }

    /// <summary>
    /// Whether to compute intersections of slices. Default: false.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Intersections look at combinations like "young AND male"
    /// rather than just "young" or "male" separately.</para>
    /// </remarks>
    public bool? ComputeSliceIntersections { get; set; }

    /// <summary>
    /// Maximum intersection size. Default: 2.
    /// </summary>
    public int? MaxIntersectionSize { get; set; }

    /// <summary>
    /// Minimum samples per slice for reliable metrics. Default: 30.
    /// </summary>
    public int? MinSliceSize { get; set; }

    /// <summary>
    /// Metrics to compute per slice. Default: same as overall evaluation.
    /// </summary>
    public IReadOnlyList<string>? MetricsToCompute { get; set; }

    /// <summary>
    /// Whether to compute confidence intervals per slice. Default: true.
    /// </summary>
    public bool? ComputeConfidenceIntervals { get; set; }

    /// <summary>
    /// Confidence level. Default: 0.95.
    /// </summary>
    public double? ConfidenceLevel { get; set; }

    /// <summary>
    /// Whether to identify underperforming slices. Default: true.
    /// </summary>
    public bool? IdentifyUnderperformingSlices { get; set; }

    /// <summary>
    /// Threshold for underperformance (relative to overall). Default: 0.9.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A threshold of 0.9 means slices performing &lt; 90% of
    /// the overall metric are flagged as underperforming.</para>
    /// </remarks>
    public double? UnderperformanceThreshold { get; set; }

    /// <summary>
    /// Whether to perform statistical test vs overall. Default: true.
    /// </summary>
    public bool? TestSignificantDifference { get; set; }

    /// <summary>
    /// Significance level for difference tests. Default: 0.05.
    /// </summary>
    public double? SignificanceLevel { get; set; }

    /// <summary>
    /// Whether to apply multiple testing correction. Default: true.
    /// </summary>
    public bool? ApplyMultipleTestingCorrection { get; set; }

    /// <summary>
    /// Whether to compute slice importance (which slices matter most). Default: false.
    /// </summary>
    public bool? ComputeSliceImportance { get; set; }

    /// <summary>
    /// Whether to generate error analysis per slice. Default: true.
    /// </summary>
    public bool? GenerateSliceErrorAnalysis { get; set; }

    /// <summary>
    /// Maximum slices to include in report. Default: 50.
    /// </summary>
    public int? MaxSlicesInReport { get; set; }

    /// <summary>
    /// Sort order for slices in report. Default: ByPerformance.
    /// </summary>
    public SliceSortOrder? SliceSortOrder { get; set; }

    /// <summary>
    /// Whether to include slice distribution statistics. Default: true.
    /// </summary>
    public bool? IncludeDistributionStats { get; set; }

    /// <summary>
    /// Report format. Default: Markdown.
    /// </summary>
    public ReportFormat? ReportFormat { get; set; }

    /// <summary>
    /// Whether to include recommendations per slice. Default: true.
    /// </summary>
    public bool? IncludeRecommendations { get; set; }

    /// <summary>
    /// Whether to run analyses in parallel. Default: true.
    /// </summary>
    public bool? ParallelExecution { get; set; }
}

/// <summary>
/// Strategy for binning continuous features.
/// </summary>
public enum ContinuousBinningStrategy
{
    /// <summary>Equal-width bins.</summary>
    Uniform = 0,

    /// <summary>Equal-count bins.</summary>
    Quantile = 1,

    /// <summary>K-means clustering.</summary>
    KMeans = 2,

    /// <summary>Custom bin edges.</summary>
    Custom = 3
}

/// <summary>
/// Sort order for slices in reports.
/// </summary>
public enum SliceSortOrder
{
    /// <summary>Sort by performance (worst first).</summary>
    ByPerformance = 0,

    /// <summary>Sort by slice size (largest first).</summary>
    BySize = 1,

    /// <summary>Sort by performance gap from overall.</summary>
    ByGap = 2,

    /// <summary>Alphabetical by slice name.</summary>
    Alphabetical = 3
}
