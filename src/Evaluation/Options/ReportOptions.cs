using AiDotNet.Evaluation.Enums;

namespace AiDotNet.Evaluation.Options;

/// <summary>
/// Configuration options for evaluation report generation.
/// </summary>
/// <remarks>
/// <para>
/// Controls how evaluation results are formatted and presented, including output format,
/// detail level, and what sections to include.
/// </para>
/// <para>
/// <b>For Beginners:</b> After evaluating your model, you need to communicate results.
/// These options control:
/// <list type="bullet">
/// <item>Format (Markdown for docs, JSON for APIs, HTML for dashboards)</item>
/// <item>Detail level (summary for stakeholders, full for data scientists)</item>
/// <item>What to include (metrics, plots, recommendations)</item>
/// </list>
/// </para>
/// </remarks>
public class ReportOptions
{
    /// <summary>
    /// Output format for the report. Default: Markdown.
    /// </summary>
    public ReportFormat? Format { get; set; }

    /// <summary>
    /// Detail level for the report. Default: Standard.
    /// </summary>
    public ReportDetailLevel? DetailLevel { get; set; }

    /// <summary>
    /// Whether to include executive summary. Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A brief overview at the top summarizing key findings.
    /// Great for stakeholders who don't need all the details.</para>
    /// </remarks>
    public bool? IncludeExecutiveSummary { get; set; }

    /// <summary>
    /// Whether to include metric tables. Default: true.
    /// </summary>
    public bool? IncludeMetricTables { get; set; }

    /// <summary>
    /// Whether to include confidence intervals in tables. Default: true.
    /// </summary>
    public bool? IncludeConfidenceIntervals { get; set; }

    /// <summary>
    /// Whether to include visualization data/references. Default: true.
    /// </summary>
    public bool? IncludeVisualizations { get; set; }

    /// <summary>
    /// Whether to include interpretation/explanation for each metric. Default: based on detail level.
    /// </summary>
    public bool? IncludeInterpretations { get; set; }

    /// <summary>
    /// Whether to include recommendations for improvement. Default: true.
    /// </summary>
    public bool? IncludeRecommendations { get; set; }

    /// <summary>
    /// Whether to include warnings (data issues, assumption violations). Default: true.
    /// </summary>
    public bool? IncludeWarnings { get; set; }

    /// <summary>
    /// Whether to include raw data exports. Default: false.
    /// </summary>
    public bool? IncludeRawData { get; set; }

    /// <summary>
    /// Whether to include computation time statistics. Default: false.
    /// </summary>
    public bool? IncludeComputationTimes { get; set; }

    /// <summary>
    /// Title for the report. Default: "Model Evaluation Report".
    /// </summary>
    public string? Title { get; set; }

    /// <summary>
    /// Subtitle or description. Default: null.
    /// </summary>
    public string? Subtitle { get; set; }

    /// <summary>
    /// Author name for the report. Default: null.
    /// </summary>
    public string? Author { get; set; }

    /// <summary>
    /// Whether to include timestamp. Default: true.
    /// </summary>
    public bool? IncludeTimestamp { get; set; }

    /// <summary>
    /// Custom timestamp format. Default: ISO 8601.
    /// </summary>
    public string? TimestampFormat { get; set; }

    /// <summary>
    /// Number of decimal places for metrics. Default: 4.
    /// </summary>
    public int? DecimalPlaces { get; set; }

    /// <summary>
    /// Whether to use percentage format where appropriate. Default: true.
    /// </summary>
    public bool? UsePercentages { get; set; }

    /// <summary>
    /// Whether to include section numbers. Default: true.
    /// </summary>
    public bool? IncludeSectionNumbers { get; set; }

    /// <summary>
    /// Whether to include table of contents. Default: based on detail level.
    /// </summary>
    public bool? IncludeTableOfContents { get; set; }

    /// <summary>
    /// Sections to include (if null, all applicable sections). Default: null.
    /// </summary>
    public ReportSection[]? SectionsToInclude { get; set; }

    /// <summary>
    /// Sections to exclude. Default: null.
    /// </summary>
    public ReportSection[]? SectionsToExclude { get; set; }

    /// <summary>
    /// Custom CSS for HTML reports. Default: null (use default styling).
    /// </summary>
    public string? CustomCss { get; set; }

    /// <summary>
    /// Whether to embed images in HTML (base64). Default: true.
    /// </summary>
    public bool? EmbedImagesInHtml { get; set; }

    /// <summary>
    /// LaTeX document class for LaTeX reports. Default: "article".
    /// </summary>
    public string? LatexDocumentClass { get; set; }

    /// <summary>
    /// Whether to make LaTeX report standalone. Default: false.
    /// </summary>
    public bool? LatexStandalone { get; set; }

    /// <summary>
    /// Output file path. Default: null (return string).
    /// </summary>
    public string? OutputPath { get; set; }

    /// <summary>
    /// Encoding for output file. Default: UTF8.
    /// </summary>
    public string? OutputEncoding { get; set; }

    /// <summary>
    /// Whether to append to existing file. Default: false.
    /// </summary>
    public bool? AppendToFile { get; set; }

    /// <summary>
    /// Maximum table rows before truncation. Default: 100.
    /// </summary>
    public int? MaxTableRows { get; set; }

    /// <summary>
    /// Whether to include "For Beginners" explanations. Default: based on detail level.
    /// </summary>
    public bool? IncludeBeginnerExplanations { get; set; }

    /// <summary>
    /// Language for the report. Default: "en".
    /// </summary>
    public string? Language { get; set; }
}

/// <summary>
/// Detail level for evaluation reports.
/// </summary>
public enum ReportDetailLevel
{
    /// <summary>
    /// Minimal report with key metrics only.
    /// </summary>
    Summary = 0,

    /// <summary>
    /// Standard report with metrics and basic analysis.
    /// </summary>
    Standard = 1,

    /// <summary>
    /// Detailed report with all metrics, tests, and visualizations.
    /// </summary>
    Detailed = 2,

    /// <summary>
    /// Full report with everything including raw data and computation details.
    /// </summary>
    Full = 3
}

/// <summary>
/// Sections that can be included in evaluation reports.
/// </summary>
public enum ReportSection
{
    /// <summary>Executive summary.</summary>
    ExecutiveSummary = 0,

    /// <summary>Dataset statistics.</summary>
    DatasetStatistics = 1,

    /// <summary>Primary metrics table.</summary>
    PrimaryMetrics = 2,

    /// <summary>All metrics table.</summary>
    AllMetrics = 3,

    /// <summary>Per-class metrics.</summary>
    PerClassMetrics = 4,

    /// <summary>Confusion matrix.</summary>
    ConfusionMatrix = 5,

    /// <summary>ROC curve analysis.</summary>
    ROCCurve = 6,

    /// <summary>Precision-recall curve.</summary>
    PRCurve = 7,

    /// <summary>Calibration analysis.</summary>
    Calibration = 8,

    /// <summary>Threshold analysis.</summary>
    ThresholdAnalysis = 9,

    /// <summary>Residual analysis.</summary>
    ResidualAnalysis = 10,

    /// <summary>Influence analysis.</summary>
    InfluenceAnalysis = 11,

    /// <summary>Cross-validation results.</summary>
    CrossValidation = 12,

    /// <summary>Learning curves.</summary>
    LearningCurves = 13,

    /// <summary>Statistical tests.</summary>
    StatisticalTests = 14,

    /// <summary>Model comparison.</summary>
    ModelComparison = 15,

    /// <summary>Fairness analysis.</summary>
    FairnessAnalysis = 16,

    /// <summary>Robustness analysis.</summary>
    RobustnessAnalysis = 17,

    /// <summary>Uncertainty analysis.</summary>
    UncertaintyAnalysis = 18,

    /// <summary>Subgroup analysis.</summary>
    SubgroupAnalysis = 19,

    /// <summary>Feature importance.</summary>
    FeatureImportance = 20,

    /// <summary>Error analysis.</summary>
    ErrorAnalysis = 21,

    /// <summary>Recommendations.</summary>
    Recommendations = 22,

    /// <summary>Warnings and issues.</summary>
    Warnings = 23,

    /// <summary>Appendix with raw data.</summary>
    Appendix = 24
}
