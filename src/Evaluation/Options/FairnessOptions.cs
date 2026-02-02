using AiDotNet.Evaluation.Enums;

namespace AiDotNet.Evaluation.Options;

/// <summary>
/// Configuration options for fairness evaluation.
/// </summary>
/// <remarks>
/// <para>
/// Fairness evaluation checks whether a model treats different demographic groups equitably.
/// This is critical for applications in lending, hiring, criminal justice, healthcare, etc.
/// </para>
/// <para>
/// <b>For Beginners:</b> Fairness metrics check if your model discriminates against certain
/// groups. For example, in loan approval:
/// <list type="bullet">
/// <item>Does the model approve men and women at similar rates? (Demographic Parity)</item>
/// <item>Is the model equally accurate for all races? (Equalized Odds)</item>
/// <item>When it predicts "will repay", is it equally reliable for all groups? (Calibration)</item>
/// </list>
/// Note: Different fairness metrics often conflict - you can't optimize for all simultaneously.
/// </para>
/// </remarks>
public class FairnessOptions
{
    /// <summary>
    /// Protected attribute column indices. Required for fairness analysis.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These are the columns containing sensitive attributes
    /// like gender, race, age, etc. that you want to check for fairness.</para>
    /// </remarks>
    public int[]? ProtectedAttributeIndices { get; set; }

    /// <summary>
    /// Protected attribute names for reporting. Default: null (use indices).
    /// </summary>
    public string[]? ProtectedAttributeNames { get; set; }

    /// <summary>
    /// Reference/privileged group values for each protected attribute. Default: null (auto-detect).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The "reference group" is typically the historically advantaged
    /// group that you compare other groups against. For example, in gender: male; in age: young.</para>
    /// </remarks>
    public object[]? PrivilegedGroupValues { get; set; }

    /// <summary>
    /// Fairness constraints to evaluate. Default: common metrics.
    /// </summary>
    public FairnessConstraint[]? MetricsToCompute { get; set; }

    /// <summary>
    /// Threshold for flagging disparity. Default: 0.8 (80% rule).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The "80% rule" from US employment law says that if a
    /// protected group has less than 80% the selection rate of the reference group, there
    /// may be adverse impact.</para>
    /// </remarks>
    public double? DisparityThreshold { get; set; }

    /// <summary>
    /// Whether to compute intersectional fairness. Default: false.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Intersectionality examines combinations of attributes
    /// (e.g., Black women vs White men) rather than one attribute at a time. Important because
    /// discrimination can compound.</para>
    /// </remarks>
    public bool? ComputeIntersectionalFairness { get; set; }

    /// <summary>
    /// Maximum intersection size. Default: 2.
    /// </summary>
    public int? MaxIntersectionSize { get; set; }

    /// <summary>
    /// Minimum group size for reliable metrics. Default: 30.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Groups smaller than this will have unreliable metrics
    /// due to small sample size.</para>
    /// </remarks>
    public int? MinGroupSize { get; set; }

    /// <summary>
    /// Whether to compute conditional fairness. Default: false.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Conditional fairness controls for legitimate factors.
    /// For example, comparing loan approval rates for people with similar credit scores,
    /// not just overall rates.</para>
    /// </remarks>
    public bool? ComputeConditionalFairness { get; set; }

    /// <summary>
    /// Legitimate feature indices for conditional fairness. Default: null.
    /// </summary>
    public int[]? LegitimateFeatureIndices { get; set; }

    /// <summary>
    /// Whether to compute confidence intervals. Default: true.
    /// </summary>
    public bool? ComputeConfidenceIntervals { get; set; }

    /// <summary>
    /// Confidence level. Default: 0.95.
    /// </summary>
    public double? ConfidenceLevel { get; set; }

    /// <summary>
    /// Number of bootstrap samples. Default: 1000.
    /// </summary>
    public int? BootstrapSamples { get; set; }

    /// <summary>
    /// Whether to perform significance tests for disparities. Default: true.
    /// </summary>
    public bool? PerformSignificanceTests { get; set; }

    /// <summary>
    /// Significance level for tests. Default: 0.05.
    /// </summary>
    public double? SignificanceLevel { get; set; }

    /// <summary>
    /// Whether to generate Theil index decomposition. Default: false.
    /// </summary>
    public bool? ComputeTheilDecomposition { get; set; }

    /// <summary>
    /// Whether to compute counterfactual fairness (requires causal model). Default: false.
    /// </summary>
    public bool? ComputeCounterfactualFairness { get; set; }

    /// <summary>
    /// Whether to flag unfair predictions. Default: false.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When true, individual predictions that may be unfair
    /// (e.g., different outcome for similar individuals from different groups) are flagged.</para>
    /// </remarks>
    public bool? FlagUnfairPredictions { get; set; }

    /// <summary>
    /// Threshold for individual unfairness. Default: 0.1.
    /// </summary>
    public double? IndividualUnfairnessThreshold { get; set; }

    /// <summary>
    /// Report format. Default: Markdown.
    /// </summary>
    public ReportFormat? ReportFormat { get; set; }

    /// <summary>
    /// Whether to include recommendations. Default: true.
    /// </summary>
    public bool? IncludeRecommendations { get; set; }
}
