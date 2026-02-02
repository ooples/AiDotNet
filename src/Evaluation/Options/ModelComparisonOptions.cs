using AiDotNet.Evaluation.Enums;

namespace AiDotNet.Evaluation.Options;

/// <summary>
/// Configuration options for comparing multiple models.
/// </summary>
/// <remarks>
/// <para>
/// Model comparison provides statistical tests to determine if one model is significantly
/// better than another, or to rank multiple models.
/// </para>
/// <para>
/// <b>For Beginners:</b> When comparing models, you might find Model A has 85% accuracy and
/// Model B has 87% accuracy. But is that 2% difference real or just noise? Statistical tests
/// help answer this by computing p-values (probability the difference is due to chance).
/// </para>
/// </remarks>
public class ModelComparisonOptions
{
    /// <summary>
    /// Primary metric for comparison. Default: Accuracy for classification, RMSE for regression.
    /// </summary>
    public string? PrimaryMetric { get; set; }

    /// <summary>
    /// Significance level (alpha) for hypothesis tests. Default: 0.05.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If p-value &lt; alpha, the difference is "statistically significant".
    /// 0.05 (5%) is standard, meaning we accept a 5% chance of a false positive.</para>
    /// </remarks>
    public double? SignificanceLevel { get; set; }

    /// <summary>
    /// Whether to apply multiple testing correction. Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When comparing many models, the chance of false positives
    /// increases. Corrections like Bonferroni adjust for this.</para>
    /// </remarks>
    public bool? ApplyMultipleTestingCorrection { get; set; }

    /// <summary>
    /// Multiple testing correction method. Default: BonferroniHolm.
    /// </summary>
    public MultipleTestingCorrectionMethod? CorrectionMethod { get; set; }

    /// <summary>
    /// Statistical test for pairwise comparison. Default: auto-select based on data.
    /// </summary>
    public PairwiseComparisonTest? PairwiseTest { get; set; }

    /// <summary>
    /// Test for comparing multiple models simultaneously. Default: FriedmanTest.
    /// </summary>
    public MultipleComparisonTest? MultipleComparisonTest { get; set; }

    /// <summary>
    /// Post-hoc test after significant Friedman test. Default: NemenyiTest.
    /// </summary>
    public PostHocTest? PostHocTest { get; set; }

    /// <summary>
    /// Cross-validation strategy for comparison. Default: RepeatedStratifiedKFold.
    /// </summary>
    public CrossValidationStrategy? CVStrategy { get; set; }

    /// <summary>
    /// Number of CV folds. Default: 10.
    /// </summary>
    public int? CVFolds { get; set; }

    /// <summary>
    /// Number of CV repetitions. Default: 10.
    /// </summary>
    public int? CVRepeats { get; set; }

    /// <summary>
    /// Whether to compute effect sizes. Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Effect size tells you how big the difference is, not just
    /// whether it's significant. A difference can be statistically significant but practically
    /// tiny (or vice versa).</para>
    /// </remarks>
    public bool? ComputeEffectSizes { get; set; }

    /// <summary>
    /// Effect size measure to use. Default: CohensD.
    /// </summary>
    public EffectSizeMeasure? EffectSizeMeasure { get; set; }

    /// <summary>
    /// Whether to compute confidence intervals for differences. Default: true.
    /// </summary>
    public bool? ComputeConfidenceIntervals { get; set; }

    /// <summary>
    /// Confidence level for intervals. Default: 0.95.
    /// </summary>
    public double? ConfidenceLevel { get; set; }

    /// <summary>
    /// Number of bootstrap samples. Default: 1000.
    /// </summary>
    public int? BootstrapSamples { get; set; }

    /// <summary>
    /// Whether to use Bayesian comparison. Default: false.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Bayesian comparison gives probability statements like
    /// "there's an 85% probability Model A is better" instead of binary yes/no decisions.</para>
    /// </remarks>
    public bool? UseBayesianComparison { get; set; }

    /// <summary>
    /// ROPE (Region of Practical Equivalence) for Bayesian comparison. Default: 0.01.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The ROPE defines what difference is "practically equivalent".
    /// If the true difference falls in the ROPE, the models are practically the same.</para>
    /// </remarks>
    public double? RopeWidth { get; set; }

    /// <summary>
    /// Whether to generate ranking of models. Default: true.
    /// </summary>
    public bool? GenerateRanking { get; set; }

    /// <summary>
    /// Whether to create critical difference diagram data. Default: true for 3+ models.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A critical difference diagram visualizes model rankings
    /// and shows which differences are statistically significant.</para>
    /// </remarks>
    public bool? GenerateCriticalDifferenceDiagram { get; set; }

    /// <summary>
    /// Random seed for reproducibility. Default: null.
    /// </summary>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Whether to run comparisons in parallel. Default: true.
    /// </summary>
    public bool? ParallelExecution { get; set; }
}

/// <summary>
/// Methods for correcting p-values when performing multiple comparisons.
/// </summary>
public enum MultipleTestingCorrectionMethod
{
    /// <summary>No correction applied.</summary>
    None = 0,

    /// <summary>Bonferroni correction (most conservative).</summary>
    Bonferroni = 1,

    /// <summary>Bonferroni-Holm step-down procedure.</summary>
    BonferroniHolm = 2,

    /// <summary>Benjamini-Hochberg false discovery rate.</summary>
    BenjaminiHochberg = 3,

    /// <summary>Benjamini-Yekutieli for dependent tests.</summary>
    BenjaminiYekutieli = 4
}

/// <summary>
/// Tests for pairwise model comparison.
/// </summary>
public enum PairwiseComparisonTest
{
    /// <summary>Paired t-test (parametric).</summary>
    PairedTTest = 0,

    /// <summary>Wilcoxon signed-rank test (non-parametric).</summary>
    WilcoxonSignedRank = 1,

    /// <summary>McNemar's test for classifiers.</summary>
    McNemarsTest = 2,

    /// <summary>Corrected resampled t-test (Nadeau-Bengio).</summary>
    CorrectedResampledTTest = 3,

    /// <summary>5x2 CV paired t-test (Dietterich).</summary>
    FiveByTwoCVTest = 4,

    /// <summary>Combined 5x2 CV F-test.</summary>
    CombinedFTest = 5,

    /// <summary>Bayesian signed-rank test.</summary>
    BayesianSignedRank = 6
}

/// <summary>
/// Tests for comparing multiple models simultaneously.
/// </summary>
public enum MultipleComparisonTest
{
    /// <summary>Friedman test (non-parametric).</summary>
    FriedmanTest = 0,

    /// <summary>Cochran's Q test.</summary>
    CochransQ = 1,

    /// <summary>Repeated measures ANOVA (parametric).</summary>
    RepeatedMeasuresAnova = 2
}

/// <summary>
/// Post-hoc tests after significant omnibus test.
/// </summary>
public enum PostHocTest
{
    /// <summary>Nemenyi test.</summary>
    NemenyiTest = 0,

    /// <summary>Bonferroni-Dunn test vs control.</summary>
    BonferroniDunn = 1,

    /// <summary>Wilcoxon-Holm step-down.</summary>
    WilcoxonHolm = 2,

    /// <summary>All pairwise comparisons.</summary>
    AllPairwise = 3
}

/// <summary>
/// Effect size measures for comparing models.
/// </summary>
public enum EffectSizeMeasure
{
    /// <summary>Cohen's d (standardized mean difference).</summary>
    CohensD = 0,

    /// <summary>Hedges' g (bias-corrected Cohen's d).</summary>
    HedgesG = 1,

    /// <summary>Glass's delta (uses control group std).</summary>
    GlassDelta = 2,

    /// <summary>Cliff's delta (non-parametric).</summary>
    CliffsDelta = 3,

    /// <summary>Common language effect size.</summary>
    CommonLanguageEffectSize = 4,

    /// <summary>Probability of superiority.</summary>
    ProbabilityOfSuperiority = 5
}
