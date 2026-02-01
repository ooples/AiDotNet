using AiDotNet.Evaluation.Enums;

namespace AiDotNet.Evaluation.Options;

/// <summary>
/// Configuration options for model evaluation.
/// </summary>
/// <remarks>
/// <para>
/// Controls which metrics are computed, confidence interval settings, and output preferences.
/// All properties are nullable with sensible defaults applied internally.
/// </para>
/// <para>
/// <b>For Beginners:</b> These options control how your model is evaluated. By default,
/// the framework auto-detects what kind of model you have (classification, regression, etc.)
/// and computes appropriate metrics. You can customize this if needed.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (float, double, etc.)</typeparam>
public class EvaluationOptions<T>
{
    /// <summary>
    /// Whether to compute confidence intervals for metrics. Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Confidence intervals show the uncertainty in your metrics.
    /// Instead of just "accuracy = 85%", you get "accuracy = 85% ± 2%".</para>
    /// </remarks>
    public bool? ComputeConfidenceIntervals { get; set; }

    /// <summary>
    /// Confidence level for intervals. Default: 0.95 (95% confidence).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Higher values (0.99) give wider but more certain intervals.
    /// Lower values (0.90) give narrower but less certain intervals. 0.95 is standard.</para>
    /// </remarks>
    public double? ConfidenceLevel { get; set; }

    /// <summary>
    /// Method for computing confidence intervals. Default: BCaBootstrap.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> BCa bootstrap is recommended for most cases as it
    /// corrects for bias and skewness in your data.</para>
    /// </remarks>
    public ConfidenceIntervalMethod? ConfidenceIntervalMethod { get; set; }

    /// <summary>
    /// Number of bootstrap samples for confidence intervals. Default: 1000.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More samples = more accurate intervals but slower.
    /// 1000 is a good balance. Use 10000+ for publication-quality results.</para>
    /// </remarks>
    public int? BootstrapSamples { get; set; }

    /// <summary>
    /// Random seed for reproducible confidence intervals. Default: null (random).
    /// </summary>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Averaging method for multi-class metrics. Default: Macro.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> For multi-class problems, this controls how per-class
    /// metrics are combined. Macro treats all classes equally, Weighted accounts for
    /// class sizes, Micro treats all samples equally.</para>
    /// </remarks>
    public AveragingMethod? MultiClassAveraging { get; set; }

    /// <summary>
    /// Which metrics to compute. Default: null (compute recommended metrics based on task type).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Leave this null to get recommended metrics automatically.
    /// Specify a list if you only want specific metrics.</para>
    /// </remarks>
    public IReadOnlyList<string>? MetricsToCompute { get; set; }

    /// <summary>
    /// Whether to compute all available metrics. Default: false.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Set to true to compute every possible metric.
    /// This is slower but gives a complete picture.</para>
    /// </remarks>
    public bool? ComputeAllMetrics { get; set; }

    /// <summary>
    /// Whether to include per-class metrics for classification. Default: true.
    /// </summary>
    public bool? IncludePerClassMetrics { get; set; }

    /// <summary>
    /// Whether to compute probability calibration metrics. Default: true for classifiers.
    /// </summary>
    public bool? ComputeCalibrationMetrics { get; set; }

    /// <summary>
    /// Whether to compute threshold analysis for binary classification. Default: true.
    /// </summary>
    public bool? ComputeThresholdAnalysis { get; set; }

    /// <summary>
    /// Threshold selection method for binary classification. Default: Youden.
    /// </summary>
    public ThresholdSelectionMethod? ThresholdSelectionMethod { get; set; }

    /// <summary>
    /// Custom classification threshold. Default: null (use optimal or 0.5).
    /// </summary>
    public T? ClassificationThreshold { get; set; }

    /// <summary>
    /// Cost of false positives for cost-sensitive evaluation. Default: 1.0.
    /// </summary>
    public T? FalsePositiveCost { get; set; }

    /// <summary>
    /// Cost of false negatives for cost-sensitive evaluation. Default: 1.0.
    /// </summary>
    public T? FalseNegativeCost { get; set; }

    /// <summary>
    /// Whether to perform residual analysis for regression. Default: true.
    /// </summary>
    public bool? PerformResidualAnalysis { get; set; }

    /// <summary>
    /// Whether to perform influence analysis for regression. Default: false (expensive).
    /// </summary>
    public bool? PerformInfluenceAnalysis { get; set; }

    /// <summary>
    /// Whether to compute feature importance via permutation. Default: false (expensive).
    /// </summary>
    public bool? ComputePermutationImportance { get; set; }

    /// <summary>
    /// Number of permutation rounds for importance. Default: 10.
    /// </summary>
    public int? PermutationRounds { get; set; }

    /// <summary>
    /// Number of parallel threads for computation. Default: null (use all available).
    /// </summary>
    public int? MaxDegreeOfParallelism { get; set; }

    /// <summary>
    /// Whether to track computation time for each metric. Default: false.
    /// </summary>
    public bool? TrackComputationTime { get; set; }

    /// <summary>
    /// Memory limit in bytes for evaluation. Default: null (no limit).
    /// </summary>
    public long? MemoryLimitBytes { get; set; }

    /// <summary>
    /// Positive class label for binary classification. Default: auto-detect.
    /// </summary>
    public T? PositiveClassLabel { get; set; }

    /// <summary>
    /// Whether to warn about potential issues (class imbalance, etc.). Default: true.
    /// </summary>
    public bool? EmitWarnings { get; set; }
}
