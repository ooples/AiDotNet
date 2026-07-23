using AiDotNet.Evaluation.Enums;

namespace AiDotNet.Evaluation.Options;

/// <summary>
/// Configuration options for learning curve analysis.
/// </summary>
/// <remarks>
/// <para>
/// Learning curves show how model performance changes as training data size increases.
/// This helps diagnose bias-variance tradeoffs and determine if more data would help.
/// </para>
/// <para>
/// <b>For Beginners:</b> A learning curve plots model performance (y-axis) against training
/// set size (x-axis). It helps answer questions like:
/// <list type="bullet">
/// <item>Will collecting more data help? (yes if training curve is still improving)</item>
/// <item>Is the model overfitting? (yes if train score >> validation score)</item>
/// <item>Is the model underfitting? (yes if both scores are low and flat)</item>
/// </list>
/// </para>
/// </remarks>
public class LearningCurveOptions
{
    /// <summary>
    /// Training set sizes to evaluate. Default: 10 evenly spaced points from 10% to 100%.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This specifies which training set sizes to test.
    /// Can be absolute numbers (e.g., [100, 500, 1000]) or ratios (e.g., [0.1, 0.5, 1.0]).</para>
    /// </remarks>
    public double[]? TrainSizes { get; set; }

    /// <summary>
    /// Whether TrainSizes are ratios (0-1) or absolute counts. Default: true (ratios).
    /// </summary>
    public bool? TrainSizesAreRatios { get; set; }

    /// <summary>
    /// Number of training sizes to auto-generate. Default: 10.
    /// </summary>
    public int? NumberOfPoints { get; set; }

    /// <summary>
    /// Minimum training size ratio. Default: 0.1 (10%).
    /// </summary>
    public double? MinTrainSizeRatio { get; set; }

    /// <summary>
    /// Cross-validation strategy for each point. Default: StratifiedKFold for classification.
    /// </summary>
    public CrossValidationStrategy? CVStrategy { get; set; }

    /// <summary>
    /// Number of CV folds. Default: 5.
    /// </summary>
    public int? CVFolds { get; set; }

    /// <summary>
    /// Whether to shuffle before splitting. Default: true.
    /// </summary>
    public bool? Shuffle { get; set; }

    /// <summary>
    /// Random seed for reproducibility. Default: null.
    /// </summary>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Metrics to track on learning curve. Default: primary metric for task type.
    /// </summary>
    public IReadOnlyList<string>? MetricsToTrack { get; set; }

    /// <summary>
    /// Whether to compute confidence intervals at each point. Default: true.
    /// </summary>
    public bool? ComputeConfidenceIntervals { get; set; }

    /// <summary>
    /// Confidence level for intervals. Default: 0.95.
    /// </summary>
    public double? ConfidenceLevel { get; set; }

    /// <summary>
    /// Whether to compute bias-variance decomposition. Default: false (expensive).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Bias-variance decomposition separates error into:
    /// <list type="bullet">
    /// <item>Bias: Error from overly simple model (underfitting)</item>
    /// <item>Variance: Error from sensitivity to training data (overfitting)</item>
    /// <item>Noise: Irreducible error in the data</item>
    /// </list>
    /// </para>
    /// </remarks>
    public bool? ComputeBiasVarianceDecomposition { get; set; }

    /// <summary>
    /// Number of bootstrap samples for bias-variance decomposition. Default: 100.
    /// </summary>
    public int? BiasVarianceBootstrapSamples { get; set; }

    /// <summary>
    /// Whether to run evaluations in parallel. Default: true.
    /// </summary>
    public bool? ParallelExecution { get; set; }

    /// <summary>
    /// Maximum degree of parallelism. Default: null (use all cores).
    /// </summary>
    public int? MaxDegreeOfParallelism { get; set; }

    /// <summary>
    /// Whether to diagnose bias-variance condition. Default: true.
    /// </summary>
    public bool? DiagnoseBiasVariance { get; set; }

    /// <summary>
    /// Threshold for high variance diagnosis (train-test gap). Default: 0.1.
    /// </summary>
    public double? HighVarianceThreshold { get; set; }

    /// <summary>
    /// Threshold for high bias diagnosis (low scores). Default: 0.7.
    /// </summary>
    public double? HighBiasThreshold { get; set; }

    /// <summary>
    /// Whether to extrapolate learning curve beyond available data. Default: false.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Extrapolation predicts how the model would perform
    /// with more data than you currently have. This is an estimate and may not be accurate.</para>
    /// </remarks>
    public bool? ExtrapolateCurve { get; set; }

    /// <summary>
    /// Extrapolation target size (ratio or absolute). Default: 2.0 (double current data).
    /// </summary>
    public double? ExtrapolationTarget { get; set; }
}
