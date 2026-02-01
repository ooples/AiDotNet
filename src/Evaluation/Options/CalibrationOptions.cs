using AiDotNet.Enums;
using AiDotNet.Evaluation.Enums;

namespace AiDotNet.Evaluation.Options;

/// <summary>
/// Configuration options for probability calibration analysis.
/// </summary>
/// <remarks>
/// <para>
/// Calibration measures whether predicted probabilities match actual frequencies.
/// A well-calibrated model that predicts 80% probability should be correct 80% of the time.
/// </para>
/// <para>
/// <b>For Beginners:</b> When your model says "80% chance this email is spam", calibration
/// checks if it's really spam 80% of the time. Poorly calibrated models might say 80% but
/// actually be right 95% or only 50% of the time. Good calibration is essential when you
/// use predicted probabilities for decisions.
/// </para>
/// </remarks>
public class CalibrationOptions
{
    /// <summary>
    /// Number of bins for reliability diagram. Default: 10.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The reliability diagram groups predictions into bins
    /// (e.g., 0-10%, 10-20%, etc.) and compares predicted vs actual rates. More bins give
    /// finer detail but noisier estimates.</para>
    /// </remarks>
    public int? NumberOfBins { get; set; }

    /// <summary>
    /// Binning strategy. Default: Uniform.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Uniform binning uses equal-width bins (0-10%, 10-20%...).
    /// Quantile binning uses equal-count bins (same number of samples in each). Quantile
    /// is better when predictions cluster in certain ranges.</para>
    /// </remarks>
    public BinningStrategy? BinningStrategy { get; set; }

    /// <summary>
    /// Calibration method to apply. Default: none (just analyze).
    /// </summary>
    public ProbabilityCalibrationMethod? CalibrationMethod { get; set; }

    /// <summary>
    /// Whether to compute Expected Calibration Error (ECE). Default: true.
    /// </summary>
    public bool? ComputeECE { get; set; }

    /// <summary>
    /// Whether to compute Maximum Calibration Error (MCE). Default: true.
    /// </summary>
    public bool? ComputeMCE { get; set; }

    /// <summary>
    /// Whether to compute Adaptive ECE. Default: false.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Adaptive ECE adjusts bin sizes based on data density,
    /// which can give more accurate calibration estimates.</para>
    /// </remarks>
    public bool? ComputeAdaptiveECE { get; set; }

    /// <summary>
    /// Whether to compute Brier score decomposition. Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Decomposes Brier score into:
    /// <list type="bullet">
    /// <item>Reliability: How close calibration is to perfect</item>
    /// <item>Resolution: How much predictions differ from overall rate</item>
    /// <item>Uncertainty: Inherent uncertainty in the data</item>
    /// </list>
    /// </para>
    /// </remarks>
    public bool? ComputeBrierDecomposition { get; set; }

    /// <summary>
    /// Whether to compute calibration slope/intercept. Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Perfect calibration has slope=1, intercept=0.
    /// Slope &lt; 1 means overconfidence, slope > 1 means underconfidence.</para>
    /// </remarks>
    public bool? ComputeCalibrationSlope { get; set; }

    /// <summary>
    /// Whether to run Hosmer-Lemeshow test. Default: true.
    /// </summary>
    public bool? RunHosmerLemeshowTest { get; set; }

    /// <summary>
    /// Whether to run Spiegelhalter's z-test. Default: true.
    /// </summary>
    public bool? RunSpiegelhalterTest { get; set; }

    /// <summary>
    /// Whether to compute per-class calibration. Default: true for multi-class.
    /// </summary>
    public bool? ComputePerClassCalibration { get; set; }

    /// <summary>
    /// Whether to generate reliability diagram data. Default: true.
    /// </summary>
    public bool? GenerateReliabilityDiagram { get; set; }

    /// <summary>
    /// Whether to generate confidence histogram data. Default: true.
    /// </summary>
    public bool? GenerateConfidenceHistogram { get; set; }

    /// <summary>
    /// Whether to use cross-validation for calibration metrics. Default: false.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Using CV for calibration gives more reliable estimates
    /// by evaluating on out-of-sample data.</para>
    /// </remarks>
    public bool? UseCrossValidation { get; set; }

    /// <summary>
    /// Number of CV folds if using cross-validation. Default: 5.
    /// </summary>
    public int? CVFolds { get; set; }

    /// <summary>
    /// Whether to compute confidence intervals for calibration metrics. Default: true.
    /// </summary>
    public bool? ComputeConfidenceIntervals { get; set; }

    /// <summary>
    /// Confidence level. Default: 0.95.
    /// </summary>
    public double? ConfidenceLevel { get; set; }

    /// <summary>
    /// Number of bootstrap samples for CIs. Default: 1000.
    /// </summary>
    public int? BootstrapSamples { get; set; }

    /// <summary>
    /// Temperature for temperature scaling calibration. Default: null (learn from data).
    /// </summary>
    public double? Temperature { get; set; }

    /// <summary>
    /// Regularization strength for calibrator fitting. Default: 0.
    /// </summary>
    public double? RegularizationStrength { get; set; }

    /// <summary>
    /// Maximum iterations for calibrator optimization. Default: 1000.
    /// </summary>
    public int? MaxIterations { get; set; }
}

/// <summary>
/// Strategies for binning predictions in calibration analysis.
/// </summary>
public enum BinningStrategy
{
    /// <summary>
    /// Uniform-width bins (0-10%, 10-20%, etc.).
    /// </summary>
    Uniform = 0,

    /// <summary>
    /// Equal-count bins (same number of samples per bin).
    /// </summary>
    Quantile = 1,

    /// <summary>
    /// Adaptive binning based on data density.
    /// </summary>
    Adaptive = 2
}
