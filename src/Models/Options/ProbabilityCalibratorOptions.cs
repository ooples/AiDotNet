namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for probability calibrator.
/// </summary>
/// <remarks>
/// <para>
/// Probability calibration ensures that predicted probabilities are reliable. When a
/// calibrated model says "70% chance", you should expect the event to occur about 70%
/// of the time across many such predictions.
/// </para>
/// <para>
/// <b>For Beginners:</b> Many models give probability-like outputs, but these aren't
/// always true probabilities. For example:
///
/// - A decision tree might output 0.9 for all "yes" predictions (overconfident)
/// - A neural network might output values between 0.3-0.7 only (underconfident)
/// - SVM outputs are not probabilities at all without calibration
///
/// Calibration transforms these outputs into reliable probabilities:
/// - If you see 1000 predictions of "60% chance", about 600 should be correct
/// - This is crucial for decision-making (medical diagnoses, financial risk, etc.)
///
/// <b>When to use:</b>
/// - Whenever you need actual probability estimates (not just rankings)
/// - When combining predictions from different models
/// - For threshold-based decisions where probability values matter
/// </para>
/// </remarks>
public class ProbabilityCalibratorOptions
{
    /// <summary>
    /// Gets or sets the calibration method.
    /// </summary>
    /// <value>Default is PlattScaling.</value>
    public ProbabilityCalibratorMethod CalibratorMethod { get; set; } = ProbabilityCalibratorMethod.PlattScaling;

    /// <summary>
    /// Gets or sets the number of bins for histogram-based methods.
    /// </summary>
    /// <value>Default is 10.</value>
    public int NumBins { get; set; } = 10;

    /// <summary>
    /// Gets or sets the maximum number of iterations for optimization-based methods.
    /// </summary>
    /// <value>Default is 100.</value>
    public int MaxIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets the convergence tolerance.
    /// </summary>
    /// <value>Default is 1e-6.</value>
    public double Tolerance { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the regularization strength.
    /// </summary>
    /// <value>Default is 0.0.</value>
    public double Regularization { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the learning rate for gradient-based methods.
    /// </summary>
    /// <value>Default is 0.01.</value>
    public double LearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    public int? Seed { get; set; }

    /// <summary>
    /// Gets or sets whether to use cross-validation for calibration.
    /// </summary>
    /// <value>Default is false.</value>
    /// <remarks>
    /// When true, the calibration is trained on cross-validated predictions to avoid
    /// overfitting to the calibration data.
    /// </remarks>
    public bool UseCrossValidation { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of cross-validation folds.
    /// </summary>
    /// <value>Default is 5.</value>
    public int NumFolds { get; set; } = 5;
}

/// <summary>
/// Types of calibration methods.
/// </summary>
public enum ProbabilityCalibratorMethod
{
    /// <summary>
    /// Platt Scaling - fits sigmoid (logistic regression) to scores.
    /// Good for most cases, especially for SVM outputs.
    /// </summary>
    PlattScaling,

    /// <summary>
    /// Isotonic Regression - non-parametric, monotonic calibration.
    /// More flexible than Platt but needs more data and may overfit.
    /// </summary>
    IsotonicRegression,

    /// <summary>
    /// Temperature Scaling - divides logits by a learned temperature.
    /// Popular for neural networks, preserves accuracy while calibrating.
    /// </summary>
    TemperatureScaling,

    /// <summary>
    /// Beta Calibration - fits beta distribution parameters.
    /// Works well when predictions are bounded in (0,1).
    /// </summary>
    BetaCalibration,

    /// <summary>
    /// Histogram Binning - assigns average probability to each bin.
    /// Simple and interpretable but may need many samples.
    /// </summary>
    HistogramBinning,

    /// <summary>
    /// Bayesian Binning into Quantiles (BBQ) - adaptive binning.
    /// Good balance between flexibility and reliability.
    /// </summary>
    BayesianBinning,

    /// <summary>
    /// Venn-ABERS - provides probability intervals, not point estimates.
    /// Useful when you need calibration guarantees.
    /// </summary>
    VennABERS
}
