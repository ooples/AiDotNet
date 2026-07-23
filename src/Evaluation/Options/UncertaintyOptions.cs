namespace AiDotNet.Evaluation.Options;

/// <summary>
/// Configuration options for uncertainty quantification.
/// </summary>
/// <remarks>
/// <para>
/// Uncertainty quantification measures how confident a model is in its predictions.
/// This is crucial for high-stakes applications where knowing "I don't know" is valuable.
/// </para>
/// <para>
/// <b>For Beginners:</b> Uncertainty tells you how confident the model is:
/// <list type="bullet">
/// <item><b>Aleatoric uncertainty:</b> Inherent noise in the data (can't be reduced)</item>
/// <item><b>Epistemic uncertainty:</b> Model's lack of knowledge (can be reduced with more data)</item>
/// </list>
/// A good uncertainty estimate lets you flag predictions the model isn't sure about,
/// which might need human review.
/// </para>
/// </remarks>
public class UncertaintyOptions
{
    /// <summary>
    /// Whether to compute predictive entropy. Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Entropy measures total uncertainty. High entropy means
    /// the model is uncertain between multiple possible outputs.</para>
    /// </remarks>
    public bool? ComputePredictiveEntropy { get; set; }

    /// <summary>
    /// Whether to decompose uncertainty into aleatoric/epistemic. Default: false (requires ensemble).
    /// </summary>
    public bool? DecomposeUncertainty { get; set; }

    /// <summary>
    /// Method for uncertainty decomposition. Default: Ensemble.
    /// </summary>
    public UncertaintyDecompositionMethod? DecompositionMethod { get; set; }

    /// <summary>
    /// Number of ensemble members for ensemble-based uncertainty. Default: 10.
    /// </summary>
    public int? EnsembleSize { get; set; }

    /// <summary>
    /// Number of Monte Carlo dropout samples. Default: 30.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> MC Dropout runs the model multiple times with dropout
    /// enabled at inference time. The variance of predictions estimates uncertainty.</para>
    /// </remarks>
    public int? MCDropoutSamples { get; set; }

    /// <summary>
    /// Dropout rate for MC Dropout. Default: 0.1.
    /// </summary>
    public double? MCDropoutRate { get; set; }

    /// <summary>
    /// Whether to compute prediction intervals. Default: true for regression.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Prediction intervals give a range like "the value will
    /// be between 45 and 55 with 95% confidence". Unlike confidence intervals for the mean,
    /// these cover individual predictions.</para>
    /// </remarks>
    public bool? ComputePredictionIntervals { get; set; }

    /// <summary>
    /// Coverage level for prediction intervals. Default: 0.95.
    /// </summary>
    public double? PredictionIntervalCoverage { get; set; }

    /// <summary>
    /// Method for prediction interval estimation. Default: Quantile.
    /// </summary>
    public PredictionIntervalMethod? IntervalMethod { get; set; }

    /// <summary>
    /// Whether to compute calibrated uncertainty. Default: true.
    /// </summary>
    public bool? CalibrateUncertainty { get; set; }

    /// <summary>
    /// Uncertainty calibration method. Default: TemperatureScaling.
    /// </summary>
    public UncertaintyCalibrationMethod? CalibrationMethod { get; set; }

    /// <summary>
    /// Whether to compute PICP (Prediction Interval Coverage Probability). Default: true.
    /// </summary>
    public bool? ComputePICP { get; set; }

    /// <summary>
    /// Whether to compute MPIW (Mean Prediction Interval Width). Default: true.
    /// </summary>
    public bool? ComputeMPIW { get; set; }

    /// <summary>
    /// Whether to compute CWC (Coverage Width Criterion). Default: true.
    /// </summary>
    public bool? ComputeCWC { get; set; }

    /// <summary>
    /// Penalty factor for CWC calculation. Default: 50.
    /// </summary>
    public double? CWCPenaltyFactor { get; set; }

    /// <summary>
    /// Whether to compute sharpness score. Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Sharpness measures how narrow the prediction intervals are.
    /// Narrower (sharper) is better, as long as coverage is maintained.</para>
    /// </remarks>
    public bool? ComputeSharpness { get; set; }

    /// <summary>
    /// Whether to check for quantile crossing. Default: true for quantile regression.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Quantile crossing happens when lower quantile predictions
    /// exceed upper quantile predictions (e.g., 25th percentile > 75th percentile), which
    /// is physically impossible.</para>
    /// </remarks>
    public bool? CheckQuantileCrossing { get; set; }

    /// <summary>
    /// Quantiles to compute for quantile regression. Default: [0.025, 0.5, 0.975].
    /// </summary>
    public double[]? Quantiles { get; set; }

    /// <summary>
    /// Whether to compute reliability diagrams for uncertainty. Default: false.
    /// </summary>
    public bool? ComputeUncertaintyReliability { get; set; }

    /// <summary>
    /// Number of bins for uncertainty reliability diagram. Default: 10.
    /// </summary>
    public int? ReliabilityBins { get; set; }

    /// <summary>
    /// Whether to identify high-uncertainty samples. Default: true.
    /// </summary>
    public bool? IdentifyHighUncertaintySamples { get; set; }

    /// <summary>
    /// Threshold for high uncertainty (percentile). Default: 0.95.
    /// </summary>
    public double? HighUncertaintyThreshold { get; set; }

    /// <summary>
    /// Whether to compute mutual information (epistemic uncertainty). Default: false.
    /// </summary>
    public bool? ComputeMutualInformation { get; set; }

    /// <summary>
    /// Random seed for reproducibility. Default: null.
    /// </summary>
    public int? RandomSeed { get; set; }
}

/// <summary>
/// Methods for decomposing predictive uncertainty into components.
/// </summary>
public enum UncertaintyDecompositionMethod
{
    /// <summary>
    /// Use ensemble of models.
    /// </summary>
    Ensemble = 0,

    /// <summary>
    /// Monte Carlo Dropout.
    /// </summary>
    MCDropout = 1,

    /// <summary>
    /// Deep ensemble (multiple trained networks).
    /// </summary>
    DeepEnsemble = 2,

    /// <summary>
    /// Bayesian neural network.
    /// </summary>
    BayesianNN = 3,

    /// <summary>
    /// Bootstrapped models.
    /// </summary>
    Bootstrap = 4
}

/// <summary>
/// Methods for computing prediction intervals.
/// </summary>
public enum PredictionIntervalMethod
{
    /// <summary>
    /// Quantile regression.
    /// </summary>
    Quantile = 0,

    /// <summary>
    /// Bootstrap prediction intervals.
    /// </summary>
    Bootstrap = 1,

    /// <summary>
    /// Conformal prediction.
    /// </summary>
    Conformal = 2,

    /// <summary>
    /// Delta method (parametric).
    /// </summary>
    DeltaMethod = 3,

    /// <summary>
    /// Gaussian process posterior.
    /// </summary>
    GaussianProcess = 4,

    /// <summary>
    /// Ensemble variance.
    /// </summary>
    EnsembleVariance = 5
}

/// <summary>
/// Methods for calibrating uncertainty estimates.
/// </summary>
public enum UncertaintyCalibrationMethod
{
    /// <summary>
    /// Temperature scaling.
    /// </summary>
    TemperatureScaling = 0,

    /// <summary>
    /// Isotonic regression.
    /// </summary>
    Isotonic = 1,

    /// <summary>
    /// Platt scaling.
    /// </summary>
    Platt = 2,

    /// <summary>
    /// Histogram binning.
    /// </summary>
    HistogramBinning = 3,

    /// <summary>
    /// No calibration.
    /// </summary>
    None = 4
}
