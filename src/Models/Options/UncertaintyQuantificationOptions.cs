using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for enabling uncertainty quantification during inference.
/// </summary>
/// <remarks>
/// <para>
/// Uncertainty quantification (UQ) augments standard point predictions with an uncertainty estimate.
/// For supported model types, the library can sample multiple stochastic predictions and aggregate them
/// into a mean prediction and an uncertainty estimate (variance).
/// </para>
/// <para><b>For Beginners:</b> This lets you ask the model not only "what is the prediction?"
/// but also "how sure are you?"</para>
/// </remarks>
public sealed class UncertaintyQuantificationOptions
{
    /// <summary>
    /// Gets or sets whether uncertainty quantification is enabled.
    /// </summary>
    /// <remarks>
    /// When disabled, calls to uncertainty APIs fall back to deterministic behavior.
    /// </remarks>
    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Gets or sets the uncertainty quantification strategy to use.
    /// </summary>
    public UncertaintyQuantificationMethod Method { get; set; } = UncertaintyQuantificationMethod.Auto;

    /// <summary>
    /// Gets or sets the number of stochastic samples to draw when using sampling-based methods.
    /// </summary>
    /// <remarks>
    /// Higher values generally improve estimate stability at the cost of increased inference latency.
    /// </remarks>
    public int NumSamples { get; set; } = 30;

    /// <summary>
    /// Gets or sets the dropout rate used when injecting Monte Carlo Dropout layers automatically.
    /// </summary>
    /// <remarks>
    /// This value is only used when the model architecture does not already contain explicit MC dropout layers.
    /// </remarks>
    public double MonteCarloDropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets an optional random seed for reproducible Monte Carlo sampling.
    /// </summary>
    public int? RandomSeed { get; set; } = null;

    /// <summary>
    /// Gets or sets whether the returned uncertainty should be denormalized to match the output scale.
    /// </summary>
    /// <remarks>
    /// When enabled, the library attempts to scale variances according to the output normalization transform.
    /// For non-linear normalization transforms, the uncertainty is returned in normalized space.
    /// </remarks>
    public bool DenormalizeUncertainty { get; set; } = true;

    /// <summary>
    /// Gets or sets the desired conformal coverage level when using conformal prediction.
    /// </summary>
    /// <remarks>
    /// This is typically expressed as a probability (e.g., 0.9 for 90% coverage).
    /// </remarks>
    public double ConformalConfidenceLevel { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the conformal calibration mode used when producing conformal intervals/sets.
    /// </summary>
    /// <remarks>
    /// Split conformal is the standard baseline. Cross-conformal can improve stability on small calibration sets.
    /// Adaptive conformal adjusts thresholds based on predicted confidence buckets.
    /// </remarks>
    public ConformalPredictionMode ConformalMode { get; set; } = ConformalPredictionMode.Split;

    /// <summary>
    /// Gets or sets the number of folds used when <see cref="ConformalMode"/> is <see cref="ConformalPredictionMode.CrossConformal"/>.
    /// </summary>
    public int CrossConformalFolds { get; set; } = 5;

    /// <summary>
    /// Gets or sets the number of confidence bins used when <see cref="ConformalMode"/> is <see cref="ConformalPredictionMode.Adaptive"/>.
    /// </summary>
    public int AdaptiveConformalBins { get; set; } = 10;

    /// <summary>
    /// Gets or sets the probability calibration method used when calibration labels are provided.
    /// </summary>
    public ProbabilityCalibrationMethod CalibrationMethod { get; set; } = ProbabilityCalibrationMethod.Auto;

    /// <summary>
    /// Gets or sets whether to fit and apply temperature scaling for classification-like outputs when calibration labels are provided.
    /// </summary>
    /// <remarks>
    /// When enabled and calibration labels are provided via the builder, the system will calibrate predicted probabilities and return
    /// calibrated probabilities as the prediction output from uncertainty APIs.
    /// </remarks>
    public bool EnableTemperatureScaling { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to fit and apply Platt scaling (binary calibration) when calibration labels are provided.
    /// </summary>
    public bool EnablePlattScaling { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to fit and apply isotonic regression calibration (binary calibration) when calibration labels are provided.
    /// </summary>
    public bool EnableIsotonicRegressionCalibration { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of independently trained models used for deep ensemble uncertainty estimation.
    /// </summary>
    /// <remarks>
    /// This value is only used when <see cref="Method"/> is <see cref="UncertaintyQuantificationMethod.DeepEnsemble"/>.
    /// </remarks>
    public int DeepEnsembleSize { get; set; } = 5;

    /// <summary>
    /// Gets or sets the standard deviation of the initial parameter perturbation applied when constructing ensemble members.
    /// </summary>
    /// <remarks>
    /// This helps ensure ensemble members do not collapse to identical solutions when created from a shared base model.
    /// </remarks>
    public double DeepEnsembleInitialNoiseStdDev { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the maximum number of samples used to fit Laplace/SWAG posteriors from calibration data.
    /// </summary>
    /// <remarks>
    /// This is a safety/performance bound to prevent extremely large calibration datasets from causing very slow builds.
    /// </remarks>
    public int PosteriorFitMaxSamples { get; set; } = 256;

    /// <summary>
    /// Gets or sets the prior precision (inverse variance) used by diagonal Laplace approximation.
    /// </summary>
    public double LaplacePriorPrecision { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the number of SWAG snapshots to collect when fitting a SWAG posterior.
    /// </summary>
    public int SwagNumSnapshots { get; set; } = 20;

    /// <summary>
    /// Gets or sets the number of SWAG update steps used to collect snapshots.
    /// </summary>
    public int SwagNumSteps { get; set; } = 60;

    /// <summary>
    /// Gets or sets the number of initial SWAG steps to skip before collecting snapshots.
    /// </summary>
    public int SwagBurnInSteps { get; set; } = 10;

    /// <summary>
    /// Gets or sets the learning rate used for SWAG posterior fitting on calibration data.
    /// </summary>
    public double SwagLearningRate { get; set; } = 0.001;

    internal void Normalize()
    {
        if (NumSamples < 1)
        {
            NumSamples = 30;
        }

        if (MonteCarloDropoutRate < 0.0 || MonteCarloDropoutRate >= 1.0)
        {
            MonteCarloDropoutRate = 0.1;
        }

        if (DeepEnsembleInitialNoiseStdDev < 0.0)
        {
            DeepEnsembleInitialNoiseStdDev = 0.01;
        }

        if (ConformalConfidenceLevel <= 0.0 || ConformalConfidenceLevel >= 1.0)
        {
            ConformalConfidenceLevel = 0.9;
        }

        if (CrossConformalFolds < 2)
        {
            CrossConformalFolds = 5;
        }

        if (AdaptiveConformalBins < 1)
        {
            AdaptiveConformalBins = 10;
        }

        if (DeepEnsembleSize < 1)
        {
            DeepEnsembleSize = 5;
        }

        if (PosteriorFitMaxSamples < 1)
        {
            PosteriorFitMaxSamples = 256;
        }

        if (LaplacePriorPrecision <= 0.0)
        {
            LaplacePriorPrecision = 1.0;
        }

        if (SwagNumSnapshots < 1)
        {
            SwagNumSnapshots = 20;
        }

        if (SwagNumSteps < 1)
        {
            SwagNumSteps = 60;
        }

        if (SwagBurnInSteps < 0)
        {
            SwagBurnInSteps = 10;
        }

        if (SwagBurnInSteps >= SwagNumSteps)
        {
            SwagBurnInSteps = SwagNumSteps > 1 ? SwagNumSteps - 1 : 0;
        }

        if (SwagLearningRate <= 0.0)
        {
            SwagLearningRate = 0.001;
        }
    }
}
