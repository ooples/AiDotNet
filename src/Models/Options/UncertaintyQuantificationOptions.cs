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
    /// Gets or sets whether to fit and apply temperature scaling for classification-like outputs when calibration labels are provided.
    /// </summary>
    /// <remarks>
    /// When enabled and calibration labels are provided via the builder, the system will calibrate predicted probabilities and return
    /// calibrated probabilities as the prediction output from uncertainty APIs.
    /// </remarks>
    public bool EnableTemperatureScaling { get; set; } = true;

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
}
