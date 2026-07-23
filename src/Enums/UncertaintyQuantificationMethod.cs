namespace AiDotNet.Enums;

/// <summary>
/// Defines the supported uncertainty quantification strategies for inference.
/// </summary>
/// <remarks>
/// <para>
/// These options control how the system estimates predictive uncertainty during inference.
/// </para>
/// <para><b>For Beginners:</b> Some models can also tell you how confident they are.
/// This enum lets you choose the strategy used to estimate that confidence.</para>
/// </remarks>
public enum UncertaintyQuantificationMethod
{
    /// <summary>
    /// Automatically selects a suitable method when possible, otherwise falls back to deterministic predictions.
    /// </summary>
    Auto,

    /// <summary>
    /// Uses Monte Carlo Dropout by enabling dropout at inference and sampling multiple forward passes.
    /// </summary>
    MonteCarloDropout,

    /// <summary>
    /// Uses a deep ensemble (multiple independently trained models) to estimate uncertainty.
    /// </summary>
    DeepEnsemble,

    /// <summary>
    /// Uses Bayesian neural network sampling (e.g., Bayes-by-Backprop style layers) to estimate uncertainty.
    /// </summary>
    BayesianNeuralNetwork,

    /// <summary>
    /// Uses a Laplace approximation (typically diagonal) over model parameters to sample predictions.
    /// </summary>
    LaplaceApproximation,

    /// <summary>
    /// Uses SWAG (Stochastic Weight Averaging-Gaussian) to sample model parameters and estimate uncertainty.
    /// </summary>
    Swag,

    /// <summary>
    /// Uses conformal prediction to produce statistically valid intervals (regression) or prediction sets (classification).
    /// </summary>
    ConformalPrediction
}
