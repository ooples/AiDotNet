using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TimeDiff (Non-autoregressive Diffusion-based Time Series Forecasting).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TimeDiff extends DDPM with novel conditioning mechanisms specifically designed for
/// time series: future-mixup for training, autoregressive initialization for inference,
/// and a transformer-based denoiser.
/// </para>
/// <para>
/// <b>Reference:</b> Shen &amp; Kwok, "Non-autoregressive Conditional Diffusion Models for Time Series Prediction", ICML 2023.
/// </para>
/// </remarks>
public class TimeDiffOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public TimeDiffOptions()
    {
        // Seed is INHERITED from ModelOptions and set here rather than shadowed with `new`,
        // which would leave anything holding a ModelOptions reference reading null.
        //
        // Defaulted so repeated predictions agree. Algorithm 2 of the paper draws x_K from
        // N(0, I) and a fresh epsilon at every reverse step, so without a seed Predict called
        // twice on the same input returns two different forecasts. Seeding restarts that
        // stream per call rather than switching the sampling off - set another value for a
        // different draw, or vary it per call for independent samples.
        Seed = 1;
    }

    public TimeDiffOptions(TimeDiffOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        // Seed is declared on ModelOptions rather than in this file, so a copy constructor
        // written from the local declarations alone misses it. Losing it on a clone silently
        // changes deterministic initialization.
        Seed = other.Seed;
        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        DiffusionSteps = other.DiffusionSteps;
        DropoutRate = other.DropoutRate;
        BetaStart = other.BetaStart;
        BetaEnd = other.BetaEnd;
        LearningRate = other.LearningRate;
        TrainingBatchSize = other.TrainingBatchSize;
        UseFutureMixup = other.UseFutureMixup;
        UseAutoregressiveInit = other.UseAutoregressiveInit;
    }

    public int ContextLength { get; set; } = 168;
    public int ForecastHorizon { get; set; } = 24;
    public int HiddenDimension { get; set; } = 128;
    public int NumLayers { get; set; } = 4;
    public int NumHeads { get; set; } = 8;
    public int DiffusionSteps { get; set; } = 100;
    public double DropoutRate { get; set; } = 0.1;
    public double BetaStart { get; set; } = 0.0001;
    /// <summary>
    /// Gets or sets the largest variance the cosine schedule is allowed to reach.
    /// </summary>
    /// <value>Defaults to 0.1, the paper's beta_K.</value>
    /// <remarks>
    /// Section 4.1: "K = 100 diffusion steps are used, with a cosine variance schedule
    /// (Rasul et al., 2021) starting from beta_1 = 10^-4 to beta_K = 10^-1." A cosine schedule
    /// is defined on alpha_bar rather than on beta, so the two endpoints are the clamp the
    /// derived betas are held inside - the same role the 0.999 cap plays in Nichol and
    /// Dhariwal 2021, which is where the cosine schedule comes from. The previous 0.5 was
    /// five times the paper's ceiling and had no source.
    /// </remarks>
    public double BetaEnd { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the Adam learning rate used when no optimizer is supplied.
    /// </summary>
    /// <value>Defaults to 1e-3, the paper's rate.</value>
    /// <remarks>
    /// Section 4.1: "We train the proposed model using Adam (Kingma and Ba, 2015) with a
    /// learning rate of 10^-3."
    /// </remarks>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>
    /// Gets or sets how many (diffusion step, noise) draws one training step averages over.
    /// </summary>
    /// <value>Defaults to 64, the paper's batch size.</value>
    /// <remarks>
    /// Algorithm 1 draws one k per example and Section 4.1 reports a batch size of 64, so a
    /// step is an average over 64 draws. A caller here hands the model a single example, so
    /// the minibatch is taken over the noise process instead: without it the gradient is a
    /// one-sample estimate of an expectation over DiffusionSteps noise levels and consecutive
    /// steps are dominated by which k came up rather than by what the model learned.
    /// </remarks>
    public int TrainingBatchSize { get; set; } = 64;

    /// <summary>
    /// Gets or sets whether to use future-mixup augmentation during training.
    /// </summary>
    /// <value>Defaults to true.</value>
    public bool UseFutureMixup { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use autoregressive initialization at inference.
    /// </summary>
    /// <value>Defaults to true.</value>
    public bool UseAutoregressiveInit { get; set; } = true;
}
