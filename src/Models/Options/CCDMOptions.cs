using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for CCDM, the Channel-aware Contrastive Conditional Diffusion model for
/// multivariate probabilistic time series forecasting (Li, Chen and Xiong, 2024).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CCDM is a DDPM forecaster. Its denoiser embeds every variable's past window and noisy future
/// independently (channel-independent dense modules), mixes the variables with channel-wise
/// diffusion transformers conditioned on the diffusion step through adaptive layer norm, and is
/// trained to predict the injected noise. An optional denoising-based InfoNCE term contrasts the
/// true future against shuffled and rescaled ones. Defaults follow the authors' reference
/// configuration (github.com/LSY-Cython/CCDM, <c>run_I48_O96.py</c>) unless noted.
/// </para>
/// <para>
/// The inherited <see cref="TimeSeriesRegressionOptions{T}.NumFeatures"/> is the number of variables
/// (D) forecast jointly. Each variable is one token of the channel-wise diffusion transformers, so
/// attention runs ACROSS variables; inputs are laid out [context, variables] (optionally with a
/// leading batch) and forecasts [horizon, variables]. With one variable the attention reduces to a
/// single token.
/// </para>
/// <para><b>For Beginners:</b> CCDM forecasts by starting from random noise and removing it step by
/// step, guided by the history it is given. Because every run starts from different noise, it draws
/// many possible futures; their median is the forecast and their spread is the uncertainty.
/// </para>
/// </remarks>
public class CCDMOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public CCDMOptions()
    {
        // Seed is INHERITED from ModelOptions and set here rather than shadowed with `new`,
        // which would leave anything holding a ModelOptions reference reading null.
        //
        // Defaulted so repeated predictions agree. This does not collapse the sampling: the
        // NumSamples paths still differ from one another, so the spread stays a real estimate.
        // It fixes only that Predict called twice on the same input returns the same answer.
        Seed = 1;
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The instance to copy from.</param>
    public CCDMOptions(CCDMOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));

        // Seed is declared on ModelOptions rather than in this file, so a copy constructor
        // written from the local declarations alone misses it.
        Seed = other.Seed;
        LagOrder = other.LagOrder;
        IncludeTrend = other.IncludeTrend;
        SeasonalPeriod = other.SeasonalPeriod;
        AutocorrelationCorrection = other.AutocorrelationCorrection;
        ModelType = other.ModelType;
        LossFunction = other.LossFunction;

        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        NumFeatures = other.NumFeatures;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        EmbeddingLayers = other.EmbeddingLayers;
        MlpRatio = other.MlpRatio;
        DiffusionSteps = other.DiffusionSteps;
        BetaSchedule = other.BetaSchedule;
        BetaStart = other.BetaStart;
        BetaEnd = other.BetaEnd;
        NumSamples = other.NumSamples;
        TrainingBatchSize = other.TrainingBatchSize;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
        AttentionDropoutRate = other.AttentionDropoutRate;
        ContrastiveWeight = other.ContrastiveWeight;
        ContrastiveTemperature = other.ContrastiveTemperature;
        NumNegatives = other.NumNegatives;
    }

    /// <summary>
    /// Gets or sets the number of historical time steps used as input context (L).
    /// </summary>
    /// <value>Defaults to 168 (one week of hourly data).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How much history the model sees before forecasting. The paper
    /// evaluates 48, 96, 192 and 336.</para>
    /// </remarks>
    public int ContextLength { get; set; } = 168;

    /// <summary>
    /// Gets or sets the number of future time steps to forecast (H).
    /// </summary>
    /// <value>Defaults to 24 (one day ahead for hourly data).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far ahead the model predicts in one pass. The paper evaluates
    /// 96, 168, 336 and 720.</para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 24;

    /// <summary>
    /// Gets or sets the embedding width of the past window, the noisy future and the diffusion step
    /// (e_hid; the transformers run at twice this width).
    /// </summary>
    /// <value>Defaults to 128, the reference configuration's width for every embedding.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The model's capacity. The paper scales it with the horizon:
    /// 128 for 96 steps up to 728 for 720.</para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the number of channel-wise diffusion transformer blocks (n_att).
    /// </summary>
    /// <value>Defaults to 2, the depth the paper fixes for every experiment.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many times the variables exchange information.</para>
    /// </remarks>
    public int NumLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>Defaults to 8.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Must divide twice <see cref="HiddenDimension"/>, the width the
    /// attention runs at.</para>
    /// </remarks>
    public int NumHeads { get; set; } = 8;

    /// <summary>
    /// Gets or sets the number of residual MLP blocks in each channel-independent dense module
    /// (n_emb; n_enc = n_dec in the paper).
    /// </summary>
    /// <value>Defaults to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How deep each variable's own encoder is before the variables are
    /// mixed. The decoder uses one fewer block followed by a linear projection.</para>
    /// </remarks>
    public int EmbeddingLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the hidden-width multiplier of the transformer MLP.
    /// </summary>
    /// <value>Defaults to 1.0, the reference configuration's mlp_ratio.</value>
    public double MlpRatio { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the number of diffusion steps (K).
    /// </summary>
    /// <value>Defaults to 50, the paper's value for a 96-step horizon.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many denoising steps a forecast takes. The paper uses 50 for
    /// short horizons and up to 200 for 720 steps.</para>
    /// </remarks>
    public int DiffusionSteps { get; set; } = 50;

    /// <summary>
    /// Gets or sets the shape of the noise schedule.
    /// </summary>
    /// <value>Defaults to <see cref="AiDotNet.Enums.BetaSchedule.ScaledLinear"/>.</value>
    /// <remarks>
    /// <para>
    /// The reference configuration's "quad" schedule is
    /// <c>betas = linspace(sqrt(beta_start), sqrt(beta_end), K)^2</c>, which is exactly
    /// <see cref="AiDotNet.Enums.BetaSchedule.ScaledLinear"/>. Linear and squared-cosine are the
    /// reference code's other two branches.
    /// </para>
    /// </remarks>
    public BetaSchedule BetaSchedule { get; set; } = BetaSchedule.ScaledLinear;

    /// <summary>
    /// Gets or sets the first beta of the noise schedule.
    /// </summary>
    /// <value>Defaults to 0.0001 (beta_1 in the paper).</value>
    public double BetaStart { get; set; } = 0.0001;

    /// <summary>
    /// Gets or sets the last beta of the noise schedule.
    /// </summary>
    /// <value>Defaults to 0.5, the reference configuration's beta_K with the quad schedule over 50
    /// steps.</value>
    /// <remarks>
    /// <para>
    /// beta_K belongs with the schedule shape and step count. Over 50 quad-scheduled steps 0.5 drives
    /// alphaBar_K to about 1e-5, so the forward process ends in the pure noise the sampler starts
    /// from. The paper uses 0.2-0.5 depending on the horizon.
    /// </para>
    /// </remarks>
    public double BetaEnd { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the number of sample paths drawn per forecast (S).
    /// </summary>
    /// <value>Defaults to 100, the paper's number of samples per test window.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The forecast is the per-position median of these paths and the
    /// quantiles come from their spread. Lower it to trade stability for speed.</para>
    /// </remarks>
    public int NumSamples { get; set; } = 100;

    /// <summary>
    /// Number of (diffusion step, noise) draws averaged into a single training step.
    /// </summary>
    /// <value>Defaults to 32.</value>
    /// <remarks>
    /// <para>
    /// Ho et al. (2020) Algorithm 1 draws one step per example and averages over a minibatch. A
    /// caller here supplies one window at a time, so the averaging happens over the noise process:
    /// the steps are drawn antithetically (k and K-1-k in pairs) as the reference implementation's
    /// "uniform" step distribution does.
    /// </para>
    /// </remarks>
    public int TrainingBatchSize { get; set; } = 32;

    /// <summary>
    /// Learning rate for the default Adam optimizer.
    /// </summary>
    /// <value>Defaults to 1e-3, the reference configuration's init_lr.</value>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>
    /// Gets or sets the dropout rate inside the residual MLP blocks of the dense modules.
    /// </summary>
    /// <value>Defaults to 0.1, the reference MLPResidual dropout.</value>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the dropout rate on the attention weights.
    /// </summary>
    /// <value>Defaults to 0.1, the reference configuration's attn_dropout.</value>
    public double AttentionDropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the weight lambda of the denoising-based temporal contrastive loss.
    /// </summary>
    /// <value>Defaults to 0 (denoising loss only), the reference configuration's default
    /// "non-contrast" mode.</value>
    /// <remarks>
    /// <para>
    /// Paper Equation 6: L = L_denoise + lambda * L_contrast. The paper reports lambda in
    /// {5e-5, 1e-4, 5e-4, 1e-3} depending on the dataset, and for large datasets first pre-trains
    /// with the denoising loss alone, then fine-tunes with the contrastive term. Setting a positive
    /// value draws <see cref="NumNegatives"/> patch-shuffled and as many rescaled negatives per
    /// training row, which multiplies the cost of a step accordingly.
    /// </para>
    /// <para><b>For Beginners:</b> An extra training signal that teaches the model to tell the real
    /// future apart from scrambled or rescaled versions of it. Off by default.</para>
    /// </remarks>
    public double ContrastiveWeight { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the InfoNCE temperature tau.
    /// </summary>
    /// <value>Defaults to 0.1, the paper's value.</value>
    public double ContrastiveTemperature { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the number of negatives per augmentation (patch shuffle and magnitude scaling).
    /// </summary>
    /// <value>Defaults to 64, the reference configuration's n_negatives.</value>
    public int NumNegatives { get; set; } = 64;
}
