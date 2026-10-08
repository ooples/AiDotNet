using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for DiffusionTS (Interpretable Diffusion for Time Series).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// DiffusionTS is an interpretable diffusion model for time series that uses seasonal-trend
/// decomposition to generate forecasts with clear interpretable components.
/// </para>
/// <para><b>For Beginners:</b> DiffusionTS focuses on making diffusion models more
/// interpretable by decomposing time series into understandable components:
///
/// <b>The Key Insight:</b>
/// Time series often have clear structure (trends, seasonality) that gets lost in
/// "black box" models. DiffusionTS preserves this structure by generating each
/// component separately and combining them.
///
/// <b>How DiffusionTS Works:</b>
/// 1. <b>Decomposition:</b> Split time series into trend, seasonal, and residual
/// 2. <b>Component Diffusion:</b> Generate each component with specialized networks
/// 3. <b>Reconstruction:</b> Combine components to form final forecast
/// 4. <b>Interpretation:</b> Each component has clear meaning
///
/// <b>DiffusionTS Architecture:</b>
/// - Trend Network: Captures long-term movements (slow, smooth)
/// - Seasonal Network: Captures periodic patterns (daily, weekly, yearly)
/// - Residual Network: Captures irregular fluctuations
/// - Fusion Module: Combines components coherently
///
/// <b>Key Benefits:</b>
/// - Interpretable decomposition of forecasts
/// - Can enforce structural constraints (smooth trends, periodic seasons)
/// - Better uncertainty quantification per component
/// - Enables "what-if" analysis by modifying components
/// </para>
/// <para>
/// <b>Reference:</b> Yuan and Qiu, "Diffusion-TS: Interpretable Diffusion for General Time Series Generation", 2024.
/// https://arxiv.org/abs/2403.01742
/// </para>
/// </remarks>
public class DiffusionTSOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>Creates options with the paper's defaults.</summary>
    public DiffusionTSOptions()
    {
        // Seed is INHERITED from ModelOptions and set here rather than shadowed with `new`,
        // so every ModelOptions reader sees the same value. A default makes sampling reproducible
        // out of the box; callers who want fresh draws per call can set it back to null.
        Seed = 1;
    }

    /// <summary>Copies every setting of <paramref name="other"/>.</summary>
    public DiffusionTSOptions(DiffusionTSOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        // Seed is declared on ModelOptions rather than in this file, so a copy constructor
        // written from the local declarations alone misses it. Losing it on a clone silently
        // changes deterministic initialization.
        Seed = other.Seed;
        NumFeatures = other.NumFeatures;
        SequenceLength = other.SequenceLength;
        ForecastHorizon = other.ForecastHorizon;
        HiddenDimension = other.HiddenDimension;
        NumHeads = other.NumHeads;
        NumEncoderLayers = other.NumEncoderLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        MlpHiddenTimes = other.MlpHiddenTimes;
        DropoutRate = other.DropoutRate;
        NumDiffusionSteps = other.NumDiffusionSteps;
        BetaSchedule = other.BetaSchedule;
        BetaStart = other.BetaStart;
        BetaEnd = other.BetaEnd;
        ReconstructionLoss = other.ReconstructionLoss;
        UseFourierLoss = other.UseFourierLoss;
        FourierLossWeight = other.FourierLossWeight;
        FourierTopKFactor = other.FourierTopKFactor;
        DenoisedClip = other.DenoisedClip;
        NumSamples = other.NumSamples;
    }

    /// <summary>
    /// Gets or sets the number of past steps a forecast is conditioned on.
    /// </summary>
    /// <value>The context length in time steps; 168 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How much history the model reads. The model generates history and horizon together as one window.
    /// </para>
    /// <para>
    /// Default: 168, one week of hourly observations.
    /// </para>
    /// </remarks>
    public int SequenceLength { get; set; } = 168;

    /// <summary>
    /// Gets or sets the number of future steps to forecast.
    /// </summary>
    /// <value>The forecast horizon in time steps; 24 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How far ahead the model predicts.
    /// </para>
    /// <para>
    /// Default: 24, a day ahead for hourly data.
    /// </para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 24;

    /// <summary>
    /// Gets or sets the transformer width d_model.
    /// </summary>
    /// <value>The model width; 64 by default. It must be even and divisible by NumHeads.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many features the transformer uses to describe each time step.
    /// </para>
    /// <para>
    /// Default: 64, the reference configurations' d_model (Yuan &amp; Qiao 2024).
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>The head count; 4 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Attention compares time steps; each head can look for a different kind of relationship.
    /// </para>
    /// <para>
    /// Default: 4, the reference configurations.
    /// </para>
    /// </remarks>
    public int NumHeads { get; set; } = 4;

    /// <summary>
    /// Gets or sets the number of encoder blocks.
    /// </summary>
    /// <value>The encoder depth; 3 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many blocks summarize the noisy window before decoding.
    /// </para>
    /// <para>
    /// Default: 3, the reference ETTh configuration.
    /// </para>
    /// </remarks>
    public int NumEncoderLayers { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of decoder blocks, each with its own trend and seasonal output.
    /// </summary>
    /// <value>The decoder depth; 2 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each decoder block adds its own trend and seasonal pieces to the final forecast.
    /// </para>
    /// <para>
    /// Default: 2, the reference ETTh configuration.
    /// </para>
    /// </remarks>
    public int NumDecoderLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the MLP expansion of each block.
    /// </summary>
    /// <value>The expansion factor; 4 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Inside each block a small network temporarily widens the features by this factor.
    /// </para>
    /// <para>
    /// Default: 4, the reference mlp_hidden_times.
    /// </para>
    /// </remarks>
    public int MlpHiddenTimes { get; set; } = 4;

    /// <summary>
    /// Gets or sets the residual and embedding dropout.
    /// </summary>
    /// <value>A probability in [0, 1); 0 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Randomly ignores some values during training so the model does not memorize the data.
    /// </para>
    /// <para>
    /// Default: 0, the reference configurations' resid_pd.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the number of diffusion steps T.
    /// </summary>
    /// <value>The length of the noising chain; 500 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many denoising steps turn noise into a forecast. More steps give finer samples but cost more time.
    /// </para>
    /// <para>
    /// Default: 500, the reference configurations' timesteps.
    /// </para>
    /// </remarks>
    public int NumDiffusionSteps { get; set; } = 500;

    /// <summary>
    /// Gets or sets the noise schedule.
    /// </summary>
    /// <value>A schedule; SquaredCosine by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The shape of the noise curve across the steps. The cosine schedule adds noise gently at first.
    /// </para>
    /// <para>
    /// Default: cosine (Nichol &amp; Dhariwal 2021), as in the paper.
    /// </para>
    /// </remarks>
    public AiDotNet.Enums.BetaSchedule BetaSchedule { get; set; } = AiDotNet.Enums.BetaSchedule.SquaredCosine;

    /// <summary>
    /// Gets or sets the first noise variance, used by the linear and scaled-linear schedules.
    /// </summary>
    /// <value>A variance in (0, 1); 1e-4 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The noise added by the first step when a linear schedule is chosen.
    /// </para>
    /// <para>
    /// Default: 1e-4, the usual DDPM value; the paper's cosine schedule does not use it.
    /// </para>
    /// </remarks>
    public double BetaStart { get; set; } = 0.0001;

    /// <summary>
    /// Gets or sets the last noise variance, used by the linear and scaled-linear schedules.
    /// </summary>
    /// <value>A variance in (0, 1); 0.02 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The noise added by the last step when a linear schedule is chosen.
    /// </para>
    /// <para>
    /// Default: 0.02, the usual DDPM value; the paper's cosine schedule does not use it.
    /// </para>
    /// </remarks>
    public double BetaEnd { get; set; } = 0.02;

    /// <summary>
    /// Gets or sets the distance between the predicted and the true clean window.
    /// </summary>
    /// <value>L1 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How training measures the error of the predicted clean series: L1 sums absolute errors, L2 sums squared errors.
    /// </para>
    /// <para>
    /// Default: L1, the reference loss_type.
    /// </para>
    /// </remarks>
    public AiDotNet.Enums.DiffusionReconstructionLoss ReconstructionLoss { get; set; } = AiDotNet.Enums.DiffusionReconstructionLoss.L1;

    /// <summary>
    /// Gets or sets whether the Fourier-based loss term is added.
    /// </summary>
    /// <value>true by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Also compares the frequencies of the predicted and true series, which helps the model learn repeating patterns.
    /// </para>
    /// <para>
    /// Default: on, as in the paper (Yuan &amp; Qiao 2024, Sec. 3).
    /// </para>
    /// </remarks>
    public bool UseFourierLoss { get; set; } = true;

    /// <summary>
    /// Gets or sets the weight of the Fourier loss term; null uses sqrt(window length) / 5.
    /// </summary>
    /// <value>A weight, or null for the reference value.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How much the frequency comparison counts against the plain error.
    /// </para>
    /// <para>
    /// Default: null, which applies the reference ff_weight of sqrt(L) / 5.
    /// </para>
    /// </remarks>
    public double? FourierLossWeight { get; set; }

    /// <summary>
    /// Gets or sets the factor of the seasonal block's frequency count: it keeps int(factor x ln K) of the K retained frequencies.
    /// </summary>
    /// <value>The factor; 1 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The seasonal part of each block is built from only the strongest few frequencies; this scales how many.
    /// </para>
    /// <para>
    /// Default: 1, the reference FourierLayer factor.
    /// </para>
    /// </remarks>
    public int FourierTopKFactor { get; set; } = 1;

    /// <summary>
    /// Gets or sets the bound, in standard deviations of the context, that each predicted clean window is clamped to during sampling; null disables it.
    /// </summary>
    /// <value>A bound in context standard deviations, or null; 5 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Stops a poor intermediate guess from growing without limit while the forecast is generated.
    /// </para>
    /// <para>
    /// The reference clamps x_0 to its data range [-1, 1]; this model standardises by the context, so the analogous bound is a number of standard deviations, and 5 leaves room for a horizon that moves well beyond the context.
    /// </para>
    /// </remarks>
    public double? DenoisedClip { get; set; } = 5.0;

    /// <summary>
    /// Gets or sets how many windows a forecast generates.
    /// </summary>
    /// <value>The number of generated windows; 100 by default.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The model draws this many possible futures; the point forecast is their mean and quantiles summarize their spread.
    /// </para>
    /// <para>
    /// Default: 100 samples, matching the probabilistic evaluation of the other diffusion forecasters here.
    /// </para>
    /// </remarks>
    public int NumSamples { get; set; } = 100;
}
