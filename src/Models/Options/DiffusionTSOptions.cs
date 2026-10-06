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

    /// <summary>Gets or sets the number of past steps the forecast is conditioned on.</summary>
    public int SequenceLength { get; set; } = 168;

    /// <summary>Gets or sets the number of future steps to forecast. The model generates context and horizon as one window.</summary>
    public int ForecastHorizon { get; set; } = 24;

    /// <summary>Gets or sets the transformer width d_model (64 in the reference configurations).</summary>
    public int HiddenDimension { get; set; } = 64;

    /// <summary>Gets or sets the number of attention heads (4 in the reference configurations).</summary>
    public int NumHeads { get; set; } = 4;

    /// <summary>Gets or sets the number of encoder blocks (3 in the reference ETTh configuration).</summary>
    public int NumEncoderLayers { get; set; } = 3;

    /// <summary>Gets or sets the number of decoder blocks, each with its own trend and seasonal output (2 in the reference ETTh configuration).</summary>
    public int NumDecoderLayers { get; set; } = 2;

    /// <summary>Gets or sets the MLP expansion of each block (4 in the reference).</summary>
    public int MlpHiddenTimes { get; set; } = 4;

    /// <summary>Gets or sets the residual and embedding dropout (0 in the reference configurations).</summary>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>Gets or sets the number of diffusion steps T (500 in the reference configurations).</summary>
    public int NumDiffusionSteps { get; set; } = 500;

    /// <summary>Gets or sets the noise schedule (cosine, Nichol &amp; Dhariwal 2021, in the paper).</summary>
    public AiDotNet.Enums.BetaSchedule BetaSchedule { get; set; } = AiDotNet.Enums.BetaSchedule.SquaredCosine;

    /// <summary>Gets or sets the first noise variance, used by the linear and scaled-linear schedules.</summary>
    public double BetaStart { get; set; } = 0.0001;

    /// <summary>Gets or sets the last noise variance, used by the linear and scaled-linear schedules.</summary>
    public double BetaEnd { get; set; } = 0.02;

    /// <summary>Gets or sets the distance between the predicted and the true clean window (L1 in the reference).</summary>
    public AiDotNet.Enums.DiffusionReconstructionLoss ReconstructionLoss { get; set; } = AiDotNet.Enums.DiffusionReconstructionLoss.L1;

    /// <summary>Gets or sets whether the Fourier-based loss term is added (on in the paper).</summary>
    public bool UseFourierLoss { get; set; } = true;

    /// <summary>
    /// Gets or sets the weight of the Fourier loss term; null uses the reference value sqrt(window length) / 5.
    /// </summary>
    public double? FourierLossWeight { get; set; }

    /// <summary>
    /// Gets or sets the factor of the seasonal block's frequency count: it keeps int(factor x ln K) of the K
    /// retained frequencies (1 in the reference).
    /// </summary>
    public int FourierTopKFactor { get; set; } = 1;

    /// <summary>
    /// Gets or sets the bound, in standard deviations of the context, that each predicted clean window is clamped to
    /// during sampling; null disables it.
    /// </summary>
    /// <remarks>
    /// The reference clamps the predicted x_0 to its data range [-1, 1] at every reverse step, which keeps an imperfect
    /// prediction from compounding through the chain. This model standardises each series by its context instead, so the
    /// analogous bound is a number of standard deviations; 5 leaves room for a horizon that moves well beyond the context.
    /// </remarks>
    public double? DenoisedClip { get; set; } = 5.0;

    /// <summary>Gets or sets how many generated windows a forecast averages (and quantile forecasts summarize).</summary>
    public int NumSamples { get; set; } = 100;
}