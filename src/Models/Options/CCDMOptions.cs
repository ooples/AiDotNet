using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for CCDM (Conditional Continuous Diffusion Model for Time Series).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CCDM extends continuous diffusion models for conditional time series generation.
/// It operates in continuous space (unlike discrete token-based approaches) and uses
/// a score-matching objective for high-quality probabilistic forecasting.
/// </para>
/// <para><b>For Beginners:</b> CCDM is a diffusion-based forecasting model that:
///
/// <b>What is Diffusion?</b>
/// Diffusion models work by learning to remove noise. During training, noise is
/// progressively added to the target series. During inference, the model starts
/// from pure noise and iteratively denoises it, conditioned on the historical
/// context, to produce a forecast.
///
/// <b>Key Advantages:</b>
/// - Produces probabilistic forecasts (uncertainty estimates) naturally
/// - Operates in continuous space (no quantization loss)
/// - Score-matching objective is stable to train
///
/// <b>Trade-offs:</b>
/// - Slower inference than direct methods (requires multiple denoising steps)
/// - More parameters to tune (noise schedule, diffusion steps)
/// </para>
/// </remarks>
public class CCDMOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public CCDMOptions() { }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The instance to copy from.</param>
    public CCDMOptions(CCDMOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));

        // Copy inherited TimeSeriesRegressionOptions properties
        LagOrder = other.LagOrder;
        IncludeTrend = other.IncludeTrend;
        SeasonalPeriod = other.SeasonalPeriod;
        AutocorrelationCorrection = other.AutocorrelationCorrection;
        ModelType = other.ModelType;
        LossFunction = other.LossFunction;

        // Copy CCDM-specific properties
        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        DiffusionSteps = other.DiffusionSteps;
        DropoutRate = other.DropoutRate;
        BetaStart = other.BetaStart;
        BetaEnd = other.BetaEnd;
        SigmaMin = other.SigmaMin;
        SigmaMax = other.SigmaMax;
    }

    /// <summary>
    /// Gets or sets the number of historical time steps used as input context.
    /// </summary>
    /// <value>Defaults to 168 (one week of hourly data).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How much historical data the model sees before making predictions.
    /// Longer context gives the model more patterns to learn from but uses more memory.</para>
    /// </remarks>
    public int ContextLength { get; set; } = 168;

    /// <summary>
    /// Gets or sets the number of future time steps to forecast.
    /// </summary>
    /// <value>Defaults to 24 (one day ahead for hourly data).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far into the future the model predicts in a single pass.</para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 24;

    /// <summary>
    /// Gets or sets the hidden dimension of the transformer layers.
    /// </summary>
    /// <value>Defaults to 128.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls the model's capacity. Larger values can capture
    /// more complex patterns but require more memory and compute.</para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the number of transformer layers.
    /// </summary>
    /// <value>Defaults to 4.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> More layers allow the model to learn deeper patterns
    /// but increase computation time and risk of overfitting on small datasets.</para>
    /// </remarks>
    public int NumLayers { get; set; } = 4;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>Defaults to 8.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each attention head focuses on different aspects of the
    /// input sequence. Must divide evenly into <see cref="HiddenDimension"/>.</para>
    /// </remarks>
    public int NumHeads { get; set; } = 8;

    /// <summary>
    /// Gets or sets the number of diffusion (denoising) steps.
    /// </summary>
    /// <value>Defaults to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> More steps generally produce better quality forecasts
    /// but increase inference time. Values between 50-200 are typical.</para>
    /// </remarks>
    public int DiffusionSteps { get; set; } = 100;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>Defaults to 0.1 (10%).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Randomly drops connections during training to prevent
    /// overfitting. Set to 0 to disable.</para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the starting beta value for the linear noise schedule.
    /// </summary>
    /// <value>Defaults to 0.0001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how much noise is added at the first diffusion step.
    /// A small value means very little noise initially.</para>
    /// </remarks>
    public double BetaStart { get; set; } = 0.0001;

    /// <summary>
    /// Gets or sets the ending beta value for the linear noise schedule.
    /// </summary>
    /// <value>Defaults to 0.5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how much noise is added at the final diffusion step.
    /// A larger value means more aggressive noise at the end of the schedule.</para>
    /// </remarks>
    public double BetaEnd { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the minimum noise level for the continuous diffusion schedule.
    /// </summary>
    /// <value>Defaults to 0.002.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The smallest noise scale used in the continuous
    /// diffusion process. Lower values preserve more detail at the finest level.</para>
    /// </remarks>
    public double SigmaMin { get; set; } = 0.002;

    /// <summary>
    /// Gets or sets the maximum noise level for the continuous diffusion schedule.
    /// </summary>
    /// <value>Defaults to 80.0 (from Song et al., "Score-Based Generative Modeling through SDEs", ICLR 2021).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The largest noise scale used. Higher values mean the
    /// model learns to recover signal from more aggressive corruption.</para>
    /// <para><b>Provenance:</b> Default noise schedule parameters (BetaStart=0.0001, BetaEnd=0.5,
    /// SigmaMin=0.002, SigmaMax=80.0) follow standard continuous diffusion practice from
    /// Song et al. (2021) and Ho et al. "Denoising Diffusion Probabilistic Models" (NeurIPS 2020).
    /// Architecture defaults (HiddenDimension=128, NumLayers=4, NumHeads=8, DiffusionSteps=100)
    /// are common baselines for time series diffusion models.</para>
    /// </remarks>
    public double SigmaMax { get; set; } = 80.0;
}
