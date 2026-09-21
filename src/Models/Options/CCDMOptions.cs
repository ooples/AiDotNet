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
    public CCDMOptions()
    {
        // Seed is INHERITED from ModelOptions and set here rather than shadowed with `new`,
        // which would leave anything holding a ModelOptions reference reading null.
        //
        // Defaulted so repeated predictions agree. This does not collapse the sampling: the
        // NumSamples paths still differ from one another, so the spread stays a real estimate
        // and the intervals the paper reports remain meaningful. It fixes only that Predict
        // called twice on the same input returns the same answer - the sampler equivalent of
        // seeding a generator before inference, not of switching sampling off. Set it to
        // another value for a different sample set, or vary it per call for independent draws.
        Seed = 1;
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The instance to copy from.</param>
    public CCDMOptions(CCDMOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));

        // Copy inherited TimeSeriesRegressionOptions properties
        // Seed is declared on ModelOptions rather than in this file, so a copy constructor
        // written from the local declarations alone misses it. Losing it on a clone silently
        // changes deterministic initialization.
        Seed = other.Seed;
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
        NumSamples = other.NumSamples;
        DropoutRate = other.DropoutRate;
        BetaStart = other.BetaStart;
        BetaEnd = other.BetaEnd;
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
    /// Gets or sets the number of sample paths drawn per forecast.
    /// </summary>
    /// <value>Defaults to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> A diffusion forecaster is generative: every call draws a random
    /// path, so a single path carries the full spread of the predictive distribution rather than
    /// its centre. Drawing several paths and reporting the per-position median gives the point
    /// forecast, and the spread across paths gives the uncertainty.</para>
    /// <para><b>Provenance:</b> 100 is the number of samples Tashiro et al. (CSDI, NeurIPS 2021)
    /// draw before taking the median, and the sibling <see cref="CSDIOptions.NumSamples"/> in this
    /// library uses the same default. Lower it to trade forecast stability for inference time.</para>
    /// </remarks>
    public int NumSamples { get; set; } = 100;

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
    /// <value>Defaults to 0.02.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how much noise is added at the final diffusion step.
    /// A larger value means more aggressive noise at the end of the schedule.</para>
    /// <para><b>Provenance:</b> 0.02 is the endpoint Ho et al., "Denoising Diffusion Probabilistic
    /// Models" (NeurIPS 2020) Section 4 give for a LINEAR beta schedule, and this model uses a
    /// linear schedule. The previous 0.5 was borrowed from Tashiro et al. (CSDI, NeurIPS 2021),
    /// where it is the endpoint of a QUADRATIC schedule over 50 steps; applied linearly over the
    /// 100 steps used here it drives the cumulative alpha product to ~5e-14, so the reverse
    /// process amplifies its input by ~5e6 before the denoiser has learned anything.</para>
    /// </remarks>
    public double BetaEnd { get; set; } = 0.02;
}
