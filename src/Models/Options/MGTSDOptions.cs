using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for MG-TSD (Multi-Granularity Time Series Diffusion).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MG-TSD introduces a multi-granularity guidance diffusion model that captures temporal
/// patterns at different scales. It uses a coarse-to-fine guidance mechanism where
/// predictions at coarser granularities guide the fine-grained diffusion process.
/// </para>
/// <para><b>For Beginners:</b> MG-TSD improves on standard diffusion models by:
///
/// <b>Multi-Granularity Guidance:</b>
/// Instead of denoising at a single resolution, MG-TSD processes the time series
/// at multiple temporal granularities (e.g., hourly, daily, weekly). Coarser
/// predictions capture long-range trends and guide finer predictions, resulting
/// in more coherent forecasts across time scales.
///
/// <b>Key Advantages:</b>
/// - Better captures patterns at different time scales
/// - Coarse-to-fine guidance improves forecast coherence
/// - Produces calibrated probabilistic forecasts
/// </para>
/// <para>
/// <b>Reference:</b> Fan et al., "MG-TSD: Multi-Granularity Time Series Diffusion Models with Guided Learning Process", ICLR 2024.
/// </para>
/// </remarks>
public class MGTSDOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public MGTSDOptions() { }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The instance to copy from.</param>
    public MGTSDOptions(MGTSDOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        DiffusionSteps = other.DiffusionSteps;
        DropoutRate = other.DropoutRate;
        BetaStart = other.BetaStart;
        BetaEnd = other.BetaEnd;
        NumGranularities = other.NumGranularities;
        GuidanceWeight = other.GuidanceWeight;
    }

    /// <summary>
    /// Gets or sets the number of historical time steps used as input context.
    /// </summary>
    /// <value>Defaults to 168 (one week of hourly data).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How much historical data the model sees before making predictions.</para>
    /// </remarks>
    public int ContextLength { get; set; } = 168;

    /// <summary>
    /// Gets or sets the number of future time steps to forecast.
    /// </summary>
    /// <value>Defaults to 24.</value>
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
    /// more complex patterns but require more memory.</para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the number of transformer layers.
    /// </summary>
    /// <value>Defaults to 4.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> More layers allow the model to learn deeper patterns
    /// but increase computation time.</para>
    /// </remarks>
    public int NumLayers { get; set; } = 4;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>Defaults to 8.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each head focuses on different aspects of the input.
    /// Must divide evenly into <see cref="HiddenDimension"/>.</para>
    /// </remarks>
    public int NumHeads { get; set; } = 8;

    /// <summary>
    /// Gets or sets the number of diffusion (denoising) steps.
    /// </summary>
    /// <value>Defaults to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> More steps produce better quality forecasts but increase
    /// inference time. Values between 50-200 are typical.</para>
    /// </remarks>
    public int DiffusionSteps { get; set; } = 100;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>Defaults to 0.1 (10%).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Randomly drops connections during training to prevent overfitting.</para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the starting beta value for the noise schedule.
    /// </summary>
    /// <value>Defaults to 0.0001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls initial noise in the diffusion process.
    /// Small values mean minimal noise at the start.</para>
    /// </remarks>
    public double BetaStart { get; set; } = 0.0001;

    /// <summary>
    /// Gets or sets the ending beta value for the noise schedule.
    /// </summary>
    /// <value>Defaults to 0.5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls maximum noise in the diffusion process.</para>
    /// </remarks>
    public double BetaEnd { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the number of temporal granularity levels for guidance.
    /// </summary>
    /// <value>Defaults to 3 (e.g., hourly, daily, weekly).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how many different time scales the model uses.
    /// More granularities capture more temporal patterns but increase computation.</para>
    /// </remarks>
    public int NumGranularities { get; set; } = 3;

    /// <summary>
    /// Gets or sets the weight for cross-granularity guidance.
    /// </summary>
    /// <value>Defaults to 0.5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how strongly coarse predictions influence
    /// fine-grained denoising. Higher values mean stronger guidance from coarser scales.</para>
    /// </remarks>
    public double GuidanceWeight { get; set; } = 0.5;
}
