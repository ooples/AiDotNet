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
    public TimeDiffOptions() { }

    public TimeDiffOptions(TimeDiffOptions<T> other)
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
    public double BetaEnd { get; set; } = 0.5;

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
