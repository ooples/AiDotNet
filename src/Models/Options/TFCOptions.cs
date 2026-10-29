using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TF-C (Time-Frequency Consistency for Self-Supervised Time Series).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TF-C learns time series representations by enforcing consistency between time-domain
/// and frequency-domain representations via contrastive learning, capturing both
/// temporal and spectral patterns.
/// </para>
/// <para>
/// <b>Reference:</b> Zhang et al., "Self-Supervised Contrastive Pre-Training For Time Series via Time-Frequency Consistency", NeurIPS 2022.
/// </para>
/// </remarks>
public class TFCOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>Creates options with the default values.</summary>
    public TFCOptions() { }

    /// <summary>Creates a copy of another options instance.</summary>
    /// <param name="other">The options to copy.</param>
    public TFCOptions(TFCOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        // Seed is declared on ModelOptions rather than in this file, so a copy constructor
        // written from the local declarations alone misses it. Losing it on a clone silently
        // changes deterministic initialization.
        Seed = other.Seed;
        ContextLength = other.ContextLength; ForecastHorizon = other.ForecastHorizon;
        HiddenDimension = other.HiddenDimension; ProjectionDimension = other.ProjectionDimension;
        NumTimeLayers = other.NumTimeLayers; NumFreqLayers = other.NumFreqLayers;
        DropoutRate = other.DropoutRate; ContrastiveTemperature = other.ContrastiveTemperature;
    }

    /// <summary>Number of past time steps the model reads as context.</summary>
    public int ContextLength { get; set; } = 200;
    /// <summary>Number of future time steps to forecast.</summary>
    public int ForecastHorizon { get; set; } = 96;
    /// <summary>Width of the encoders' hidden representations.</summary>
    public int HiddenDimension { get; set; } = 128;
    /// <summary>Size of the shared time-frequency space both encoders project into.</summary>
    public int ProjectionDimension { get; set; } = 64;
    /// <summary>Number of layers of the time-domain encoder.</summary>
    public int NumTimeLayers { get; set; } = 4;
    /// <summary>Number of layers of the frequency-domain encoder.</summary>
    public int NumFreqLayers { get; set; } = 4;
    /// <summary>Dropout probability applied during training.</summary>
    public double DropoutRate { get; set; } = 0.1;
    /// <summary>Temperature of the contrastive (NT-Xent) loss.</summary>
    public double ContrastiveTemperature { get; set; } = 0.07;
}
