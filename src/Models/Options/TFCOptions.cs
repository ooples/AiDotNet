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
    public TFCOptions() { }

    public TFCOptions(TFCOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        ContextLength = other.ContextLength; ForecastHorizon = other.ForecastHorizon;
        HiddenDimension = other.HiddenDimension; ProjectionDimension = other.ProjectionDimension;
        NumTimeLayers = other.NumTimeLayers; NumFreqLayers = other.NumFreqLayers;
        DropoutRate = other.DropoutRate; ContrastiveTemperature = other.ContrastiveTemperature;
    }

    public int ContextLength { get; set; } = 200;
    public int ForecastHorizon { get; set; } = 96;
    public int HiddenDimension { get; set; } = 128;
    public int ProjectionDimension { get; set; } = 64;
    public int NumTimeLayers { get; set; } = 4;
    public int NumFreqLayers { get; set; } = 4;
    public double DropoutRate { get; set; } = 0.1;
    public double ContrastiveTemperature { get; set; } = 0.07;
}
