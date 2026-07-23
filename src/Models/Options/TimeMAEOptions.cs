using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TimeMAE (Masked Autoencoder for Time Series).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TimeMAE applies masked autoencoding to time series, randomly masking patches of the input
/// and training a transformer to reconstruct the missing patches, learning rich temporal representations.
/// </para>
/// <para>
/// <b>Reference:</b> Cheng et al., "TimeMAE: Self-Supervised Representations of Time Series with Decoupled Masked Autoencoders", 2023.
/// </para>
/// </remarks>
public class TimeMAEOptions<T> : TimeSeriesRegressionOptions<T>
{
    public TimeMAEOptions() { }

    public TimeMAEOptions(TimeMAEOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        ContextLength = other.ContextLength; ForecastHorizon = other.ForecastHorizon;
        PatchLength = other.PatchLength; HiddenDimension = other.HiddenDimension;
        NumEncoderLayers = other.NumEncoderLayers; NumDecoderLayers = other.NumDecoderLayers;
        NumHeads = other.NumHeads; MaskRatio = other.MaskRatio;
        DropoutRate = other.DropoutRate;
    }

    public int ContextLength { get; set; } = 512;
    public int ForecastHorizon { get; set; } = 96;
    public int PatchLength { get; set; } = 16;
    public int HiddenDimension { get; set; } = 256;
    public int NumEncoderLayers { get; set; } = 6;
    public int NumDecoderLayers { get; set; } = 2;
    public int NumHeads { get; set; } = 8;
    public double MaskRatio { get; set; } = 0.75;
    public double DropoutRate { get; set; } = 0.1;
}
