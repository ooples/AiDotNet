using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for SimMTM (Simple Pre-Training Framework for Masked Time-Series Modeling).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SimMTM combines masked time series modeling with series-level similarity learning,
/// recovering masked series by aggregating from similar unmasked series in the batch.
/// </para>
/// <para>
/// <b>Reference:</b> Dong et al., "SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling", NeurIPS 2023.
/// </para>
/// </remarks>
public class SimMTMOptions<T> : TimeSeriesRegressionOptions<T>
{
    public SimMTMOptions() { }

    public SimMTMOptions(SimMTMOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        ContextLength = other.ContextLength; ForecastHorizon = other.ForecastHorizon;
        PatchLength = other.PatchLength; HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers; NumHeads = other.NumHeads;
        MaskRatio = other.MaskRatio; DropoutRate = other.DropoutRate;
        SimilarityTemperature = other.SimilarityTemperature;
    }

    public int ContextLength { get; set; } = 512;
    public int ForecastHorizon { get; set; } = 96;
    public int PatchLength { get; set; } = 16;
    public int HiddenDimension { get; set; } = 256;
    public int NumLayers { get; set; } = 6;
    public int NumHeads { get; set; } = 8;
    public double MaskRatio { get; set; } = 0.5;
    public double DropoutRate { get; set; } = 0.1;
    public double SimilarityTemperature { get; set; } = 0.07;
}
