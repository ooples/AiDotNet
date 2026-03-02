using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for YingLong (Alibaba's Enterprise Time Series Foundation Model).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// YingLong is Alibaba's transformer-based time series foundation model designed for
/// general-purpose forecasting with a focus on cloud and enterprise workloads.
/// Pre-trained on large-scale data from Alibaba's data infrastructure.
/// </para>
/// </remarks>
public class YingLongOptions<T> : TimeSeriesRegressionOptions<T>
{
    public YingLongOptions() { }

    public YingLongOptions(YingLongOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        PatchLength = other.PatchLength;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        IntermediateSize = other.IntermediateSize;
        DropoutRate = other.DropoutRate;
        ModelSize = other.ModelSize;
    }

    public int ContextLength { get; set; } = 1024;
    public int ForecastHorizon { get; set; } = 96;
    public int PatchLength { get; set; } = 32;
    public int HiddenDimension { get; set; } = 768;
    public int NumLayers { get; set; } = 12;
    public int NumHeads { get; set; } = 12;
    public int IntermediateSize { get; set; } = 3072;
    public double DropoutRate { get; set; } = 0.1;
    public FoundationModelSize ModelSize { get; set; } = FoundationModelSize.Base;
}
