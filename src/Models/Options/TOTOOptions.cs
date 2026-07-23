using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TOTO (Datadog's Time Series Foundation Model for Observability).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TOTO is Datadog's domain-specific time series foundation model optimized for IT operations,
/// infrastructure monitoring, and observability. Pre-trained on 1 trillion data points from
/// the Datadog observability platform, it excels at SRE metrics and anomaly detection.
/// </para>
/// <para>
/// <b>Reference:</b> Datadog, "Introducing Toto: A state-of-the-art time series foundation model", 2025.
/// </para>
/// </remarks>
public class TOTOOptions<T> : TimeSeriesRegressionOptions<T>
{
    public TOTOOptions() { }

    public TOTOOptions(TOTOOptions<T> other)
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

    public int ContextLength { get; set; } = 2048;
    public int ForecastHorizon { get; set; } = 96;
    public int PatchLength { get; set; } = 32;
    public int HiddenDimension { get; set; } = 768;
    public int NumLayers { get; set; } = 12;
    public int NumHeads { get; set; } = 12;
    public int IntermediateSize { get; set; } = 3072;
    public double DropoutRate { get; set; } = 0.1;
    public FoundationModelSize ModelSize { get; set; } = FoundationModelSize.Base;
}
