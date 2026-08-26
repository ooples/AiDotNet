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
        // Seed is declared on ModelOptions rather than in this file, so a copy constructor
        // written from the local declarations alone misses it. Losing it on a clone silently
        // changes deterministic initialization.
        Seed = other.Seed;
        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        PatchLength = other.PatchLength;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        IntermediateSize = other.IntermediateSize;
        DropoutRate = other.DropoutRate;
        ModelSize = other.ModelSize;
        LearningRate = other.LearningRate;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        WeightDecay = other.WeightDecay;
        WarmupSteps = other.WarmupSteps;
        TotalTrainingSteps = other.TotalTrainingSteps;
    }

    public int ContextLength { get; set; } = 2048;
    public int ForecastHorizon { get; set; } = 96;
    public int PatchLength { get; set; } = 32;
    public int HiddenDimension { get; set; } = 512;
    public int NumLayers { get; set; } = 24;
    public int NumHeads { get; set; } = 8;
    public int IntermediateSize { get; set; } = 2048;
    public double DropoutRate { get; set; } = 0.1;
    public FoundationModelSize ModelSize { get; set; } = FoundationModelSize.Base;
    public double LearningRate { get; set; } = 0.001;
    public double Beta1 { get; set; } = 0.9;
    public double Beta2 { get; set; } = 0.95;
    public double WeightDecay { get; set; } = 0.01;
    public int WarmupSteps { get; set; } = 5000;
    public int TotalTrainingSteps { get; set; } = 193000;
}
