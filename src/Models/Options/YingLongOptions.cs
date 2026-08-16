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
        : base(other)
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
        WeightDecay = other.WeightDecay;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        WarmupSteps = other.WarmupSteps;
        TotalTrainingSteps = other.TotalTrainingSteps;
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

    /// <summary>Gets or sets the AdamW learning rate. The paper uses 1e-4.</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the decoupled AdamW weight decay. The paper uses 0.1.</summary>
    public double WeightDecay { get; set; } = 0.1;

    /// <summary>Gets or sets AdamW's first-moment decay. The paper uses 0.9.</summary>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>Gets or sets AdamW's second-moment decay. The paper uses 0.95.</summary>
    public double Beta2 { get; set; } = 0.95;

    /// <summary>Gets or sets the linear warmup duration. The paper uses 2,000 steps.</summary>
    public int WarmupSteps { get; set; } = 2_000;

    /// <summary>Gets or sets the cosine schedule duration. The paper trains for 100,000 steps.</summary>
    public int TotalTrainingSteps { get; set; } = 100_000;
}
