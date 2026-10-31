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
    /// <summary>Creates options with the default values.</summary>
    public YingLongOptions() { }

    /// <summary>Creates a copy of another options instance.</summary>
    /// <param name="other">The options to copy.</param>
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

    /// <summary>Number of past time steps the model reads as context.</summary>
    public int ContextLength { get; set; } = 1024;
    /// <summary>Number of future time steps to forecast.</summary>
    public int ForecastHorizon { get; set; } = 96;
    /// <summary>Number of time steps in each patch the series is split into.</summary>
    public int PatchLength { get; set; } = 32;
    /// <summary>Width of the model's hidden representations.</summary>
    public int HiddenDimension { get; set; } = 768;
    /// <summary>Number of transformer layers.</summary>
    public int NumLayers { get; set; } = 12;
    /// <summary>Number of attention heads.</summary>
    public int NumHeads { get; set; } = 12;
    /// <summary>Width of the feed-forward layer inside each transformer block.</summary>
    public int IntermediateSize { get; set; } = 3072;
    /// <summary>Dropout probability applied during training.</summary>
    public double DropoutRate { get; set; } = 0.1;
    /// <summary>Preset size of the foundation model.</summary>
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
