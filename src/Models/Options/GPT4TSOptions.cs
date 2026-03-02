using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for GPT4TS (One Fits All: Power General Time Series Analysis by Pretrained LM).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// GPT4TS uses a frozen GPT-2 backbone with task-specific heads for time series forecasting,
/// classification, and anomaly detection. It demonstrates that pretrained language models
/// transfer effectively to time series tasks without fine-tuning the backbone.
/// </para>
/// <para>
/// <b>Reference:</b> Zhou et al., "One Fits All: Power General Time Series Analysis by Pretrained LM", 2023.
/// </para>
/// </remarks>
public class GPT4TSOptions<T> : TimeSeriesRegressionOptions<T>
{
    public GPT4TSOptions() { }

    public GPT4TSOptions(GPT4TSOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        PatchLength = other.PatchLength;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        DropoutRate = other.DropoutRate;
        ModelSize = other.ModelSize;
        Task = other.Task;
        FreezeBackbone = other.FreezeBackbone;
    }

    public int ContextLength { get; set; } = 512;
    public int ForecastHorizon { get; set; } = 96;
    public int PatchLength { get; set; } = 16;
    public int HiddenDimension { get; set; } = 768;
    public int NumLayers { get; set; } = 12;
    public int NumHeads { get; set; } = 12;
    public double DropoutRate { get; set; } = 0.1;
    public FoundationModelSize ModelSize { get; set; } = FoundationModelSize.Base;
    public TimeSeriesFoundationModelTask Task { get; set; } = TimeSeriesFoundationModelTask.Forecasting;

    /// <summary>
    /// Gets or sets whether to freeze the GPT-2 backbone weights.
    /// </summary>
    /// <value>Defaults to true (frozen backbone, only train task heads).</value>
    public bool FreezeBackbone { get; set; } = true;
}
