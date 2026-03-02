using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for LLM-Time (Zero-Shot Time Series Forecasting via LLM Tokenization).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// LLM-Time converts numeric time series into text strings and uses pretrained LLMs (GPT-3, LLaMA)
/// for zero-shot forecasting by treating the task as next-token prediction on numerical text.
/// No fine-tuning is required—the LLM backbone is frozen.
/// </para>
/// <para>
/// <b>Reference:</b> Gruver et al., "Large Language Models Are Zero-Shot Time Series Forecasters", NeurIPS 2023.
/// </para>
/// </remarks>
public class LLMTimeOptions<T> : TimeSeriesRegressionOptions<T>
{
    public LLMTimeOptions() { }

    public LLMTimeOptions(LLMTimeOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        DropoutRate = other.DropoutRate;
        ModelSize = other.ModelSize;
        NumDecimalPlaces = other.NumDecimalPlaces;
        NumSamples = other.NumSamples;
        Temperature = other.Temperature;
    }

    public int ContextLength { get; set; } = 512;
    public int ForecastHorizon { get; set; } = 96;
    public int HiddenDimension { get; set; } = 768;
    public int NumLayers { get; set; } = 12;
    public int NumHeads { get; set; } = 12;
    public double DropoutRate { get; set; } = 0.0;
    public FoundationModelSize ModelSize { get; set; } = FoundationModelSize.Base;

    /// <summary>
    /// Gets or sets the number of decimal places for numeric-to-text conversion.
    /// </summary>
    /// <value>Defaults to 3.</value>
    public int NumDecimalPlaces { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of samples for probabilistic forecasting.
    /// </summary>
    /// <value>Defaults to 20.</value>
    public int NumSamples { get; set; } = 20;

    /// <summary>
    /// Gets or sets the LLM sampling temperature.
    /// </summary>
    /// <value>Defaults to 0.7.</value>
    public double Temperature { get; set; } = 0.7;
}
