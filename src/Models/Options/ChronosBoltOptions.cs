using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Chronos-Bolt (Fast Non-Autoregressive Time Series Forecasting).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Chronos-Bolt is part of the Amazon Chronos family but uses an encoder-decoder architecture
/// with direct quantile forecasting (non-autoregressive), making it significantly faster than
/// the autoregressive Chronos v1/v2 models while maintaining competitive accuracy.
/// </para>
/// <para><b>For Beginners:</b> Chronos-Bolt trades autoregressive generation for speed:
///
/// <b>Key Difference from Chronos v1/v2:</b>
/// - Chronos v1/v2: Generates one token at a time (autoregressive, slow)
/// - Chronos-Bolt: Generates all forecast steps at once (non-autoregressive, fast)
///
/// <b>Architecture:</b>
/// - Encoder: Processes the input context
/// - Decoder: Directly outputs all forecast quantiles in one pass
/// - No autoregressive loop = much faster inference
///
/// <b>When to Use:</b>
/// - When you need fast inference (production/real-time)
/// - When Chronos v1/v2 is too slow for your use case
/// - When you need quantile forecasts (uncertainty estimates)
/// </para>
/// <para>
/// <b>Reference:</b> Part of Chronos family, Amazon, Nov 2024.
/// </para>
/// </remarks>
public class ChronosBoltOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public ChronosBoltOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    public ChronosBoltOptions(ChronosBoltOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        PatchLength = other.PatchLength;
        EncoderHiddenDim = other.EncoderHiddenDim;
        DecoderHiddenDim = other.DecoderHiddenDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        NumHeads = other.NumHeads;
        DropoutRate = other.DropoutRate;
        ModelSize = other.ModelSize;
        NumQuantiles = other.NumQuantiles;
    }

    /// <summary>
    /// Gets or sets the context length.
    /// </summary>
    /// <value>Defaults to 512.</value>
    public int ContextLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets the forecast horizon.
    /// </summary>
    /// <value>Defaults to 64.</value>
    public int ForecastHorizon { get; set; } = 64;

    /// <summary>
    /// Gets or sets the patch length for input tokenization.
    /// </summary>
    /// <value>Defaults to 16.</value>
    public int PatchLength { get; set; } = 16;

    /// <summary>
    /// Gets or sets the encoder hidden dimension.
    /// </summary>
    /// <value>Defaults to 512.</value>
    public int EncoderHiddenDim { get; set; } = 512;

    /// <summary>
    /// Gets or sets the decoder hidden dimension.
    /// </summary>
    /// <value>Defaults to 512.</value>
    public int DecoderHiddenDim { get; set; } = 512;

    /// <summary>
    /// Gets or sets the number of encoder layers.
    /// </summary>
    /// <value>Defaults to 6.</value>
    public int NumEncoderLayers { get; set; } = 6;

    /// <summary>
    /// Gets or sets the number of decoder layers.
    /// </summary>
    /// <value>Defaults to 6.</value>
    public int NumDecoderLayers { get; set; } = 6;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>Defaults to 8.</value>
    public int NumHeads { get; set; } = 8;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    /// <value>Defaults to 0.1.</value>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the model size variant.
    /// </summary>
    /// <value>Defaults to <see cref="FoundationModelSize.Base"/>.</value>
    public FoundationModelSize ModelSize { get; set; } = FoundationModelSize.Base;

    /// <summary>
    /// Gets or sets the number of quantiles for direct quantile forecasting.
    /// </summary>
    /// <value>Defaults to 9.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Chronos-Bolt directly outputs multiple quantile
    /// predictions (e.g., 10th, 20th, ..., 90th percentile) in a single forward pass.
    /// </para>
    /// </remarks>
    public int NumQuantiles { get; set; } = 9;
}
