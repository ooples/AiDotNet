using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Sundial (A Family of Highly Capable Time Series Foundation Models).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// Sundial is a decoder-only time series foundation model that achieves state-of-the-art
/// performance with fewer parameters than competing models like Time-MoE. It uses a
/// GPT-style autoregressive architecture with patch-based tokenization.
/// </para>
/// <para><b>For Beginners:</b> Sundial is a highly efficient forecasting model:
///
/// <b>Key Innovation:</b>
/// Sundial outperforms Time-MoE (which has up to 2.4B params) with significantly fewer
/// parameters, achieving a 4.71% average MSE reduction.
///
/// <b>Architecture:</b>
/// - Decoder-only transformer (like GPT)
/// - Patch-based input tokenization
/// - Autoregressive generation for forecasting
/// - Efficient scaling through architectural improvements
///
/// <b>When to Use:</b>
/// - High-accuracy forecasting with moderate compute
/// - When you need better accuracy than Time-MoE with fewer parameters
/// - General-purpose time series forecasting across domains
/// </para>
/// <para>
/// <b>Reference:</b> "Sundial: A Family of Highly Capable Time Series Foundation Models", 2025.
/// https://arxiv.org/abs/2502.00816
/// </para>
/// </remarks>
public class SundialOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public SundialOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SundialOptions(SundialOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        PatchLength = other.PatchLength;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        IntermediateSize = other.IntermediateSize;
        DropoutRate = other.DropoutRate;
        ModelSize = other.ModelSize;
        NumQuantiles = other.NumQuantiles;
        UseFlashAttention = other.UseFlashAttention;
    }

    /// <summary>
    /// Gets or sets the context length (input sequence length).
    /// </summary>
    /// <value>Defaults to 2048.</value>
    public int ContextLength { get; set; } = 2048;

    /// <summary>
    /// Gets or sets the forecast horizon (prediction length).
    /// </summary>
    /// <value>Defaults to 96.</value>
    public int ForecastHorizon { get; set; } = 96;

    /// <summary>
    /// Gets or sets the patch length for input tokenization.
    /// </summary>
    /// <value>Defaults to 32.</value>
    public int PatchLength { get; set; } = 32;

    /// <summary>
    /// Gets or sets the hidden dimension of the transformer.
    /// </summary>
    /// <value>Defaults to 1024.</value>
    public int HiddenDimension { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the number of transformer layers.
    /// </summary>
    /// <value>Defaults to 24.</value>
    public int NumLayers { get; set; } = 24;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>Defaults to 16.</value>
    public int NumHeads { get; set; } = 16;

    /// <summary>
    /// Gets or sets the intermediate size for the feed-forward network.
    /// </summary>
    /// <value>Defaults to 4096 (4x hidden dimension).</value>
    public int IntermediateSize { get; set; } = 4096;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>Defaults to 0.1.</value>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the model size variant.
    /// </summary>
    /// <value>Defaults to <see cref="FoundationModelSize.Base"/>.</value>
    public FoundationModelSize ModelSize { get; set; } = FoundationModelSize.Base;

    /// <summary>
    /// Gets or sets the number of quantiles for probabilistic forecasting.
    /// </summary>
    /// <value>Defaults to 9.</value>
    public int NumQuantiles { get; set; } = 9;

    /// <summary>
    /// Gets or sets whether to use flash attention for efficient computation.
    /// </summary>
    /// <value>Defaults to true.</value>
    public bool UseFlashAttention { get; set; } = true;
}
