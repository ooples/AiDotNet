using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Informer model (Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// Informer addresses the computational complexity challenges of vanilla Transformers for long-sequence forecasting.
/// Key innovations include:
/// - ProbSparse self-attention mechanism (O(L log L) complexity instead of O(L²))
/// - Self-attention distilling for efficient stacking
/// - Generative style decoder for one-forward prediction
/// </para>
/// <para><b>For Beginners:</b> Informer is an efficient version of the Transformer architecture
/// designed specifically for long time series. Traditional transformers become very slow with long sequences,
/// but Informer uses smart tricks to be much faster while maintaining accuracy. It's particularly
/// good for forecasting that requires looking far back in history (like predicting next month based on
/// the past year).
/// </para>
/// </remarks>
public class InformerOptions<T> : TimeSeriesRegressionOptions<T>
{
    public InformerOptions() { }

    public InformerOptions(InformerOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        LookbackWindow = other.LookbackWindow;
        ForecastHorizon = other.ForecastHorizon;
        EmbeddingDim = other.EmbeddingDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        DropoutRate = other.DropoutRate;
        LearningRate = other.LearningRate;
        Epochs = other.Epochs;
        BatchSize = other.BatchSize;
        DistillingFactor = other.DistillingFactor;
    }

    /// <summary>
    /// Gets or sets the lookback window (encoder input length).
    /// </summary>
    public int LookbackWindow { get; set; } = 96;

    /// <summary>
    /// Gets or sets the forecast horizon (decoder output length).
    /// </summary>
    public int ForecastHorizon { get; set; } = 24;

    /// <summary>
    /// Gets or sets the embedding dimension.
    /// </summary>
    public int EmbeddingDim { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of encoder layers.
    /// </summary>
    public int NumEncoderLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of decoder layers.
    /// </summary>
    public int NumDecoderLayers { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    public int NumAttentionHeads { get; set; } = 8;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    public double DropoutRate { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets the learning rate.
    /// </summary>
    public double LearningRate { get; set; } = 0.0001;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    public int Epochs { get; set; } = 10;

    /// <summary>
    /// Gets or sets the batch size.
    /// </summary>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets the distilling factor for self-attention distilling.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This controls how much the model compresses
    /// information between layers. A factor of 2 means each layer has half
    /// the sequence length of the previous one.
    /// </para>
    /// </remarks>
    public int DistillingFactor { get; set; } = 2;
}
