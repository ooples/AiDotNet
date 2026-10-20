using System;
using AiDotNet.TimeSeries;

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
        // Seed is declared on ModelOptions rather than in this file, so a copy constructor
        // written from the local declarations alone misses it. Losing it on a clone silently
        // changes deterministic initialization.
        Seed = other.Seed;
        LookbackWindow = other.LookbackWindow;
        _labelLength = other._labelLength;
        ForecastHorizon = other.ForecastHorizon;
        EmbeddingDim = other.EmbeddingDim;
        FeedForwardDim = other.FeedForwardDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        SamplingFactor = other.SamplingFactor;
        Attention = other.Attention;
        Distil = other.Distil;
        Mix = other.Mix;
        UseGelu = other.UseGelu;
        DropoutRate = other.DropoutRate;
        LearningRate = other.LearningRate;
        Epochs = other.Epochs;
        BatchSize = other.BatchSize;
    }

    /// <summary>
    /// Gets or sets the lookback window, the encoder input length (seq_len; the reference's default 96).
    /// </summary>
    public int LookbackWindow { get; set; } = 96;

    /// <summary>
    /// Gets or sets the start-token length: the last steps of the lookback window that open the decoder input
    /// (label_len). Unless set, half of <see cref="LookbackWindow"/>: the reference's 48 for its 96-step window, and a
    /// valid start for any shorter window.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Informer's decoder predicts the whole horizon in one pass. It is given the end of
    /// the known history as a "start", followed by empty placeholders it fills in.</para>
    /// </remarks>
    public int LabelLength
    {
        get => _labelLength ?? LookbackWindow / 2;
        set => _labelLength = value;
    }

    private int? _labelLength;

    /// <summary>
    /// Gets or sets the forecast horizon, the number of predicted steps (pred_len, 24).
    /// </summary>
    public int ForecastHorizon { get; set; } = 24;

    /// <summary>
    /// Gets or sets the model width (d_model, 512).
    /// </summary>
    public int EmbeddingDim { get; set; } = 512;

    /// <summary>
    /// Gets or sets the feed-forward width (d_ff, 2048).
    /// </summary>
    public int FeedForwardDim { get; set; } = 2048;

    /// <summary>
    /// Gets or sets the number of encoder layers (e_layers, 2).
    /// </summary>
    public int NumEncoderLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of decoder layers (d_layers, 1).
    /// </summary>
    public int NumDecoderLayers { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of attention heads (n_heads, 8).
    /// </summary>
    public int NumAttentionHeads { get; set; } = 8;

    /// <summary>
    /// Gets or sets the ProbSparse sampling factor c (factor, 5): c·⌈ln L⌉ keys are sampled per query and as many
    /// queries attend.
    /// </summary>
    public int SamplingFactor { get; set; } = 5;

    /// <summary>
    /// Gets or sets the self-attention of the encoder and decoder (attn, ProbSparse).
    /// </summary>
    public InformerAttention Attention { get; set; } = InformerAttention.ProbSparse;

    /// <summary>
    /// Gets or sets whether a distilling layer (conv, batch norm, ELU, max-pool halving the length) follows every
    /// encoder layer but the last (distil, true).
    /// </summary>
    public bool Distil { get; set; } = true;

    /// <summary>
    /// Gets or sets whether the decoder self-attention mixes its heads as the reference does (mix, true).
    /// </summary>
    public bool Mix { get; set; } = true;

    /// <summary>
    /// Gets or sets whether the feed-forward activation is GELU (activation 'gelu'; false: ReLU).
    /// </summary>
    public bool UseGelu { get; set; } = true;

    /// <summary>
    /// Gets or sets the dropout rate (dropout, 0.05).
    /// </summary>
    public double DropoutRate { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets the initial learning rate of Adam (learning_rate, 1e-4), halved after every epoch (lradj 'type1').
    /// </summary>
    public double LearningRate { get; set; } = 0.0001;

    /// <summary>
    /// Gets or sets the number of training epochs (train_epochs, 6).
    /// </summary>
    public int Epochs { get; set; } = 6;

    /// <summary>
    /// Gets or sets the batch size (batch_size, 32).
    /// </summary>
    public int BatchSize { get; set; } = 32;
}
