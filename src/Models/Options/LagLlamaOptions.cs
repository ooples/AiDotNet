using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Lag-Llama (Large Language Model for time series forecasting).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// Lag-Llama is a foundation model that adapts LLM architecture for time series forecasting.
/// It uses a decoder-only transformer with lag-based features to capture temporal dependencies
/// across multiple time scales.
/// </para>
/// <para><b>For Beginners:</b> Lag-Llama brings large language model innovations to time series:
///
/// <b>The Lag Feature Idea:</b>
/// Instead of just using recent values, Lag-Llama looks at values from specific past points:
/// - Lag-1: Value from 1 step ago (yesterday)
/// - Lag-7: Value from 7 steps ago (last week, same day)
/// - Lag-365: Value from 365 steps ago (last year, same day)
///
/// These "lag features" help capture patterns at different time scales.
///
/// <b>Why Llama-style Architecture?</b>
/// Llama introduced efficient transformer improvements:
/// - RMSNorm: Simpler, faster layer normalization
/// - RoPE: Rotary Position Embeddings for better position encoding
/// - SwiGLU: Improved activation function
/// - Grouped Query Attention: More efficient attention
///
/// <b>Zero-Shot Capability:</b>
/// Like other foundation models, Lag-Llama is pre-trained on diverse time series
/// and can forecast new series without fine-tuning.
///
/// <b>Probabilistic Forecasting:</b>
/// Lag-Llama outputs a distribution (not just point estimates):
/// - Predicts parameters of a probability distribution
/// - Allows uncertainty quantification
/// - Enables risk-aware decision making
///
/// <b>When to Use:</b>
/// - Time series with multiple seasonal patterns (daily, weekly, yearly)
/// - When you need uncertainty estimates
/// - Cross-domain zero-shot forecasting
/// </para>
/// <para>
/// <b>Reference:</b> Rasul et al., "Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting", 2024.
/// https://arxiv.org/abs/2310.08278
/// </para>
/// </remarks>
public class LagLlamaOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="LagLlamaOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default Lag-Llama configuration optimized for
    /// general-purpose probabilistic time series forecasting.
    /// </para>
    /// </remarks>
    public LagLlamaOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a copy of existing options.
    /// </para>
    /// </remarks>
    public LagLlamaOptions(LagLlamaOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        IntermediateSize = other.IntermediateSize;
        LagIndices = (int[])other.LagIndices.Clone();
        DropoutRate = other.DropoutRate;
        DistributionOutput = other.DistributionOutput;
        UseRoPE = other.UseRoPE;
    }

    /// <summary>
    /// Gets or sets the context length (input sequence length).
    /// </summary>
    /// <value>The context length, defaulting to 96.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps the model looks at.
    /// Lag-Llama uses lag features, so the effective context can be much longer.
    /// </para>
    /// </remarks>
    public int ContextLength { get; set; } = 96;

    /// <summary>
    /// Gets or sets the forecast horizon (prediction length).
    /// </summary>
    /// <value>The forecast horizon, defaulting to 24.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far into the future to predict.
    /// </para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 24;

    /// <summary>
    /// Gets or sets the hidden dimension of the transformer.
    /// </summary>
    /// <value>The hidden dimension, defaulting to 256.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The internal representation size.
    /// Larger dimensions = more capacity but more computation.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the number of transformer layers.
    /// </summary>
    /// <value>The number of layers, defaulting to 8.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How deep the transformer stack is.
    /// More layers = more capacity but more computation.
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 8;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>The number of heads, defaulting to 4.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Multi-head attention allows the model to focus on
    /// different aspects simultaneously. Each head learns different patterns.
    /// </para>
    /// </remarks>
    public int NumHeads { get; set; } = 4;

    /// <summary>
    /// Gets or sets the intermediate size for the feed-forward network.
    /// </summary>
    /// <value>The intermediate size, defaulting to 1024 (4x hidden dimension).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The expansion factor in the MLP blocks.
    /// Typically 4x the hidden dimension, following Llama convention.
    /// </para>
    /// </remarks>
    public int IntermediateSize { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the lag indices used for feature extraction.
    /// </summary>
    /// <value>Array of lag indices, defaulting to common seasonal lags.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Which past time points to look at:
    /// - 1, 2, 3: Recent history (short-term patterns)
    /// - 7: Weekly seasonality (same day last week)
    /// - 14: Bi-weekly patterns
    /// - 28: Monthly patterns (approximately)
    /// - 365: Yearly seasonality (same day last year)
    ///
    /// Custom lags can be set based on domain knowledge.
    /// </para>
    /// </remarks>
    public int[] LagIndices { get; set; } = [1, 2, 3, 7, 14, 28, 365];

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout helps prevent overfitting.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the distribution type for probabilistic output.
    /// </summary>
    /// <value>The distribution output type, defaulting to "StudentT".</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Instead of single point predictions, Lag-Llama
    /// outputs parameters of a probability distribution:
    /// - "StudentT": Student's t-distribution (handles heavy tails)
    /// - "Normal": Gaussian distribution (mean and variance)
    /// - "NegativeBinomial": For count data
    /// </para>
    /// </remarks>
    public string DistributionOutput { get; set; } = "StudentT";

    /// <summary>
    /// Gets or sets whether to use Rotary Position Embeddings (RoPE).
    /// </summary>
    /// <value>True to use RoPE, defaulting to true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> RoPE is a modern position encoding method that:
    /// - Encodes relative positions, not absolute
    /// - Generalizes better to different sequence lengths
    /// - Is more efficient than learned position embeddings
    /// </para>
    /// </remarks>
    public bool UseRoPE { get; set; } = true;
}
