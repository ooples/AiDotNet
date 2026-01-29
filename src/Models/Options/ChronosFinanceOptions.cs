using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Chronos Finance (Amazon's time series foundation model for financial forecasting).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// Chronos Finance is an implementation of Amazon's Chronos foundation model optimized for
/// financial time series forecasting. It tokenizes time series values using scaling and quantization,
/// then uses a language model to generate probabilistic forecasts.
/// </para>
/// <para><b>For Beginners:</b> Chronos treats time series forecasting as a language modeling problem:
///
/// <b>The Tokenization Idea:</b>
/// Just like text is converted to tokens for GPT, Chronos converts time series:
/// - Scales values to a standard range (e.g., [-1, 1])
/// - Quantizes into discrete bins (e.g., 4096 bins)
/// - Each bin becomes a "token" like words in text
///
/// <b>Why Tokenize?</b>
/// - Leverages powerful pretrained language models (T5, GPT)
/// - No need to design specialized time series architectures
/// - Benefits from LLM's pattern recognition capabilities
/// - Easy to handle different scales and magnitudes
///
/// <b>Probabilistic via Sampling:</b>
/// Like GPT generating text, Chronos samples from the predicted token distribution:
/// - Each token prediction is a probability over all bins
/// - Multiple samples give uncertainty estimates
/// - More diverse samples = higher uncertainty
///
/// <b>Architecture Variants:</b>
/// Chronos comes in different sizes (like GPT-2 vs GPT-3):
/// - Mini: Fast, lightweight
/// - Small: Balanced
/// - Base: More capacity
/// - Large: Best accuracy, more compute
/// </para>
/// <para>
/// <b>Reference:</b> Ansari et al., "Chronos: Learning the Language of Time Series", 2024.
/// https://arxiv.org/abs/2403.07815
/// </para>
/// </remarks>
public class ChronosFinanceOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ChronosFinanceOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default Chronos configuration for
    /// general-purpose time series forecasting with tokenization-based approach.
    /// </para>
    /// </remarks>
    public ChronosFinanceOptions()
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
    public ChronosFinanceOptions(ChronosFinanceOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        NumTokens = other.NumTokens;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        IntermediateSize = other.IntermediateSize;
        NumSamples = other.NumSamples;
        DropoutRate = other.DropoutRate;
        Temperature = other.Temperature;
        ModelSize = other.ModelSize;
    }

    /// <summary>
    /// Gets or sets the context length (input sequence length).
    /// </summary>
    /// <value>The context length, defaulting to 512.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps the model can look at.
    /// Chronos uses efficient tokenization, allowing longer contexts.
    /// </para>
    /// </remarks>
    public int ContextLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets the forecast horizon (prediction length).
    /// </summary>
    /// <value>The forecast horizon, defaulting to 64.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far into the future to predict.
    /// </para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of discrete tokens (bins) for quantization.
    /// </summary>
    /// <value>The number of tokens, defaulting to 4096.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Chronos converts continuous values to discrete tokens.
    /// More tokens = finer granularity = better precision but larger vocabulary.
    /// 4096 is a good balance (like BPE vocabulary size in text).
    /// </para>
    /// </remarks>
    public int NumTokens { get; set; } = 4096;

    /// <summary>
    /// Gets or sets the hidden dimension of the transformer.
    /// </summary>
    /// <value>The hidden dimension, defaulting to 768.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The internal representation size.
    /// Chronos uses standard transformer dimensions (768 for base).
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 768;

    /// <summary>
    /// Gets or sets the number of transformer layers.
    /// </summary>
    /// <value>The number of layers, defaulting to 12.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How deep the transformer stack is.
    /// More layers = more capacity but more computation.
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 12;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>The number of heads, defaulting to 12.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Multi-head attention allows the model to attend to
    /// different patterns simultaneously.
    /// </para>
    /// </remarks>
    public int NumHeads { get; set; } = 12;

    /// <summary>
    /// Gets or sets the intermediate size for the feed-forward network.
    /// </summary>
    /// <value>The intermediate size, defaulting to 3072 (4x hidden dimension).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The expansion factor in the MLP blocks.
    /// Typically 4x the hidden dimension.
    /// </para>
    /// </remarks>
    public int IntermediateSize { get; set; } = 3072;

    /// <summary>
    /// Gets or sets the number of forecast samples for uncertainty estimation.
    /// </summary>
    /// <value>The number of samples, defaulting to 20.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Chronos generates multiple forecast samples
    /// to estimate uncertainty. More samples = better uncertainty estimates
    /// but slower inference.
    /// </para>
    /// </remarks>
    public int NumSamples { get; set; } = 20;

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
    /// Gets or sets the temperature for sampling.
    /// </summary>
    /// <value>The temperature, defaulting to 1.0.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls sampling randomness:
    /// - Temperature = 1.0: Standard sampling
    /// - Temperature &lt; 1.0: More confident (less random)
    /// - Temperature &gt; 1.0: More diverse (more random)
    /// </para>
    /// </remarks>
    public double Temperature { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the model size variant.
    /// </summary>
    /// <value>The model size, defaulting to "base".</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Chronos comes in different sizes:
    /// - "mini": Smallest, fastest (20M params)
    /// - "small": Light (46M params)
    /// - "base": Balanced (200M params)
    /// - "large": Best accuracy (710M params)
    /// </para>
    /// </remarks>
    public string ModelSize { get; set; } = "base";
}
