using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Time-LLM (Large Language Model Reprogramming for Time Series).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// Time-LLM repurposes frozen large language models for time series forecasting by
/// learning a reprogramming layer that translates time series into text-like representations
/// that the LLM can understand.
/// </para>
/// <para><b>For Beginners:</b> Time-LLM is a clever way to use powerful language models for time series:
///
/// <b>The Key Insight:</b>
/// LLMs like GPT/LLaMA are amazing at pattern recognition in sequences.
/// Time-LLM asks: "Can we make time series 'speak' the language of LLMs?"
///
/// <b>How It Works:</b>
/// 1. <b>Patch Reprogramming:</b> Convert time series patches into "prompt-like" tokens
/// 2. <b>Text Prototypes:</b> Learn embeddings that bridge numeric and text domains
/// 3. <b>Frozen LLM:</b> The LLM weights stay fixed (no fine-tuning needed)
/// 4. <b>Output Projection:</b> Map LLM output back to forecast values
///
/// <b>Advantages:</b>
/// - Leverages powerful pretrained LLMs without expensive fine-tuning
/// - Works with any LLM backbone (GPT-2, LLaMA, etc.)
/// - Only trains small reprogramming layers
/// - Zero-shot transfer to new domains
///
/// <b>Architecture:</b>
/// [Time Series] → [Patch] → [Reprogram] → [Frozen LLM] → [Project] → [Forecast]
/// </para>
/// <para>
/// <b>Reference:</b> Jin et al., "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models", 2024.
/// https://arxiv.org/abs/2310.01728
/// </para>
/// </remarks>
public class TimeLLMOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TimeLLMOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default Time-LLM configuration for
    /// reprogramming-based time series forecasting.
    /// </para>
    /// </remarks>
    public TimeLLMOptions()
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
    public TimeLLMOptions(TimeLLMOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        PatchLength = other.PatchLength;
        PatchStride = other.PatchStride;
        LLMDimension = other.LLMDimension;
        NumPrototypes = other.NumPrototypes;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        DropoutRate = other.DropoutRate;
        LLMBackbone = other.LLMBackbone;
    }

    /// <summary>
    /// Gets or sets the context length (input sequence length).
    /// </summary>
    /// <value>The context length, defaulting to 512.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps the model can look at.
    /// </para>
    /// </remarks>
    public int ContextLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets the forecast horizon (prediction length).
    /// </summary>
    /// <value>The forecast horizon, defaulting to 96.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far into the future to predict.
    /// </para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 96;

    /// <summary>
    /// Gets or sets the patch length for input segmentation.
    /// </summary>
    /// <value>The patch length, defaulting to 16.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Time series is divided into patches, like words in text.
    /// Each patch becomes a token for the LLM to process.
    /// </para>
    /// </remarks>
    public int PatchLength { get; set; } = 16;

    /// <summary>
    /// Gets or sets the patch stride.
    /// </summary>
    /// <value>The patch stride, defaulting to 8.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How much to slide the window for each patch.
    /// Stride less than patch length means overlapping patches.
    /// </para>
    /// </remarks>
    public int PatchStride { get; set; } = 8;

    /// <summary>
    /// Gets or sets the LLM hidden dimension.
    /// </summary>
    /// <value>The LLM dimension, defaulting to 768.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The internal representation size of the LLM.
    /// Must match the frozen LLM backbone (e.g., 768 for GPT-2).
    /// </para>
    /// </remarks>
    public int LLMDimension { get; set; } = 768;

    /// <summary>
    /// Gets or sets the number of text prototypes.
    /// </summary>
    /// <value>The number of prototypes, defaulting to 10.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Text prototypes are learned embeddings that help
    /// translate time series patterns into the LLM's "language". More prototypes
    /// can capture more diverse patterns.
    /// </para>
    /// </remarks>
    public int NumPrototypes { get; set; } = 10;

    /// <summary>
    /// Gets or sets the number of transformer layers in the reprogramming module.
    /// </summary>
    /// <value>The number of layers, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The reprogramming module uses a few transformer layers
    /// to learn the mapping from time series to LLM space. Fewer layers since most
    /// processing is done by the frozen LLM.
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>The number of heads, defaulting to 8.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Multi-head attention in the reprogramming module.
    /// </para>
    /// </remarks>
    public int NumHeads { get; set; } = 8;

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
    /// Gets or sets the LLM backbone type.
    /// </summary>
    /// <value>The LLM backbone, defaulting to "gpt2".</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Which pretrained LLM to use as the backbone:
    /// - "gpt2": GPT-2 (768 dim, 12 layers)
    /// - "llama": LLaMA-style (various sizes)
    /// - "bert": BERT-style encoder
    /// </para>
    /// </remarks>
    public string LLMBackbone { get; set; } = "gpt2";
}
