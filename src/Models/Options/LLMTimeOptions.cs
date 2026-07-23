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
/// <para><b>For Beginners:</b> LLM-Time takes a surprising approach:
///
/// <b>How It Works:</b>
/// 1. Convert numbers to text: [1.5, 2.3, 3.1] → "1.500, 2.300, 3.100"
/// 2. Feed the text to a pretrained LLM (GPT-3, LLaMA)
/// 3. The LLM predicts the next "tokens" (which are digits of future values)
/// 4. Parse the generated text back into numbers
///
/// <b>Key Advantages:</b>
/// - Zero-shot: no training required at all
/// - Leverages the vast pattern knowledge of large language models
/// - Produces probabilistic forecasts via sampling
///
/// <b>Trade-offs:</b>
/// - Requires access to a large language model API
/// - Limited precision (controlled by decimal places)
/// - Slower than specialized time series models
/// </para>
/// <para>
/// <b>Reference:</b> Gruver et al., "Large Language Models Are Zero-Shot Time Series Forecasters", NeurIPS 2023.
/// </para>
/// </remarks>
public class LLMTimeOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public LLMTimeOptions() { }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The instance to copy from.</param>
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

    /// <summary>
    /// Gets or sets the maximum number of historical time steps for the text prompt.
    /// </summary>
    /// <value>Defaults to 512.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past values are included in the text prompt.
    /// Limited by the LLM's context window (each number uses multiple text tokens).</para>
    /// </remarks>
    public int ContextLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets the number of future time steps to forecast.
    /// </summary>
    /// <value>Defaults to 96.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many future values the LLM generates as text.</para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 96;

    /// <summary>
    /// Gets or sets the hidden dimension of the LLM backbone.
    /// </summary>
    /// <value>Defaults to 768.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Must match the pretrained LLM being used. This is
    /// for native mode simulation only; in practice, LLM-Time calls an external LLM API.</para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 768;

    /// <summary>
    /// Gets or sets the number of transformer layers.
    /// </summary>
    /// <value>Defaults to 12.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Must match the pretrained LLM architecture.</para>
    /// </remarks>
    public int NumLayers { get; set; } = 12;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>Defaults to 12.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Must match the pretrained LLM architecture.</para>
    /// </remarks>
    public int NumHeads { get; set; } = 12;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    /// <value>Defaults to 0.0 (no dropout for frozen backbone).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Set to 0 because LLM-Time uses a frozen backbone
    /// with no training. Dropout is only relevant during training.</para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the model size variant.
    /// </summary>
    /// <value>Defaults to <see cref="FoundationModelSize.Base"/>.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls which LLM checkpoint to use.
    /// Larger models give better forecasts but cost more to run.</para>
    /// </remarks>
    public FoundationModelSize ModelSize { get; set; } = FoundationModelSize.Base;

    /// <summary>
    /// Gets or sets the number of decimal places for numeric-to-text conversion.
    /// </summary>
    /// <value>Defaults to 3.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls precision of the text representation.
    /// More decimals = more precision but more tokens per number (slower, uses more
    /// context window). 3 decimal places works well for most applications.</para>
    /// </remarks>
    public int NumDecimalPlaces { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of samples for probabilistic forecasting.
    /// </summary>
    /// <value>Defaults to 20.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> LLM-Time generates multiple forecast samples by
    /// running the LLM multiple times with different random seeds. More samples give
    /// better uncertainty estimates but increase inference time linearly.</para>
    /// </remarks>
    public int NumSamples { get; set; } = 20;

    /// <summary>
    /// Gets or sets the LLM sampling temperature.
    /// </summary>
    /// <value>Defaults to 0.7.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls randomness in the LLM's predictions.
    /// Lower values (e.g., 0.1) make predictions more deterministic;
    /// higher values (e.g., 1.0) produce more diverse samples.
    /// 0.7 is a good balance between diversity and quality.</para>
    /// </remarks>
    public double Temperature { get; set; } = 0.7;
}
