using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the TimesFM (Time Series Foundation Model).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// TimesFM is Google's foundation model for time series forecasting. It uses a decoder-only
/// transformer architecture pre-trained on a massive dataset of diverse time series, enabling
/// zero-shot and few-shot forecasting across different domains without fine-tuning.
/// </para>
/// <para><b>For Beginners:</b> TimesFM is a pre-trained model for general-purpose forecasting:
///
/// <b>What is a Foundation Model?</b>
/// Like GPT for text, foundation models for time series are:
/// - Pre-trained on massive, diverse datasets
/// - Can generalize to new forecasting tasks without fine-tuning
/// - Work across different domains (finance, weather, retail, etc.)
///
/// <b>Zero-Shot Forecasting:</b>
/// TimesFM can forecast a time series it has never seen before:
/// - No training required on your specific data
/// - Just provide historical values and get predictions
/// - Works because it learned general patterns during pre-training
///
/// <b>Architecture:</b>
/// TimesFM uses a decoder-only transformer (like GPT):
/// - Processes historical time steps as "tokens"
/// - Each token attends to all previous tokens
/// - Generates forecast tokens autoregressively
///
/// <b>Input Patching:</b>
/// Instead of processing one time step at a time, TimesFM:
/// - Groups consecutive time steps into "patches"
/// - Each patch becomes one input token
/// - Reduces sequence length, enables longer context
///
/// <b>When to Use:</b>
/// - Quick forecasting without model training
/// - New domains with limited historical data
/// - Baseline comparisons for custom models
/// </para>
/// <para>
/// <b>Reference:</b> Das et al., "A decoder-only foundation model for time-series forecasting", 2024.
/// https://arxiv.org/abs/2310.10688
/// </para>
/// </remarks>
public class TimesFMOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TimesFMOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default TimesFM configuration optimized for
    /// general-purpose time series forecasting.
    /// </para>
    /// </remarks>
    public TimesFMOptions()
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
    public TimesFMOptions(TimesFMOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        PatchLength = other.PatchLength;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        DropoutRate = other.DropoutRate;
        UsePretrainedWeights = other.UsePretrainedWeights;
    }

    /// <summary>
    /// Gets or sets the context length (input sequence length).
    /// </summary>
    /// <value>The context length, defaulting to 512.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps the model can look at.
    /// TimesFM supports longer contexts than typical models due to patching.
    /// 512 time steps is a good default for most applications.
    /// </para>
    /// </remarks>
    public int ContextLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets the forecast horizon (prediction length).
    /// </summary>
    /// <value>The forecast horizon, defaulting to 96.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far into the future to predict.
    /// TimesFM can generate forecasts of variable length.
    /// </para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 96;

    /// <summary>
    /// Gets or sets the patch length for input tokenization.
    /// </summary>
    /// <value>The patch length, defaulting to 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many consecutive time steps are grouped into one token.
    /// - Larger patches = fewer tokens = faster but less granular
    /// - Smaller patches = more tokens = slower but more detailed
    ///
    /// The number of tokens = ContextLength / PatchLength.
    /// With defaults: 512 / 32 = 16 tokens for the transformer.
    /// </para>
    /// </remarks>
    public int PatchLength { get; set; } = 32;

    /// <summary>
    /// Gets or sets the hidden dimension of the transformer.
    /// </summary>
    /// <value>The hidden dimension, defaulting to 256.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The internal representation size.
    /// Larger dimensions = more capacity but more computation.
    /// The pre-trained model uses specific dimensions.
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
    /// <para><b>For Beginners:</b> Multi-head attention allows the model to attend to
    /// different patterns simultaneously. Each head learns different relationships.
    /// </para>
    /// </remarks>
    public int NumHeads { get; set; } = 4;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout helps prevent overfitting.
    /// For pre-trained models, this is mainly used during fine-tuning.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether to use pre-trained weights.
    /// </summary>
    /// <value>True to use pre-trained weights, defaulting to true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When true, the model loads weights from pre-training.
    /// Set to false to train from scratch (not recommended for most use cases).
    /// </para>
    /// </remarks>
    public bool UsePretrainedWeights { get; set; } = true;
}
