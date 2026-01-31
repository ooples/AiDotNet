using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Timer (Generative Pre-Training for Time Series).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// Timer is a generative pre-training approach for time series that uses
/// autoregressive generation combined with masked modeling to learn rich
/// temporal representations from diverse time series datasets.
/// </para>
/// <para><b>For Beginners:</b> Timer brings GPT-style pre-training to time series:
///
/// <b>The Key Insight:</b>
/// Just like GPT learns language by predicting the next token, Timer learns
/// time series patterns by predicting future values. Pre-training on diverse
/// datasets enables strong zero-shot transfer.
///
/// <b>How It Works:</b>
/// 1. <b>Autoregressive Pre-training:</b> Learn to predict future from past
/// 2. <b>Masked Modeling:</b> Learn to reconstruct masked portions
/// 3. <b>Multi-scale Processing:</b> Handle different temporal granularities
/// 4. <b>Fine-tuning:</b> Adapt to specific domains with minimal data
///
/// <b>Architecture:</b>
/// - Patch-based tokenization (like PatchTST)
/// - GPT-style decoder transformer
/// - Autoregressive generation head
/// - Optional masked modeling objective
///
/// <b>Advantages:</b>
/// - Strong zero-shot and few-shot performance
/// - Generalizes across domains and frequencies
/// - Efficient fine-tuning with minimal labeled data
/// - Handles variable sequence lengths
/// </para>
/// <para>
/// <b>Reference:</b> Liu et al., "Timer: Generative Pre-Training of Time Series", 2024.
/// https://arxiv.org/abs/2402.02368
/// </para>
/// </remarks>
public class TimerOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TimerOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default Timer configuration for
    /// generative time series pre-training and forecasting.
    /// </para>
    /// </remarks>
    public TimerOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public TimerOptions(TimerOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        PatchLength = other.PatchLength;
        PatchStride = other.PatchStride;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        DropoutRate = other.DropoutRate;
        MaskRatio = other.MaskRatio;
        UseAutoregressiveDecoding = other.UseAutoregressiveDecoding;
        GenerationTemperature = other.GenerationTemperature;
    }

    /// <summary>
    /// Gets or sets the context length (input sequence length).
    /// </summary>
    /// <value>The context length, defaulting to 512.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps the model can look at.
    /// Longer context allows learning longer-range patterns.
    /// </para>
    /// </remarks>
    public int ContextLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets the forecast horizon (prediction length).
    /// </summary>
    /// <value>The forecast horizon, defaulting to 96.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far into the future to predict.
    /// Can be overridden at inference time for flexible forecasting.
    /// </para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 96;

    /// <summary>
    /// Gets or sets the patch length for tokenization.
    /// </summary>
    /// <value>The patch length, defaulting to 16.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Time series is divided into patches (like words in text).
    /// Each patch becomes a "token" that Timer processes.
    /// </para>
    /// </remarks>
    public int PatchLength { get; set; } = 16;

    /// <summary>
    /// Gets or sets the patch stride.
    /// </summary>
    /// <value>The patch stride, defaulting to 8.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How much to slide the window for each patch.
    /// Stride less than patch length creates overlapping patches.
    /// </para>
    /// </remarks>
    public int PatchStride { get; set; } = 8;

    /// <summary>
    /// Gets or sets the hidden dimension size.
    /// </summary>
    /// <value>The hidden dimension, defaulting to 768.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The internal representation size.
    /// Larger dimensions can capture more complex patterns but require more memory.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 768;

    /// <summary>
    /// Gets or sets the number of transformer layers.
    /// </summary>
    /// <value>The number of layers, defaulting to 12.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How deep the transformer stack is.
    /// Deeper models can learn more abstract representations.
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
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout helps prevent overfitting by randomly
    /// dropping connections during training.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the mask ratio for masked modeling pre-training.
    /// </summary>
    /// <value>The mask ratio, defaulting to 0.4.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> During pre-training, this fraction of patches
    /// is masked and the model learns to reconstruct them. Higher values make
    /// the task harder but can lead to better representations.
    /// </para>
    /// </remarks>
    public double MaskRatio { get; set; } = 0.4;

    /// <summary>
    /// Gets or sets whether to use autoregressive decoding during generation.
    /// </summary>
    /// <value>True to use autoregressive decoding; false for parallel decoding. Default: true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Autoregressive decoding generates one step at a time,
    /// using each prediction as input for the next. Parallel decoding generates all
    /// steps at once (faster but potentially less accurate for long horizons).
    /// </para>
    /// </remarks>
    public bool UseAutoregressiveDecoding { get; set; } = true;

    /// <summary>
    /// Gets or sets the temperature for sampling during generation.
    /// </summary>
    /// <value>The temperature, defaulting to 1.0.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Temperature controls randomness in generation:
    /// - Lower values (e.g., 0.1) = more deterministic/confident predictions
    /// - Higher values (e.g., 2.0) = more diverse/random predictions
    /// - 1.0 = balanced
    /// Use lower temperature for point forecasts, higher for sampling diverse scenarios.
    /// </para>
    /// </remarks>
    public double GenerationTemperature { get; set; } = 1.0;
}
