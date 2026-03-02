using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for GPT4TS (One Fits All: Power General Time Series Analysis by Pretrained LM).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// GPT4TS uses a frozen GPT-2 backbone with task-specific heads for time series forecasting,
/// classification, and anomaly detection. It demonstrates that pretrained language models
/// transfer effectively to time series tasks without fine-tuning the backbone.
/// </para>
/// <para><b>For Beginners:</b> GPT4TS repurposes GPT-2 (a language model) for time series:
///
/// <b>How It Works:</b>
/// 1. Time series are split into patches (like "words")
/// 2. Patches are fed through a frozen GPT-2 backbone (no weight updates)
/// 3. A lightweight task-specific head is trained on top
///
/// <b>Key Advantages:</b>
/// - Leverages pretrained language model knowledge
/// - Only the small task head needs training (fast + low data)
/// - Supports multiple tasks: forecasting, classification, anomaly detection
///
/// <b>When to Use:</b>
/// - When you have limited training data (the frozen backbone provides strong priors)
/// - When you need multi-task support from a single model
/// </para>
/// <para>
/// <b>Reference:</b> Zhou et al., "One Fits All: Power General Time Series Analysis by Pretrained LM", 2023.
/// </para>
/// </remarks>
public class GPT4TSOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public GPT4TSOptions() { }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The instance to copy from.</param>
    public GPT4TSOptions(GPT4TSOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        PatchLength = other.PatchLength;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        DropoutRate = other.DropoutRate;
        ModelSize = other.ModelSize;
        Task = other.Task;
        FreezeBackbone = other.FreezeBackbone;
    }

    /// <summary>
    /// Gets or sets the number of historical time steps used as input context.
    /// </summary>
    /// <value>Defaults to 512.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How much historical data the model sees. Must be
    /// divisible by <see cref="PatchLength"/>.</para>
    /// </remarks>
    public int ContextLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets the number of future time steps to forecast.
    /// </summary>
    /// <value>Defaults to 96.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far into the future the model predicts.</para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 96;

    /// <summary>
    /// Gets or sets the patch length for input tokenization.
    /// </summary>
    /// <value>Defaults to 16.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Time series are split into non-overlapping patches of this
    /// size, similar to how text is split into tokens. Smaller patches capture finer detail
    /// but produce more tokens.</para>
    /// </remarks>
    public int PatchLength { get; set; } = 16;

    /// <summary>
    /// Gets or sets the hidden dimension of the GPT-2 backbone.
    /// </summary>
    /// <value>Defaults to 768 (GPT-2 base).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This should match the pretrained GPT-2 model size.
    /// GPT-2 base uses 768, GPT-2 medium uses 1024.</para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 768;

    /// <summary>
    /// Gets or sets the number of transformer layers.
    /// </summary>
    /// <value>Defaults to 12 (GPT-2 base).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Should match the pretrained GPT-2 model. GPT-2 base has 12 layers.</para>
    /// </remarks>
    public int NumLayers { get; set; } = 12;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>Defaults to 12 (GPT-2 base).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Should match the pretrained GPT-2 model. GPT-2 base has 12 heads.</para>
    /// </remarks>
    public int NumHeads { get; set; } = 12;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>Defaults to 0.1 (10%).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Applied only to the trainable task head, not the frozen backbone.</para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the model size variant.
    /// </summary>
    /// <value>Defaults to <see cref="FoundationModelSize.Base"/>.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls which GPT-2 checkpoint to load.
    /// Larger models have more capacity but require more memory.</para>
    /// </remarks>
    public FoundationModelSize ModelSize { get; set; } = FoundationModelSize.Base;

    /// <summary>
    /// Gets or sets the downstream task for the model.
    /// </summary>
    /// <value>Defaults to <see cref="TimeSeriesFoundationModelTask.Forecasting"/>.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> GPT4TS supports forecasting, classification, and anomaly
    /// detection. The task determines which output head is used.</para>
    /// </remarks>
    public TimeSeriesFoundationModelTask Task { get; set; } = TimeSeriesFoundationModelTask.Forecasting;

    /// <summary>
    /// Gets or sets whether to freeze the GPT-2 backbone weights.
    /// </summary>
    /// <value>Defaults to true (frozen backbone, only train task heads).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When true, only the lightweight task head is trained,
    /// preserving the pretrained GPT-2 knowledge. Set to false for full fine-tuning
    /// (requires more data and compute).</para>
    /// </remarks>
    public bool FreezeBackbone { get; set; } = true;
}
