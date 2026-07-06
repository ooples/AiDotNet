using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Chronos-Bolt (Fast Non-Autoregressive Time Series Forecasting).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Chronos-Bolt is part of the Amazon Chronos family but uses an encoder-decoder architecture
/// with direct quantile forecasting (non-autoregressive), making it significantly faster than
/// the autoregressive Chronos v1/v2 models while maintaining competitive accuracy.
/// </para>
/// <para><b>For Beginners:</b> Chronos-Bolt trades autoregressive generation for speed:
///
/// <b>Key Difference from Chronos v1/v2:</b>
/// - Chronos v1/v2: Generates one token at a time (autoregressive, slow)
/// - Chronos-Bolt: Generates all forecast steps at once (non-autoregressive, fast)
///
/// <b>Architecture:</b>
/// - Encoder: Processes the input context
/// - Decoder: Directly outputs all forecast quantiles in one pass
/// - No autoregressive loop = much faster inference
///
/// <b>When to Use:</b>
/// - When you need fast inference (production/real-time)
/// - When Chronos v1/v2 is too slow for your use case
/// - When you need quantile forecasts (uncertainty estimates)
/// </para>
/// <para>
/// <b>Reference:</b> Part of Chronos family, Amazon, Nov 2024.
/// </para>
/// </remarks>
public class ChronosBoltOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public ChronosBoltOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    public ChronosBoltOptions(ChronosBoltOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        PatchLength = other.PatchLength;
        EncoderHiddenDim = other.EncoderHiddenDim;
        DecoderHiddenDim = other.DecoderHiddenDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        NumHeads = other.NumHeads;
        DropoutRate = other.DropoutRate;
        ModelSize = other.ModelSize;
        NumQuantiles = other.NumQuantiles;
    }

    /// <summary>
    /// Gets or sets the context length.
    /// </summary>
    /// <value>Defaults to 512.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps the encoder processes.</para>
    /// </remarks>
    public int ContextLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets the forecast horizon.
    /// </summary>
    /// <value>Defaults to 64.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many future steps the decoder outputs in one pass.</para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 64;

    /// <summary>
    /// Gets or sets the patch length for input tokenization.
    /// </summary>
    /// <value>Defaults to 16.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Groups consecutive time steps into patches. Must divide evenly into <see cref="ContextLength"/>.</para>
    /// </remarks>
    public int PatchLength { get; set; } = 16;

    /// <summary>
    /// Gets or sets the encoder hidden dimension.
    /// </summary>
    /// <value>Defaults to 512.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Internal representation size for the encoder that processes historical data.</para>
    /// </remarks>
    public int EncoderHiddenDim { get; set; } = 512;

    /// <summary>
    /// Gets or sets the decoder hidden dimension.
    /// </summary>
    /// <value>Defaults to 512.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Internal representation size for the decoder that produces forecasts.</para>
    /// </remarks>
    public int DecoderHiddenDim { get; set; } = 512;

    /// <summary>
    /// Gets or sets the number of encoder layers.
    /// </summary>
    /// <value>Defaults to 6.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Depth of the encoder stack. More layers capture more complex input patterns.</para>
    /// </remarks>
    public int NumEncoderLayers { get; set; } = 6;

    /// <summary>
    /// Gets or sets the number of decoder layers.
    /// </summary>
    /// <value>Defaults to 6.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Depth of the decoder stack. More layers can produce more nuanced forecasts.</para>
    /// </remarks>
    public int NumDecoderLayers { get; set; } = 6;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>Defaults to 8.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each head focuses on different temporal patterns. Must divide into encoder/decoder hidden dims.</para>
    /// </remarks>
    public int NumHeads { get; set; } = 8;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    /// <value>Defaults to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Regularization to prevent overfitting. Set to 0 to disable.</para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the model size variant.
    /// </summary>
    /// <value>Defaults to <see cref="FoundationModelSize.Base"/>.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls overall model capacity. Available sizes: Mini, Small, Base.</para>
    /// </remarks>
    public FoundationModelSize ModelSize { get; set; } = FoundationModelSize.Base;

    /// <summary>
    /// Gets or sets the number of quantiles for direct quantile forecasting.
    /// </summary>
    /// <value>Defaults to 9.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Chronos-Bolt directly outputs multiple quantile
    /// predictions (e.g., 10th, 20th, ..., 90th percentile) in a single forward pass.
    /// </para>
    /// </remarks>
    public int NumQuantiles { get; set; } = 9;

    /// <summary>
    /// Gets or sets the base (peak) learning rate for the built-in Adam optimizer with linear warmup.
    /// </summary>
    /// <value>Defaults to 0.001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The learning rate controls how big a step the optimizer takes on each
    /// update. This is the value the warmup ramps <i>up</i> to (from a tiny <c>1e-6</c>) and the value the
    /// cosine decay then winds <i>down</i> from toward <see cref="EndLearningRate"/>.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of warmup steps (batches) over which the learning rate ramps from a tiny
    /// initial value up to <see cref="LearningRate"/>.
    /// </summary>
    /// <value>Defaults to 10.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Warmup starts training with tiny steps so the first few noisy batches
    /// do not destabilize the model, then reaches the full <see cref="LearningRate"/> after this many steps.
    /// </para>
    /// </remarks>
    public int WarmupSteps { get; set; } = 10;

    /// <summary>
    /// Gets or sets the total number of scheduler steps (batches) over which the learning rate decays from
    /// <see cref="LearningRate"/> down to <see cref="EndLearningRate"/>.
    /// </summary>
    /// <value>Defaults to 1000.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> After this many steps the learning rate clamps to
    /// <see cref="EndLearningRate"/>, so size it to your actual run length (batches-per-epoch multiplied by
    /// epochs). If you train for more batches than this, the extra batches run at
    /// <see cref="EndLearningRate"/> and the model effectively stops learning — raise this value for long runs.
    /// </para>
    /// </remarks>
    public int TotalSteps { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the final learning rate the cosine decay winds down to at <see cref="TotalSteps"/>.
    /// </summary>
    /// <value>Defaults to 0.0.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is where the learning rate ends up once decay completes. Leaving it
    /// at 0.0 fully anneals the model; set a small positive value (e.g. <c>1e-5</c>) if you want it to keep
    /// making tiny updates past <see cref="TotalSteps"/>.
    /// </para>
    /// </remarks>
    public double EndLearningRate { get; set; } = 0.0;
}
