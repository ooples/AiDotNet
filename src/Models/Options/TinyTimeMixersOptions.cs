using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Tiny Time Mixers (TTM) foundation model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// Tiny Time Mixers (TTM) is IBM Research's compact foundation model for time series forecasting
/// that uses an MLP-Mixer architecture instead of attention-based transformers. Despite having
/// only 1-5 million parameters, TTM outperforms models 20-40x its size on standard benchmarks.
/// </para>
/// <para><b>For Beginners:</b> TTM is designed to be small, fast, and surprisingly powerful:
///
/// <b>Key Innovation — MLP-Mixer Architecture:</b>
/// Instead of expensive attention mechanisms (like in GPT or Chronos), TTM uses simple
/// MLP (Multi-Layer Perceptron) blocks that mix information across two dimensions:
/// 1. <b>Temporal mixing</b>: Exchanges information across time steps within each patch
/// 2. <b>Channel mixing</b>: Exchanges information across different features/variables
///
/// <b>Why This Works:</b>
/// - MLP-Mixers are 10-100x faster than attention mechanisms
/// - For time series, you don't always need the flexibility of attention
/// - The model can be trained on much less data and compute
/// - Perfect for edge deployment and real-time applications
///
/// <b>Performance Highlights:</b>
/// - 1-5M parameters (vs 200-700M for Chronos/MOIRAI)
/// - Outperforms or matches much larger foundation models
/// - Can forecast on CPU in real-time
/// - Trains in minutes instead of hours/days
///
/// <b>Adaptive Patching:</b>
/// TTM can automatically adjust its patch size based on the input data characteristics,
/// allowing it to handle different frequencies without manual tuning.
/// </para>
/// <para>
/// <b>Reference:</b> Ekambaram et al., "Tiny Time Mixers (TTMs): Fast Pre-trained Models
/// for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series", NeurIPS 2024.
/// https://arxiv.org/abs/2401.03955
/// </para>
/// </remarks>
public class TinyTimeMixersOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public TinyTimeMixersOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public TinyTimeMixersOptions(TinyTimeMixersOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        PatchLength = other.PatchLength;
        HiddenDimension = other.HiddenDimension;
        NumMixerLayers = other.NumMixerLayers;
        ExpansionFactor = other.ExpansionFactor;
        DropoutRate = other.DropoutRate;
        ModelSize = other.ModelSize;
        UseAdaptivePatching = other.UseAdaptivePatching;
        NumFeatures = other.NumFeatures;
    }

    /// <summary>
    /// Gets or sets the context length (input sequence length).
    /// </summary>
    /// <value>Defaults to 512.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps the model can look at.
    /// </para>
    /// </remarks>
    public int ContextLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets the forecast horizon (prediction length).
    /// </summary>
    /// <value>Defaults to 96.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far into the future to predict.
    /// </para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 96;

    /// <summary>
    /// Gets or sets the patch length for input tokenization.
    /// </summary>
    /// <value>Defaults to 64.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Groups consecutive time steps into patches for the mixer.
    /// With context=512 and patch=64, the model processes 8 patches.
    /// </para>
    /// </remarks>
    public int PatchLength { get; set; } = 64;

    /// <summary>
    /// Gets or sets the hidden dimension of the mixer layers.
    /// </summary>
    /// <value>Defaults to 64.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The internal representation size. TTM intentionally
    /// uses small hidden dimensions (64-128) to stay compact yet effective.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of mixer layers.
    /// </summary>
    /// <value>Defaults to 4.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many temporal-channel mixing blocks are stacked.
    /// Each block contains a temporal mixing MLP and a channel mixing MLP.
    /// </para>
    /// </remarks>
    public int NumMixerLayers { get; set; } = 4;

    /// <summary>
    /// Gets or sets the expansion factor for the mixer feed-forward networks.
    /// </summary>
    /// <value>Defaults to 4.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each MLP in the mixer block expands the hidden dimension
    /// by this factor and then projects back. A factor of 4 means a 64-dim hidden space
    /// expands to 256 inside the MLP.
    /// </para>
    /// </remarks>
    public int ExpansionFactor { get; set; } = 4;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>Defaults to 0.1.</value>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the model size variant.
    /// </summary>
    /// <value>Defaults to <see cref="FoundationModelSize.Tiny"/>.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> TTM is designed to be tiny — even the largest variant
    /// has only ~5M parameters. The size controls the hidden dimension and layer count.
    /// </para>
    /// </remarks>
    public FoundationModelSize ModelSize { get; set; } = FoundationModelSize.Tiny;

    /// <summary>
    /// Gets or sets whether to use adaptive patching.
    /// </summary>
    /// <value>Defaults to null (auto-detect based on data characteristics).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When enabled, TTM automatically adjusts the patch size
    /// based on the input data frequency. When null (default), the model decides based
    /// on the data characteristics. Set to true to force adaptive patching or false to
    /// use the fixed <see cref="PatchLength"/>.
    /// </para>
    /// </remarks>
    public bool? UseAdaptivePatching { get; set; }

    /// <summary>
    /// Gets or sets the number of input features (channels) for multivariate forecasting.
    /// </summary>
    /// <value>Defaults to 1 (univariate).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Set this to the number of variables in your time series.
    /// For example, if you have price, volume, and sentiment, set this to 3.
    /// TTM's channel mixing MLP learns relationships between features.
    /// </para>
    /// </remarks>
    public int NumFeatures { get; set; } = 1;
}
