using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for MOIRAI (Salesforce's Universal Time Series Foundation Model).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// MOIRAI (Masked EncOder-based UnIveRsAl TIme Series Foundation Model) is Salesforce's
/// foundation model for universal time series forecasting. It uses masked encoder-based
/// training with multiple patches to handle any-to-any forecasting across different
/// frequencies and domains without fine-tuning.
/// </para>
/// <para><b>For Beginners:</b> MOIRAI is designed to be a truly universal time series model:
///
/// <b>Key Innovations:</b>
/// - Multi-patch embeddings: Uses multiple patch sizes simultaneously
/// - Any-variate forecasting: Handles univariate and multivariate series
/// - Distribution mixture outputs: Combines multiple distributions for flexibility
/// - Unified masked encoder: Single architecture for all forecasting tasks
///
/// <b>Architecture:</b>
/// 1. Multi-scale patching: Creates tokens at different time scales
/// 2. Masked encoder: Transformer encoder with masking
/// 3. Distribution head: Outputs mixture of distributions
/// 4. Any-to-any: Can predict any horizon from any context
///
/// <b>Model Sizes:</b>
/// - Small: Lightweight (parameters: ~14M)
/// - Base: Balanced (parameters: ~91M)
/// - Large: Maximum capacity (parameters: ~311M)
/// </para>
/// <para>
/// <b>Reference:</b> Woo et al., "Unified Training of Universal Time Series Forecasting Transformers", 2024.
/// https://arxiv.org/abs/2402.02592
/// </para>
/// </remarks>
public class MOIRAIOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MOIRAIOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default MOIRAI configuration for
    /// universal time series forecasting with multi-scale patching.
    /// </para>
    /// </remarks>
    public MOIRAIOptions()
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
    public MOIRAIOptions(MOIRAIOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        PatchSizes = other.PatchSizes;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        IntermediateSize = other.IntermediateSize;
        NumMixtures = other.NumMixtures;
        DropoutRate = other.DropoutRate;
        MaskRatio = other.MaskRatio;
        ModelSize = other.ModelSize;
    }

    /// <summary>
    /// Gets or sets the context length (input sequence length).
    /// </summary>
    /// <value>The context length, defaulting to 512.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps the model can look at.
    /// MOIRAI uses efficient multi-scale patching for long contexts.
    /// </para>
    /// </remarks>
    public int ContextLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets the forecast horizon (prediction length).
    /// </summary>
    /// <value>The forecast horizon, defaulting to 96.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far into the future to predict.
    /// MOIRAI can handle variable horizons without retraining.
    /// </para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 96;

    /// <summary>
    /// Gets or sets the patch sizes for multi-scale patching.
    /// </summary>
    /// <value>Array of patch sizes, defaulting to [8, 16, 32, 64].</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> MOIRAI uses multiple patch sizes simultaneously
    /// to capture patterns at different time scales. Smaller patches capture
    /// fine-grained patterns, larger patches capture trends.
    /// </para>
    /// </remarks>
    public int[] PatchSizes { get; set; } = new[] { 8, 16, 32, 64 };

    /// <summary>
    /// Gets or sets the hidden dimension of the transformer.
    /// </summary>
    /// <value>The hidden dimension, defaulting to 768.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The internal representation size.
    /// Larger values increase capacity but require more memory.
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
    /// different patterns simultaneously across the multi-scale patches.
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
    /// Gets or sets the number of mixture components for distribution output.
    /// </summary>
    /// <value>The number of mixtures, defaulting to 10.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> MOIRAI outputs a mixture of distributions
    /// for probabilistic forecasting. More mixtures can capture more complex
    /// distributions but increase computation.
    /// </para>
    /// </remarks>
    public int NumMixtures { get; set; } = 10;

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
    /// Gets or sets the mask ratio for training.
    /// </summary>
    /// <value>The mask ratio, defaulting to 0.3.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> During training, a portion of patches are masked
    /// and the model learns to predict them. This is similar to BERT-style
    /// masked language modeling but for time series.
    /// </para>
    /// </remarks>
    public double MaskRatio { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets the model size variant.
    /// </summary>
    /// <value>The model size, defaulting to "base".</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> MOIRAI comes in different sizes:
    /// - "small": Lightweight (~14M params)
    /// - "base": Balanced (~91M params)
    /// - "large": Maximum capacity (~311M params)
    /// </para>
    /// </remarks>
    public string ModelSize { get; set; } = "base";
}
