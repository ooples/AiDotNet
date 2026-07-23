using AiDotNet.ActivationFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TabDPT (Tabular Data Pre-Training).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// TabDPT is a foundation model approach for tabular data that uses pre-training
/// on diverse datasets to learn transferable representations.
/// </para>
/// <para>
/// <b>For Beginners:</b> TabDPT brings the power of foundation models to tabular data:
///
/// - **Pre-training**: The model learns patterns from many tabular datasets
/// - **Transfer learning**: These patterns help on new, unseen datasets
/// - **In-context learning**: Can adapt to new tasks without fine-tuning
///
/// Example:
/// <code>
/// var options = new TabDPTOptions&lt;double&gt;
/// {
///     EmbeddingDimension = 128,
///     NumLayers = 6,
///     NumHeads = 4
/// };
/// var model = new TabDPTClassifier&lt;double&gt;(numFeatures: 20, numClasses: 3, options);
/// </code>
/// </para>
/// <para>
/// Reference: "TabDPT: Scaling Tabular Foundation Models" (2025)
/// </para>
/// </remarks>
public class TabDPTOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the embedding dimension.
    /// </summary>
    /// <value>The embedding dimension, defaulting to 128.</value>
    public int EmbeddingDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the number of transformer layers.
    /// </summary>
    /// <value>The number of layers, defaulting to 6.</value>
    public int NumLayers { get; set; } = 6;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>The number of heads, defaulting to 4.</value>
    public int NumHeads { get; set; } = 4;

    /// <summary>
    /// Gets or sets the feed-forward dimension multiplier.
    /// </summary>
    /// <value>The multiplier, defaulting to 4.</value>
    public int FeedForwardMultiplier { get; set; } = 4;

    /// <summary>
    /// Gets the feed-forward dimension.
    /// </summary>
    public int FeedForwardDimension => EmbeddingDimension * FeedForwardMultiplier;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.1.</value>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the maximum number of features supported.
    /// </summary>
    /// <value>The maximum features, defaulting to 100.</value>
    public int MaxFeatures { get; set; } = 100;

    /// <summary>
    /// Gets or sets the context length (number of examples for in-context learning).
    /// </summary>
    /// <value>The context length, defaulting to 32.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In-context learning uses training examples as context.
    /// The model sees these examples and learns to make predictions based on them.
    /// </para>
    /// </remarks>
    public int ContextLength { get; set; } = 32;

    /// <summary>
    /// Gets or sets whether to use layer normalization.
    /// </summary>
    /// <value>True to use layer normalization; false otherwise. Defaults to true.</value>
    public bool UseLayerNorm { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use pre-norm (norm before attention).
    /// </summary>
    /// <value>True for pre-norm; false for post-norm. Defaults to true.</value>
    public bool UsePreNorm { get; set; } = true;

    /// <summary>
    /// Gets or sets the hidden dimensions for the output head.
    /// </summary>
    /// <value>Array of hidden dimensions, defaulting to [64].</value>
    public int[] OutputHeadDimensions { get; set; } = [64];

    /// <summary>
    /// Gets or sets the initialization scale.
    /// </summary>
    /// <value>The initialization scale, defaulting to 0.02.</value>
    public double InitScale { get; set; } = 0.02;

    /// <summary>
    /// Gets or sets whether to use feature-wise attention.
    /// </summary>
    /// <value>True to use feature attention; false otherwise. Defaults to true.</value>
    public bool UseFeatureAttention { get; set; } = true;

    /// <summary>
    /// Gets or sets the cardinalities of categorical features.
    /// </summary>
    public int[]? CategoricalCardinalities { get; set; }

    /// <summary>
    /// Gets or sets the hidden layer activation function.
    /// </summary>
    /// <value>The activation function, defaulting to GELU.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> GELU (Gaussian Error Linear Unit) is commonly used
    /// in foundation models and transformers for smooth gradient flow.
    /// </para>
    /// </remarks>
    public IActivationFunction<T>? HiddenActivation { get; set; } = new GELUActivation<T>();

    /// <summary>
    /// Gets or sets the hidden layer vector activation function (alternative to scalar activation).
    /// </summary>
    /// <value>The vector activation function, or null to use scalar activation.</value>
    public IVectorActivationFunction<T>? HiddenVectorActivation { get; set; }

    /// <summary>
    /// Gets or sets the input projection activation function.
    /// </summary>
    /// <value>The activation function, defaulting to ReLU.</value>
    public IActivationFunction<T>? InputActivation { get; set; } = new ReLUActivation<T>();
}
