using AiDotNet.ActivationFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for SAINT (Self-Attention and Intersample Attention Transformer).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// SAINT combines two types of attention:
/// 1. Self-attention over features (column attention, like FT-Transformer)
/// 2. Inter-sample attention (row attention, comparing samples within a batch)
/// </para>
/// <para>
/// <b>For Beginners:</b> SAINT is powerful because it learns two things:
///
/// - **Column attention**: Which features are related to each other?
/// - **Row attention**: Which training samples are similar?
///
/// This dual attention makes SAINT especially good when similar samples
/// share patterns that are useful for prediction.
///
/// Example:
/// <code>
/// var options = new SAINTOptions&lt;double&gt;
/// {
///     EmbeddingDimension = 32,
///     NumLayers = 6,
///     NumHeads = 8,
///     UseIntersampleAttention = true
/// };
/// var model = new SAINTClassifier&lt;double&gt;(numFeatures: 10, numClasses: 3, options);
/// </code>
/// </para>
/// <para>
/// Reference: "SAINT: Improved Neural Networks for Tabular Data via Row Attention
/// and Contrastive Pre-Training" (2021)
/// </para>
/// </remarks>
public class SAINTOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the embedding dimension for features.
    /// </summary>
    /// <value>The embedding dimension, defaulting to 32.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each feature is converted to a vector of this size.
    /// Larger values can capture more complex patterns but cost more to train.
    /// </para>
    /// </remarks>
    public int EmbeddingDimension { get; set; } = 32;

    /// <summary>
    /// Hidden dimension size for feed-forward networks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Higher values let the model learn more complex
    /// patterns but require more compute.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 128;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multiple heads allow the model to focus on
    /// different relationships at the same time. Must evenly divide EmbeddingDimension.
    /// </para>
    /// </remarks>
    public int NumHeads { get; set; } = 8;

    /// <summary>
    /// Number of transformer layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> More layers mean a deeper model that can capture
    /// richer interactions, but it trains more slowly.
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 6;

    /// <summary>
    /// Gets or sets the feed-forward dimension multiplier.
    /// </summary>
    /// <value>The multiplier, defaulting to 4.</value>
    public int FeedForwardMultiplier { get; set; } = 4;

    /// <summary>
    /// Gets the feed-forward network dimension.
    /// </summary>
    public int FeedForwardDimension => EmbeddingDimension * FeedForwardMultiplier;

    /// <summary>
    /// Dropout rate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dropout helps prevent overfitting by randomly
    /// turning off parts of the network during training.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Batch size (sequence length for inter-sample attention).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SAINT compares rows in a batch to each other.
    /// This value controls how many rows it can compare at once.
    /// </para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets whether to use inter-sample (row) attention.
    /// </summary>
    /// <value>True to use inter-sample attention; false otherwise. Defaults to true.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Inter-sample attention compares each sample in a batch
    /// to other samples. This helps when similar samples share useful patterns.
    /// Set to false for faster training if your samples are independent.
    /// </para>
    /// </remarks>
    public bool UseIntersampleAttention { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use layer normalization.
    /// </summary>
    /// <value>True to use layer normalization; false otherwise. Defaults to true.</value>
    public bool UseLayerNorm { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use pre-norm (norm before attention) or post-norm.
    /// </summary>
    /// <value>True for pre-norm; false for post-norm. Defaults to true.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pre-norm places layer normalization before attention,
    /// which often leads to more stable training with deeper models.
    /// </para>
    /// </remarks>
    public bool UsePreNorm { get; set; } = true;

    /// <summary>
    /// Gets or sets the hidden dimensions for the MLP head.
    /// </summary>
    /// <value>Array of hidden dimensions, defaulting to [128, 64].</value>
    public int[] MLPHiddenDimensions { get; set; } = [128, 64];

    /// <summary>
    /// Gets or sets the cardinalities of categorical features.
    /// </summary>
    /// <value>Array of cardinalities, or null if no categorical features.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If you have categorical features, specify how many
    /// unique values each one can have. Set to null for purely numerical data.
    /// </para>
    /// </remarks>
    public int[]? CategoricalCardinalities { get; set; }

    /// <summary>
    /// Gets or sets the initialization scale for embeddings.
    /// </summary>
    /// <value>The initialization scale, defaulting to 0.02.</value>
    public double EmbeddingInitScale { get; set; } = 0.02;

    /// <summary>
    /// Gets or sets the attention dropout rate (separate from general dropout).
    /// </summary>
    /// <value>The attention dropout rate, defaulting to 0.0.</value>
    public double AttentionDropoutRate { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the hidden layer activation function.
    /// </summary>
    /// <value>The activation function, defaulting to GELU.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> GELU is commonly used in transformer architectures
    /// and provides smooth gradients for better training.
    /// </para>
    /// </remarks>
    public IActivationFunction<T>? HiddenActivation { get; set; } = new GELUActivation<T>();

    /// <summary>
    /// Gets or sets the hidden layer vector activation function (alternative to scalar activation).
    /// </summary>
    /// <value>The vector activation function, or null to use scalar activation.</value>
    public IVectorActivationFunction<T>? HiddenVectorActivation { get; set; }
}
