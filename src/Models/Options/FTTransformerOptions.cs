namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for FT-Transformer, a Feature Tokenizer + Transformer for tabular data.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// FT-Transformer applies the transformer architecture to tabular data by treating each feature
/// as a token. It tokenizes numerical and categorical features into embeddings and processes
/// them with standard transformer encoder layers.
/// </para>
/// <para>
/// <b>For Beginners:</b> FT-Transformer is a way to use the powerful Transformer architecture
/// (the technology behind ChatGPT) on traditional tabular data like spreadsheets.
///
/// Key concepts:
/// - **Feature Tokenization**: Each column in your data becomes a "token" (like words in text)
/// - **[CLS] Token**: A special token added to aggregate information for the final prediction
/// - **Self-Attention**: Allows the model to learn relationships between different features
/// - **No Manual Feature Engineering**: The model learns which features interact automatically
///
/// Advantages:
/// - Captures complex feature interactions through attention
/// - Works well with both numerical and categorical features
/// - Often outperforms gradient boosting on larger datasets
/// - Provides attention weights for interpretability
///
/// Example usage:
/// <code>
/// var options = new FTTransformerOptions&lt;double&gt;
/// {
///     EmbeddingDimension = 192,
///     NumHeads = 8,
///     NumLayers = 3,
///     FeedForwardMultiplier = 4
/// };
/// var model = new FTTransformerClassifier&lt;double&gt;(numFeatures: 10, numClasses: 3, options);
/// </code>
/// </para>
/// <para>
/// Reference: "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., NeurIPS 2021)
/// </para>
/// </remarks>
public class FTTransformerOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the embedding dimension for feature tokens.
    /// </summary>
    /// <value>The embedding dimension, defaulting to 192.</value>
    /// <remarks>
    /// <para>
    /// This determines the size of the embedding vector for each feature token.
    /// Must be divisible by NumHeads.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The embedding dimension is how much information each feature
    /// can carry after being converted to a vector.
    ///
    /// Common values: 64, 128, 192, 256, 384
    /// - Small datasets (&lt;10K rows): 64-128
    /// - Medium datasets (10K-100K): 128-256
    /// - Large datasets (&gt;100K): 192-384
    ///
    /// Must be divisible by the number of attention heads.
    /// </para>
    /// </remarks>
    public int EmbeddingDimension { get; set; } = 192;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>The number of attention heads, defaulting to 8.</value>
    /// <remarks>
    /// <para>
    /// Multi-head attention allows the model to jointly attend to information
    /// from different representation subspaces at different positions.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Attention heads are like different "experts" that look
    /// at the relationships between features from different perspectives.
    ///
    /// - More heads: Can capture more diverse relationships (but EmbeddingDimension must be divisible by NumHeads)
    /// - Common values: 4, 6, 8, 12
    ///
    /// For example, one head might focus on correlations between age and income,
    /// while another focuses on geographic features.
    /// </para>
    /// </remarks>
    public int NumHeads { get; set; } = 8;

    /// <summary>
    /// Gets or sets the number of transformer encoder layers.
    /// </summary>
    /// <value>The number of layers, defaulting to 3.</value>
    /// <remarks>
    /// <para>
    /// More layers allow the model to learn more complex feature interactions,
    /// but increase computation and risk of overfitting.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Each layer processes the features again, allowing
    /// the model to build up more complex understanding.
    ///
    /// - 2-3 layers: Good starting point for most datasets
    /// - 4-6 layers: For very complex feature interactions
    /// - More layers: Risk overfitting on smaller datasets
    ///
    /// Start with 3 layers and increase if underfitting.
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 3;

    /// <summary>
    /// Gets or sets the feed-forward network dimension multiplier.
    /// </summary>
    /// <value>The multiplier, defaulting to 4.</value>
    /// <remarks>
    /// <para>
    /// The feed-forward network dimension is EmbeddingDimension * FeedForwardMultiplier.
    /// A value of 4 is standard in transformer architectures.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This controls the "width" of the internal processing
    /// in each transformer layer.
    ///
    /// FeedForwardDim = EmbeddingDimension × FeedForwardMultiplier
    ///
    /// - Standard value: 4 (from original Transformer paper)
    /// - Lower (2-3): Faster but less capacity
    /// - Higher (6-8): More capacity but slower
    /// </para>
    /// </remarks>
    public int FeedForwardMultiplier { get; set; } = 4;

    /// <summary>
    /// Gets the feed-forward network dimension.
    /// </summary>
    /// <value>EmbeddingDimension times FeedForwardMultiplier.</value>
    public int FeedForwardDimension => EmbeddingDimension * FeedForwardMultiplier;

    /// <summary>
    /// Gets or sets the dropout rate for attention and feed-forward layers.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dropout randomly turns off parts of the network during
    /// training to prevent overfitting.
    ///
    /// - 0.0: No dropout (may overfit)
    /// - 0.1: Light dropout (good default)
    /// - 0.2-0.3: Heavier dropout for small datasets
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the dropout rate specifically for attention weights.
    /// </summary>
    /// <value>The attention dropout rate, defaulting to 0.0.</value>
    /// <remarks>
    /// <para>
    /// Separate dropout applied to attention weights. Set to 0 in the original
    /// FT-Transformer paper.
    /// </para>
    /// </remarks>
    public double AttentionDropoutRate { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the dropout rate applied to the residual connections.
    /// </summary>
    /// <value>The residual dropout rate, defaulting to 0.0.</value>
    public double ResidualDropoutRate { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets whether to use layer normalization before attention (Pre-LN) or after (Post-LN).
    /// </summary>
    /// <value>True to use Pre-LN; false for Post-LN. Defaults to true.</value>
    /// <remarks>
    /// <para>
    /// Pre-LN (layer norm before attention) tends to be more stable during training.
    /// This is the recommended setting and was used in the original FT-Transformer.
    /// </para>
    /// </remarks>
    public bool UsePreLayerNorm { get; set; } = true;

    /// <summary>
    /// Gets or sets the epsilon value for layer normalization numerical stability.
    /// </summary>
    /// <value>The epsilon value, defaulting to 1e-6.</value>
    public double LayerNormEpsilon { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the initialization scale for embeddings.
    /// </summary>
    /// <value>The embedding initialization scale, defaulting to 0.01.</value>
    public double EmbeddingInitScale { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets whether to use a bias term in the numerical feature embedding.
    /// </summary>
    /// <value>True to use bias; false otherwise. Defaults to true.</value>
    public bool UseNumericalBias { get; set; } = true;

    /// <summary>
    /// Gets or sets the categorical feature cardinalities (number of unique values per categorical feature).
    /// </summary>
    /// <value>An array of cardinalities for each categorical feature, or null if no categorical features.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If you have categorical features (like "red", "green", "blue"),
    /// specify how many unique values each one has.
    ///
    /// Example: If you have two categorical features:
    /// - Color with 3 values: red, green, blue
    /// - Size with 4 values: XS, S, M, L
    /// Then: CategoricalCardinalities = [3, 4]
    ///
    /// Leave null or empty if all features are numerical.
    /// </para>
    /// </remarks>
    public int[]? CategoricalCardinalities { get; set; }

    /// <summary>
    /// Gets or sets whether to enable gradient clipping.
    /// </summary>
    /// <value>True to enable gradient clipping; false otherwise. Defaults to true.</value>
    public bool EnableGradientClipping { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum gradient norm for clipping.
    /// </summary>
    /// <value>The maximum gradient norm, defaulting to 1.0.</value>
    public double MaxGradientNorm { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the weight decay (L2 regularization) coefficient.
    /// </summary>
    /// <value>The weight decay, defaulting to 1e-5.</value>
    public double WeightDecay { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets whether to use ReGLU activation in the feed-forward network.
    /// </summary>
    /// <value>True to use ReGLU; false to use standard GELU. Defaults to true.</value>
    /// <remarks>
    /// <para>
    /// ReGLU (Rectified Gated Linear Unit) often improves transformer performance
    /// on tabular data. It multiplies the output of a ReLU with a linear gating mechanism.
    /// </para>
    /// </remarks>
    public bool UseReGLU { get; set; } = true;

    /// <summary>
    /// Creates a copy of the options.
    /// </summary>
    /// <returns>A new FTTransformerOptions instance with the same values.</returns>
    public FTTransformerOptions<T> Clone()
    {
        return new FTTransformerOptions<T>
        {
            EmbeddingDimension = EmbeddingDimension,
            NumHeads = NumHeads,
            NumLayers = NumLayers,
            FeedForwardMultiplier = FeedForwardMultiplier,
            DropoutRate = DropoutRate,
            AttentionDropoutRate = AttentionDropoutRate,
            ResidualDropoutRate = ResidualDropoutRate,
            UsePreLayerNorm = UsePreLayerNorm,
            LayerNormEpsilon = LayerNormEpsilon,
            EmbeddingInitScale = EmbeddingInitScale,
            UseNumericalBias = UseNumericalBias,
            CategoricalCardinalities = CategoricalCardinalities?.ToArray(),
            EnableGradientClipping = EnableGradientClipping,
            MaxGradientNorm = MaxGradientNorm,
            WeightDecay = WeightDecay,
            UseReGLU = UseReGLU
        };
    }
}
