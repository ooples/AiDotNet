namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TabTransformer.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// TabTransformer applies transformer self-attention only to categorical features,
/// while keeping numerical features in their original form. The transformed categorical
/// embeddings are concatenated with numerical features for prediction.
/// </para>
/// <para>
/// <b>For Beginners:</b> TabTransformer treats categorical features specially:
///
/// Key ideas:
/// 1. **Categorical Embeddings**: Each category value gets a learned vector
/// 2. **Transformer for Categories**: Self-attention captures relationships between categories
/// 3. **Numerical Features Unchanged**: Numbers pass through directly
/// 4. **Concatenation**: Transformed categories + numbers → MLP → prediction
///
/// Why this works:
/// - Categorical features often have complex interactions (e.g., city + job type)
/// - Transformer attention can learn these interactions automatically
/// - Numerical features don't need the same treatment
///
/// Example:
/// <code>
/// var options = new TabTransformerOptions&lt;double&gt;
/// {
///     EmbeddingDimension = 32,
///     NumHeads = 8,
///     NumLayers = 6,
///     CategoricalCardinalities = new[] { 5, 10, 20 }  // 3 categorical features
/// };
/// var model = new TabTransformerClassifier&lt;double&gt;(numNumerical: 5, numClasses: 3, options);
/// </code>
/// </para>
/// <para>
/// Reference: "TabTransformer: Tabular Data Modeling Using Contextual Embeddings" (2020)
/// </para>
/// </remarks>
public class TabTransformerOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the cardinalities of categorical features.
    /// </summary>
    /// <value>Array of cardinalities, or null if no categorical features.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cardinality is the number of unique values a category can have.
    /// For example:
    /// - Gender (Male, Female, Other) has cardinality 3
    /// - US State has cardinality 50
    ///
    /// Set this to null if you only have numerical features.
    /// </para>
    /// </remarks>
    public int[]? CategoricalCardinalities { get; set; }

    /// <summary>
    /// Gets or sets the embedding dimension for categorical features.
    /// </summary>
    /// <value>The embedding dimension, defaulting to 32.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each category value is represented as a vector of this size:
    /// - Smaller (16-32): Faster, good for simpler categorical relationships
    /// - Larger (32-64): More capacity for complex relationships
    /// </para>
    /// </remarks>
    public int EmbeddingDimension { get; set; } = 32;

    /// <summary>
    /// Hidden dimension size for transformer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the width of the internal representations.
    /// Larger values can capture more complex patterns but cost more to train.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 128;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multiple heads let the model focus on different
    /// relationships at the same time. Must evenly divide EmbeddingDimension.
    /// </para>
    /// </remarks>
    public int NumHeads { get; set; } = 8;

    /// <summary>
    /// Number of transformer layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> More layers means a deeper model that can learn
    /// richer patterns, but it is slower to train.
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 6;

    /// <summary>
    /// Dropout rate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dropout helps prevent overfitting by randomly
    /// turning off some neurons during training.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

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
    /// Gets or sets whether to use layer normalization.
    /// </summary>
    /// <value>True to use layer normalization; false otherwise. Defaults to true.</value>
    public bool UseLayerNorm { get; set; } = true;

    /// <summary>
    /// Gets or sets the hidden dimensions for the MLP head.
    /// </summary>
    /// <value>Array of hidden dimensions, defaulting to [128, 64].</value>
    public int[] MLPHiddenDimensions { get; set; } = [128, 64];

    /// <summary>
    /// Gets or sets whether to use column embedding (add learnable column-specific vectors).
    /// </summary>
    /// <value>True to use column embedding; false otherwise. Defaults to true.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Column embedding adds position information:
    /// - The model knows which column each embedding came from
    /// - Helps distinguish between categories from different features
    /// </para>
    /// </remarks>
    public bool UseColumnEmbedding { get; set; } = true;

    /// <summary>
    /// Gets or sets the initialization scale for embeddings.
    /// </summary>
    /// <value>The initialization scale, defaulting to 0.01.</value>
    public double EmbeddingInitScale { get; set; } = 0.01;

    // Backing field for NumCategoricalFeatures
    private int? _numCategoricalFeatures;

    /// <summary>
    /// Gets or sets the number of categorical features.
    /// </summary>
    /// <value>
    /// Defaults to the length of <see cref="CategoricalCardinalities"/> if set, otherwise 0
    /// (meaning no categorical features).
    /// </value>
    /// <remarks>
    /// <para>
    /// When CategoricalCardinalities is set, this property returns its length.
    /// Can also be set directly to override the inferred value.
    /// Setting to 0 resets to inference mode (infer from CategoricalCardinalities).
    /// A positive value permanently overrides inference.
    /// </para>
    /// </remarks>
    public int NumCategoricalFeatures
    {
        get => _numCategoricalFeatures ?? CategoricalCardinalities?.Length ?? 0;
        set => _numCategoricalFeatures = value == 0 ? null : value;
    }
}
