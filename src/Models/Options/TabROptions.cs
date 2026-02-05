namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TabR, a retrieval-augmented model for tabular data.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// TabR combines neural networks with instance-based learning by retrieving similar
/// training examples and using their information to help make predictions. It encodes
/// both the query sample and retrieved neighbors, then aggregates the information
/// using attention.
/// </para>
/// <para>
/// <b>For Beginners:</b> TabR is like a student who looks at similar past problems
/// to help solve a new one.
///
/// How it works:
/// 1. **Encode**: Convert features to a learned representation
/// 2. **Retrieve**: Find the K most similar training samples
/// 3. **Attend**: Use attention to aggregate information from neighbors
/// 4. **Predict**: Make a prediction using both query and neighbor information
///
/// Why this works well for tabular data:
/// - Tabular data often has local structure (similar inputs → similar outputs)
/// - Retrieval adds "memory" without needing to memorize in network weights
/// - Combines strengths of neural networks (feature learning) and k-NN (locality)
/// - Naturally handles rare patterns by finding similar examples
///
/// Example usage:
/// <code>
/// var options = new TabROptions&lt;double&gt;
/// {
///     NumNeighbors = 96,
///     EmbeddingDimension = 256,
///     NumLayers = 4
/// };
/// var model = new TabRClassifier&lt;double&gt;(numFeatures: 10, numClasses: 3, options);
/// </code>
/// </para>
/// <para>
/// Reference: "TabR: Tabular Deep Learning Meets Nearest Neighbors" (2023)
/// </para>
/// </remarks>
public class TabROptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the number of nearest neighbors to retrieve.
    /// </summary>
    /// <value>The number of neighbors, defaulting to 96.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is how many similar training examples to look at:
    /// - Fewer neighbors (32-64): Faster, but might miss relevant examples
    /// - More neighbors (96-128): More context, but slower and might include less relevant samples
    ///
    /// The default of 96 works well for most datasets. Increase for larger datasets.
    /// </para>
    /// </remarks>
    public int NumNeighbors { get; set; } = 96;

    /// <summary>
    /// Gets or sets the embedding dimension for encoding features.
    /// </summary>
    /// <value>The embedding dimension, defaulting to 256.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This controls the size of the learned representation:
    /// - Smaller (128-192): Faster, good for smaller datasets
    /// - Larger (256-384): More capacity, better for complex relationships
    /// </para>
    /// </remarks>
    public int EmbeddingDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the number of MLP layers in the feature encoder.
    /// </summary>
    /// <value>The number of layers, defaulting to 4.</value>
    public int NumLayers { get; set; } = 4;

    /// <summary>
    /// Gets or sets the number of attention heads for neighbor aggregation.
    /// </summary>
    /// <value>The number of attention heads, defaulting to 8.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Attention heads let the model weigh different neighbors
    /// differently from multiple perspectives.
    ///
    /// Must divide EmbeddingDimension evenly.
    /// </para>
    /// </remarks>
    public int NumAttentionHeads { get; set; } = 8;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.1.</value>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether to include target values of retrieved neighbors.
    /// </summary>
    /// <value>True to include neighbor targets; false otherwise. Defaults to true.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When true, the model can directly see what the outputs
    /// were for similar training samples. This helps especially when:
    /// - The relationship between features and target is complex
    /// - There are rare patterns that benefit from direct memory
    ///
    /// During training, leave-one-out ensures we don't use the sample's own target.
    /// </para>
    /// </remarks>
    public bool IncludeNeighborTargets { get; set; } = true;

    /// <summary>
    /// Gets or sets the temperature for retrieval softmax.
    /// </summary>
    /// <value>The temperature, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Controls how "sharp" the attention to neighbors is:
    /// - Lower temperature (&lt;1): Focus more on the most similar neighbors
    /// - Higher temperature (&gt;1): Spread attention more evenly across neighbors
    /// </para>
    /// </remarks>
    public double RetrievalTemperature { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether to normalize embeddings for retrieval.
    /// </summary>
    /// <value>True to normalize embeddings; false otherwise. Defaults to true.</value>
    /// <remarks>
    /// <para>
    /// Normalizing embeddings to unit length makes cosine similarity equivalent
    /// to dot product, which is more stable for retrieval.
    /// </para>
    /// </remarks>
    public bool NormalizeEmbeddings { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of context encoder layers.
    /// </summary>
    /// <value>The number of context layers, defaulting to 2.</value>
    /// <remarks>
    /// <para>
    /// Context encoder processes the retrieved neighbors and aggregates
    /// their information with the query sample.
    /// </para>
    /// </remarks>
    public int NumContextLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets whether to use layer normalization.
    /// </summary>
    /// <value>True to use layer normalization; false otherwise. Defaults to true.</value>
    public bool UseLayerNorm { get; set; } = true;

    /// <summary>
    /// Gets or sets the activation function type.
    /// </summary>
    /// <value>The activation type, defaulting to "ReLU".</value>
    public string ActivationType { get; set; } = "ReLU";

    /// <summary>
    /// Gets or sets whether to use feature-wise linear modulation.
    /// </summary>
    /// <value>True to use FiLM; false otherwise. Defaults to false.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> FiLM (Feature-wise Linear Modulation) allows the context
    /// from neighbors to modulate the feature processing. It can help when the
    /// relationship between features changes based on context.
    /// </para>
    /// </remarks>
    public bool UseFiLM { get; set; } = false;

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
    /// Gets or sets the weight decay coefficient.
    /// </summary>
    /// <value>The weight decay, defaulting to 1e-5.</value>
    public double WeightDecay { get; set; } = 1e-5;

    /// <summary>
    /// Creates a copy of the options.
    /// </summary>
    public TabROptions<T> Clone()
    {
        return new TabROptions<T>
        {
            NumNeighbors = NumNeighbors,
            EmbeddingDimension = EmbeddingDimension,
            NumLayers = NumLayers,
            NumAttentionHeads = NumAttentionHeads,
            DropoutRate = DropoutRate,
            IncludeNeighborTargets = IncludeNeighborTargets,
            RetrievalTemperature = RetrievalTemperature,
            NormalizeEmbeddings = NormalizeEmbeddings,
            NumContextLayers = NumContextLayers,
            UseLayerNorm = UseLayerNorm,
            ActivationType = ActivationType,
            UseFiLM = UseFiLM,
            FeedForwardMultiplier = FeedForwardMultiplier,
            EnableGradientClipping = EnableGradientClipping,
            MaxGradientNorm = MaxGradientNorm,
            WeightDecay = WeightDecay
        };
    }
}
