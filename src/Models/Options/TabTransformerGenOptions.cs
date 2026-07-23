namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TabTransformer-Gen, a generative model that uses contextual embeddings
/// from multi-head self-attention over categorical columns to generate realistic tabular data.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// TabTransformer-Gen adapts the TabTransformer architecture for data generation:
/// - Categorical columns get learned embeddings that attend to each other
/// - Continuous columns pass through with optional normalization
/// - A masked prediction objective trains the model to reconstruct missing columns
/// - Generation uses iterative masked prediction (like masked language modeling)
/// </para>
/// <para>
/// <b>For Beginners:</b> TabTransformer-Gen works like a fill-in-the-blank game:
///
/// 1. Each categorical column gets its own "word" (embedding)
/// 2. These words "talk to each other" through attention (e.g., "Occupation"
///    pays attention to "Education" because they're related)
/// 3. During training, we randomly mask some columns and ask the model to guess them
/// 4. During generation, we start with everything masked and iteratively fill in columns
///
/// Example:
/// <code>
/// var options = new TabTransformerGenOptions&lt;double&gt;
/// {
///     NumLayers = 6,
///     NumHeads = 8,
///     EmbeddingDimension = 32,
///     Epochs = 100
/// };
/// var gen = new TabTransformerGenGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// <para>
/// Reference: "TabTransformer: Tabular Data Modeling Using Contextual Embeddings"
/// (Huang et al., 2020) — adapted for generation with masked prediction
/// </para>
/// </remarks>
public class TabTransformerGenOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the number of transformer layers.
    /// </summary>
    /// <value>Number of layers, defaulting to 6.</value>
    public int NumLayers { get; set; } = 6;

    /// <summary>
    /// Gets or sets the number of attention heads per layer.
    /// </summary>
    /// <value>Number of heads, defaulting to 8.</value>
    public int NumHeads { get; set; } = 8;

    /// <summary>
    /// Gets or sets the embedding dimension for each column.
    /// </summary>
    /// <value>Embedding dimension, defaulting to 32.</value>
    public int EmbeddingDimension { get; set; } = 32;

    /// <summary>
    /// Gets or sets the feed-forward dimension in each transformer block.
    /// </summary>
    /// <value>FFN dimension, defaulting to 128.</value>
    public int FeedForwardDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    /// <value>Dropout rate, defaulting to 0.1.</value>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the fraction of columns to mask during training.
    /// </summary>
    /// <value>Mask ratio, defaulting to 0.3.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> During training, this fraction of columns are randomly hidden
    /// and the model learns to predict them. 0.3 means 30% of columns are masked.
    /// Higher values make training harder but can improve generation quality.
    /// </para>
    /// </remarks>
    public double MaskRatio { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets the number of iterative refinement steps during generation.
    /// </summary>
    /// <value>Number of generation steps, defaulting to 10.</value>
    public int GenerationSteps { get; set; } = 10;

    /// <summary>
    /// Gets or sets the training batch size.
    /// </summary>
    /// <value>The batch size, defaulting to 64.</value>
    public int BatchSize { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <value>Number of epochs, defaulting to 100.</value>
    public int Epochs { get; set; } = 100;

    /// <summary>
    /// Gets or sets the learning rate.
    /// </summary>
    /// <value>The learning rate, defaulting to 1e-3.</value>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>
    /// Gets or sets the number of VGM modes for continuous column transformation.
    /// </summary>
    /// <value>Number of modes, defaulting to 10.</value>
    public int VGMModes { get; set; } = 10;
}
