namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TabLLM-Gen, an LLM-style tabular data generator that uses
/// schema-aware tokenization and autoregressive transformers with column prompts.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// TabLLM-Gen adapts large language model techniques for tabular data:
/// - <b>Schema-aware tokenization</b>: Column names and types become special tokens
/// - <b>Column prompts</b>: Each column is prompted with its name/type for context
/// - <b>Autoregressive generation</b>: Generates one column at a time, left to right
/// - <b>Few-shot learning</b>: Can condition on example rows for in-context learning
/// </para>
/// <para>
/// <b>For Beginners:</b> TabLLM-Gen treats table generation like text generation:
///
/// 1. A row becomes a "sentence": "[Age: 35] [Income: 75000] [Education: MS]"
/// 2. The model learns to predict each column value given previous columns
/// 3. Column names and types serve as "instructions" to guide generation
///
/// This approach naturally captures column dependencies and produces coherent rows.
///
/// Example:
/// <code>
/// var options = new TabLLMGenOptions&lt;double&gt;
/// {
///     NumLayers = 4,
///     NumHeads = 4,
///     EmbeddingDimension = 128,
///     Epochs = 100
/// };
/// var gen = new TabLLMGenGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// <para>
/// Reference: "LLM-based Tabular Data Generation" (2024)
/// </para>
/// </remarks>
public class TabLLMGenOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the number of transformer layers.
    /// </summary>
    /// <value>Number of layers, defaulting to 4.</value>
    public int NumLayers { get; set; } = 4;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>Number of heads, defaulting to 4.</value>
    public int NumHeads { get; set; } = 4;

    /// <summary>
    /// Gets or sets the embedding dimension.
    /// </summary>
    /// <value>Embedding dimension, defaulting to 128.</value>
    public int EmbeddingDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the feed-forward dimension.
    /// </summary>
    /// <value>FFN dimension, defaulting to 256.</value>
    public int FeedForwardDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    /// <value>Dropout rate, defaulting to 0.1.</value>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the number of bins for discretizing continuous values.
    /// </summary>
    /// <value>Number of bins, defaulting to 100.</value>
    public int NumBins { get; set; } = 100;

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
    /// Gets or sets the sampling temperature for generation.
    /// </summary>
    /// <value>Temperature, defaulting to 0.7.</value>
    public double Temperature { get; set; } = 0.7;

    /// <summary>
    /// Gets or sets the number of extra schema tokens per column.
    /// </summary>
    /// <value>Number of schema tokens, defaulting to 2 (name + type).</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each column gets special tokens for its name and data type.
    /// These act like "instructions" telling the model what kind of value to generate.
    /// </para>
    /// </remarks>
    public int SchemaTokensPerColumn { get; set; } = 2;
}
