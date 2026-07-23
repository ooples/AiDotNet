namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for REaLTabFormer, a GPT-2 style autoregressive transformer
/// for generating realistic tabular data by treating columns as a sequence of tokens.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// REaLTabFormer tokenizes each column value and generates table rows left-to-right,
/// one column at a time, using a causal (autoregressive) transformer architecture.
/// </para>
/// <para>
/// <b>For Beginners:</b> REaLTabFormer treats each row like a sentence and each column
/// value like a word. It generates data by predicting one column at a time:
///
/// 1. Start with a special [START] token
/// 2. Predict the first column's value: "Age = 35"
/// 3. Given Age=35, predict next: "Income = 75000"
/// 4. Given Age=35, Income=75000, predict: "Education = MS"
/// 5. Continue until all columns are filled
///
/// This captures column dependencies naturally because each prediction
/// is conditioned on all previously generated columns.
///
/// Example:
/// <code>
/// var options = new REaLTabFormerOptions&lt;double&gt;
/// {
///     NumLayers = 4,
///     NumHeads = 4,
///     EmbeddingDimension = 128,
///     Epochs = 100
/// };
/// var realtab = new REaLTabFormerGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// <para>
/// Reference: "REaLTabFormer: Generating Realistic Relational and Tabular Data
/// using Transformers" (Solatorio and Dupriez, 2023)
/// </para>
/// </remarks>
public class REaLTabFormerOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the number of transformer layers.
    /// </summary>
    /// <value>Number of layers, defaulting to 4.</value>
    public int NumLayers { get; set; } = 4;

    /// <summary>
    /// Gets or sets the number of attention heads per layer.
    /// </summary>
    /// <value>Number of heads, defaulting to 4.</value>
    public int NumHeads { get; set; } = 4;

    /// <summary>
    /// Gets or sets the embedding dimension for tokens and positions.
    /// </summary>
    /// <value>Embedding dimension, defaulting to 128.</value>
    public int EmbeddingDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the dimension of the feed-forward network in each transformer block.
    /// </summary>
    /// <value>FFN dimension, defaulting to 256.</value>
    public int FeedForwardDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the dropout rate for attention and feed-forward layers.
    /// </summary>
    /// <value>Dropout rate, defaulting to 0.1.</value>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the number of bins for discretizing continuous values.
    /// </summary>
    /// <value>Number of bins, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Continuous numbers are split into bins (like a histogram)
    /// so the transformer can treat them as categories. More bins = finer granularity
    /// but larger vocabulary. 50-200 is typical.
    /// </para>
    /// </remarks>
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
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Controls randomness during generation:
    /// - Lower (0.1-0.5): More deterministic, less diverse
    /// - Medium (0.5-1.0): Good balance
    /// - Higher (1.0+): More random, more diverse but potentially less realistic
    /// </para>
    /// </remarks>
    public double Temperature { get; set; } = 0.7;
}
