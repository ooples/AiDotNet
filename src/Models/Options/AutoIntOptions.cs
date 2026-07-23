using AiDotNet.ActivationFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for AutoInt (Automatic Feature Interaction Learning).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// AutoInt uses multi-head self-attention to automatically learn high-order
/// feature interactions without manual feature engineering.
/// </para>
/// <para>
/// <b>For Beginners:</b> AutoInt is designed to find interactions between features:
///
/// - **Feature embeddings**: Each feature is converted to a vector
/// - **Self-attention layers**: Features attend to each other to learn interactions
/// - **Explicit interactions**: The model learns "feature A combined with feature B"
///
/// Example use case: In click-through rate prediction, "age + product_category"
/// might have a strong interaction that AutoInt can automatically discover.
///
/// Example:
/// <code>
/// var options = new AutoIntOptions&lt;double&gt;
/// {
///     EmbeddingDimension = 16,
///     NumLayers = 3,
///     NumHeads = 2
/// };
/// var model = new AutoIntClassifier&lt;double&gt;(numFeatures: 10, numClasses: 2, options);
/// </code>
/// </para>
/// <para>
/// Reference: "AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks" (2018)
/// </para>
/// </remarks>
public class AutoIntOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the embedding dimension for features.
    /// </summary>
    /// <value>The embedding dimension, defaulting to 16.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each feature is represented as a vector of this size.
    /// Smaller values (8-16) work well for simpler datasets.
    /// </para>
    /// </remarks>
    public int EmbeddingDimension { get; set; } = 16;

    /// <summary>
    /// Gets or sets the number of interacting (self-attention) layers.
    /// </summary>
    /// <value>The number of layers, defaulting to 3.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each layer captures higher-order interactions:
    /// - Layer 1: Pairwise interactions (A+B)
    /// - Layer 2: Triple interactions (A+B+C)
    /// - Layer 3: Quadruple interactions, and so on
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of attention heads per layer.
    /// </summary>
    /// <value>The number of heads, defaulting to 2.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multiple heads let the model learn different
    /// types of interactions simultaneously.
    /// </para>
    /// </remarks>
    public int NumHeads { get; set; } = 2;

    /// <summary>
    /// Gets or sets the attention dimension per head.
    /// </summary>
    /// <value>The attention dimension, defaulting to 32.</value>
    public int AttentionDimension { get; set; } = 32;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.0.</value>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets whether to use residual connections.
    /// </summary>
    /// <value>True to use residual connections; false otherwise. Defaults to true.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Residual connections add the original feature embeddings
    /// back to the interaction outputs, preserving individual feature information.
    /// </para>
    /// </remarks>
    public bool UseResidual { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use layer normalization.
    /// </summary>
    /// <value>True to use layer normalization; false otherwise. Defaults to false.</value>
    public bool UseLayerNorm { get; set; } = false;

    /// <summary>
    /// Gets or sets the hidden dimensions for the MLP output layer.
    /// </summary>
    /// <value>Array of hidden dimensions, defaulting to [64, 32].</value>
    public int[] MLPHiddenDimensions { get; set; } = [64, 32];

    /// <summary>
    /// Gets or sets the cardinalities of categorical features.
    /// </summary>
    /// <value>Array of cardinalities, or null if no categorical features.</value>
    public int[]? CategoricalCardinalities { get; set; }

    /// <summary>
    /// Gets or sets the initialization scale for embeddings.
    /// </summary>
    /// <value>The initialization scale, defaulting to 0.01.</value>
    public double EmbeddingInitScale { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the hidden layer activation function.
    /// </summary>
    /// <value>The activation function, defaulting to ReLU.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The activation function adds non-linearity to the MLP layers.
    /// ReLU is a good default that works well for most tabular problems.
    /// </para>
    /// </remarks>
    public IActivationFunction<T>? HiddenActivation { get; set; } = new ReLUActivation<T>();

    /// <summary>
    /// Gets or sets the hidden layer vector activation function (alternative to scalar activation).
    /// </summary>
    /// <value>The vector activation function, or null to use scalar activation.</value>
    public IVectorActivationFunction<T>? HiddenVectorActivation { get; set; }
}
