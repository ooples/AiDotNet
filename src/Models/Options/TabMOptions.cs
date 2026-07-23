namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TabM, a parameter-efficient ensemble model for tabular data.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// TabM uses BatchEnsemble-style parameter sharing to create multiple ensemble members
/// with minimal parameter overhead. Each member shares base weights but has its own
/// small rank vectors that modulate the shared weights.
/// </para>
/// <para>
/// <b>For Beginners:</b> TabM is a smart way to get the benefits of model ensembles
/// (multiple models voting together) without the huge computational cost.
///
/// Key concepts:
/// - **Ensembles**: Multiple models combined usually give better predictions
/// - **Parameter Sharing**: TabM shares most weights across ensemble members
/// - **Rank Vectors**: Small per-member vectors that customize the shared weights
/// - **Efficient**: Gets ensemble benefits with minimal extra parameters
///
/// Why this matters:
/// - Traditional ensembles need N times the parameters for N models
/// - TabM only adds about 1-5% extra parameters per ensemble member
/// - Same computational cost as a single model during inference
/// - Often outperforms both single models and traditional ensembles
///
/// Example usage:
/// <code>
/// var options = new TabMOptions&lt;double&gt;
/// {
///     NumEnsembleMembers = 4,
///     HiddenDimensions = [256, 256],
///     DropoutRate = 0.1
/// };
/// var model = new TabMClassifier&lt;double&gt;(numFeatures: 10, numClasses: 3, options);
/// </code>
/// </para>
/// <para>
/// Reference: "TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling" (2024)
/// </para>
/// </remarks>
public class TabMOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the number of ensemble members.
    /// </summary>
    /// <value>The number of ensemble members, defaulting to 4.</value>
    /// <remarks>
    /// <para>
    /// More members generally improve performance but add computation.
    /// The parameter overhead is small due to weight sharing.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This controls how many "models" are combined:
    /// - 2-4 members: Good starting point, minimal overhead
    /// - 4-8 members: Usually optimal for most tasks
    /// - 8+ members: Diminishing returns, more computation
    ///
    /// Unlike traditional ensembles, you can use more members with TabM
    /// because the parameter overhead is so small.
    /// </para>
    /// </remarks>
    public int NumEnsembleMembers { get; set; } = 4;

    /// <summary>
    /// Gets or sets the hidden layer dimensions.
    /// </summary>
    /// <value>Array of hidden dimensions, defaulting to [256, 256].</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These are the sizes of the internal processing layers.
    /// More or larger layers = more capacity to learn complex patterns.
    ///
    /// Examples:
    /// - [128, 128]: Smaller, faster, good for smaller datasets
    /// - [256, 256]: Default, good balance
    /// - [512, 256, 128]: Larger, pyramid shape, for complex problems
    /// </para>
    /// </remarks>
    public int[] HiddenDimensions { get; set; } = [256, 256];

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dropout randomly turns off neurons during training
    /// to prevent overfitting.
    ///
    /// - 0.0: No dropout
    /// - 0.1: Light dropout (default)
    /// - 0.2-0.3: Heavier dropout for small datasets
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether to use layer normalization.
    /// </summary>
    /// <value>True to use layer normalization; false otherwise. Defaults to true.</value>
    /// <remarks>
    /// <para>
    /// Layer normalization helps stabilize training and often improves performance.
    /// </para>
    /// </remarks>
    public bool UseLayerNorm { get; set; } = true;

    /// <summary>
    /// Gets or sets the initialization scale for rank vectors.
    /// </summary>
    /// <value>The rank initialization scale, defaulting to 0.5.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This controls how different the ensemble members are initially:
    /// - Lower values (0.1-0.3): Members start more similar
    /// - Higher values (0.5-1.0): Members start more diverse
    ///
    /// The default 0.5 provides good diversity while maintaining stability.
    /// </para>
    /// </remarks>
    public double RankInitScale { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets whether to use bias in the ensemble layers.
    /// </summary>
    /// <value>True to use bias; false otherwise. Defaults to true.</value>
    public bool UseBias { get; set; } = true;

    /// <summary>
    /// Gets or sets the activation function type.
    /// </summary>
    /// <value>The activation type, defaulting to "ReLU".</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The activation function controls how neurons respond:
    /// - "ReLU": Fast and effective, most common choice
    /// - "GELU": Smoother, often better for deep networks
    /// - "SiLU": Similar to GELU, good for modern architectures
    /// </para>
    /// </remarks>
    public string ActivationType { get; set; } = "ReLU";

    /// <summary>
    /// Gets or sets whether to average ensemble predictions or concatenate.
    /// </summary>
    /// <value>True to average predictions; false to concatenate then project. Defaults to true.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How to combine predictions from ensemble members:
    /// - Average (true): Simple average of all member predictions
    /// - Concatenate (false): Concatenate and learn a combination
    ///
    /// Averaging is simpler and usually works well. Concatenation gives
    /// more flexibility but adds parameters.
    /// </para>
    /// </remarks>
    public bool AverageEnsemble { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use numerical feature embeddings.
    /// </summary>
    /// <value>True to embed numerical features; false to use them directly. Defaults to true.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Feature embeddings can help the model learn better
    /// representations, especially when features have different scales.
    ///
    /// With embeddings: Each feature gets its own learned transformation
    /// Without: Features are used directly (faster but sometimes worse)
    /// </para>
    /// </remarks>
    public bool UseFeatureEmbeddings { get; set; } = true;

    /// <summary>
    /// Gets or sets the embedding dimension for feature embeddings.
    /// </summary>
    /// <value>The embedding dimension, defaulting to 32.</value>
    public int FeatureEmbeddingDimension { get; set; } = 32;

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
    /// Creates a copy of the options.
    /// </summary>
    /// <returns>A new TabMOptions instance with the same values.</returns>
    public TabMOptions<T> Clone()
    {
        return new TabMOptions<T>
        {
            NumEnsembleMembers = NumEnsembleMembers,
            HiddenDimensions = HiddenDimensions.ToArray(),
            DropoutRate = DropoutRate,
            UseLayerNorm = UseLayerNorm,
            RankInitScale = RankInitScale,
            UseBias = UseBias,
            ActivationType = ActivationType,
            AverageEnsemble = AverageEnsemble,
            UseFeatureEmbeddings = UseFeatureEmbeddings,
            FeatureEmbeddingDimension = FeatureEmbeddingDimension,
            EnableGradientClipping = EnableGradientClipping,
            MaxGradientNorm = MaxGradientNorm,
            WeightDecay = WeightDecay
        };
    }
}
