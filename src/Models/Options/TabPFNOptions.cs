using AiDotNet.ActivationFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TabPFN (Prior-Fitted Networks for Tabular Data).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// TabPFN is a transformer model trained on synthetic data that can perform
/// in-context learning on tabular classification tasks. It approximates
/// Bayesian inference through attention mechanisms.
/// </para>
/// <para>
/// <b>For Beginners:</b> TabPFN is unique because it learns from synthetic data:
///
/// - **Meta-learning**: Trained on millions of synthetic classification tasks
/// - **In-context learning**: Given training data as input, predicts test labels
/// - **No fine-tuning**: Works zero-shot on new datasets
/// - **Fast inference**: No training loop needed for new tasks
///
/// Example:
/// <code>
/// var options = new TabPFNOptions&lt;double&gt;
/// {
///     EmbeddingDimension = 128,
///     NumLayers = 12,
///     NumHeads = 4
/// };
/// var model = new TabPFNClassifier&lt;double&gt;(numFeatures: 20, numClasses: 3, options);
/// </code>
/// </para>
/// <para>
/// Reference: "TabPFN: A Transformer That Solves Small Tabular Classification
/// Problems in a Second" (2022)
/// </para>
/// </remarks>
public class TabPFNOptions<T> : RiskModelOptions<T>
{
    /// <summary>Initializes a new instance with default values.</summary>
    public TabPFNOptions() { }

    /// <summary>Initializes a new instance by copying every property from another instance.</summary>
    /// <param name="other">The instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    /// <remarks>
    /// <para>
    /// EVERY property is carried, including the inherited <c>ModelOptions.Seed</c> -- which is
    /// declared on a base class rather than in this file, so a constructor written from the local
    /// declarations alone would silently drop it and the clone would lose its configured seed.
    /// </para>
    /// <para>
    /// Collection properties are CLONED, not assigned: a bare assignment leaves the clone and
    /// the original writing through the same buffer, so mutating one silently reconfigures the
    /// other. A null source stays null rather than becoming an empty collection.
    /// </para>
    /// </remarks>
    public TabPFNOptions(TabPFNOptions<T> other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        LearningRate = other.LearningRate;
        EmbeddingDimension = other.EmbeddingDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        FeedForwardMultiplier = other.FeedForwardMultiplier;
        DropoutRate = other.DropoutRate;
        MaxFeatures = other.MaxFeatures;
        MaxContextSamples = other.MaxContextSamples;
        MaxClasses = other.MaxClasses;
        UsePositionalEncoding = other.UsePositionalEncoding;
        UsePreNorm = other.UsePreNorm;
        InitScale = other.InitScale;
        // An empty array rather than a suppressed null. OutputHeadDimensions is declared
        // non-nullable, so a null source already violates its own contract, and propagating that
        // with a null-forgiving operator just moves the failure downstream to whoever reads it.
        OutputHeadDimensions = other.OutputHeadDimensions is null
            ? []
            : (int[])other.OutputHeadDimensions.Clone();
        UseEnsemble = other.UseEnsemble;
        NumEnsembles = other.NumEnsembles;
        CategoricalCardinalities = other.CategoricalCardinalities is null ? null : (int[])other.CategoricalCardinalities.Clone();
        HiddenActivation = other.HiddenActivation;
        HiddenVectorActivation = other.HiddenVectorActivation;
    }

    /// <summary>
    /// Gets or sets the Adam learning rate.
    /// </summary>
    /// <value>The learning rate, defaulting to the paper's 1e-4 (Hollmann et al., ICLR 2023).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How big a step the model takes each time it learns. The default is
    /// the value the TabPFN authors used. Lower it if the loss stops improving and starts drifting
    /// back up over long training runs.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the embedding dimension.
    /// </summary>
    /// <value>The embedding dimension, defaulting to 128.</value>
    public int EmbeddingDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the number of transformer layers.
    /// </summary>
    /// <value>The number of layers, defaulting to 12.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TabPFN uses many layers (originally 12) because
    /// it needs to learn complex meta-learning patterns from synthetic data.
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 12;

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
    /// <value>The dropout rate, defaulting to 0.0.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TabPFN typically uses no dropout during inference
    /// since it's a pre-trained model that doesn't need regularization.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the maximum number of features supported.
    /// </summary>
    /// <value>The maximum features, defaulting to 100.</value>
    public int MaxFeatures { get; set; } = 100;

    /// <summary>
    /// Gets or sets the maximum number of training samples in context.
    /// </summary>
    /// <value>The maximum context samples, defaulting to 1024.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TabPFN uses training data as context for in-context
    /// learning. This limit ensures efficient inference.
    /// </para>
    /// </remarks>
    public int MaxContextSamples { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the maximum number of classes supported.
    /// </summary>
    /// <value>The maximum classes, defaulting to 10.</value>
    public int MaxClasses { get; set; } = 10;

    /// <summary>
    /// Gets or sets whether to use positional encoding.
    /// </summary>
    /// <value>True to use positional encoding; false otherwise. Defaults to true.</value>
    public bool UsePositionalEncoding { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use pre-norm (norm before attention).
    /// </summary>
    /// <value>True for pre-norm; false for post-norm. Defaults to true.</value>
    public bool UsePreNorm { get; set; } = true;

    /// <summary>
    /// Gets or sets the initialization scale.
    /// </summary>
    /// <value>The initialization scale, defaulting to 0.02.</value>
    public double InitScale { get; set; } = 0.02;

    /// <summary>
    /// Gets or sets the hidden dimensions for the output head.
    /// </summary>
    /// <value>Array of hidden dimensions, defaulting to [64].</value>
    public int[] OutputHeadDimensions { get; set; } = [64];

    /// <summary>
    /// Gets or sets whether to use ensemble predictions.
    /// </summary>
    /// <value>True to use ensembles; false otherwise. Defaults to false.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Ensembles average predictions from multiple
    /// random permutations of the input data for more robust predictions.
    /// </para>
    /// </remarks>
    public bool UseEnsemble { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of ensemble members.
    /// </summary>
    /// <value>The number of ensembles, defaulting to 16.</value>
    public int NumEnsembles { get; set; } = 16;

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
    /// <b>For Beginners:</b> GELU is the standard activation for transformer-based
    /// models like TabPFN and provides smooth gradients.
    /// </para>
    /// </remarks>
    public IActivationFunction<T>? HiddenActivation { get; set; } = new GELUActivation<T>();

    /// <summary>
    /// Gets or sets the hidden layer vector activation function (alternative to scalar activation).
    /// </summary>
    /// <value>The vector activation function, or null to use scalar activation.</value>
    public IVectorActivationFunction<T>? HiddenVectorActivation { get; set; }
}
