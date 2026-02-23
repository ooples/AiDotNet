namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for GANDALF (Gated Additive Neural Decision Forest).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// GANDALF combines gated feature selection with neural decision trees in an additive
/// ensemble. It learns which features are important through attention-based gating
/// and makes predictions through soft decision trees.
/// </para>
/// <para>
/// <b>For Beginners:</b> GANDALF is like having a smart feature selector combined
/// with a forest of decision trees.
///
/// Key ideas:
/// 1. **Gated Feature Selection**: Learns which features matter for each prediction
/// 2. **Soft Decision Trees**: Trees with smooth (differentiable) decisions
/// 3. **Additive Ensemble**: Trees are combined by adding their predictions
///
/// Why this works well:
/// - Automatic feature importance learning (no manual selection needed)
/// - Interpretable structure (can see which features and paths are used)
/// - Combines benefits of neural networks and decision trees
/// - Good handling of both numerical and categorical features
///
/// Example:
/// <code>
/// var options = new GANDALFOptions&lt;double&gt;
/// {
///     NumTrees = 20,
///     TreeDepth = 6,
///     GatingHiddenDimension = 128
/// };
/// var model = new GANDALFClassifier&lt;double&gt;(numFeatures: 10, numClasses: 3, options);
/// </code>
/// </para>
/// <para>
/// Reference: "GANDALF: Gated Adaptive Network for Deep Automated Learning of Features" (2022)
/// </para>
/// </remarks>
public class GANDALFOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the number of trees in the ensemble.
    /// </summary>
    /// <value>The number of trees, defaulting to 20.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> More trees = more capacity but slower training:
    /// - 10-20 trees: Good for smaller datasets
    /// - 20-50 trees: Good for medium datasets
    /// - 50-100 trees: Good for large, complex datasets
    /// </para>
    /// </remarks>
    public int NumTrees { get; set; } = 20;

    /// <summary>
    /// Gets or sets the depth of each tree.
    /// </summary>
    /// <value>The tree depth, defaulting to 6.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Deeper trees can model more complex patterns:
    /// - Depth 4-5: Simpler patterns, faster
    /// - Depth 6-8: Good balance for most problems
    /// - Depth 8+: Complex patterns but risk of overfitting
    ///
    /// Each tree has 2^depth leaf nodes.
    /// </para>
    /// </remarks>
    public int TreeDepth { get; set; } = 6;

    /// <summary>
    /// Gets or sets the hidden dimension for gating network.
    /// </summary>
    /// <value>The gating hidden dimension, defaulting to 128.</value>
    public int GatingHiddenDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the number of gating layers.
    /// </summary>
    /// <value>The number of gating layers, defaulting to 2.</value>
    public int NumGatingLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the temperature for soft tree decisions.
    /// </summary>
    /// <value>The temperature, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Controls how "soft" the tree decisions are:
    /// - Lower temperature (&lt;1): Sharper decisions, closer to hard trees
    /// - Higher temperature (&gt;1): Softer decisions, more uncertainty
    ///
    /// Start with 1.0 and adjust if needed.
    /// </para>
    /// </remarks>
    public double Temperature { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.1.</value>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether to use batch normalization.
    /// </summary>
    /// <value>True to use batch normalization; false otherwise. Defaults to true.</value>
    public bool UseBatchNorm { get; set; } = true;

    /// <summary>
    /// Gets or sets the leaf dimension (output dimension per leaf).
    /// </summary>
    /// <value>The leaf dimension, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each leaf can output multiple values:
    /// - LeafDimension = 1: Standard tree (one value per leaf)
    /// - LeafDimension &gt; 1: Rich leaf representations (more expressive)
    /// </para>
    /// </remarks>
    public int LeafDimension { get; set; } = 1;

    /// <summary>
    /// Gets or sets whether to use feature-specific gating.
    /// </summary>
    /// <value>True for feature-specific gating; false for shared gating. Defaults to true.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Feature-specific gating learns separate importance
    /// weights for each feature, while shared gating uses a single attention
    /// mechanism across all features.
    /// </para>
    /// </remarks>
    public bool UseFeatureGating { get; set; } = true;

    /// <summary>
    /// Gets or sets the initialization scale for tree parameters.
    /// </summary>
    /// <value>The initialization scale, defaulting to 0.01.</value>
    public double InitScale { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the weight decay for regularization.
    /// </summary>
    /// <value>The weight decay, defaulting to 1e-5.</value>
    public double WeightDecay { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets whether to use residual connections in gating.
    /// </summary>
    /// <value>True to use residual connections; false otherwise. Defaults to true.</value>
    public bool UseResidualGating { get; set; } = true;

    /// <summary>
    /// Gets the number of leaves per tree (2^depth).
    /// </summary>
    public int NumLeaves => 1 << TreeDepth;

    /// <summary>
    /// Gets the number of internal nodes per tree (2^depth - 1).
    /// </summary>
    public int NumInternalNodes => NumLeaves - 1;
}
