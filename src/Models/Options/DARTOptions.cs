namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for DART (Dropouts meet Multiple Additive Regression Trees).
/// </summary>
/// <remarks>
/// <para>
/// DART applies the dropout concept from neural networks to gradient boosting. During each
/// boosting iteration, a random subset of existing trees is dropped, and the new tree is fit
/// to the residuals considering only the non-dropped trees. This helps prevent overfitting
/// and improves generalization.
/// </para>
/// <para>
/// <b>For Beginners:</b> Regular gradient boosting can overfit by making each new tree
/// perfectly complement all previous trees. DART introduces randomness by "dropping out"
/// some trees during training:
///
/// - Imagine you have a team of experts (trees) making predictions together
/// - In regular boosting, each new expert learns to fix exactly what the whole team got wrong
/// - In DART, when training a new expert, some existing experts are temporarily removed
/// - This forces the new expert to be more versatile, not just filling a specific gap
/// - The result is a more robust ensemble that generalizes better to new data
///
/// DART is especially useful when:
/// - You're overfitting with regular gradient boosting
/// - You want more robust predictions
/// - You have enough computational budget (DART is slower than regular boosting)
/// </para>
/// <para>
/// Reference: Rashmi, K.V. &amp; Gilad-Bachrach, R. (2015). "DART: Dropouts meet Multiple
/// Additive Regression Trees". AISTATS 2015.
/// </para>
/// </remarks>
public class DARTOptions
{
    /// <summary>
    /// Gets or sets the number of boosting iterations (trees to build).
    /// </summary>
    /// <value>Default is 100.</value>
    public int NumberOfIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets the learning rate (shrinkage factor).
    /// </summary>
    /// <value>Default is 0.1.</value>
    /// <remarks>
    /// Lower values require more trees but often produce better results.
    /// </remarks>
    public double LearningRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the maximum depth of each tree.
    /// </summary>
    /// <value>Default is 6.</value>
    public int MaxDepth { get; set; } = 6;

    /// <summary>
    /// Gets or sets the minimum number of samples required to split a node.
    /// </summary>
    /// <value>Default is 20.</value>
    public int MinSamplesLeaf { get; set; } = 20;

    /// <summary>
    /// Gets or sets the dropout rate (probability of dropping a tree).
    /// </summary>
    /// <value>Default is 0.1 (10% of trees are dropped).</value>
    /// <remarks>
    /// Higher values provide more regularization but may require more trees.
    /// Typical values range from 0.1 to 0.5.
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the dropout mode.
    /// </summary>
    /// <value>Default is Uniform.</value>
    public DARTDropoutMode DropoutMode { get; set; } = DARTDropoutMode.Uniform;

    /// <summary>
    /// Gets or sets the normalization strategy after dropout.
    /// </summary>
    /// <value>Default is Tree.</value>
    public DARTNormalization Normalization { get; set; } = DARTNormalization.Tree;

    /// <summary>
    /// Gets or sets the fraction of features to consider for each split.
    /// </summary>
    /// <value>Default is 1.0 (all features).</value>
    public double FeatureFraction { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the fraction of samples to use for each tree.
    /// </summary>
    /// <value>Default is 1.0 (all samples).</value>
    public double SubsampleFraction { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the L2 regularization strength for leaf weights.
    /// </summary>
    /// <value>Default is 0.0.</value>
    public double L2Regularization { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the minimum loss reduction required to make a split.
    /// </summary>
    /// <value>Default is 0.0.</value>
    public double MinSplitGain { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    public int? Seed { get; set; }

    /// <summary>
    /// Gets or sets whether to skip dropout during the first k iterations.
    /// </summary>
    /// <value>Default is 0.</value>
    /// <remarks>
    /// Skipping dropout early allows the ensemble to build up some base trees
    /// before introducing the dropout regularization.
    /// </remarks>
    public int SkipDropoutIterations { get; set; } = 0;

    /// <summary>
    /// Gets or sets whether to use one-drop strategy.
    /// </summary>
    /// <value>Default is false.</value>
    /// <remarks>
    /// When true, at least one tree is always dropped if the dropout rate is > 0.
    /// This ensures dropout is always applied after the initial iterations.
    /// </remarks>
    public bool OneDrop { get; set; } = false;

    /// <summary>
    /// Gets or sets the probability of adding a dropped tree back to the ensemble.
    /// </summary>
    /// <value>Default is 1.0 (all dropped trees are kept).</value>
    /// <remarks>
    /// Values less than 1.0 can be used for model compression, but may hurt performance.
    /// </remarks>
    public double SampleDroppedTreesProbability { get; set; } = 1.0;
}

/// <summary>
/// Dropout modes for DART.
/// </summary>
public enum DARTDropoutMode
{
    /// <summary>
    /// Each tree has an equal probability of being dropped.
    /// </summary>
    Uniform,

    /// <summary>
    /// Newer trees are less likely to be dropped (protect recent trees).
    /// </summary>
    Age,

    /// <summary>
    /// Trees with higher weights are more likely to be dropped.
    /// </summary>
    Weighted
}

/// <summary>
/// Normalization strategies for DART.
/// </summary>
public enum DARTNormalization
{
    /// <summary>
    /// Normalize by dividing by the number of dropped trees plus one.
    /// </summary>
    Tree,

    /// <summary>
    /// Normalize by dividing by the sum of weights of dropped trees plus the new tree weight.
    /// </summary>
    Forest
}
