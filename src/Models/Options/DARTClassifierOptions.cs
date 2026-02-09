using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for DART (Dropouts meet Multiple Additive Regression Trees) classifier.
/// </summary>
/// <remarks>
/// <para>
/// DART applies dropout regularization to gradient boosting classification. During each iteration,
/// a random subset of existing trees is dropped, helping prevent overfitting.
/// </para>
/// <para>
/// <b>For Beginners:</b> DART is gradient boosting with dropout - it randomly "forgets" some trees
/// during training to prevent overfitting. This is similar to dropout in neural networks.
///
/// Key options:
/// - DropoutRate: Fraction of trees to drop each iteration (higher = more regularization)
/// - DropoutType: How to select which trees to drop
/// - NormalizationType: How to scale predictions after dropout
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DARTClassifierOptions<T> : ClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the number of boosting iterations.
    /// </summary>
    /// <value>Default is 100.</value>
    public int NumberOfIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets the learning rate (shrinkage factor).
    /// </summary>
    /// <value>Default is 0.1.</value>
    public double LearningRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the dropout rate - fraction of trees to drop each iteration.
    /// </summary>
    /// <value>Default is 0.1 (10% of trees dropped).</value>
    /// <remarks>
    /// <para>
    /// Higher dropout rates provide more regularization but slower learning.
    /// Typical values range from 0.05 to 0.5.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many trees to "forget" each round.
    /// A rate of 0.1 means 10% of trees are ignored when training a new tree.
    /// Higher values help prevent overfitting but may need more iterations.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the type of dropout selection.
    /// </summary>
    /// <value>Default is Uniform.</value>
    public DARTDropoutType DropoutType { get; set; } = DARTDropoutType.Uniform;

    /// <summary>
    /// Gets or sets the normalization type after dropout.
    /// </summary>
    /// <value>Default is Tree.</value>
    public DARTNormalizationType NormalizationType { get; set; } = DARTNormalizationType.Tree;

    /// <summary>
    /// Gets or sets the maximum depth of trees.
    /// </summary>
    /// <value>Default is 3.</value>
    public int MaxDepth { get; set; } = 3;

    /// <summary>
    /// Gets or sets the minimum number of samples required to split.
    /// </summary>
    /// <value>Default is 2.</value>
    public int MinSamplesSplit { get; set; } = 2;

    /// <summary>
    /// Gets or sets the minimum number of samples at a leaf node.
    /// </summary>
    /// <value>Default is 1.</value>
    public int MinSamplesLeaf { get; set; } = 1;

    /// <summary>
    /// Gets or sets the fraction of features to consider for splits.
    /// </summary>
    /// <value>Default is 1.0 (use all features).</value>
    /// <remarks>
    /// Value should be between 0 and 1. Set to 1.0 to use all features.
    /// </remarks>
    public double MaxFeatures { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the split criterion for the base trees.
    /// </summary>
    /// <value>Default is VarianceReduction.</value>
    public SplitCriterion SplitCriterion { get; set; } = SplitCriterion.VarianceReduction;

    /// <summary>
    /// Gets or sets early stopping rounds.
    /// </summary>
    /// <value>Default is null (no early stopping).</value>
    public int? EarlyStoppingRounds { get; set; }

    /// <summary>
    /// Gets or sets whether to print verbose output.
    /// </summary>
    public bool Verbose { get; set; }

    /// <summary>
    /// Gets or sets how often to print progress.
    /// </summary>
    public int VerboseEval { get; set; } = 10;
}

/// <summary>
/// Types of dropout selection for DART.
/// </summary>
public enum DARTDropoutType
{
    /// <summary>
    /// Uniform dropout - always drop a fixed fraction of trees.
    /// </summary>
    Uniform,

    /// <summary>
    /// Binomial dropout - each tree has independent probability of being dropped.
    /// </summary>
    Binomial
}

/// <summary>
/// Types of normalization after dropout in DART.
/// </summary>
public enum DARTNormalizationType
{
    /// <summary>
    /// Tree normalization - scale by number of dropped trees.
    /// </summary>
    Tree,

    /// <summary>
    /// Forest normalization - scale by total forest contribution.
    /// </summary>
    Forest
}
