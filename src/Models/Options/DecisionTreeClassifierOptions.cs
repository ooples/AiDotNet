namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for decision tree classifiers.
/// </summary>
/// <typeparam name="T">The data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Decision trees are supervised learning algorithms that learn a hierarchy of
/// if/else rules from training data. They are easy to interpret and can handle
/// both numerical and categorical features.
/// </para>
/// <para><b>For Beginners:</b> Decision trees are like a game of 20 questions.
///
/// At each step, the tree asks a question about a feature:
/// "Is age > 30?" -> Yes: "Is income > 50000?" -> No: "Deny loan"
///                -> No: "Is student?" -> Yes: "Approve loan"
///
/// Key settings:
/// - MaxDepth: Limits how many questions deep the tree can go
/// - MinSamplesSplit: Minimum samples needed to continue splitting
/// - MaxFeatures: How many features to consider at each split
/// </para>
/// </remarks>
public class DecisionTreeClassifierOptions<T> : ClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the maximum depth of the tree.
    /// </summary>
    /// <value>
    /// The maximum depth, or null for unlimited depth. Default is null.
    /// </value>
    /// <remarks>
    /// <para>
    /// Limiting tree depth prevents overfitting by restricting the complexity
    /// of the learned model. Deeper trees can capture more complex patterns
    /// but are more prone to overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> MaxDepth limits how many decisions the tree can make.
    ///
    /// - MaxDepth = 2: Tree asks at most 2 questions before deciding
    /// - MaxDepth = 10: Tree can ask up to 10 questions
    /// - MaxDepth = null: No limit (careful - can lead to overfitting!)
    ///
    /// Start with a smaller depth (3-5) and increase if needed.
    /// </para>
    /// </remarks>
    public int? MaxDepth { get; set; } = null;

    /// <summary>
    /// Gets or sets the minimum number of samples required to split an internal node.
    /// </summary>
    /// <value>
    /// The minimum number of samples. Default is 2.
    /// </value>
    /// <remarks>
    /// <para>
    /// Increasing this value prevents the tree from learning patterns specific
    /// to very small groups of samples, which helps prevent overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This prevents the tree from making decisions based on too few examples.
    ///
    /// If MinSamplesSplit = 10, the tree won't split a node unless it has at least
    /// 10 samples. This makes the tree more robust and less likely to memorize noise.
    /// </para>
    /// </remarks>
    public int MinSamplesSplit { get; set; } = 2;

    /// <summary>
    /// Gets or sets the minimum number of samples required at a leaf node.
    /// </summary>
    /// <value>
    /// The minimum number of samples at leaf nodes. Default is 1.
    /// </value>
    /// <remarks>
    /// <para>
    /// This parameter ensures that each leaf node represents at least this many
    /// training samples, which helps prevent overfitting to individual samples.
    /// </para>
    /// <para><b>For Beginners:</b> Each final decision (leaf) must apply to at least this many examples.
    ///
    /// If MinSamplesLeaf = 5, every leaf must have at least 5 training samples.
    /// This prevents the tree from creating rules for individual outliers.
    /// </para>
    /// </remarks>
    public int MinSamplesLeaf { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of features to consider when looking for the best split.
    /// </summary>
    /// <value>
    /// The number of features, or null to use all features. Default is null.
    /// </value>
    /// <remarks>
    /// <para>
    /// Using a subset of features at each split can improve generalization and
    /// reduce training time for high-dimensional datasets.
    /// </para>
    /// <para><b>For Beginners:</b> At each decision point, the tree considers this many features.
    ///
    /// - null: Consider all features (default)
    /// - sqrt(n_features): Common for random forests
    /// - A specific number: Limits the features considered
    ///
    /// Using fewer features speeds up training and can prevent overfitting.
    /// </para>
    /// </remarks>
    public int? MaxFeatures { get; set; } = null;

    /// <summary>
    /// Gets or sets the criterion used to measure the quality of a split.
    /// </summary>
    /// <value>
    /// The split criterion. Default is Gini impurity.
    /// </value>
    /// <remarks>
    /// <para>
    /// The criterion determines how the tree evaluates potential splits.
    /// Gini impurity and entropy (information gain) are common choices.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how the tree decides which question to ask.
    ///
    /// - Gini: Measures how "pure" the groups are after a split
    /// - Entropy: Measures information gain from a split
    ///
    /// Both work well in practice; Gini is slightly faster to compute.
    /// </para>
    /// </remarks>
    public ClassificationSplitCriterion Criterion { get; set; } = ClassificationSplitCriterion.Gini;

    /// <summary>
    /// Gets or sets the random state for reproducibility.
    /// </summary>
    /// <value>
    /// The random seed, or null for non-deterministic behavior. Default is null.
    /// </value>
    /// <remarks>
    /// <para>
    /// When MaxFeatures is set or when there are ties in split decisions,
    /// randomness is used. Setting this value ensures reproducible results.
    /// </para>
    /// </remarks>
    public int? RandomState { get; set; } = null;

    /// <summary>
    /// Gets or sets the minimum impurity decrease required for a split.
    /// </summary>
    /// <value>
    /// The minimum impurity decrease. Default is 0.0.
    /// </value>
    /// <remarks>
    /// <para>
    /// A node will only be split if the split induces a decrease in impurity
    /// greater than or equal to this value. This can be used for pre-pruning.
    /// </para>
    /// <para><b>For Beginners:</b> Only split if it significantly improves the classification.
    ///
    /// A higher value (e.g., 0.01) means the tree will only split when it really helps,
    /// resulting in a simpler tree that generalizes better.
    /// </para>
    /// </remarks>
    public double MinImpurityDecrease { get; set; } = 0.0;
}

/// <summary>
/// Criterion used to measure the quality of a split in classification decision trees.
/// </summary>
public enum ClassificationSplitCriterion
{
    /// <summary>
    /// Gini impurity measures how often a randomly chosen element would be incorrectly
    /// labeled if it was randomly labeled according to the distribution of labels in the subset.
    /// </summary>
    Gini = 0,

    /// <summary>
    /// Entropy (information gain) measures the average rate at which information is
    /// produced by the stochastic source of data.
    /// </summary>
    Entropy = 1,

    /// <summary>
    /// Log loss criterion useful for probability estimation.
    /// </summary>
    LogLoss = 2
}
