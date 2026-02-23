namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Extra Trees (Extremely Randomized Trees) classifier.
/// </summary>
/// <typeparam name="T">The data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Extra Trees is similar to Random Forest but with even more randomization.
/// Instead of finding the best split, it selects splits at random, which can
/// lead to better generalization and faster training.
/// </para>
/// <para><b>For Beginners:</b> Extra Trees is a "more random" version of Random Forest!
///
/// The key differences from Random Forest:
/// 1. Random Forest finds the BEST split among random features
/// 2. Extra Trees picks RANDOM splits for random features
///
/// This extra randomness:
/// - Makes training even faster
/// - Often generalizes better (less overfitting)
/// - Creates more diverse trees
///
/// When to use Extra Trees:
/// - When Random Forest is overfitting
/// - When you need faster training
/// - As an alternative to try alongside Random Forest
/// </para>
/// </remarks>
public class ExtraTreesClassifierOptions<T> : ClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the number of trees in the forest.
    /// </summary>
    /// <value>
    /// The number of trees. Default is 100.
    /// </value>
    public int NEstimators { get; set; } = 100;

    /// <summary>
    /// Gets or sets the maximum depth of each tree.
    /// </summary>
    /// <value>
    /// The maximum depth, or null for unlimited. Default is null.
    /// </value>
    public int? MaxDepth { get; set; } = null;

    /// <summary>
    /// Gets or sets the minimum samples required to split.
    /// </summary>
    /// <value>
    /// The minimum samples. Default is 2.
    /// </value>
    public int MinSamplesSplit { get; set; } = 2;

    /// <summary>
    /// Gets or sets the minimum samples required at a leaf.
    /// </summary>
    /// <value>
    /// The minimum samples. Default is 1.
    /// </value>
    public int MinSamplesLeaf { get; set; } = 1;

    /// <summary>
    /// Gets or sets the maximum number of features to consider.
    /// </summary>
    /// <value>
    /// "sqrt", "log2", "all", or a number. Default is "sqrt".
    /// </value>
    public string MaxFeatures { get; set; } = "sqrt";

    /// <summary>
    /// Gets or sets the split criterion.
    /// </summary>
    /// <value>
    /// The split quality measure. Default is Gini.
    /// </value>
    public ClassificationSplitCriterion Criterion { get; set; } = ClassificationSplitCriterion.Gini;

    /// <summary>
    /// Gets or sets whether to bootstrap samples.
    /// </summary>
    /// <value>
    /// True for bootstrap sampling, false for full dataset. Default is false.
    /// </value>
    /// <remarks>
    /// Unlike Random Forest, Extra Trees typically uses the full dataset
    /// for each tree (Bootstrap = false), relying on random split selection
    /// for diversity instead of bootstrap sampling.
    /// </remarks>
    public bool Bootstrap { get; set; } = false;

    /// <summary>
    /// Gets or sets the minimum impurity decrease for splitting.
    /// </summary>
    /// <value>
    /// The threshold. Default is 0.0.
    /// </value>
    public double MinImpurityDecrease { get; set; } = 0.0;
}
