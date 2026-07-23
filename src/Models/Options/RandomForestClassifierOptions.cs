namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Random Forest classifiers.
/// </summary>
/// <typeparam name="T">The data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Random Forest is an ensemble learning method that constructs multiple decision trees
/// during training and outputs the class that is the mode (most frequent) of the classes
/// predicted by individual trees.
/// </para>
/// <para><b>For Beginners:</b> Random Forest is like asking many decision trees for their opinion!
///
/// Imagine you're trying to classify a flower species:
/// 1. Create many decision trees, each trained on a random subset of your data
/// 2. Each tree considers only a random subset of features at each split
/// 3. To classify a new flower, ask all trees and take a vote
///
/// Why does this work so well?
/// - Each tree is slightly different due to random sampling
/// - Averaging many "weak" trees often creates a "strong" classifier
/// - It's much harder to overfit than a single deep tree
///
/// Random Forest is one of the most popular algorithms because it:
/// - Works well with default settings
/// - Handles both classification and regression
/// - Is robust to outliers and noise
/// - Provides feature importance scores
/// </para>
/// </remarks>
public class RandomForestClassifierOptions<T> : ClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the number of trees in the forest.
    /// </summary>
    /// <value>
    /// The number of decision trees. Default is 100.
    /// </value>
    /// <remarks>
    /// <para>
    /// More trees generally improve performance but increase training time and memory usage.
    /// The relationship between number of trees and accuracy has diminishing returns.
    /// </para>
    /// <para><b>For Beginners:</b> How many trees to grow in your forest.
    ///
    /// - 10-50 trees: Quick training, may not be fully stable
    /// - 100 trees: Good default, usually sufficient
    /// - 200-500 trees: Better accuracy, slower training
    /// - 1000+ trees: Rarely needed, diminishing returns
    ///
    /// Start with 100 and increase if you need more accuracy.
    /// </para>
    /// </remarks>
    public int NEstimators { get; set; } = 100;

    /// <summary>
    /// Gets or sets the maximum depth of each tree.
    /// </summary>
    /// <value>
    /// The maximum depth, or null for unlimited depth. Default is null.
    /// </value>
    /// <remarks>
    /// <para>
    /// Limiting depth prevents overfitting. With many trees, shallow trees often work well
    /// because the ensemble can still capture complex patterns.
    /// </para>
    /// <para><b>For Beginners:</b> How deep each tree can grow.
    ///
    /// In a Random Forest, you often don't need to limit depth because:
    /// - Averaging many trees reduces overfitting
    /// - Random feature selection at each split adds diversity
    ///
    /// However, limiting depth can speed up training significantly.
    /// Try MaxDepth = 10-20 if training is too slow.
    /// </para>
    /// </remarks>
    public int? MaxDepth { get; set; } = null;

    /// <summary>
    /// Gets or sets the minimum number of samples required to split an internal node.
    /// </summary>
    /// <value>
    /// The minimum number of samples. Default is 2.
    /// </value>
    public int MinSamplesSplit { get; set; } = 2;

    /// <summary>
    /// Gets or sets the minimum number of samples required at a leaf node.
    /// </summary>
    /// <value>
    /// The minimum number of samples. Default is 1.
    /// </value>
    public int MinSamplesLeaf { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of features to consider when looking for the best split.
    /// </summary>
    /// <value>
    /// The number of features, a string specifier, or null for auto.
    /// Default is "sqrt" (square root of total features).
    /// </value>
    /// <remarks>
    /// <para>
    /// Using a subset of features at each split is key to Random Forest's success.
    /// It introduces randomness that decorrelates the trees.
    /// </para>
    /// <para><b>For Beginners:</b> How many features each tree considers at each decision point.
    ///
    /// Common settings:
    /// - "sqrt": Square root of features (default for classification)
    /// - "log2": Log base 2 of features
    /// - null or "all": All features (but this loses some randomness!)
    /// - A number: Exactly that many features
    ///
    /// Using fewer features = more random, more different trees.
    /// </para>
    /// </remarks>
    public string MaxFeatures { get; set; } = "sqrt";

    /// <summary>
    /// Gets or sets the criterion used to measure the quality of a split.
    /// </summary>
    /// <value>
    /// The split criterion. Default is Gini impurity.
    /// </value>
    public ClassificationSplitCriterion Criterion { get; set; } = ClassificationSplitCriterion.Gini;

    /// <summary>
    /// Gets or sets whether to use bootstrap sampling.
    /// </summary>
    /// <value>
    /// True to use bootstrap sampling (default), false to use the whole dataset.
    /// </value>
    /// <remarks>
    /// <para>
    /// Bootstrap sampling means each tree is trained on a random sample of the data
    /// (with replacement). This is a key part of the Random Forest algorithm.
    /// </para>
    /// <para><b>For Beginners:</b> Should each tree see a different random sample of data?
    ///
    /// With Bootstrap = true (default):
    /// - Each tree trains on a random sample (with some duplicates)
    /// - About 63% of the data is used by each tree
    /// - The remaining 37% (out-of-bag) can be used for validation
    ///
    /// With Bootstrap = false:
    /// - Each tree trains on the full dataset
    /// - Less diversity between trees
    /// - Generally not recommended
    /// </para>
    /// </remarks>
    public bool Bootstrap { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to compute out-of-bag score during training.
    /// </summary>
    /// <value>
    /// True to compute OOB score. Default is false.
    /// </value>
    /// <remarks>
    /// <para>
    /// OOB score provides a validation estimate without needing a separate validation set.
    /// Only available when Bootstrap is true.
    /// </para>
    /// <para><b>For Beginners:</b> Get a "free" accuracy estimate during training!
    ///
    /// Because each tree only sees about 63% of the data, the remaining 37%
    /// can be used to test that tree. Averaging these gives the OOB score.
    ///
    /// It's like cross-validation but comes "for free" during training.
    /// </para>
    /// </remarks>
    public bool OobScore { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of jobs for parallel training.
    /// </summary>
    /// <value>
    /// The number of parallel jobs. -1 means use all processors. Default is 1.
    /// </value>
    /// <remarks>
    /// <para>
    /// Since trees in a Random Forest are independent, they can be trained in parallel.
    /// This can significantly speed up training on multi-core machines.
    /// </para>
    /// </remarks>
    public int NJobs { get; set; } = 1;

    /// <summary>
    /// Gets or sets the minimum impurity decrease required for a split.
    /// </summary>
    /// <value>
    /// The minimum impurity decrease. Default is 0.0.
    /// </value>
    public double MinImpurityDecrease { get; set; } = 0.0;
}
