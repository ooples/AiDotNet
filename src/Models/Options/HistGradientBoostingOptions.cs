namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Histogram-based Gradient Boosting, a fast ensemble learning technique
/// that uses binned features for efficient tree building.
/// </summary>
/// <remarks>
/// <para>
/// Histogram-based Gradient Boosting is an optimization of traditional gradient boosting that
/// discretizes continuous features into a small number of bins. This dramatically speeds up the
/// algorithm by reducing the number of split points to consider, similar to algorithms like
/// LightGBM and XGBoost's "hist" tree method.
/// </para>
/// <para><b>For Beginners:</b> Think of Histogram Gradient Boosting as a smarter, faster version
/// of regular gradient boosting. Instead of checking every possible split point for each feature
/// (which can be millions of operations), it first groups similar values into "bins" (like putting
/// temperatures into ranges: 0-10, 10-20, 20-30, etc.). Then it only needs to check splits between
/// bins, which is much faster.
///
/// This approach:
/// - Trains 10-100x faster than traditional gradient boosting on large datasets
/// - Uses less memory by storing bin indices instead of full feature values
/// - Often achieves similar or better accuracy due to the regularization effect of binning
/// - Handles missing values natively without imputation
///
/// Use Histogram Gradient Boosting when:
/// - You have a large dataset (millions of rows)
/// - Training time is a concern
/// - You want state-of-the-art performance on tabular data
/// </para>
/// </remarks>
public class HistGradientBoostingOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the maximum number of bins for discretizing feature values.
    /// </summary>
    /// <value>The maximum number of bins, defaulting to 256.</value>
    /// <remarks>
    /// <para>
    /// Features with fewer unique values than MaxBins will have fewer bins. More bins
    /// means more precision but slower training. 256 is the maximum allowed, matching
    /// LightGBM's default, which allows storing bin indices as bytes for efficiency.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how finely the algorithm divides up your
    /// continuous features. Imagine dividing all ages into bins: you could have 10 bins
    /// (0-10, 10-20, etc.) or 256 bins for more precision.
    ///
    /// - More bins (e.g., 256): More precise splits, slower training
    /// - Fewer bins (e.g., 32): Coarser splits, faster training, more regularization
    ///
    /// The default of 256 works well for most cases. Only reduce it if you need faster
    /// training and are willing to sacrifice some precision.
    /// </para>
    /// </remarks>
    public int MaxBins { get; set; } = 256;

    /// <summary>
    /// Gets or sets the number of boosting iterations (trees).
    /// </summary>
    /// <value>The number of trees, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// Each iteration adds a new tree that corrects the errors of the previous ensemble.
    /// More iterations generally improve performance but increase training time and risk
    /// of overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many trees will work together in your model.
    /// Each tree learns from the mistakes of all previous trees.
    ///
    /// - Too few trees: Model may underfit (not capture all patterns)
    /// - Too many trees: Model may overfit (memorize training data)
    ///
    /// Start with 100 and increase if validation performance is still improving.
    /// Use early stopping to automatically find the optimal number.
    /// </para>
    /// </remarks>
    public int NumberOfIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets the learning rate (shrinkage factor).
    /// </summary>
    /// <value>The learning rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// The learning rate scales the contribution of each tree. Lower values require
    /// more trees but often generalize better. Common values range from 0.01 to 0.3.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much each tree influences the final
    /// prediction. Think of it as "confidence" in each tree's contribution.
    ///
    /// - Lower rate (0.01-0.05): Cautious learning, needs more trees, often better generalization
    /// - Higher rate (0.1-0.3): Faster learning, fewer trees needed, higher overfitting risk
    ///
    /// A common strategy is to set a low learning rate and use many trees with early stopping.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the maximum depth of each tree.
    /// </summary>
    /// <value>The maximum depth, defaulting to 6.</value>
    /// <remarks>
    /// <para>
    /// Limits how many times the tree can split. Deeper trees can capture more complex
    /// patterns but are more prone to overfitting. Use MaxLeafNodes for finer control.
    /// </para>
    /// <para><b>For Beginners:</b> This limits how "tall" each decision tree can grow.
    /// A depth of 3 means at most 3 yes/no questions before making a prediction.
    ///
    /// - Shallow trees (2-4): Simpler patterns, less overfitting
    /// - Deeper trees (6-10): Complex patterns, more overfitting risk
    ///
    /// For gradient boosting, shallow trees (3-6) often work well because the ensemble
    /// can combine many simple trees to capture complex patterns.
    /// </para>
    /// </remarks>
    public int MaxDepth { get; set; } = 6;

    /// <summary>
    /// Gets or sets the maximum number of leaf nodes per tree.
    /// </summary>
    /// <value>The maximum leaf nodes, defaulting to 31 (2^5 - 1).</value>
    /// <remarks>
    /// <para>
    /// An alternative to MaxDepth that directly controls tree complexity. Trees will
    /// grow best-first until hitting this limit. Set to null to use MaxDepth instead.
    /// </para>
    /// <para><b>For Beginners:</b> This limits the number of "final answers" each tree
    /// can give. A tree with 31 leaves can make 31 different predictions based on
    /// different combinations of feature values.
    ///
    /// This is often more effective than MaxDepth because:
    /// - The tree grows where it's most useful (best-first)
    /// - Gives more direct control over complexity
    ///
    /// Common values: 15, 31, 63 (one less than powers of 2)
    /// </para>
    /// </remarks>
    public int? MaxLeafNodes { get; set; } = 31;

    /// <summary>
    /// Gets or sets the minimum number of samples required at a leaf node.
    /// </summary>
    /// <value>The minimum samples per leaf, defaulting to 20.</value>
    /// <remarks>
    /// <para>
    /// A split will only be made if it leaves at least this many samples in each child.
    /// Higher values prevent the tree from learning overly specific patterns.
    /// </para>
    /// <para><b>For Beginners:</b> This ensures each "final answer" in the tree is based
    /// on at least this many training examples. If a potential split would create a leaf
    /// with fewer samples, that split is rejected.
    ///
    /// - Lower values (1-10): More detailed splits, higher overfitting risk
    /// - Higher values (20-50): More robust predictions, may miss fine details
    ///
    /// This acts as regularization, preventing the model from memorizing individual
    /// training examples.
    /// </para>
    /// </remarks>
    public int MinSamplesLeaf { get; set; } = 20;

    /// <summary>
    /// Gets or sets the L2 regularization parameter.
    /// </summary>
    /// <value>The L2 regularization strength, defaulting to 0.</value>
    /// <remarks>
    /// <para>
    /// Adds a penalty proportional to the square of leaf values, encouraging smaller
    /// predictions. Higher values increase regularization.
    /// </para>
    /// <para><b>For Beginners:</b> L2 regularization penalizes extreme predictions,
    /// encouraging the model to make more moderate, stable predictions.
    ///
    /// - 0: No regularization (default)
    /// - Small values (0.001-0.1): Mild regularization
    /// - Larger values (1-10): Strong regularization
    ///
    /// Use this if your model is overfitting (good training score, poor validation score).
    /// </para>
    /// </remarks>
    public double L2Regularization { get; set; } = 0;

    /// <summary>
    /// Gets or sets the minimum improvement required to make a split.
    /// </summary>
    /// <value>The minimum gain to split, defaulting to 0.</value>
    /// <remarks>
    /// <para>
    /// A split is only made if it improves the loss by at least this amount. Acts as
    /// pre-pruning to prevent splits that provide minimal benefit.
    /// </para>
    /// <para><b>For Beginners:</b> This prevents the tree from making splits that don't
    /// help much. If a potential split only improves predictions by a tiny amount,
    /// it's probably not worth the added complexity.
    ///
    /// - 0: Allow all beneficial splits (default)
    /// - Small values (0.001-0.01): Filter out marginal improvements
    /// - Larger values (0.1+): Only allow substantial improvements
    ///
    /// This is another form of regularization that can reduce overfitting.
    /// </para>
    /// </remarks>
    public double MinGainToSplit { get; set; } = 0;

    /// <summary>
    /// Gets or sets the fraction of samples to use for each tree (row subsampling).
    /// </summary>
    /// <value>The subsample fraction, defaulting to 1.0 (use all samples).</value>
    /// <remarks>
    /// <para>
    /// Implements stochastic gradient boosting by training each tree on a random subset
    /// of the data. Values less than 1.0 add randomness and can reduce overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This randomly selects a portion of your data to train
    /// each tree. Using 0.8 means each tree sees a different 80% of the data.
    ///
    /// Benefits of subsampling:
    /// - Reduces overfitting
    /// - Speeds up training
    /// - Can improve generalization
    ///
    /// Common values: 0.5-0.9. Start with 1.0 and reduce if overfitting.
    /// </para>
    /// </remarks>
    public double SubsampleRatio { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the fraction of features to consider for each split.
    /// </summary>
    /// <value>The feature fraction, defaulting to 1.0 (use all features).</value>
    /// <remarks>
    /// <para>
    /// Limits the features considered at each split to a random subset. This adds
    /// diversity to trees and can reduce overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This randomly limits which features each split can use.
    /// Using 0.5 means each split considers only 50% of available features.
    ///
    /// Benefits of feature subsampling:
    /// - Reduces correlation between trees
    /// - Can handle many features efficiently
    /// - Often improves generalization
    ///
    /// Common values: 0.5-1.0. More useful when you have many features.
    /// </para>
    /// </remarks>
    public double ColsampleByTree { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether to use early stopping to prevent overfitting.
    /// </summary>
    /// <value>True to enable early stopping, defaulting to false.</value>
    /// <remarks>
    /// <para>
    /// When enabled, training stops if validation performance doesn't improve for
    /// EarlyStoppingRounds consecutive iterations. Requires validation data to be provided.
    /// </para>
    /// <para><b>For Beginners:</b> Early stopping monitors how well the model performs on
    /// data it hasn't seen during training. If performance stops improving, training stops
    /// early to prevent overfitting.
    ///
    /// This is very useful because:
    /// - You don't need to guess the right number of iterations
    /// - Saves training time
    /// - Often produces better models
    ///
    /// Enable this when you have validation data available.
    /// </para>
    /// </remarks>
    public bool UseEarlyStopping { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of rounds to wait for improvement before stopping.
    /// </summary>
    /// <value>The early stopping patience, defaulting to 10.</value>
    /// <remarks>
    /// <para>
    /// If validation performance doesn't improve for this many consecutive iterations,
    /// training stops. Only used when UseEarlyStopping is true.
    /// </para>
    /// <para><b>For Beginners:</b> This is how patient the algorithm is while waiting
    /// for improvement. With a value of 10, training stops if no improvement is seen
    /// for 10 consecutive trees.
    ///
    /// - Lower values (5): Stops quickly, may stop too early
    /// - Higher values (20-50): More patient, less likely to stop prematurely
    ///
    /// If your validation score is noisy, use a higher patience value.
    /// </para>
    /// </remarks>
    public int EarlyStoppingRounds { get; set; } = 10;

}
