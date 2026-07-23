namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Gradient Boosting classifier.
/// </summary>
/// <typeparam name="T">The data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Gradient Boosting builds an ensemble of trees where each tree corrects the errors
/// of the previous ones by fitting to the gradient of the loss function.
/// </para>
/// <para><b>For Beginners:</b> Gradient Boosting is like iteratively fixing mistakes!
///
/// The process:
/// 1. Start with a simple prediction (like the average)
/// 2. Calculate how wrong we are (the residuals)
/// 3. Train a tree to predict those errors
/// 4. Add the tree's predictions (scaled down) to improve our model
/// 5. Repeat many times
///
/// Key parameters to tune:
/// - n_estimators: More trees = potentially better but slower
/// - learning_rate: Lower values need more trees but often work better
/// - max_depth: Usually 3-8 works well (shallower than random forest)
///
/// Gradient Boosting often gives the best accuracy but:
/// - Takes longer to train
/// - More sensitive to hyperparameters
/// - More prone to overfitting if not tuned properly
/// </para>
/// </remarks>
public class GradientBoostingClassifierOptions<T> : ClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the number of boosting stages.
    /// </summary>
    /// <value>
    /// The number of estimators. Default is 100.
    /// </value>
    public int NEstimators { get; set; } = 100;

    /// <summary>
    /// Gets or sets the learning rate (shrinkage).
    /// </summary>
    /// <value>
    /// The learning rate. Default is 0.1.
    /// </value>
    /// <remarks>
    /// <para>
    /// Shrinks the contribution of each tree. Trade-off with n_estimators:
    /// Lower learning rate requires more estimators for same performance
    /// but often achieves better generalization.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the maximum depth of each tree.
    /// </summary>
    /// <value>
    /// The maximum depth. Default is 3.
    /// </value>
    /// <remarks>
    /// Gradient Boosting typically uses shallow trees (depth 3-8).
    /// This is different from Random Forest which often uses deeper trees.
    /// </remarks>
    public int MaxDepth { get; set; } = 3;

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
    /// Gets or sets the fraction of samples used for each tree.
    /// </summary>
    /// <value>
    /// The subsample fraction (0.0 to 1.0). Default is 1.0.
    /// </value>
    /// <remarks>
    /// Values less than 1.0 result in Stochastic Gradient Boosting.
    /// This can reduce overfitting and speed up training.
    /// </remarks>
    public double Subsample { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the fraction of features used for each tree.
    /// </summary>
    /// <value>
    /// The max features fraction or count. Default is 1.0 (all features).
    /// </value>
    public double MaxFeatures { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the loss function.
    /// </summary>
    /// <value>
    /// The loss function. Default is Deviance (log loss).
    /// </value>
    public GradientBoostingLoss Loss { get; set; } = GradientBoostingLoss.Deviance;

    /// <summary>
    /// Gets or sets the minimum impurity decrease for splitting.
    /// </summary>
    /// <value>
    /// The threshold. Default is 0.0.
    /// </value>
    public double MinImpurityDecrease { get; set; } = 0.0;
}

/// <summary>
/// Loss functions for Gradient Boosting classifier.
/// </summary>
public enum GradientBoostingLoss
{
    /// <summary>
    /// Deviance (log loss) - standard for classification.
    /// Equivalent to logistic regression loss.
    /// </summary>
    Deviance,

    /// <summary>
    /// Exponential loss - similar to AdaBoost.
    /// More sensitive to outliers than deviance.
    /// </summary>
    Exponential
}
