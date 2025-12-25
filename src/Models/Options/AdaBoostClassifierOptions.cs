namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for AdaBoost classifier.
/// </summary>
/// <typeparam name="T">The data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// AdaBoost (Adaptive Boosting) is a meta-algorithm that combines multiple weak classifiers
/// into a strong classifier. Each subsequent classifier focuses more on the samples that
/// were misclassified by previous classifiers.
/// </para>
/// <para><b>For Beginners:</b> AdaBoost is like a team of experts that learns from mistakes!
///
/// Imagine you have a series of simple decision makers:
/// 1. The first one makes some mistakes
/// 2. The second one focuses on fixing those mistakes
/// 3. The third one focuses on fixing the remaining mistakes
/// 4. And so on...
///
/// Each decision maker gets a "vote weight" based on how accurate it is.
/// The final prediction combines all their votes.
///
/// AdaBoost is great because:
/// - It automatically focuses on hard-to-classify samples
/// - It combines many simple rules into a complex decision boundary
/// - It's resistant to overfitting (in most cases)
/// - It provides a natural confidence measure
/// </para>
/// </remarks>
public class AdaBoostClassifierOptions<T> : ClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the maximum number of estimators (weak learners).
    /// </summary>
    /// <value>
    /// The number of boosting stages. Default is 50.
    /// </value>
    /// <remarks>
    /// More estimators can improve accuracy but increase training time.
    /// AdaBoost is less prone to overfitting than other methods, so more
    /// estimators usually help until a plateau is reached.
    /// </remarks>
    public int NEstimators { get; set; } = 50;

    /// <summary>
    /// Gets or sets the learning rate (shrinkage).
    /// </summary>
    /// <value>
    /// The learning rate. Default is 1.0.
    /// </value>
    /// <remarks>
    /// <para>
    /// The learning rate shrinks the contribution of each classifier.
    /// Lower values require more estimators but can give better results.
    /// </para>
    /// <para><b>For Beginners:</b> The learning rate controls how much each weak learner contributes.
    ///
    /// - learning_rate = 1.0: Each learner contributes fully (default, faster)
    /// - learning_rate = 0.1: Each learner contributes 10% (needs more estimators, often better)
    ///
    /// A common strategy is to use a smaller learning rate with more estimators.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the algorithm variant to use.
    /// </summary>
    /// <value>
    /// The algorithm: SAMME or SAMME.R. Default is SAMME.R.
    /// </value>
    /// <remarks>
    /// SAMME.R (Real) uses probability estimates and typically performs better.
    /// SAMME uses class predictions only and is the original algorithm.
    /// </remarks>
    public AdaBoostAlgorithm Algorithm { get; set; } = AdaBoostAlgorithm.SAMME_R;

    /// <summary>
    /// Gets or sets the random state for reproducibility.
    /// </summary>
    /// <value>
    /// The random seed, or null for non-deterministic. Default is null.
    /// </value>
    public int? RandomState { get; set; } = null;
}

/// <summary>
/// AdaBoost algorithm variants.
/// </summary>
public enum AdaBoostAlgorithm
{
    /// <summary>
    /// SAMME algorithm using discrete class predictions.
    /// Original AdaBoost algorithm.
    /// </summary>
    SAMME,

    /// <summary>
    /// SAMME.R algorithm using probability estimates.
    /// Generally performs better than SAMME.
    /// </summary>
    SAMME_R
}
