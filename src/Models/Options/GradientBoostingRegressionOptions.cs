namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Gradient Boosting Regression, an ensemble learning technique that combines
/// multiple decision trees to create a powerful regression model.
/// </summary>
/// <remarks>
/// <para>
/// Gradient Boosting is an ensemble machine learning technique that builds multiple decision trees
/// sequentially, with each tree correcting the errors made by the previous trees. This approach typically
/// produces more accurate models than single decision trees, at the cost of increased complexity and
/// training time. This class inherits from DecisionTreeOptions, so all options for configuring individual
/// trees are also available.
/// </para>
/// <para><b>For Beginners:</b> Think of Gradient Boosting as a team of decision trees working together to
/// make predictions. Instead of relying on just one tree (which might make mistakes), gradient boosting
/// builds trees one after another, with each new tree focusing specifically on correcting the mistakes
/// made by all the previous trees.
/// 
/// Imagine you're trying to predict house prices. The first tree might make rough predictions, getting
/// some houses right but being way off on others. The second tree doesn't try to predict the full house
/// price again - instead, it specifically focuses on the houses where the first tree was wrong, trying to
/// predict the error. This process continues, with each new tree fixing more subtle mistakes, until you
/// have a collection of trees that work together to make very accurate predictions.
/// 
/// Gradient boosting models are among the most powerful and widely used machine learning algorithms,
/// especially for structured/tabular data, because they often achieve excellent accuracy while being
/// relatively easy to use.</para>
/// </remarks>
public class GradientBoostingRegressionOptions : DecisionTreeOptions
{
    /// <summary>
    /// Gets or sets the number of trees (estimators) in the ensemble.
    /// </summary>
    /// <value>The number of trees, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls how many decision trees are built sequentially in the ensemble. More trees
    /// generally improve model performance up to a point, after which returns diminish and the risk of
    /// overfitting increases. The optimal number depends on the dataset size, complexity, and other
    /// hyperparameters like the learning rate.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how many trees will work together in your model.
    /// With the default value of 100, the model will build 100 trees sequentially, with each new tree
    /// focusing on correcting the mistakes of all previous trees.
    /// 
    /// Think of it like getting multiple opinions before making a decision - more opinions (trees) generally
    /// leads to better decisions, but there's a point where adding more opinions doesn't help much and just
    /// makes the process take longer. Similarly, more trees usually improve accuracy but make your model
    /// slower to train and use.
    /// 
    /// If you're getting poor results, you might increase this number (e.g., to 200 or 500). If training is
    /// too slow, you might decrease it (e.g., to 50). The optimal number of trees is often related to your
    /// learning rate - a smaller learning rate typically requires more trees to achieve good performance.</para>
    /// </remarks>
    public int NumberOfTrees { get; set; } = 100;

    /// <summary>
    /// Gets or sets the learning rate (shrinkage) applied to each tree's contribution.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// The learning rate controls how much each tree contributes to the final prediction. Lower values mean
    /// each tree has less influence, requiring more trees to achieve the same level of performance but
    /// typically resulting in better generalization. Higher values make each tree more influential but may
    /// lead to overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how strongly each tree's predictions affect the
    /// final result. With the default value of 0.1, each tree's contribution is scaled down to 10% of its
    /// original prediction.
    /// 
    /// Think of it like taking cautious steps when navigating in the dark - smaller steps (lower learning rate)
    /// mean you're less likely to make a big mistake, but you'll need more steps (trees) to reach your
    /// destination. A higher learning rate is like taking bigger steps - you might reach your destination
    /// faster, but you're more likely to stumble.
    /// 
    /// Common values range from 0.01 to 0.3:
    /// - Smaller values (0.01-0.05): More conservative, less likely to overfit, but require more trees
    /// - Medium values (0.1): A good default balancing performance and training time
    /// - Larger values (0.2-0.3): Faster training, but higher risk of overfitting
    /// 
    /// The learning rate and number of trees are closely related - if you decrease the learning rate, you
    /// typically need to increase the number of trees.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the fraction of samples used for fitting each tree.
    /// </summary>
    /// <value>The subsample ratio, defaulting to 1.0 (use all samples).</value>
    /// <remarks>
    /// <para>
    /// This parameter controls the random subsampling of the training data before building each tree.
    /// Values less than 1.0 implement stochastic gradient boosting, which can improve model robustness
    /// and reduce overfitting by introducing randomness. A value of 1.0 means all samples are used for
    /// every tree (standard gradient boosting).
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines what fraction of your training data is used to
    /// build each individual tree. With the default value of 1.0, every tree uses all of your training data.
    /// If you set it to a lower value like 0.8, each tree would be built using a random 80% of your data.
    /// 
    /// Think of it like getting opinions from people who have seen different parts of a movie - each person
    /// has incomplete information, but together they might give you a more robust understanding than if they
    /// all saw exactly the same scenes. Similarly, training trees on different subsets of data can make your
    /// model more robust and less likely to memorize the training data.
    /// 
    /// Setting this below 1.0 (typically 0.5-0.8) can:
    /// - Help prevent overfitting
    /// - Make training faster
    /// - Introduce helpful randomness that improves generalization
    /// - Make the model more robust to outliers
    /// 
    /// This technique is sometimes called "stochastic gradient boosting" and is particularly helpful for
    /// larger datasets.</para>
    /// </remarks>
    public double SubsampleRatio { get; set; } = 1.0;
}
