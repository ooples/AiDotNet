namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Random Forest Regression, an ensemble learning method that combines
/// multiple decision trees to improve prediction accuracy and control overfitting.
/// </summary>
/// <remarks>
/// <para>
/// Random Forest Regression is an ensemble learning technique that constructs multiple decision trees
/// during training and outputs the average prediction of the individual trees for regression tasks.
/// This method combines the concepts of bagging (bootstrap aggregating) and feature randomization
/// to create a diverse set of trees. Each tree is trained on a random bootstrap sample of the original
/// data, and at each node, only a random subset of features is considered for splitting. These randomization
/// techniques help reduce correlation between trees, which is essential for the ensemble's performance.
/// Random Forests generally provide higher accuracy than single decision trees, are more robust to noise
/// and outliers, handle high-dimensional data well, and are less prone to overfitting. They also provide
/// built-in estimates of feature importance, making them valuable for feature selection and understanding
/// the underlying data structure.
/// </para>
/// <para><b>For Beginners:</b> Random Forest Regression is like getting predictions from a group of experts instead of just one person.
/// 
/// Think about house price prediction:
/// - A single decision tree is like asking one real estate agent to estimate a house's value
/// - A Random Forest is like asking 100 different agents and taking their average estimate
/// - Each agent (tree) looks at slightly different aspects of the house and has seen different houses before
/// - The combined wisdom of many agents usually gives a more reliable prediction than any single agent
/// 
/// What this technique does:
/// - It builds many decision trees (like a "forest")
/// - Each tree is built using a random sample of your data
/// - Each tree also considers a random subset of features at each decision point
/// - The final prediction is the average of all individual tree predictions
/// 
/// This is especially useful when:
/// - You need more accurate predictions than a single decision tree can provide
/// - Your data has complex relationships that are hard to capture in one model
/// - You want to understand which features are most important for prediction
/// - You're concerned about overfitting (when a model works well on training data but poorly on new data)
/// 
/// For example, in medical diagnosis, a Random Forest might combine the "opinions" of many decision trees
/// to predict patient outcomes more accurately than any single diagnostic approach.
///
/// This class lets you configure how the Random Forest ensemble is constructed.
/// </para>
/// </remarks>
public class RandomForestRegressionOptions : DecisionTreeOptions
{
    /// <summary>
    /// Gets or sets the number of trees to grow in the forest.
    /// </summary>
    /// <value>The number of trees, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines how many individual decision trees will be constructed in the Random Forest
    /// ensemble. Each tree is built using a bootstrap sample of the training data and a random subset of
    /// features at each split. A larger number of trees generally improves prediction accuracy and provides
    /// more stable results, at the cost of increased computation time and memory usage. The improvement in
    /// performance typically diminishes with increasing numbers of trees, with diminishing returns often
    /// observed beyond a few hundred trees. The optimal number depends on the complexity of the problem,
    /// the size and characteristics of the dataset, and the available computational resources.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many different decision trees the algorithm builds.
    /// 
    /// The default value of 100 means:
    /// - The algorithm will create 100 different trees
    /// - Each tree is trained on a randomly selected subset of your data
    /// - Each tree makes slightly different decisions because it sees different examples
    /// - The final prediction averages all 100 tree predictions together
    /// 
    /// Think of it like taking a survey:
    /// - Each tree is like asking one person for their opinion
    /// - With 100 trees, you're surveying 100 different people
    /// - More opinions (trees) generally give you a more reliable consensus
    /// - But at some point, adding more people to your survey doesn't change the average much
    /// 
    /// You might want more trees (like 500 or 1000):
    /// - When you need maximum possible accuracy
    /// - When you have a complex problem with many variables
    /// - When you have plenty of computational resources
    /// - When you're using the model for critical decisions
    /// 
    /// You might want fewer trees (like 50 or 20):
    /// - When you need faster training and prediction times
    /// - When you have limited computational resources
    /// - When you're doing initial exploration and don't need optimal performance
    /// - When your dataset is relatively simple
    /// 
    /// Adding more trees almost never hurts performance (just computation time), so this is 
    /// one of the easier parameters to tune: start with 100 and increase if needed.
    /// </para>
    /// </remarks>
    public int NumberOfTrees { get; set; } = 100;
}
