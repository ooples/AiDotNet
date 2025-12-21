namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the M5 Model Tree algorithm, which combines decision trees
/// with linear regression models at the leaf nodes.
/// </summary>
/// <remarks>
/// <para>
/// The M5 Model Tree is an extension of decision trees for regression problems, originally proposed by
/// Quinlan. Unlike traditional regression trees that store a constant value at each leaf, M5 Model Trees
/// fit a multivariate linear regression model at each leaf node. This combination allows the algorithm to
/// model both non-linear and linear relationships in the data efficiently. The algorithm includes pruning
/// mechanisms to prevent overfitting and smoothing techniques to improve predictions at the boundaries
/// between different linear models.
/// </para>
/// <para><b>For Beginners:</b> The M5 Model Tree is a powerful algorithm that combines two approaches
/// to make predictions about continuous values (like prices, temperatures, or heights).
/// 
/// Imagine you're trying to predict house prices:
/// - A regular decision tree would divide houses into categories (like "small houses in good neighborhoods")
///   and assign an average price to each category
/// - The M5 Model Tree does something smarter: it divides the houses into categories, but then creates
///   a custom formula for each category
/// 
/// Think of it like this:
/// - First, it groups similar houses together (like a traditional decision tree)
/// - Then, within each group, it creates a formula that considers factors like exact square footage,
///   number of bathrooms, etc. (using linear regression)
/// - This gives you more precise predictions than a simple average for each group
/// 
/// This class allows you to configure how the tree is built, pruned, and how its predictions are smoothed.
/// </para>
/// </remarks>
public class M5ModelTreeOptions : DecisionTreeOptions
{
    /// <summary>
    /// Gets or sets the minimum number of training instances required at each leaf node.
    /// </summary>
    /// <value>The minimum number of instances per leaf, defaulting to 4.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls the minimum number of training examples that must be present in a node
    /// for it to be considered a leaf. If a potential split would result in a leaf with fewer instances
    /// than this threshold, the split is not performed. Higher values lead to smaller trees and help
    /// prevent overfitting, while lower values allow the tree to capture more detailed patterns in the data
    /// but increase the risk of overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how detailed your decision tree can get.
    /// 
    /// Think of it like organizing a library:
    /// - You could have very specific categories (like "Science Fiction Books About Space Travel Published After 2010")
    /// - But if only 1-2 books fit that category, it might not be a useful way to organize
    /// 
    /// This setting requires each "category" (leaf node) to contain at least a certain number of examples:
    /// - The default value of 4 means each final category must have at least 4 training examples
    /// - Higher values (like 10) create broader categories, making the model simpler but possibly less precise
    /// - Lower values (like 2) allow for more specific categories, potentially capturing more detail but
    ///   risking "memorization" of the training data rather than learning general patterns
    /// 
    /// If your model seems to be memorizing the training data but performing poorly on new data,
    /// try increasing this value. If your model seems too simplistic and misses important patterns,
    /// consider decreasing it.
    /// </para>
    /// </remarks>
    public int MinInstancesPerLeaf { get; set; } = 4;

    /// <summary>
    /// Gets or sets the pruning factor that controls the trade-off between model complexity and error.
    /// </summary>
    /// <value>The pruning factor, defaulting to 0.05 (5%).</value>
    /// <remarks>
    /// <para>
    /// The pruning factor is used during the post-pruning phase of the M5 algorithm. It represents the
    /// acceptable increase in error when replacing a subtree with a leaf node containing a linear model.
    /// A higher value leads to more aggressive pruning, resulting in smaller trees, while a lower value
    /// preserves more of the tree structure. The pruning process helps reduce overfitting by removing
    /// branches that do not significantly contribute to the model's predictive performance.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how aggressively the algorithm simplifies
    /// the tree after building it.
    /// 
    /// Imagine you've created a complex flowchart:
    /// - After it's done, you look for sections that could be simplified
    /// - You might replace a complicated section with a simpler approach if the results are similar enough
    /// 
    /// The pruning factor determines what "similar enough" means:
    /// - The default value of 0.05 means you'll simplify parts of the tree if doing so increases the error
    ///   by less than 5%
    /// - Higher values (like 0.1) allow more simplification, potentially making the model easier to understand
    ///   and less prone to overfitting, but possibly less accurate
    /// - Lower values (like 0.01) preserve more of the tree's complexity, potentially maintaining higher
    ///   accuracy on the training data but increasing the risk of overfitting
    /// 
    /// This is like editing a document: aggressive pruning cuts more content for clarity while
    /// minimal pruning preserves more details but might be harder to follow.
    /// </para>
    /// </remarks>
    public double PruningFactor { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets whether to use linear regression models at leaf nodes instead of constant values.
    /// </summary>
    /// <value>Flag indicating whether to use linear regression at leaves, defaulting to true.</value>
    /// <remarks>
    /// <para>
    /// This is a fundamental parameter of the M5 Model Tree algorithm. When set to true, each leaf node
    /// will contain a multivariate linear regression model fitted to the instances that reach that leaf.
    /// When set to false, the algorithm behaves more like a traditional regression tree, using the average
    /// target value of training instances at each leaf. Using linear models at the leaves generally improves
    /// prediction accuracy but increases model complexity and computation time.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines whether the algorithm uses simple averages
    /// or custom formulas at the end points of the decision tree.
    /// 
    /// Returning to our house price example:
    /// - When set to true (the default): Each category of houses gets its own formula for predicting price
    ///   (like: price = $100,000 + $100 × sq.ft + $10,000 × #bathrooms)
    /// - When set to false: Each category just gets an average price (like: all 3-bedroom suburban houses
    ///   are predicted to cost $350,000)
    /// 
    /// The default (true) typically gives more accurate predictions because:
    /// - It can capture more subtle relationships within each category
    /// - It makes better use of the numeric features in your data
    /// - It creates smoother transitions between different parts of your prediction space
    /// 
    /// You might set this to false if:
    /// - You want a simpler, more interpretable model
    /// - Your dataset is very small and at risk of overfitting
    /// - You're primarily interested in understanding the main factors that influence your target variable
    /// </para>
    /// </remarks>
    public bool UseLinearRegressionAtLeaves { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to apply pruning to the tree after it is fully grown.
    /// </summary>
    /// <value>Flag indicating whether to use pruning, defaulting to true.</value>
    /// <remarks>
    /// <para>
    /// Pruning is an important step in the M5 algorithm that helps prevent overfitting. When enabled,
    /// the algorithm first grows a full tree and then examines each non-leaf node, starting from the
    /// bottom, to determine if replacing the subtree with a linear model would improve the error-complexity
    /// trade-off. Disabling pruning allows the tree to grow to its maximum size based on other constraints,
    /// which may capture more detailed patterns but increases the risk of overfitting to the training data.
    /// </para>
    /// <para><b>For Beginners:</b> This setting simply turns the pruning process on or off.
    /// 
    /// Pruning is like editing a first draft:
    /// - First, you write everything down in great detail (growing the full tree)
    /// - Then, you go back and remove sections that don't add enough value (pruning)
    /// 
    /// The default value (true) enables this editing process, which typically:
    /// - Makes the model simpler and easier to understand
    /// - Reduces the risk of overfitting (memorizing the training data too precisely)
    /// - Often improves performance on new, unseen data
    /// 
    /// You might set this to false if:
    /// - You've carefully tuned other parameters to prevent overfitting
    /// - You're working with a dataset where very detailed patterns are crucial
    /// - You want to analyze the full, unpruned tree for research purposes
    /// 
    /// For most practical applications, keeping pruning enabled (true) is recommended.
    /// </para>
    /// </remarks>
    public bool UsePruning { get; set; } = true;

    /// <summary>
    /// Gets or sets the smoothing constant that controls the blending of predictions across different
    /// models in the tree.
    /// </summary>
    /// <value>The smoothing constant, defaulting to 15.0.</value>
    /// <remarks>
    /// <para>
    /// The smoothing process in M5 Model Trees helps improve predictions by combining the predictions of
    /// models along the path from the root to a leaf. The smoothing constant determines the weight of this
    /// blending, with higher values giving more weight to the models at interior nodes and lower values
    /// favoring the leaf model. This smoothing helps reduce discontinuities at the boundaries between
    /// different linear models and often improves overall prediction accuracy.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how the algorithm blends predictions from
    /// different parts of the tree.
    /// 
    /// Imagine your decision tree has created several prediction models:
    /// - One general model for all houses
    /// - More specific models for different types of houses
    /// - Very specific models for particular neighborhoods
    /// 
    /// The smoothing constant determines how to combine these predictions:
    /// - A higher value (like 30.0) gives more influence to the general models
    /// - A lower value (like 5.0) gives more influence to the specific models
    /// - The default value of 15.0 provides a balanced blend
    /// 
    /// This smoothing helps prevent "sharp edges" in your predictions. For example:
    /// - Without smoothing, houses on opposite sides of a neighborhood boundary might have very
    ///   different predicted prices
    /// - With smoothing, the transition becomes more gradual and natural
    /// 
    /// If your model's predictions seem to have abrupt jumps or inconsistencies, try increasing this value.
    /// If your model seems too generalized and misses important local variations, try decreasing it.
    /// </para>
    /// </remarks>
    public double SmoothingConstant { get; set; } = 15.0;
}
