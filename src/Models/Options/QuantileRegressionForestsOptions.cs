namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Quantile Regression Forests, an extension of Random Forests that enables
/// prediction of conditional quantiles rather than just the conditional mean.
/// </summary>
/// <remarks>
/// <para>
/// Quantile Regression Forests extend the Random Forest algorithm to provide full conditional distributions
/// instead of just point estimates. While standard Random Forests estimate the conditional mean E(Y|X),
/// Quantile Regression Forests can estimate any conditional quantile Q(a|X) for a ? (0,1), including
/// medians and prediction intervals. This is achieved by keeping track of all target values in the leaf
/// nodes of each tree, rather than just their averages. The algorithm provides a non-parametric way to
/// estimate conditional distributions, making it particularly valuable for problems where uncertainty
/// quantification is important or where the conditional distribution is non-Gaussian, skewed, or
/// heteroscedastic (having non-constant variance). Quantile Regression Forests inherit many advantages of
/// Random Forests, including handling of non-linear relationships, robustness to outliers, and minimal
/// parameter tuning requirements.
/// </para>
/// <para><b>For Beginners:</b> Quantile Regression Forests help predict not just a single value, but a range of possible values with their probabilities.
/// 
/// Think about weather forecasting:
/// - A regular forecast might say "tomorrow's temperature will be 75°F"
/// - But Quantile Regression Forests could tell you:
///   - "There's a 10% chance it will be below 70°F"
///   - "There's a 50% chance it will be below 75°F" (the median)
///   - "There's a 90% chance it will be below 80°F"
/// 
/// What this algorithm does:
/// - It builds many decision trees, just like a regular Random Forest
/// - But instead of averaging their predictions to get a single answer
/// - It keeps track of all possible outcomes and their distributions
/// - This lets you understand the uncertainty in your predictions
/// 
/// This is especially useful when:
/// - You need to know the range of possible outcomes, not just the average
/// - Your data has varying levels of uncertainty in different regions
/// - The distribution of possible outcomes is not symmetric
/// - Risk assessment is as important as the prediction itself
/// 
/// For example, in financial forecasting, knowing there's a 5% chance of losing $10,000
/// is very different information than just knowing the average expected return.
/// 
/// This class lets you configure how the forest of trees is built and processed.
/// </para>
/// </remarks>
public class QuantileRegressionForestsOptions : DecisionTreeOptions
{
    /// <summary>
    /// Gets or sets the number of trees to grow in the forest.
    /// </summary>
    /// <value>The number of trees, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines how many individual decision trees will be constructed in the Quantile
    /// Regression Forest ensemble. Each tree is built using a bootstrap sample of the training data and
    /// a random subset of features at each split. A larger number of trees generally improves prediction
    /// quality and provides more stable quantile estimates, at the cost of increased computation time and
    /// memory usage. The improvement in performance typically diminishes with increasing numbers of trees,
    /// with diminishing returns beyond a certain point. For quantile estimation, having more trees can be
    /// particularly important to obtain smooth and stable estimates of the conditional distribution.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many different decision trees the algorithm builds.
    /// 
    /// The default value of 100 means:
    /// - The algorithm will create 100 different trees
    /// - Each tree is slightly different because it uses randomly selected data and features
    /// - The final prediction combines information from all 100 trees
    /// 
    /// Think of it like getting opinions from multiple experts:
    /// - Each tree is like a different expert with a slightly different perspective
    /// - More experts (trees) generally give you more reliable opinions
    /// - But there's a point where adding more experts doesn't help much more
    /// 
    /// You might want more trees (like 500 or 1000):
    /// - When you need more precise quantile estimates
    /// - When you have a complex problem with many variables
    /// - When you have enough computational resources available
    /// - When the stability of quantile estimates is particularly important
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

    /// <summary>
    /// Gets or sets the maximum degree of parallelism for tree building.
    /// </summary>
    /// <value>The maximum degree of parallelism, defaulting to the number of processor cores.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls how many trees can be built in parallel during the training process.
    /// Since each tree in the forest can be built independently, Quantile Regression Forests are naturally
    /// parallelizable. By default, this value is set to the number of logical processors available on
    /// the machine, which often provides a good balance between computation speed and resource utilization.
    /// Higher values can improve training speed on machines with many cores, while lower values reduce
    /// resource consumption. Setting this to 1 disables parallelism, which may be necessary in memory-constrained
    /// environments or when debugging. This parameter affects only training speed, not the model's
    /// predictive capabilities.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many trees can be built simultaneously using multiple processor cores.
    /// 
    /// The default value (Environment.ProcessorCount) means:
    /// - The algorithm will use all available CPU cores on your machine
    /// - If you have a 4-core processor, it can build 4 trees at the same time
    /// - This makes training much faster than building trees one at a time
    /// 
    /// Think of it like a construction project:
    /// - Each tree is like a house that needs to be built
    /// - Each processor core is like a construction team
    /// - With multiple teams working in parallel, you can build houses much faster
    /// - But using all your teams means you can't use them for other tasks
    /// 
    /// You might want a higher value:
    /// - This setting automatically uses all available cores, so increasing it beyond your core count won't help
    /// 
    /// You might want a lower value (like 2 or 1):
    /// - When you want to leave CPU resources for other applications
    /// - When you're running into memory limitations (fewer parallel trees = less memory usage)
    /// - When you notice your system becoming unresponsive during training
    /// 
    /// This setting only affects training speed and resource usage - it doesn't change how the model performs
    /// once trained. It's a practical consideration rather than a modeling decision.
    /// </para>
    /// </remarks>
    public int MaxDegreeOfParallelism { get; set; } = Environment.ProcessorCount;
}
