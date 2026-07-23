namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for model statistics and diagnostics calculations, which help evaluate
/// the quality, reliability, and performance of machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// The ModelStatsOptions class provides configuration parameters for various statistical measures
/// and diagnostics that assess model quality. These include tests for multicollinearity (when features
/// are too closely related), metrics for ranking quality (MAP and NDCG), and tools for time series
/// analysis (ACF and PACF). Different diagnostic tools are appropriate for different types of models,
/// and this class allows customization of how these diagnostics are calculated.
/// </para>
/// <para><b>For Beginners:</b> When building machine learning models, we need ways to check if they're
/// working well and to identify potential problems. This is similar to how a doctor uses various tests
/// to check your health.
/// 
/// Think of this class as a collection of settings for these "health checks" for your models:
/// - Some settings help detect when input variables are too similar (which can confuse models)
/// - Some settings configure how we evaluate recommendation or ranking systems
/// - Some settings control how we analyze patterns over time in time series data
/// 
/// By adjusting these settings, you can customize how thorough or sensitive these diagnostic
/// checks should be, similar to how medical tests can be adjusted for different levels of sensitivity.
/// The default values work well for most situations, but sometimes you'll want to adjust them based
/// on your specific data and model.
/// </para>
/// </remarks>
public class ModelStatsOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the method used to calculate the condition number, which measures how numerically
    /// well-behaved a matrix is.
    /// </summary>
    /// <value>The condition number calculation method, defaulting to ConditionNumberMethod.SVD.</value>
    /// <remarks>
    /// <para>
    /// The condition number is a measure of how sensitive a linear system is to changes or errors in the
    /// input data. High condition numbers indicate an ill-conditioned matrix, which can lead to numerical
    /// instability in model fitting. Different methods for calculating the condition number offer trade-offs
    /// between accuracy and computational efficiency. SVD (Singular Value Decomposition) is generally the
    /// most reliable method but can be computationally intensive for large matrices.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how the system calculates a measure called
    /// the "condition number" - a value that tells you if your model might have numerical problems.
    /// 
    /// Imagine trying to solve a wobbly table:
    /// - Some tables are easy to fix (well-conditioned problems)
    /// - Others are so unstable that any small adjustment makes them worse (ill-conditioned problems)
    /// 
    /// The condition number helps identify these "wobbly table" situations in your data. A high
    /// condition number means your model might be unstable and sensitive to tiny changes in data.
    /// 
    /// The default method (SVD) is like using a precise level to measure the wobble - it's very
    /// accurate but takes more time. Other methods might be faster but less precise.
    /// 
    /// Most users won't need to change this setting unless they're dealing with very large datasets
    /// where computation time becomes an issue.
    /// </para>
    /// </remarks>
    public ConditionNumberMethod ConditionNumberMethod { get; set; } = ConditionNumberMethod.SVD;

    /// <summary>
    /// Gets or sets the correlation threshold above which two variables are considered to have
    /// problematic multicollinearity.
    /// </summary>
    /// <value>The multicollinearity threshold, defaulting to 0.8 (80% correlation).</value>
    /// <remarks>
    /// <para>
    /// Multicollinearity occurs when two or more predictor variables in a model are highly correlated,
    /// meaning they contain redundant information. This can cause issues with model stability, interpretability,
    /// and the reliability of coefficient estimates. This parameter sets the correlation threshold at which
    /// the system will flag variables as potentially problematic. The correlation coefficient ranges from
    /// -1 to 1, with values closer to -1 or 1 indicating stronger relationships.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines when the system should warn you that two
    /// of your input variables are too similar to each other.
    /// 
    /// Imagine you're trying to predict house prices using both "square footage" and "number of rooms":
    /// - These two variables are often highly correlated (bigger houses have more rooms)
    /// - Including both might confuse your model because they provide redundant information
    /// - This is called "multicollinearity" and can make your model less reliable
    /// 
    /// The default value of 0.8 means:
    /// - If two variables have a correlation of 80% or higher (positive or negative)
    /// - The system will flag them as potentially problematic
    /// 
    /// You might want to lower this value (like to 0.7) if:
    /// - You need a very stable, interpretable model
    /// - You're in a field where variable independence is critical
    /// 
    /// You might want to raise this value (like to 0.9) if:
    /// - You're more concerned with prediction power than interpretability
    /// - You have theoretical reasons to include correlated variables
    /// - You're using techniques that can handle multicollinearity well
    /// 
    /// This check helps you create more stable and interpretable models by identifying redundant variables.
    /// </para>
    /// </remarks>
    public double MulticollinearityThreshold { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets the maximum allowed Variance Inflation Factor (VIF), which measures how much
    /// the variance of a regression coefficient is increased due to multicollinearity.
    /// </summary>
    /// <value>The maximum acceptable VIF value, defaulting to 10.</value>
    /// <remarks>
    /// <para>
    /// The Variance Inflation Factor (VIF) quantifies the severity of multicollinearity in regression analysis.
    /// It provides an index that measures how much the variance of an estimated regression coefficient is
    /// increased because of collinearity with other predictors. A VIF of 1 means no multicollinearity,
    /// while higher values indicate increasing levels of multicollinearity. This parameter sets the threshold
    /// above which a variable is flagged as having problematic multicollinearity.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines when to flag a variable as being too redundant
    /// with a combination of your other variables.
    /// 
    /// While the MulticollinearityThreshold looks at pairs of variables, the VIF (Variance Inflation Factor)
    /// looks at how each variable relates to all other variables combined:
    /// 
    /// Think of it like this:
    /// - A low VIF (close to 1) means a variable contains unique information
    /// - A high VIF means a variable's information is mostly redundant with other variables
    /// - The higher the VIF, the less reliable that variable's estimated effect will be
    /// 
    /// The default value of 10 means:
    /// - Variables with a VIF over 10 will be flagged as problematic
    /// - This indicates that the variable's estimated effect is about 10 times less reliable than it
    ///   would be without multicollinearity
    /// 
    /// You might want to lower this value (like to 5) if:
    /// - You need very precise estimates of each variable's effect
    /// - You're building a model where interpretation is critical
    /// 
    /// You might keep or raise this value if:
    /// - You're mainly focused on prediction rather than interpretation
    /// - Removing variables would throw away important theoretical information
    /// 
    /// This is a more sophisticated check for redundancy than simple correlation and helps ensure
    /// your model's reliability.
    /// </para>
    /// </remarks>
    public int MaxVIF { get; set; } = 10;

    /// <summary>
    /// Gets or sets the number of top items to consider when calculating Mean Average Precision (MAP).
    /// </summary>
    /// <value>The number of top items for MAP calculations, defaulting to 10.</value>
    /// <remarks>
    /// <para>
    /// Mean Average Precision (MAP) is an evaluation metric for ranking algorithms that considers both
    /// the relevance and the order of recommended items. MAP@k calculates this metric considering only
    /// the top k recommendations. This parameter defines what 'k' value to use. A smaller k focuses the
    /// evaluation more on the highest-ranked items, while a larger k takes more of the ranking into account.
    /// MAP is particularly important in recommendation systems, search engines, and other ranking problems.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how many top recommendations to consider
    /// when evaluating a recommendation or ranking system.
    /// 
    /// Imagine you're evaluating a movie recommendation system:
    /// - Mean Average Precision (MAP) measures how good the recommendations are
    /// - But users typically only look at the first few recommendations
    /// - This setting specifies how many top recommendations to include in the evaluation
    /// 
    /// The default value of 10 means:
    /// - Only the top 10 recommendations are considered when calculating this metric
    /// - This focuses the evaluation on what the user is most likely to see
    /// 
    /// You might want to lower this value (like to 5) if:
    /// - Users in your application typically only look at the first few items
    /// - You want to put extra emphasis on getting the very top recommendations right
    /// 
    /// You might want to raise this value (like to 20 or 50) if:
    /// - Users commonly browse through many recommendations
    /// - You want a more comprehensive evaluation of the ranking quality
    /// 
    /// This setting helps ensure your evaluation metric aligns with how users actually interact
    /// with your system.
    /// </para>
    /// </remarks>
    public int MapTopK { get; set; } = 10;

    /// <summary>
    /// Gets or sets the number of top items to consider when calculating Normalized Discounted Cumulative Gain (NDCG).
    /// </summary>
    /// <value>The number of top items for NDCG calculations, defaulting to 10.</value>
    /// <remarks>
    /// <para>
    /// Normalized Discounted Cumulative Gain (NDCG) is a measure of ranking quality that takes into account
    /// both the relevance of items and their position in the result list. The discount factor penalizes
    /// relevant items appearing lower in the ranking. NDCG@k calculates this metric considering only the
    /// top k recommendations. This parameter defines what 'k' value to use. Similar to MAP, the choice of k
    /// depends on how many recommendations users typically consider in your application.
    /// </para>
    /// <para><b>For Beginners:</b> This setting is similar to MapTopK, but for a different evaluation
    /// metric called NDCG.
    /// 
    /// Continuing with the movie recommendation example:
    /// - NDCG (Normalized Discounted Cumulative Gain) is another way to measure recommendation quality
    /// - Unlike MAP, NDCG considers not just if recommendations are relevant, but how relevant they are
    /// - It also gives more weight to getting the order right at the top of the list
    /// 
    /// The default value of 10 means:
    /// - Only the top 10 recommendations are considered when calculating this metric
    /// - Getting the order right within these 10 items is important
    /// 
    /// The considerations for changing this value are similar to those for MapTopK:
    /// - Lower it if users only look at a few recommendations
    /// - Raise it if users browse through many items
    /// 
    /// Having both MAP and NDCG gives you a more complete picture of your ranking system's quality.
    /// MAP focuses on whether relevant items are included, while NDCG additionally considers their
    /// order and degree of relevance.
    /// </para>
    /// </remarks>
    public int NdcgTopK { get; set; } = 10;

    /// <summary>
    /// Gets or sets the maximum lag to use for the ACF calculation. Default is 20.
    /// </summary>
    /// <value>The maximum lag for autocorrelation function calculations, defaulting to 20.</value>
    /// <remarks>
    /// <para>
    /// The Autocorrelation Function (ACF) measures the correlation between a time series and lagged versions
    /// of itself. This is crucial for understanding the temporal dependence structure in time series data.
    /// This parameter sets how many lags (previous time points) to consider when calculating autocorrelations.
    /// Higher values allow detection of longer-term patterns but require more data and computation. The
    /// appropriate maximum lag depends on the time scale of your data and the patterns you expect to find.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how far back in time the system should look
    /// when analyzing patterns in time series data.
    /// 
    /// Imagine you're analyzing daily temperature data:
    /// - The Autocorrelation Function (ACF) helps detect patterns like:
    ///   - Does today's temperature relate to yesterday's? (lag 1)
    ///   - Does it relate to the temperature a week ago? (lag 7)
    ///   - Does it relate to the temperature a month ago? (lag 30)
    /// 
    /// The default value of 20 means:
    /// - The system will check for relationships with up to 20 time periods in the past
    /// - For daily data, this means up to 20 days back
    /// 
    /// You might want to increase this value if:
    /// - You suspect longer-term patterns exist in your data
    /// - You have seasonal patterns that repeat over a longer period
    /// - You have sufficient data to support looking further back
    /// 
    /// You might want to decrease this value if:
    /// - You're only interested in short-term dependencies
    /// - You have limited data (as a rule of thumb, you want at least 50 data points beyond the maximum lag)
    /// 
    /// This setting helps you discover how the past influences the present in your time series data.
    /// </para>
    /// </remarks>
    public int AcfMaxLag { get; set; } = 20;

    /// <summary>
    /// Gets or sets the maximum lag to use for the PACF calculation. Default is 20.
    /// </summary>
    /// <value>The maximum lag for partial autocorrelation function calculations, defaulting to 20.</value>
    /// <remarks>
    /// <para>
    /// The Partial Autocorrelation Function (PACF) measures the correlation between a time series and lagged
    /// versions of itself after removing the effects of intermediate lags. This helps identify the direct
    /// relationship between observations separated by a specific time lag. This parameter sets how many lags
    /// to consider when calculating partial autocorrelations. The PACF is particularly useful for determining
    /// the appropriate order of autoregressive (AR) models in time series analysis.
    /// </para>
    /// <para><b>For Beginners:</b> This setting is similar to AcfMaxLag, but for a more specific type
    /// of relationship called "partial autocorrelation."
    /// 
    /// Going back to the temperature example:
    /// - Regular autocorrelation (ACF) might show today's temperature relates to both yesterday's and
    ///   the day before
    /// - But this could be because yesterday's temperature relates to the day before, creating an
    ///   indirect chain
    /// - Partial autocorrelation (PACF) removes these indirect effects to show only direct relationships
    /// 
    /// The default value of 20 means:
    /// - The system will check for direct relationships with up to 20 time periods in the past
    /// 
    /// You would typically set this to the same value as AcfMaxLag since they're complementary analyses:
    /// - ACF helps identify moving average (MA) patterns
    /// - PACF helps identify autoregressive (AR) patterns
    /// 
    /// Together, these two tools help you understand the complex patterns in your time series data
    /// and are essential for building accurate forecasting models.
    /// </para>
    /// </remarks>
    public int PacfMaxLag { get; set; } = 20;
}
