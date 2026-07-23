using AiDotNet.Models.Options;

namespace AiDotNet.Models;

/// <summary>
/// Configuration options for the Shapley Value Fit Detector, which evaluates model fit quality
/// by analyzing feature importance using Shapley values.
/// </summary>
/// <remarks>
/// <para>
/// Shapley values, derived from cooperative game theory, provide a method for fairly distributing the 
/// "contribution" of each feature to the prediction made by a machine learning model. The Shapley Value 
/// Fit Detector uses these values to assess model fit by analyzing the distribution of feature importance. 
/// It identifies which features contribute significantly to the model's predictions and uses this information 
/// to detect potential overfitting (when the model relies too heavily on too few features) or underfitting 
/// (when the model distributes importance too evenly across many features). This class provides configuration 
/// options for the thresholds used in this analysis, including the cumulative importance threshold for 
/// identifying significant features, the number of Monte Carlo samples for calculating Shapley values, 
/// and thresholds for detecting overfitting and underfitting based on the ratio of important features.
/// </para>
/// <para><b>For Beginners:</b> This class helps evaluate if your model is using features appropriately.
/// 
/// Shapley values measure how much each feature contributes to your model's predictions:
/// - They come from game theory and provide a fair way to distribute "credit" among features
/// - They show which features are doing the heavy lifting in your model
/// - They can reveal if your model is using features in a balanced way
/// 
/// The detector uses these values to identify potential problems:
/// - Overfitting: When your model relies too heavily on just a few features (like memorizing the data)
/// - Underfitting: When your model spreads importance too evenly across many features (not finding strong patterns)
/// 
/// For example, in a house price prediction model:
/// - Overfitting might show up as the model relying almost entirely on exact address
/// - Underfitting might show up as the model giving similar importance to crucial factors (like location)
///   and irrelevant ones (like the day of week the house was listed)
/// 
/// This class lets you configure how the detector evaluates feature importance distribution
/// to identify these potential issues.
/// </para>
/// </remarks>
public class ShapleyValueFitDetectorOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the threshold for cumulative importance to determine significant features.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.8 (80%).</value>
    /// <remarks>
    /// <para>
    /// This property defines the cumulative importance threshold used to identify the subset of features that 
    /// are considered significant. Features are sorted by their Shapley values (importance) in descending order, 
    /// and features are added to the "significant" set until their cumulative importance exceeds this threshold. 
    /// For example, with the default value of 0.8, features are added until they collectively account for 80% 
    /// of the total importance. This approach focuses the analysis on the features that have the most impact 
    /// on the model's predictions. A higher threshold will include more features in the significant set, while 
    /// a lower threshold will be more selective, including only the most important features.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines which features are considered "important" based on their combined contribution.
    /// 
    /// The detector works by:
    /// - Ranking all features from most to least important based on their Shapley values
    /// - Adding up importance values starting from the top until reaching this threshold
    /// - Considering only those features as "significant" for further analysis
    /// 
    /// The default value of 0.8 means:
    /// - Features are added to the "important" list until they collectively account for 80% of total importance
    /// - This typically identifies the vital few features that drive most of the model's predictions
    /// 
    /// Think of it like the Pareto principle (80/20 rule):
    /// - Often a small subset of features contributes most of the predictive power
    /// - This threshold helps identify that critical subset
    /// 
    /// When to adjust this value:
    /// - Increase it (e.g., to 0.9) to include more features in your "important" set
    /// - Decrease it (e.g., to 0.7) to focus only on the most critical features
    /// 
    /// For example, in a customer churn model with 50 features, this threshold might identify that
    /// just 8 features account for 80% of the model's predictive power.
    /// </para>
    /// </remarks>
    public double ImportanceThreshold { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets the number of Monte Carlo samples to use when calculating Shapley values.
    /// </summary>
    /// <value>A positive integer, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the number of Monte Carlo samples used to approximate Shapley values. Exact 
    /// computation of Shapley values is computationally expensive, with complexity growing exponentially with 
    /// the number of features. Monte Carlo sampling provides an efficient approximation by randomly sampling 
    /// feature coalitions. A larger number of samples provides more accurate approximations but requires more 
    /// computational resources. The default value of 1000 is generally sufficient for most applications to 
    /// provide stable estimates while maintaining reasonable computational efficiency. For models with many 
    /// features or when high precision is required, a larger number of samples might be appropriate, while for 
    /// exploratory analysis or when computational resources are limited, a smaller number might be used.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how accurately the Shapley values are calculated.
    /// 
    /// Calculating exact Shapley values is extremely computationally intensive:
    /// - The exact calculation would require testing all possible combinations of features
    /// - This becomes impossible with more than a few features
    /// - Monte Carlo sampling provides an efficient approximation
    /// 
    /// The default value of 1000 means:
    /// - The algorithm will take 1000 random samples of feature combinations
    /// - It uses these samples to estimate each feature's contribution
    /// - More samples give more accurate estimates but take longer to compute
    /// 
    /// When to adjust this value:
    /// - Increase it (e.g., to 2000 or 5000) when you need very precise importance values
    ///   and have computational resources to spare
    /// - Decrease it (e.g., to 500 or 200) when you need faster results and can accept
    ///   slightly less precise estimates
    /// 
    /// For most applications, 1000 samples provides a good balance between accuracy and
    /// computational efficiency. This is similar to how polling works - more samples give
    /// more accurate results, but with diminishing returns.
    /// </para>
    /// </remarks>
    public int MonteCarloSamples { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the threshold for the ratio of important features to total features,
    /// below which the model is considered to be overfitting.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.2 (20%).</value>
    /// <remarks>
    /// <para>
    /// This property defines the threshold used to detect overfitting based on the ratio of significant features 
    /// to total features. If this ratio falls below the threshold, the model is considered to be overfitting, 
    /// indicating that it is relying too heavily on too few features. Overfitting often occurs when a model 
    /// learns the training data too well, including its noise and outliers, resulting in poor generalization to 
    /// new data. The default value of 0.2 means that if less than 20% of the features account for the cumulative 
    /// importance specified by ImportanceThreshold, the model is flagged as potentially overfitting. This threshold 
    /// may need adjustment based on the specific domain, the total number of features, and the expected distribution 
    /// of feature importance for well-fitted models in the particular application.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps detect if your model is relying too heavily on too few features.
    /// 
    /// Overfitting often manifests as a model that:
    /// - Puts too much importance on a small subset of features
    /// - Essentially "memorizes" patterns in those few features
    /// - Fails to generalize well to new data
    /// 
    /// The OverfitThreshold value (default 0.2) means:
    /// - If less than 20% of your features account for the important portion of predictions
    ///   (as defined by ImportanceThreshold), the model may be overfitting
    /// - This suggests the model is ignoring potentially useful information in other features
    /// 
    /// For example, if you have 100 features and only 10 of them (10%) account for 80% of the
    /// model's predictions, this would fall below the default threshold of 20% and trigger
    /// an overfitting warning.
    /// 
    /// When to adjust this value:
    /// - Decrease it when you expect a very small number of features to legitimately dominate
    ///   (like in some physics or chemistry models where a few variables truly determine outcomes)
    /// - Increase it when you expect importance to be more evenly distributed across features
    /// 
    /// This helps identify models that might perform well on training data but fail when
    /// deployed to real-world scenarios.
    /// </para>
    /// </remarks>
    public double OverfitThreshold { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets the threshold for the ratio of important features to total features,
    /// above which the model is considered to be underfitting.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.8 (80%).</value>
    /// <remarks>
    /// <para>
    /// This property defines the threshold used to detect underfitting based on the ratio of significant features 
    /// to total features. If this ratio exceeds the threshold, the model is considered to be underfitting, 
    /// indicating that it is distributing importance too evenly across many features without focusing sufficiently 
    /// on the most predictive ones. Underfitting often occurs when a model is too simple to capture the underlying 
    /// patterns in the data. The default value of 0.8 means that if more than 80% of the features are needed to 
    /// account for the cumulative importance specified by ImportanceThreshold, the model is flagged as potentially 
    /// underfitting. This threshold may need adjustment based on the specific domain, the total number of features, 
    /// and the expected distribution of feature importance for well-fitted models in the particular application.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps detect if your model is spreading importance too evenly across too many features.
    /// 
    /// Underfitting often manifests as a model that:
    /// - Distributes importance too evenly across many or all features
    /// - Fails to identify which features are truly predictive
    /// - Is too simple to capture important patterns in the data
    /// 
    /// The UnderfitThreshold value (default 0.8) means:
    /// - If more than 80% of your features are needed to account for the important portion of predictions
    ///   (as defined by ImportanceThreshold), the model may be underfitting
    /// - This suggests the model isn't effectively distinguishing between important and unimportant features
    /// 
    /// For example, if you have 100 features and need 85 of them (85%) to account for 80% of the
    /// model's predictions, this would exceed the default threshold of 80% and trigger
    /// an underfitting warning.
    /// 
    /// When to adjust this value:
    /// - Increase it when you expect most features to legitimately contribute similarly
    ///   (like in some complex systems where many variables truly matter equally)
    /// - Decrease it when you expect a clearer distinction between important and unimportant features
    /// 
    /// This helps identify models that might be too simplistic and miss important patterns
    /// in your data.
    /// </para>
    /// </remarks>
    public double UnderfitThreshold { get; set; } = 0.8;
}
