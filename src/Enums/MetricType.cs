namespace AiDotNet.Enums;

/// <summary>
/// Defines the types of metrics used to evaluate machine learning models.
/// </summary>
public enum MetricType
{
    /// <summary>
    /// Coefficient of determination, measuring how well the model explains the variance in the data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For Beginners: R² (R-squared) tells you how well your model fits the data, on a scale from 0 to 1.
    /// A value of 1 means your model perfectly predicts the data, while 0 means it's no better than
    /// just guessing the average value. For example, an R² of 0.75 means your model explains 75% of
    /// the variation in the data.
    /// </para>
    /// </remarks>
    R2,
    
    /// <summary>
    /// A modified version of R² that accounts for the number of predictors in the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For Beginners: Adjusted R² is similar to R², but it penalizes you for adding too many input variables
    /// that don't help much. This prevents "overfitting" - when your model becomes too complex and starts
    /// memorizing the training data rather than learning general patterns. Use this instead of regular R²
    /// when comparing models with different numbers of input variables.
    /// </para>
    /// </remarks>
    AdjustedR2,
    
    /// <summary>
    /// Measures the proportion of variance in the dependent variable explained by the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For Beginners: Explained Variance Score measures how much of the variation in your data is captured
    /// by your model. Like R², it ranges from 0 to 1, with higher values being better. The main difference
    /// is that this metric focuses purely on variance explained, while R² also considers how far predictions
    /// are from the actual values.
    /// </para>
    /// </remarks>
    ExplainedVarianceScore,
    
    /// <summary>
    /// The average difference between predicted values and actual values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For Beginners: Mean Prediction Error simply calculates the average difference between what your model
    /// predicted and what the actual values were. A lower value is better. This metric helps you understand
    /// if your model tends to overestimate or underestimate the results, as positive and negative errors
    /// don't cancel each other out.
    /// </para>
    /// </remarks>
    MeanPredictionError,
    
    /// <summary>
    /// The middle value of all differences between predicted values and actual values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For Beginners: Median Prediction Error finds the middle value of all the prediction errors when sorted.
    /// Unlike the mean, the median isn't affected by extreme outliers, so it gives you a more robust measure
    /// of your model's typical error when some predictions are way off.
    /// </para>
    /// </remarks>
    MedianPredictionError,
    
    /// <summary>
    /// The proportion of correct predictions among all predictions made.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For Beginners: Accuracy simply measures what percentage of your predictions were exactly right.
    /// For example, if your model made 100 predictions and got 85 correct, the accuracy is 85%.
    /// This metric is most useful when all types of errors are equally important and your data is balanced.
    /// </para>
    /// </remarks>
    Accuracy,
    
    /// <summary>
    /// The proportion of true positive predictions among all positive predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For Beginners: Precision measures how many of the items your model identified as positive were actually positive.
    /// For example, if your spam filter marked 10 emails as spam, but only 8 were actually spam, your precision is 80%.
    /// High precision means few false positives - you're not incorrectly flagging things that are actually negative.
    /// </para>
    /// </remarks>
    Precision,
    
    /// <summary>
    /// The proportion of true positive predictions among all actual positives.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For Beginners: Recall measures how many of the actual positive items your model correctly identified.
    /// For example, if there were 20 spam emails, and your filter caught 15 of them, your recall is 75%.
    /// High recall means few false negatives - you're not missing things that should be flagged as positive.
    /// </para>
    /// </remarks>
    Recall,
    
    /// <summary>
    /// The harmonic mean of precision and recall, providing a balance between the two metrics.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For Beginners: F1 Score combines precision and recall into a single number. It's useful when you need
    /// to balance between not missing positives (recall) and not incorrectly flagging negatives (precision).
    /// The score ranges from 0 to 1, with higher values being better. It's especially useful when your data
    /// has an uneven distribution of classes.
    /// </para>
    /// </remarks>
    F1Score,
    
    /// <summary>
    /// The percentage of actual values that fall within the model's prediction intervals.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For Beginners: Prediction Interval Coverage checks if your model's uncertainty estimates are reliable.
    /// Instead of just making a single prediction, some models provide a range (like "between 10-15 units").
    /// This metric tells you what percentage of actual values fall within these predicted ranges. Ideally,
    /// a 95% prediction interval should contain the actual value 95% of the time.
    /// </para>
    /// </remarks>
    PredictionIntervalCoverage,
    
    /// <summary>
    /// Measures the linear correlation between predicted and actual values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For Beginners: Pearson Correlation measures how well the relationship between your predictions and
    /// actual values can be described with a straight line. It ranges from -1 to 1, where:
    /// • 1 means perfect positive correlation (when actual values increase, predictions increase)
    /// • 0 means no correlation
    /// • -1 means perfect negative correlation (when actual values increase, predictions decrease)
    /// A high positive value indicates your model is capturing the right patterns, even if the exact values differ.
    /// </para>
    /// </remarks>
    PearsonCorrelation,
    
    /// <summary>
    /// Measures the monotonic relationship between predicted and actual values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For Beginners: Spearman Correlation is similar to Pearson, but it measures whether predictions and
    /// actual values increase or decrease together, without requiring a straight-line relationship.
    /// It works by ranking the values and then comparing the ranks. This makes it useful when your data
    /// has outliers or when the relationship isn't strictly linear but still follows a pattern.
    /// Like Pearson, it ranges from -1 to 1.
    /// </para>
    /// </remarks>
    SpearmanCorrelation,
    
    /// <summary>
    /// Measures the ordinal association between predicted and actual values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For Beginners: Kendall Tau measures how well your model preserves the correct ordering of values.
    /// It compares every possible pair of data points and checks if your model predicts the same relationship
    /// (is A greater than B, less than B, or equal to B?). This is useful when you care more about getting
    /// the ranking right than the exact values. For example, in a recommendation system, you might care more
    /// about showing the most relevant items first, rather than predicting exact relevance scores.
    /// </para>
    /// </remarks>
    KendallTau
}