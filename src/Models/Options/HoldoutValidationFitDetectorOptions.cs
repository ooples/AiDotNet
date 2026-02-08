namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Holdout Validation Fit Detector, which analyzes model performance
/// on separate training and validation datasets to identify overfitting, underfitting, and other
/// model quality issues.
/// </summary>
/// <remarks>
/// <para>
/// Holdout validation is a technique where a portion of the available data is "held out" from training
/// and used only for validation. By comparing model performance on training data versus this held-out
/// validation data, we can detect various issues like overfitting (performing much better on training
/// than validation data) or underfitting (performing poorly on both datasets).
/// </para>
/// <para><b>For Beginners:</b> This detector helps you understand if your machine learning model is
/// learning properly by comparing how well it performs on data it has seen during training versus new
/// data it hasn't seen before.
/// 
/// Think of it like testing a student's understanding: if they can only answer questions they've seen
/// before but struggle with new questions on the same topic, they've memorized answers rather than
/// truly understanding the subject. Similarly, a good model should perform well not just on its training
/// data but also on new, unseen data.
/// 
/// This detector uses several thresholds to identify common problems:
/// - Overfitting: The model performs much better on training data than validation data (memorization)
/// - Underfitting: The model performs poorly on both training and validation data (not learning enough)
/// - High Variance: The model's performance varies significantly across different validation sets
/// - Good Fit: The model performs well on both training and validation data (proper learning)
/// - Stability: The model's performance is consistent across different validation sets
/// 
/// By detecting these issues early, you can adjust your model or training approach to get better results.</para>
/// </remarks>
public class HoldoutValidationFitDetectorOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the threshold for detecting overfitting based on the relative difference between
    /// training and validation performance.
    /// </summary>
    /// <value>The overfit threshold, defaulting to 0.1 (10%).</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when a model is considered to be overfitting. If the relative difference
    /// between training and validation performance exceeds this threshold, the model is likely overfitting
    /// to the training data. The difference is typically calculated as (training_score - validation_score) / training_score.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps identify when your model is "memorizing" the training
    /// data instead of learning general patterns. With the default value of 0.1, if your model performs
    /// more than 10% better on the training data than on the validation data, it's flagged as overfitting.
    /// 
    /// For example, if your model achieves 90% accuracy on training data but only 80% on validation data
    /// (a relative difference of about 11%), it would be considered overfitting. This suggests your model
    /// has learned patterns that are specific to your training data but don't generalize well to new data.
    /// 
    /// When overfitting is detected, you might want to:
    /// - Use more regularization to penalize complexity
    /// - Reduce model complexity (fewer features, simpler model)
    /// - Gather more training data
    /// - Use techniques like early stopping or dropout
    /// 
    /// If you want to be more strict about preventing overfitting, you could lower this threshold (e.g., to 0.05).
    /// If you're willing to accept more potential overfitting, you could increase it (e.g., to 0.15).</para>
    /// </remarks>
    public double OverfitThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the threshold for detecting underfitting based on the absolute performance on training data.
    /// </summary>
    /// <value>The underfit threshold, defaulting to 0.5 (50%).</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when a model is considered to be underfitting. If the model's performance
    /// on the training data is below this threshold, the model is likely too simple to capture the underlying
    /// patterns in the data. The exact interpretation depends on the performance metric being used.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps identify when your model is "not learning enough"
    /// from the training data. With the default value of 0.5, if your model's performance score on the
    /// training data is below 50%, it's flagged as underfitting.
    /// 
    /// For example, if you're using accuracy as your metric and your model only achieves 45% accuracy even
    /// on the training data, it would be considered underfitting. This suggests your model is too simple
    /// to capture the patterns in your data, or there might be issues with your features or training process.
    /// 
    /// When underfitting is detected, you might want to:
    /// - Increase model complexity (more features, more complex model)
    /// - Train for more iterations or epochs
    /// - Reduce regularization strength
    /// - Engineer better features
    /// - Try a different type of model altogether
    /// 
    /// The appropriate threshold depends on your specific problem and performance metric. For some difficult
    /// problems, even 40% accuracy might be good, while for others, anything below 80% might indicate underfitting.</para>
    /// </remarks>
    public double UnderfitThreshold { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the threshold for detecting high variance based on the relative difference between
    /// multiple validation runs.
    /// </summary>
    /// <value>The high variance threshold, defaulting to 0.1 (10%).</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when a model is considered to have high variance. If the relative standard
    /// deviation of performance across multiple validation runs exceeds this threshold, the model likely
    /// has high variance and is sensitive to the specific data split used for validation.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps identify when your model's performance is too
    /// inconsistent across different validation datasets. With the default value of 0.1, if your model's
    /// performance varies by more than 10% when validated on different subsets of your data, it's flagged
    /// as having high variance.
    /// 
    /// For example, if you run validation 5 times with different random splits and get accuracy scores of
    /// 82%, 75%, 88%, 79%, and 84%, the standard deviation relative to the mean would be about 0.06 or 6%.
    /// This would be acceptable. But if the scores varied more widely, like 65%, 85%, 72%, 90%, and 78%,
    /// the relative standard deviation would be higher than 10%, indicating high variance.
    /// 
    /// High variance suggests that your model's performance depends too much on which specific data points
    /// it sees during training and validation. This is often a sign of overfitting or insufficient data.
    /// 
    /// When high variance is detected, you might want to:
    /// - Use cross-validation instead of a single validation split
    /// - Gather more training data
    /// - Simplify your model
    /// - Use ensemble methods to reduce variance</para>
    /// </remarks>
    public double HighVarianceThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the threshold for confirming good fit based on the absolute performance on validation data.
    /// </summary>
    /// <value>The good fit threshold, defaulting to 0.7 (70%).</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when a model is considered to have a good fit. If the model's performance
    /// on the validation data exceeds this threshold, and it's not overfitting, the model is likely capturing
    /// the underlying patterns in the data well. The exact interpretation depends on the performance metric being used.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps identify when your model is performing well enough
    /// to be considered successful. With the default value of 0.7, if your model's performance score on
    /// the validation data is above 70%, and it's not overfitting, it's considered to have a good fit.
    /// 
    /// For example, if you're using accuracy as your metric and your model achieves 75% accuracy on validation
    /// data (and doesn't show signs of overfitting), it would be considered to have a good fit. This suggests
    /// your model has learned meaningful patterns that generalize well to new data.
    /// 
    /// The appropriate threshold depends on your specific problem and performance metric:
    /// - For some difficult problems, even 60% accuracy might be excellent
    /// - For easier problems, you might want to set this threshold higher (e.g., 0.85 or 0.9)
    /// - For metrics like mean squared error (where lower is better), you would need to invert the logic
    /// 
    /// A good fit means your model has struck a good balance between underfitting and overfitting - it's
    /// complex enough to learn from the data but not so complex that it just memorizes it.</para>
    /// </remarks>
    public double GoodFitThreshold { get; set; } = 0.7;

    /// <summary>
    /// Gets or sets the threshold for confirming model stability based on the relative difference between
    /// multiple validation runs.
    /// </summary>
    /// <value>The stability threshold, defaulting to 0.05 (5%).</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when a model is considered to be stable. If the relative standard deviation
    /// of performance across multiple validation runs is below this threshold, the model is likely stable
    /// and not overly sensitive to the specific data split used for validation.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps identify when your model's performance is consistent
    /// enough across different validation datasets to be considered reliable. With the default value of 0.05,
    /// if your model's performance varies by less than 5% when validated on different subsets of your data,
    /// it's considered stable.
    /// 
    /// For example, if you run validation 5 times with different random splits and get accuracy scores of
    /// 82%, 80%, 83%, 81%, and 84%, the standard deviation relative to the mean would be about 0.02 or 2%.
    /// This would indicate a stable model since it's below the 5% threshold.
    /// 
    /// Stability is important because it suggests your model will perform consistently when deployed in
    /// the real world, rather than having unpredictable performance that depends on which specific data
    /// it encounters.
    /// 
    /// A stable model with good performance is the ideal outcome of the machine learning process. It means
    /// your model has learned general patterns that apply consistently across different subsets of data,
    /// which is exactly what we want.</para>
    /// </remarks>
    public double StabilityThreshold { get; set; } = 0.05;
}
