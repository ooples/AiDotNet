namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for detecting overfitting, underfitting, and model stability using
/// stratified k-fold cross-validation.
/// </summary>
/// <remarks>
/// <para>
/// Stratified K-Fold Cross-Validation is a technique that divides the dataset into k folds (subsets) while 
/// maintaining the same class distribution in each fold as in the complete dataset. This is particularly 
/// important for imbalanced datasets where some classes have significantly fewer samples than others. The 
/// fit detector uses the performance metrics across these folds to assess whether a model is overfitting 
/// (performing much better on training data than validation data), underfitting (performing poorly on both 
/// training and validation data), or has high variance (performance varies significantly across different 
/// folds). This class provides configuration options for the thresholds used to make these determinations.
/// </para>
/// <para><b>For Beginners:</b> This class helps you detect common model training problems using cross-validation results.
/// 
/// When training machine learning models:
/// - Overfitting: Model learns the training data too well but doesn't generalize
/// - Underfitting: Model is too simple and doesn't capture important patterns
/// - High variance: Model performance changes dramatically with different data subsets
/// 
/// Stratified k-fold cross-validation:
/// - Splits your data into k subsets (folds)
/// - Maintains the same class distribution in each fold (stratified)
/// - Trains k different models, each using k-1 folds for training and 1 for validation
/// - Helps assess how well your model will generalize to new data
/// 
/// This class provides thresholds to automatically detect these issues based on
/// cross-validation results, helping you diagnose and fix model training problems.
/// </para>
/// </remarks>
public class StratifiedKFoldCrossValidationFitDetectorOptions
{
    /// <summary>
    /// Gets or sets the threshold for detecting overfitting.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the maximum acceptable difference between the training and validation performance 
    /// metrics as a proportion of the training metric. If the difference exceeds this threshold, the model is 
    /// considered to be overfitting. For example, with the default value of 0.1 (10%), if the training F1 score 
    /// is 0.9 and the validation F1 score is 0.8 or lower, the model would be flagged as overfitting. A smaller 
    /// threshold is more strict, flagging smaller differences as overfitting, while a larger threshold is more 
    /// lenient, allowing larger differences before flagging overfitting. The appropriate value depends on the 
    /// specific application and the expected variability between training and validation performance.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how much better a model can perform on training data versus validation data before it's considered overfitting.
    /// 
    /// Overfitting occurs when:
    /// - A model performs significantly better on training data than on validation data
    /// - It has essentially "memorized" the training examples rather than learning general patterns
    /// 
    /// The default value of 0.1 means:
    /// - If performance on validation data is more than 10% worse than on training data, the model is overfitting
    /// - For example, if training accuracy is 0.95 and validation accuracy is 0.84, that's overfitting
    /// 
    /// Think of it like this:
    /// - Lower values (e.g., 0.05): More strict, flags smaller differences as overfitting
    /// - Higher values (e.g., 0.2): More lenient, allows larger differences before flagging overfitting
    /// 
    /// When to adjust this value:
    /// - Decrease it when working with simple datasets where training and validation should be very close
    /// - Increase it for complex problems where some gap is expected and acceptable
    /// 
    /// For example, in image classification with limited data, you might increase this to 0.15-0.2
    /// since some gap between training and validation performance is normal.
    /// </para>
    /// </remarks>
    public double OverfitThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the threshold for detecting underfitting.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.6.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the minimum acceptable value for the training performance metric. If the training 
    /// metric falls below this threshold, the model is considered to be underfitting. For example, with the default 
    /// value of 0.6 (60%), if the training F1 score is below 0.6, the model would be flagged as underfitting. A 
    /// higher threshold is more strict, requiring better training performance to avoid being flagged as underfitting, 
    /// while a lower threshold is more lenient, allowing poorer training performance before flagging underfitting. 
    /// The appropriate value depends on the specific application, the complexity of the problem, and the expected 
    /// level of performance for a well-fitted model.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how well a model must perform on training data to avoid being considered underfitting.
    /// 
    /// Underfitting occurs when:
    /// - A model performs poorly on both training and validation data
    /// - It's too simple to capture the underlying patterns in the data
    /// 
    /// The default value of 0.6 means:
    /// - If performance on training data is below 60%, the model is underfitting
    /// - For example, if training accuracy is only 0.55, that's underfitting
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 0.7): More strict, requires better training performance
    /// - Lower values (e.g., 0.5): More lenient, allows poorer training performance
    /// 
    /// When to adjust this value:
    /// - Increase it for problems where high performance is expected and achievable
    /// - Decrease it for very difficult problems where even good models achieve lower metrics
    /// - Adjust based on the specific metric being used (F1, accuracy, etc.)
    /// 
    /// For example, in a medical diagnosis model where high accuracy is critical,
    /// you might increase this to 0.8 or higher to ensure the model is learning effectively.
    /// </para>
    /// </remarks>
    public double UnderfitThreshold { get; set; } = 0.6;

    /// <summary>
    /// Gets or sets the threshold for detecting high variance.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the maximum acceptable standard deviation of the validation performance metrics 
    /// across the k folds. If the standard deviation exceeds this threshold, the model is considered to have high 
    /// variance. For example, with the default value of 0.1 (10%), if the standard deviation of the validation F1 
    /// scores across the folds is greater than 0.1, the model would be flagged as having high variance. A smaller 
    /// threshold is more strict, flagging smaller variations as high variance, while a larger threshold is more 
    /// lenient, allowing larger variations before flagging high variance. The appropriate value depends on the 
    /// specific application, the size and diversity of the dataset, and the expected stability of the model across 
    /// different subsets of the data.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how much a model's performance can vary across different data folds before it's considered unstable.
    /// 
    /// High variance occurs when:
    /// - A model's performance changes significantly when trained on different subsets of data
    /// - It's too sensitive to the specific examples it sees during training
    /// 
    /// The default value of 0.1 means:
    /// - If the standard deviation of performance across folds exceeds 0.1, the model has high variance
    /// - For example, if validation scores across 5 folds are [0.7, 0.9, 0.65, 0.85, 0.75], the standard deviation is high
    /// 
    /// Think of it like this:
    /// - Lower values (e.g., 0.05): More strict, requires more consistent performance across folds
    /// - Higher values (e.g., 0.15): More lenient, allows more variation across folds
    /// 
    /// When to adjust this value:
    /// - Decrease it when stability across different data subsets is critical
    /// - Increase it for smaller datasets where some variation across folds is expected
    /// 
    /// For example, in financial risk models where consistent performance is essential,
    /// you might decrease this to 0.05 to ensure the model is stable across different data subsets.
    /// </para>
    /// </remarks>
    public double HighVarianceThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the threshold for determining a good fit.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.8.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the minimum acceptable value for the validation performance metric for a model to be 
    /// considered a good fit. Even if a model is not overfitting, underfitting, or showing high variance, it might 
    /// still not perform well enough for the specific application. For example, with the default value of 0.8 (80%), 
    /// if the average validation F1 score across the folds is below 0.8, the model would not be considered a good fit, 
    /// even if it doesn't exhibit other problems. A higher threshold is more strict, requiring better validation 
    /// performance to be considered a good fit, while a lower threshold is more lenient. The appropriate value depends 
    /// on the specific application and the minimum acceptable performance for the model to be useful.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how well a model must perform on validation data to be considered a good fit.
    /// 
    /// A good fit means:
    /// - The model performs well on validation data
    /// - It has learned useful patterns that generalize beyond the training examples
    /// - It meets the performance requirements for your application
    /// 
    /// The default value of 0.8 means:
    /// - The average validation performance must be at least 80% to be considered a good fit
    /// - For example, if average validation accuracy is 0.75, that's not considered good enough
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 0.9): More strict, requires excellent validation performance
    /// - Lower values (e.g., 0.7): More lenient, accepts moderate validation performance
    /// 
    /// When to adjust this value:
    /// - Increase it for applications where high performance is critical
    /// - Decrease it for difficult problems where even state-of-the-art models achieve lower metrics
    /// - Adjust based on the specific metric being used (F1, accuracy, etc.)
    /// 
    /// For example, in spam detection where high accuracy is expected,
    /// you might increase this to 0.95, while for a complex recommendation system,
    /// you might decrease it to 0.7.
    /// </para>
    /// </remarks>
    public double GoodFitThreshold { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets the threshold for determining model stability.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.05.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the maximum acceptable coefficient of variation (standard deviation divided by mean) 
    /// of the validation performance metrics across the k folds. This is an alternative measure of stability to the 
    /// HighVarianceThreshold, which considers the standard deviation relative to the mean performance rather than in 
    /// absolute terms. For example, with the default value of 0.05 (5%), if the coefficient of variation of the 
    /// validation F1 scores across the folds is greater than 0.05, the model would be flagged as unstable. A smaller 
    /// threshold is more strict, requiring more consistent relative performance across folds, while a larger threshold 
    /// is more lenient. The appropriate value depends on the specific application and the expected relative stability 
    /// of the model across different subsets of the data.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how consistent a model's performance must be relative to its average performance.
    /// 
    /// Stability in this context means:
    /// - The model performs consistently across different subsets of data
    /// - The variation in performance is small relative to the average performance
    /// 
    /// The default value of 0.05 means:
    /// - If the coefficient of variation (standard deviation ÷ mean) exceeds 5%, the model is considered unstable
    /// - This is a relative measure, unlike HighVarianceThreshold which is absolute
    /// 
    /// Think of it like this:
    /// - Lower values (e.g., 0.03): More strict, requires more consistent relative performance
    /// - Higher values (e.g., 0.1): More lenient, allows more relative variation
    /// 
    /// When to adjust this value:
    /// - Decrease it when consistent relative performance is critical
    /// - Increase it when some relative variation is acceptable
    /// - This is particularly useful for comparing stability across different metrics with different scales
    /// 
    /// For example, in a clinical prediction model where consistent performance is essential,
    /// you might decrease this to 0.03 to ensure the model performs reliably across different patient subgroups.
    /// </para>
    /// </remarks>
    public double StabilityThreshold { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets the primary metric used for evaluating model fit.
    /// </summary>
    /// <value>A value from the MetricType enumeration, defaulting to MetricType.F1Score.</value>
    /// <remarks>
    /// <para>
    /// This property specifies which performance metric should be used as the primary criterion for evaluating the 
    /// model's fit. Different metrics emphasize different aspects of model performance and are appropriate for 
    /// different types of problems. The F1 score (the default) is a harmonic mean of precision and recall, providing 
    /// a balanced measure for classification problems, especially with imbalanced classes. Other common options might 
    /// include accuracy (proportion of correct predictions), precision (proportion of positive identifications that 
    /// were correct), recall (proportion of actual positives that were identified), or area under the ROC curve (AUC). 
    /// The choice of metric should align with the specific goals of the modeling task and the relative importance of 
    /// different types of errors.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines which performance metric is used to evaluate your model.
    /// 
    /// The primary metric:
    /// - Is the main measure used to assess model performance
    /// - Should align with what's most important for your specific problem
    /// 
    /// The default F1Score:
    /// - Balances precision (accuracy of positive predictions) and recall (ability to find all positives)
    /// - Works well for classification problems, especially with imbalanced classes
    /// 
    /// Common alternatives include:
    /// - Accuracy: Simple percentage of correct predictions (good for balanced classes)
    /// - Precision: Focus on minimizing false positives
    /// - Recall: Focus on minimizing false negatives
    /// - AUC: Area under the ROC curve (good for ranking performance)
    /// 
    /// When to adjust this value:
    /// - Choose based on what type of errors are most important to minimize in your application
    /// - For imbalanced classes, usually prefer F1Score, Precision, Recall, or AUC over Accuracy
    /// 
    /// For example, in spam detection, you might use Precision if false positives (marking legitimate emails as spam)
    /// are more problematic, or Recall if false negatives (missing actual spam) are more problematic.
    /// </para>
    /// </remarks>
    public MetricType PrimaryMetric { get; set; } = MetricType.F1Score;
}
