namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Confusion Matrix Fit Detector, which evaluates how well a classification model performs.
/// </summary>
/// <remarks>
/// <para>
/// The Confusion Matrix Fit Detector analyzes classification results to determine if a model is performing adequately.
/// It uses various thresholds and metrics to categorize model performance as good, moderate, or poor,
/// and can detect issues like class imbalance that might affect model reliability.
/// </para>
/// <para><b>For Beginners:</b> When you build a model that predicts categories (like "spam" vs "not spam" 
/// or "dog" vs "cat" vs "bird"), you need a way to check how good your model is. A confusion matrix is a table 
/// that shows how often your model was right or wrong for each category. This class provides settings that help 
/// automatically evaluate your model's performance based on that table. It can tell you if your model is doing 
/// well overall, if it's struggling with certain categories, or if your data is unbalanced (having way more 
/// examples of one category than others). Think of it like an automated grading system for your AI model.</para>
/// </remarks>
public class ConfusionMatrixFitDetectorOptions
{
    /// <summary>
    /// Gets or sets the threshold above which a model's performance is considered good.
    /// </summary>
    /// <value>The good fit threshold as a decimal between 0 and 1, defaulting to 0.8 (80%).</value>
    /// <remarks>
    /// <para>
    /// This threshold applies to the primary metric (by default, F1 score).
    /// Models with a primary metric value above this threshold are considered to have good performance.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines when your model gets an "A grade." The default 
    /// value (0.8 or 80%) means that if your model's score is above 80%, it's considered to be performing well. 
    /// You might adjust this higher (like 0.9) if you need very high accuracy, or lower (like 0.7) if your 
    /// problem is particularly difficult and even modest performance is valuable. For medical or safety-critical 
    /// applications, you'd typically want this threshold to be higher.</para>
    /// </remarks>
    public double GoodFitThreshold { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets the threshold above which a model's performance is considered moderate.
    /// </summary>
    /// <value>The moderate fit threshold as a decimal between 0 and 1, defaulting to 0.6 (60%).</value>
    /// <remarks>
    /// <para>
    /// This threshold applies to the primary metric (by default, F1 score).
    /// Models with a primary metric value above this threshold but below the GoodFitThreshold
    /// are considered to have moderate performance.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines when your model gets a "C grade." The default 
    /// value (0.6 or 60%) means that if your model's score is between 60% and the good fit threshold (80% by default), 
    /// it's considered to be performing adequately but not great. Models below this threshold are considered poor 
    /// performers. This middle ground is useful for identifying models that might be worth improving rather than 
    /// discarding entirely. You might adjust this based on your specific needs and the difficulty of your problem.</para>
    /// </remarks>
    public double ModerateFitThreshold { get; set; } = 0.6;

    /// <summary>
    /// Gets or sets the threshold that determines when class imbalance is considered significant.
    /// </summary>
    /// <value>The class imbalance threshold as a decimal between 0 and 1, defaulting to 0.2 (20%).</value>
    /// <remarks>
    /// <para>
    /// Class imbalance occurs when some categories have significantly more examples than others.
    /// If the proportion of the smallest class is below this threshold compared to the largest class,
    /// the detector will flag potential class imbalance issues.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps detect when your data is unbalanced. For example, 
    /// if you're building a model to detect fraud, you might have 1000 examples of normal transactions but 
    /// only 10 examples of fraudulent ones. The default value (0.2 or 20%) means that if your smallest category 
    /// makes up less than 20% compared to your largest category, the detector will warn you about this imbalance. 
    /// Imbalanced data can make models appear more accurate than they really are, because they can get high 
    /// accuracy just by always predicting the most common category.</para>
    /// </remarks>
    public double ClassImbalanceThreshold { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets the primary metric used to evaluate model performance.
    /// </summary>
    /// <value>The primary metric type, defaulting to F1Score.</value>
    /// <remarks>
    /// <para>
    /// Different metrics emphasize different aspects of model performance:
    /// - Accuracy: Overall correctness across all classes
    /// - Precision: Ability to avoid false positives
    /// - Recall: Ability to find all positive instances
    /// - F1Score: Balanced measure of precision and recall
    /// - AUC: Area under the ROC curve, measuring discrimination ability
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines which measurement is used to grade your model. 
    /// The default (F1Score) is a balanced measure that works well for most situations because it considers 
    /// both precision (how often the model is right when it makes a positive prediction) and recall (how many 
    /// of the actual positives the model found). Other options include:
    /// - Accuracy: Simple percentage of correct predictions (can be misleading with imbalanced data)
    /// - Precision: Focus on minimizing false positives (good when false alarms are costly)
    /// - Recall: Focus on finding all positive cases (good when missing a positive is costly, like disease detection)
    /// - AUC: How well the model distinguishes between classes (good for ranking performance)</para>
    /// </remarks>
    public MetricType PrimaryMetric { get; set; } = MetricType.F1Score;

    /// <summary>
    /// Gets or sets the confidence threshold used for converting probability predictions to class labels.
    /// </summary>
    /// <value>The confidence threshold as a decimal between 0 and 1, defaulting to 0.5 (50%).</value>
    /// <remarks>
    /// <para>
    /// Many classification models output probabilities rather than direct class predictions.
    /// This threshold determines the cutoff point for converting those probabilities to class labels.
    /// For binary classification, predictions with probability above this threshold are assigned to the positive class.
    /// </para>
    /// <para><b>For Beginners:</b> When your model makes predictions, it often calculates a confidence score 
    /// (like "I'm 75% sure this email is spam"). This setting determines how confident the model needs to be 
    /// before making a positive prediction. The default (0.5 or 50%) means that if the model is more than 50% 
    /// confident, it will make the positive prediction. You might increase this (like to 0.7) if false positives 
    /// are costly, or decrease it (like to 0.3) if false negatives are more problematic. For example, in spam 
    /// detection, you might use a higher threshold to avoid marking important emails as spam.</para>
    /// </remarks>
    public double ConfidenceThreshold { get; set; } = 0.5;
}
