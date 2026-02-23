using AiDotNet.Models.Options;

namespace AiDotNet.Models;

/// <summary>
/// Configuration options for the ROC Curve Fit Detector, which evaluates classification model quality
/// using Receiver Operating Characteristic (ROC) curve analysis.
/// </summary>
/// <remarks>
/// <para>
/// The ROC Curve Fit Detector assesses classification model performance by analyzing the Receiver Operating 
/// Characteristic (ROC) curve and its associated Area Under the Curve (AUC) metric. The ROC curve plots the 
/// True Positive Rate against the False Positive Rate at various classification thresholds, providing a 
/// comprehensive view of classifier performance across all possible decision thresholds. The AUC value ranges 
/// from 0 to 1, where 1 represents a perfect classifier and 0.5 represents a classifier that performs no better 
/// than random guessing. This class provides configuration options for thresholds that determine what constitutes 
/// good, moderate, and poor model fit based on AUC values, as well as parameters for confidence scaling and 
/// handling class imbalance. These settings allow users to customize the fit detection criteria according to 
/// their specific application requirements and domain knowledge.
/// </para>
/// <para><b>For Beginners:</b> This class helps evaluate how well your classification model performs.
/// 
/// Classification models predict categories (like "spam/not spam" or "will buy/won't buy"):
/// - We need ways to measure how good these predictions are
/// - The ROC curve is a powerful tool for this evaluation
/// - It shows the tradeoff between correctly identifying positives and incorrectly flagging negatives
/// 
/// The key metric is AUC (Area Under the Curve):
/// - AUC ranges from 0 to 1
/// - 1.0 means perfect predictions
/// - 0.5 means the model is no better than random guessing
/// - Values below 0.5 suggest the model is worse than guessing
/// 
/// This class lets you set thresholds for what AUC values are considered:
/// - Good performance
/// - Moderate performance
/// - Poor performance
/// 
/// It also includes settings to adjust for confidence levels and data imbalance (when you have
/// many more examples of one category than another).
/// </para>
/// </remarks>
public class ROCCurveFitDetectorOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the AUC threshold for considering a model to have good fit.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.8.</value>
    /// <remarks>
    /// <para>
    /// This property defines the minimum Area Under the ROC Curve (AUC) value required for a classification 
    /// model to be considered as having a good fit. AUC values range from 0 to 1, with higher values indicating 
    /// better classifier performance. The default threshold of 0.8 is commonly used in practice to indicate good 
    /// discriminative ability. Models with AUC values above this threshold are generally considered to have strong 
    /// predictive power and reliable performance across different decision thresholds. The appropriate threshold 
    /// may vary depending on the specific domain, the consequences of misclassification, and the baseline performance 
    /// for the particular problem being addressed.
    /// </para>
    /// <para><b>For Beginners:</b> This setting defines what AUC value is needed for your model to be considered "good."
    /// 
    /// The default value of 0.8 means:
    /// - Models with AUC = 0.8 are considered to have good performance
    /// - This is a commonly used threshold in many fields
    /// - It indicates the model is correct about 80% of the time (roughly speaking)
    /// 
    /// In practical terms:
    /// - AUC of 0.8: If you randomly select one positive case and one negative case, the model will correctly 
    ///   identify which is which 80% of the time
    /// - This level of performance is usually considered strong in most applications
    /// 
    /// When to adjust this value:
    /// - Increase it (e.g., to 0.85 or 0.9) for critical applications where high accuracy is essential
    ///   (like medical diagnostics or fraud detection)
    /// - Decrease it (e.g., to 0.75) for problems that are inherently difficult to predict
    ///   or where the current state-of-the-art performance is lower
    /// 
    /// For example, in email spam detection, an AUC of 0.8 would indicate a model that does a good job
    /// of separating spam from legitimate emails across various threshold settings.
    /// </para>
    /// </remarks>
    public double GoodFitThreshold { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets the AUC threshold for considering a model to have moderate fit.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.7.</value>
    /// <remarks>
    /// <para>
    /// This property defines the minimum Area Under the ROC Curve (AUC) value required for a classification 
    /// model to be considered as having a moderate fit. Models with AUC values below the GoodFitThreshold but 
    /// above this threshold are classified as having moderate performance. The default value of 0.7 represents 
    /// a classifier with acceptable discriminative ability that may be useful in many applications, particularly 
    /// when combined with other information or when perfect classification is not required. As with the 
    /// GoodFitThreshold, the appropriate value for this threshold may depend on the specific domain and application 
    /// requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This setting defines what AUC value is needed for your model to be considered "moderate."
    /// 
    /// The default value of 0.7 means:
    /// - Models with AUC between 0.7 and 0.8 are considered to have moderate performance
    /// - This range indicates useful but not exceptional predictive power
    /// 
    /// In practical terms:
    /// - AUC of 0.7: The model will correctly rank a random positive case above a random negative case 
    ///   70% of the time
    /// - This level of performance is acceptable for many applications, especially when:
    ///   * The problem is inherently difficult
    ///   * The model is one of several decision inputs
    ///   * Perfect accuracy isn't critical
    /// 
    /// When to adjust this value:
    /// - Increase it if you want to be more stringent about what constitutes "moderate" performance
    /// - Decrease it if you're working in a domain where even modest predictive power is valuable
    /// 
    /// For example, in customer churn prediction, an AUC of 0.75 might be considered moderate -
    /// useful for identifying customers at risk of leaving, but not reliable enough to be the
    /// sole basis for expensive retention campaigns.
    /// </para>
    /// </remarks>
    public double ModerateFitThreshold { get; set; } = 0.7;

    /// <summary>
    /// Gets or sets the AUC threshold for considering a model to have poor fit.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.6.</value>
    /// <remarks>
    /// <para>
    /// This property defines the minimum Area Under the ROC Curve (AUC) value required for a classification 
    /// model to be considered as having a poor (but potentially still useful) fit. Models with AUC values below 
    /// the ModerateFitThreshold but above this threshold are classified as having poor performance. Models with 
    /// AUC values below this threshold are considered to have very poor performance, potentially no better than 
    /// random guessing (especially as the AUC approaches 0.5). The default value of 0.6 represents a classifier 
    /// with limited discriminative ability that may still provide some information in certain contexts but should 
    /// generally be improved or replaced. As with the other thresholds, the appropriate value may depend on the 
    /// specific domain and application requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This setting defines what AUC value is needed for your model to be considered "poor" rather than "very poor."
    /// 
    /// The default value of 0.6 means:
    /// - Models with AUC between 0.6 and 0.7 are considered to have poor performance
    /// - Models with AUC below 0.6 are considered to have very poor performance
    /// - As AUC approaches 0.5, the model becomes no better than random guessing
    /// 
    /// In practical terms:
    /// - AUC of 0.6: The model is only slightly better than chance
    /// - This level of performance indicates the model has found some patterns in the data,
    ///   but not enough to be reliably useful in most applications
    /// 
    /// When to adjust this value:
    /// - Increase it if you want to be more stringent about what constitutes usable performance
    /// - Decrease it (closer to 0.5) if even slight improvements over random guessing are valuable in your domain
    /// 
    /// For example, in a difficult problem like predicting stock market movements, even an AUC of 0.55
    /// might be considered valuable, so you might lower this threshold to 0.55 in such cases.
    /// </para>
    /// </remarks>
    public double PoorFitThreshold { get; set; } = 0.6;

    /// <summary>
    /// Gets or sets the scaling factor for confidence intervals when evaluating model fit.
    /// </summary>
    /// <value>A positive double value, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// This property allows adjusting the width of confidence intervals used when evaluating the statistical 
    /// significance of AUC values. A larger scaling factor results in wider confidence intervals, making the 
    /// detector more conservative in its assessments (requiring stronger evidence to classify a model as having 
    /// good fit). A smaller scaling factor results in narrower confidence intervals, making the detector more 
    /// liberal in its assessments. The default value of 1.0 uses standard confidence intervals. This parameter 
    /// is particularly useful when working with smaller datasets, where AUC estimates may have higher variance, 
    /// or in applications where the cost of incorrectly assessing model fit is asymmetric.
    /// </para>
    /// <para><b>For Beginners:</b> This setting adjusts how certain the detector needs to be before classifying a model's performance.
    /// 
    /// AUC values have statistical uncertainty, especially with smaller datasets:
    /// - The detector calculates confidence intervals around the AUC estimate
    /// - These intervals represent the range where the true AUC likely falls
    /// - This scaling factor adjusts the width of these intervals
    /// 
    /// The default value of 1.0 means:
    /// - Standard statistical confidence intervals are used
    /// - This provides a balanced approach to uncertainty
    /// 
    /// When to adjust this value:
    /// - Increase it (e.g., to 1.5 or 2.0) to be more conservative, requiring more certainty before
    ///   classifying a model as "good" or "moderate"
    /// - Decrease it (e.g., to 0.8 or 0.5) to be more lenient, accepting more uncertainty in the classification
    /// 
    /// For example, in a high-stakes application like medical diagnosis, you might increase this value
    /// to ensure that only models with very reliable performance are classified as "good."
    /// 
    /// This is an advanced setting that most beginners can leave at the default value unless they
    /// have specific reasons to adjust the confidence level.
    /// </para>
    /// </remarks>
    public double ConfidenceScalingFactor { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the threshold for determining if a dataset is balanced.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.5.</value>
    /// <remarks>
    /// <para>
    /// This property defines what proportion of the minority class is required for a dataset to be considered 
    /// balanced. Class imbalance can affect the interpretation of ROC curves and AUC values, as well as the 
    /// appropriate thresholds for good, moderate, and poor fit. The default value of 0.5 considers a dataset 
    /// balanced if the minority class represents at least 50% of the size of the majority class (i.e., the 
    /// classes are in a ratio of at least 1:2). For highly imbalanced datasets, alternative metrics like 
    /// Precision-Recall curves or balanced accuracy might be more appropriate than ROC curves. This threshold 
    /// allows the detector to adjust its evaluation criteria or provide warnings when working with imbalanced data.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps the detector account for imbalanced datasets.
    /// 
    /// In classification, you often have two classes (like "fraud/not fraud" or "click/no click"):
    /// - When one class is much more common than the other, the dataset is "imbalanced"
    /// - This imbalance can affect how we interpret model performance
    /// - ROC curves are generally robust to imbalance, but extreme imbalance can still be problematic
    /// 
    /// The default value of 0.5 means:
    /// - The dataset is considered "balanced enough" if the minority class is at least half the size of the majority class
    /// - For example, if you have 100 positive examples and 200 negative examples (a 1:2 ratio)
    /// - This is a relatively strict definition of balance
    /// 
    /// When to adjust this value:
    /// - Increase it (e.g., to 0.7 or 0.8) to require more balanced datasets before applying standard interpretations
    /// - Decrease it (e.g., to 0.3 or 0.2) to be more tolerant of imbalance
    /// 
    /// For example, in fraud detection where fraudulent transactions might be only 1% of all transactions,
    /// you might lower this threshold to 0.1 or less to acknowledge that extreme imbalance is expected.
    /// 
    /// When datasets are identified as imbalanced, you might want to consider additional metrics beyond
    /// AUC, such as precision-recall curves or F1 scores.
    /// </para>
    /// </remarks>
    public double BalancedDatasetThreshold { get; set; } = 0.5;
}
