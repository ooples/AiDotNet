namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Gradient Boosting Fit Detector, which analyzes model fit quality
/// to detect overfitting in gradient boosting models.
/// </summary>
/// <remarks>
/// <para>
/// The Gradient Boosting Fit Detector monitors the performance difference between training and validation
/// data to identify when a gradient boosting model is overfitting. Overfitting occurs when a model performs
/// significantly better on training data than on new, unseen data, indicating that it has memorized the
/// training examples rather than learning generalizable patterns.
/// </para>
/// <para><b>For Beginners:</b> Think of this as a quality control tool specifically designed for gradient
/// boosting models (like XGBoost, LightGBM, or similar algorithms). These models are powerful but can easily
/// "memorize" your training data instead of learning general patterns. This detector helps you identify when
/// that's happening by comparing how well your model performs on the data it was trained on versus new data
/// it hasn't seen before.
/// 
/// It's like testing if someone truly understands a subject versus just memorizing answers to specific test
/// questions. If they score 95% on questions they've seen before but only 65% on new questions about the same
/// topic, they've probably memorized answers rather than understanding the subject. Similarly, this detector
/// helps identify when your model is "memorizing" rather than "learning," which would make it perform poorly
/// on new data in real-world applications.</para>
/// </remarks>
public class GradientBoostingFitDetectorOptions
{
    /// <summary>
    /// Gets or sets the threshold for detecting overfitting based on the difference between training and validation performance.
    /// </summary>
    /// <value>The overfit threshold, defaulting to 0.05 (5%).</value>
    /// <remarks>
    /// <para>
    /// When the difference between training and validation performance exceeds this threshold, the model is
    /// considered to be overfitting. This is typically measured as the relative difference in error metrics
    /// or performance scores between the two datasets.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines when your model is considered to be overfitting.
    /// With the default value of 0.05, if your model performs 5% better on the training data than on validation
    /// data, it's flagged as potentially overfitting. 
    /// 
    /// For example, if your model achieves 90% accuracy on training data but only 85% on validation data
    /// (a 5.6% relative difference), it would be flagged as overfitting. This suggests your model might be
    /// starting to memorize specific examples rather than learning general patterns.
    /// 
    /// If you want to be more strict about preventing overfitting, you could lower this threshold (e.g., to 0.03).
    /// If you're willing to accept more potential overfitting for the sake of model performance, you could
    /// increase it (e.g., to 0.07).</para>
    /// </remarks>
    public double OverfitThreshold { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets the threshold for detecting severe overfitting based on the difference between training and validation performance.
    /// </summary>
    /// <value>The severe overfit threshold, defaulting to 0.1 (10%).</value>
    /// <remarks>
    /// <para>
    /// When the difference between training and validation performance exceeds this threshold, the model is
    /// considered to be severely overfitting. This indicates a more serious problem that likely requires
    /// immediate attention through regularization, early stopping, or model simplification.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines when your model is considered to be severely
    /// overfitting. With the default value of 0.1, if your model performs 10% better on the training data
    /// than on validation data, it's flagged as having a serious overfitting problem.
    /// 
    /// For example, if your model achieves 95% accuracy on training data but only 85% on validation data
    /// (an 11.8% relative difference), it would be flagged as severely overfitting. This suggests your model
    /// has definitely memorized specific examples rather than learning general patterns, and you should take
    /// immediate action.
    /// 
    /// When severe overfitting is detected, you should consider:
    /// - Adding more regularization to your model
    /// - Using early stopping to prevent additional overfitting
    /// - Simplifying your model by reducing its complexity
    /// - Gathering more training data
    /// - Using techniques like cross-validation to get more reliable performance estimates</para>
    /// </remarks>
    public double SevereOverfitThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the threshold for considering model fit as good based on the similarity of training and validation performance.
    /// </summary>
    /// <value>The good fit threshold, defaulting to 0.1 (10%).</value>
    /// <remarks>
    /// <para>
    /// When the absolute difference between training and validation performance is below this threshold,
    /// the model is considered to have a good fit. This indicates that the model is generalizing well
    /// from the training data to unseen data.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines when your model is considered to have a good fit.
    /// With the default value of 0.1, if the absolute difference between your model's performance on training
    /// and validation data is less than 10%, it's considered to be fitting well.
    /// 
    /// For example, if your model achieves 87% accuracy on training data and 85% on validation data
    /// (a 2.3% relative difference), it would be flagged as having a good fit. This suggests your model is
    /// learning general patterns that apply well to new data, rather than just memorizing the training examples.
    /// 
    /// A good fit means your model is likely to perform similarly on new, unseen data as it does on your
    /// training data, which is exactly what we want in machine learning. If your model has a good fit but
    /// the overall performance is still not satisfactory, you might need a more complex model or better
    /// features, rather than worrying about overfitting.</para>
    /// </remarks>
    public double GoodFitThreshold { get; set; } = 0.1;
}
