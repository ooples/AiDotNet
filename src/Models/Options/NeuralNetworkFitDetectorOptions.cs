namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Neural Network Fit Detector, which evaluates the quality of a neural network's
/// fit to data by analyzing performance metrics and detecting issues like underfitting and overfitting.
/// </summary>
/// <remarks>
/// <para>
/// The Neural Network Fit Detector provides automated detection and classification of model fit quality
/// based on configurable thresholds. It analyzes the discrepancy between training and validation performance
/// to identify issues such as underfitting (poor performance on both training and validation data) and
/// overfitting (good performance on training data but poor performance on validation data). This tool
/// helps data scientists and machine learning engineers quickly assess model quality and make informed
/// decisions about architecture adjustments, regularization techniques, or data augmentation strategies.
/// </para>
/// <para><b>For Beginners:</b> The Neural Network Fit Detector helps you understand if your AI model is learning properly.
/// 
/// When training an AI model, three common problems can occur:
/// - Underfitting: The model is too simple and performs poorly on all data
/// - Overfitting: The model has "memorized" the training data instead of learning general patterns
/// - Just Right: The model has learned general patterns that work well on new data
/// 
/// Imagine you're teaching someone to play chess:
/// - Underfitting is like they only learned how pieces move, but no strategies
/// - Overfitting is like they memorized specific games but can't adapt to new situations
/// - Just Right is when they learned general principles they can apply to any game
/// 
/// This class lets you set thresholds that determine:
/// - What level of error is acceptable for a "good" model
/// - When a model is considered "moderately good"
/// - When a model is performing poorly
/// - When a model shows signs of overfitting (performing much better on training data than on new data)
/// 
/// These thresholds help automatically classify models during development and training,
/// so you can quickly identify and fix issues in your neural network.
/// </para>
/// </remarks>
public class NeuralNetworkFitDetectorOptions
{
    /// <summary>
    /// Gets or sets the maximum error threshold for classifying a model's fit as "good".
    /// </summary>
    /// <value>The good fit threshold, defaulting to 0.05 (5%).</value>
    /// <remarks>
    /// <para>
    /// This threshold defines the maximum acceptable error rate for a model to be classified as having a
    /// "good" fit. Error rates below this threshold indicate that the model has successfully learned
    /// patterns in the data and is generalizing well. The specific metric used for the error rate
    /// depends on the task (e.g., mean squared error for regression, cross-entropy for classification),
    /// but in all cases, lower values indicate better performance. This threshold should be set based
    /// on domain-specific requirements and the quality standards for your particular application.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how accurate a model must be
    /// to be considered "good."
    /// 
    /// The default value of 0.05 means:
    /// - If the model's error rate is below 5%
    /// - It's classified as having a "good" fit
    /// 
    /// Think of it like a test score:
    /// - A "good" model gets at least 95% of answers correct (error less than 5%)
    /// 
    /// You might want a lower value (stricter, like 0.03) if:
    /// - Your application requires very high accuracy
    /// - You're working with a relatively simple problem
    /// - You have high-quality, consistent data
    /// 
    /// You might want a higher value (more lenient, like 0.08) if:
    /// - Your problem is inherently noisy or difficult
    /// - You're working with limited data
    /// - Perfect predictions aren't required for your application
    /// 
    /// The right threshold depends on your specific problem - what's "good" for predicting
    /// stock prices might be different from what's "good" for image classification.
    /// </para>
    /// </remarks>
    public double GoodFitThreshold { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets the maximum error threshold for classifying a model's fit as "moderate".
    /// </summary>
    /// <value>The moderate fit threshold, defaulting to 0.1 (10%).</value>
    /// <remarks>
    /// <para>
    /// This threshold defines the maximum acceptable error rate for a model to be classified as having
    /// a "moderate" fit. Error rates below this threshold but above the GoodFitThreshold indicate that
    /// the model has learned useful patterns but may benefit from further refinement. Models in this
    /// category typically demonstrate reasonable performance but might be improved through additional
    /// training, architectural adjustments, or data preprocessing techniques. This intermediate category
    /// helps distinguish models that require minor refinements from those that need significant rework.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines the threshold for a model to be
    /// considered "moderately good."
    /// 
    /// The default value of 0.1 means:
    /// - If the model's error rate is between 5% and 10%
    /// - It's classified as having a "moderate" fit
    /// 
    /// Continuing with our test score analogy:
    /// - A "moderate" model gets between 90% and 95% of answers correct
    /// 
    /// You might want a lower value (like 0.07) if:
    /// - You have high standards for your application
    /// - You want to push more models into the "poor" category for further improvement
    /// - The problem is well-understood and solvable
    /// 
    /// You might want a higher value (like 0.15) if:
    /// - The problem is challenging with inherent uncertainty
    /// - You want more tolerance for moderately-performing models
    /// - You need to balance accuracy with other considerations like model simplicity
    /// 
    /// Models in the "moderate" category are usually worth keeping and refining further,
    /// while those below this threshold might need more substantial redesign.
    /// </para>
    /// </remarks>
    public double ModerateFitThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the maximum error threshold for classifying a model's fit as "poor" rather than "very poor".
    /// </summary>
    /// <value>The poor fit threshold, defaulting to 0.2 (20%).</value>
    /// <remarks>
    /// <para>
    /// This threshold defines the boundary between "poor" and "very poor" model performance. Models with
    /// error rates below this threshold but above the ModerateFitThreshold are classified as having a
    /// "poor" fit, while those exceeding this threshold are classified as "very poor". Models in the
    /// "poor" category typically show some learning but fail to capture important patterns in the data,
    /// while "very poor" models may be no better than random guessing or simple heuristics. This
    /// distinction helps prioritize which underperforming models require complete redesign versus
    /// those that might be salvageable with substantial modifications.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines when a model moves from
    /// "poor" to "very poor" classification.
    /// 
    /// The default value of 0.2 means:
    /// - If the model's error rate is between 10% and 20%
    /// - It's classified as having a "poor" fit
    /// - If the error rate exceeds 20%
    /// - It's classified as having a "very poor" fit
    /// 
    /// In test score terms:
    /// - A "poor" model gets between 80% and 90% of answers correct
    /// - A "very poor" model gets less than 80% correct
    /// 
    /// You might want a lower value (like 0.15) if:
    /// - Even moderately bad performance is unacceptable
    /// - You want to identify more models as needing complete redesign
    /// - The baseline for your problem is already decent
    /// 
    /// You might want a higher value (like 0.3) if:
    /// - The problem is extremely difficult or noisy
    /// - You're dealing with very limited or low-quality data
    /// - You want to give more models a chance for improvement before complete rejection
    /// 
    /// This threshold helps you decide which underperforming models might be worth
    /// trying to improve versus which ones should be completely rethought.
    /// </para>
    /// </remarks>
    public double PoorFitThreshold { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets the threshold for detecting overfitting based on the difference between training
    /// and validation performance.
    /// </summary>
    /// <value>The overfitting threshold, defaulting to 0.2 (20%).</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when a model is classified as overfitting by measuring the relative
    /// difference between training and validation error rates. If the validation error exceeds the
    /// training error by this percentage or more, the model is flagged as overfitting. Overfitting
    /// occurs when a model learns the training data too precisely, including noise and outliers,
    /// resulting in poor generalization to new data. Early detection of overfitting allows for timely
    /// application of regularization techniques, data augmentation, or early stopping to improve
    /// model generalizability.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps detect when a model has "memorized"
    /// the training data instead of learning general patterns.
    /// 
    /// The default value of 0.2 means:
    /// - If the model performs 20% or more better on training data than on validation data
    /// - It's flagged as "overfitting"
    /// 
    /// Using our chess learning analogy:
    /// - Overfitting is like a player who performs great when replaying games they've studied
    /// - But performs much worse when playing new opponents with different strategies
    /// 
    /// You might want a lower value (like 0.1) if:
    /// - You want to be very conservative about overfitting
    /// - Your application requires consistent performance across all data
    /// - You're working with limited validation data
    /// 
    /// You might want a higher value (like 0.3) if:
    /// - Some overfitting is acceptable in your application
    /// - Your training and validation data naturally have different characteristics
    /// - You want to focus on other issues before addressing mild overfitting
    /// 
    /// Detecting overfitting early helps you apply the right techniques (like regularization,
    /// dropout, or early stopping) to improve your model's ability to generalize to new data.
    /// </para>
    /// </remarks>
    public double OverfittingThreshold { get; set; } = 0.2;
}
