namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Precision-Recall Curve Fit Detector, which evaluates model quality
/// using precision-recall metrics particularly valuable for imbalanced classification problems.
/// </summary>
/// <remarks>
/// <para>
/// The Precision-Recall Curve Fit Detector assesses model performance using metrics derived from the
/// precision-recall curve, which plots precision against recall at various classification thresholds.
/// Unlike accuracy, which can be misleading for imbalanced datasets, precision and recall metrics provide
/// more meaningful insights into model performance when class distributions are skewed. The Area Under
/// the Precision-Recall Curve (AUC-PR) and F1 Score are combined with customizable weights to produce
/// a composite fitness score. This detector is particularly valuable for applications where false positives
/// and false negatives have different implications, such as fraud detection, medical diagnosis, or anomaly
/// detection. The thresholds and weights configured in this class determine whether a model is considered
/// adequately fitted based on these metrics.
/// </para>
/// <para><b>For Beginners:</b> The Precision-Recall Curve Fit Detector helps evaluate how well your model is performing, especially when you have imbalanced data.
/// 
/// Imagine you're building a system to detect rare events (like fraud):
/// - You might have 1,000 normal transactions for every 1 fraudulent one
/// - A model that always predicts "not fraud" would be 99.9% accurate, but useless!
/// - This is why we need better ways to evaluate models with imbalanced data
/// 
/// Instead of simple accuracy, this detector uses two important metrics:
/// 
/// 1. Precision: When the model predicts something is positive (like fraud), how often is it correct?
///    - High precision means fewer false alarms
///    - Think of it as: "When the model raises an alert, how trustworthy is that alert?"
/// 
/// 2. Recall: Out of all the actual positive cases, how many did the model correctly identify?
///    - High recall means fewer missed cases
///    - Think of it as: "What percentage of fraudulent transactions did the model catch?"
/// 
/// The precision-recall curve shows the trade-off between these metrics at different thresholds.
/// This class lets you configure how the detector evaluates model quality based on these metrics.
/// </para>
/// </remarks>
public class PrecisionRecallCurveFitDetectorOptions
{
    /// <summary>
    /// Gets or sets the minimum acceptable Area Under the Precision-Recall Curve (AUC-PR) value.
    /// </summary>
    /// <value>The AUC-PR threshold, defaulting to 0.7.</value>
    /// <remarks>
    /// <para>
    /// This parameter defines the threshold for the Area Under the Precision-Recall Curve (AUC-PR)
    /// below which a model may be considered underperforming. The AUC-PR measures the overall
    /// quality of the model across all possible classification thresholds, with values ranging
    /// from 0 to 1. Higher values indicate better performance. Unlike the ROC curve, the baseline
    /// for the PR curve depends on the class imbalance ratio, making it particularly suitable for
    /// imbalanced datasets. The appropriate threshold depends on the specific domain, the degree
    /// of class imbalance, and the consequences of classification errors in the application.
    /// </para>
    /// <para><b>For Beginners:</b> This setting defines the minimum acceptable score for the Area Under the Precision-Recall Curve.
    /// 
    /// The default value of 0.7 means:
    /// - The model's AUC-PR score should be at least 0.7 (on a scale from 0 to 1)
    /// - Below this threshold, the model might be considered inadequate
    /// 
    /// Think of AUC-PR like a student's overall test score:
    /// - 1.0 is a perfect score (the model perfectly separates positive and negative cases)
    /// - 0.5 might be mediocre performance (for moderately imbalanced datasets)
    /// - 0.0 is terrible performance (the model gets everything wrong)
    /// - The default threshold of 0.7 is like requiring at least a "C" grade
    /// 
    /// You might want a higher threshold (like 0.8 or 0.9):
    /// - For critical applications where mistakes are costly
    /// - When you have high-quality data that should enable better models
    /// - When previous models have consistently achieved higher scores
    /// 
    /// You might accept a lower threshold (like 0.6 or 0.5):
    /// - For extremely imbalanced datasets where even good models have lower AUC-PR
    /// - In early stages of model development
    /// - When the problem is inherently difficult to predict
    /// 
    /// Note: Unlike accuracy, AUC-PR accounts for class imbalance, so a "good" score
    /// depends on your specific dataset and domain.
    /// </para>
    /// </remarks>
    public double AreaUnderCurveThreshold { get; set; } = 0.7;

    /// <summary>
    /// Gets or sets the minimum acceptable F1 Score for the model.
    /// </summary>
    /// <value>The F1 Score threshold, defaulting to 0.6.</value>
    /// <remarks>
    /// <para>
    /// This parameter defines the threshold for the F1 Score below which a model may be considered
    /// underperforming. The F1 Score is the harmonic mean of precision and recall, providing a
    /// single metric that balances these two aspects of classification performance. Values range
    /// from 0 to 1, with higher values indicating better performance. Unlike the AUC-PR, which
    /// evaluates performance across all possible thresholds, the F1 Score is typically calculated
    /// at a specific classification threshold (often 0.5). This makes it a more practical metric
    /// for evaluating model performance at the operating point that will be used in production.
    /// </para>
    /// <para><b>For Beginners:</b> This setting defines the minimum acceptable F1 Score for your model.
    /// 
    /// The default value of 0.6 means:
    /// - The model's F1 Score should be at least 0.6 (on a scale from 0 to 1)
    /// - Below this threshold, the model might be considered inadequate
    /// 
    /// The F1 Score combines precision and recall into a single number:
    /// - It gives you a balanced view of both metrics
    /// - It's especially useful when you care about both false positives and false negatives
    /// - A high F1 Score means both good precision AND good recall
    /// 
    /// Think of it like a balanced meal score:
    /// - You need both proteins (precision) and vegetables (recall) for a healthy meal
    /// - The F1 Score ensures you're not just loading up on one and ignoring the other
    /// - A score of 1.0 means perfect precision and recall
    /// - A score of 0.0 means either precision or recall (or both) is terrible
    /// 
    /// You might want a higher threshold (like 0.7 or 0.8):
    /// - When balanced performance is critical to your application
    /// - In mature systems where models should achieve good results on both metrics
    /// 
    /// You might accept a lower threshold (like 0.5 or 0.4):
    /// - In highly imbalanced datasets where even good models have lower F1 Scores
    /// - When you're more concerned about one metric (precision or recall) than perfect balance
    /// - In early development phases
    /// </para>
    /// </remarks>
    public double F1ScoreThreshold { get; set; } = 0.6;

    /// <summary>
    /// Gets or sets the weight applied to the Area Under the Precision-Recall Curve in the composite fitness score.
    /// </summary>
    /// <value>The AUC-PR weight, defaulting to 0.6.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls the relative importance of the Area Under the Precision-Recall Curve
    /// in the composite fitness score. The composite score is calculated as a weighted average
    /// of the AUC-PR and F1 Score, with AucWeight and F1ScoreWeight determining the contribution of
    /// each metric. Higher values give more influence to the AUC-PR, which evaluates model performance
    /// across all possible thresholds. This weighting should reflect the relative importance of
    /// overall model quality versus performance at the specific operating point in the application context.
    /// Note that AucWeight and F1ScoreWeight should sum to 1.0 to maintain a consistent scale.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much importance is given to the AUC-PR score when calculating the overall model quality.
    /// 
    /// The default value of 0.6 means:
    /// - The AUC-PR contributes 60% to the final quality score
    /// - The F1 Score contributes the remaining 40%
    /// 
    /// Think of it like grading a student:
    /// - AUC-PR is like their overall course performance across many assignments
    /// - F1 Score is like their performance on the final exam
    /// - This weight determines if you care more about consistent performance (AUC-PR) or performance at a specific threshold (F1 Score)
    /// 
    /// You might want a higher AUC weight (like 0.8):
    /// - When you want to reward models that perform well across many thresholds
    /// - When you're still exploring the best threshold to use in production
    /// - When you might need to adjust thresholds frequently based on changing conditions
    /// 
    /// You might want a lower AUC weight (like 0.4 or 0.2):
    /// - When you care more about performance at your specific operating threshold
    /// - When you have a fixed threshold that won't change in production
    /// - When optimizing for the F1 Score is more important in your application
    /// 
    /// Note: AucWeight and F1ScoreWeight should add up to 1.0 to maintain a proper scale.
    /// </para>
    /// </remarks>
    public double AucWeight { get; set; } = 0.6;

    /// <summary>
    /// Gets or sets the weight applied to the F1 Score in the composite fitness score.
    /// </summary>
    /// <value>The F1 Score weight, defaulting to 0.4.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls the relative importance of the F1 Score in the composite fitness score.
    /// The composite score is calculated as a weighted average of the AUC-PR and F1 Score, with
    /// AucWeight and F1ScoreWeight determining the contribution of each metric. Higher values give
    /// more influence to the F1 Score, which evaluates model performance at a specific operating point.
    /// This weighting should reflect the relative importance of performance at the specific operating
    /// point versus overall model quality across all thresholds. Note that AucWeight and F1ScoreWeight
    /// should sum to 1.0 to maintain a consistent scale.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much importance is given to the F1 Score when calculating the overall model quality.
    /// 
    /// The default value of 0.4 means:
    /// - The F1 Score contributes 40% to the final quality score
    /// - The AUC-PR contributes the remaining 60%
    /// 
    /// Think of it like hiring an employee:
    /// - The F1 Score is like how well they perform on the specific job tasks they'll do daily
    /// - AUC-PR is like their overall skill set across many potential tasks
    /// - This weight determines if you care more about specific job performance (F1 Score) or overall capabilities (AUC-PR)
    /// 
    /// You might want a higher F1 Score weight (like 0.6 or 0.8):
    /// - When you have a fixed classification threshold in production
    /// - When you care most about performance at that specific threshold
    /// - When balance between precision and recall at your operating point is critical
    /// 
    /// You might want a lower F1 Score weight (like 0.2):
    /// - When you want to emphasize overall model quality across all thresholds
    /// - When you frequently adjust classification thresholds
    /// - When exploring different models during development
    /// 
    /// Note: AucWeight and F1ScoreWeight should add up to 1.0 to maintain a proper scale.
    /// </para>
    /// </remarks>
    public double F1ScoreWeight { get; set; } = 0.4;
}
