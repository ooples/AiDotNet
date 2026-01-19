namespace AiDotNet.Models;

/// <summary>
/// Configuration options for prediction statistics generation, which provides statistical analysis
/// and reporting for model predictions including confidence intervals and learning curve analysis.
/// </summary>
/// <remarks>
/// <para>
/// The PredictionStatsOptions class controls how statistical information is calculated and presented
/// for model predictions. It enables the generation of confidence intervals to quantify prediction
/// uncertainty and learning curves to track model improvement over increasing training data sizes.
/// These statistical measures are crucial for understanding model reliability, evaluating prediction
/// robustness, and determining whether additional training data would improve model performance.
/// The statistical analysis is particularly valuable for applications in scientific research, 
/// decision support systems, and critical domains where understanding prediction uncertainty is essential.
/// </para>
/// <para><b>For Beginners:</b> Prediction statistics help you understand how reliable your model's predictions are and how your model improves with more data.
/// 
/// Think of prediction statistics like weather forecasting:
/// - Weather forecasts don't just say "tomorrow will be 75°F"
/// - They often say "75°F with a 90% chance of being between 72-78°F"
/// - They also show how forecast accuracy improves with more data points
/// 
/// What these statistics do:
/// 
/// 1. Confidence Intervals: Show the range where the true value is likely to fall
///    - Instead of a single prediction like "house price will be $300,000"
///    - You get "house price will be $300,000 ± $15,000 with 95% confidence"
///    - This helps you understand how certain or uncertain each prediction is
/// 
/// 2. Learning Curves: Show how your model improves as you give it more training data
///    - This helps you decide if collecting more data would help your model
///    - It can reveal if your model has reached its potential or needs more examples
/// 
/// This class lets you configure these statistical measures to better understand your model's performance.
/// </para>
/// </remarks>
public class PredictionStatsOptions
{
    /// <summary>
    /// Gets or sets the confidence level used for generating prediction confidence intervals.
    /// </summary>
    /// <value>The confidence level, defaulting to 0.95 (95%).</value>
    /// <remarks>
    /// <para>
    /// This parameter determines the confidence level used when calculating prediction intervals.
    /// The confidence level represents the probability that the true value falls within the calculated
    /// interval. For example, a 95% confidence level means that, over many predictions, approximately
    /// 95% of the calculated intervals will contain the true value. Higher confidence levels result
    /// in wider intervals, while lower confidence levels produce narrower intervals. The appropriate
    /// confidence level depends on the specific application's requirements for certainty versus precision.
    /// Common values are 0.90 (90%), 0.95 (95%), and 0.99 (99%).
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how certain you want to be about your prediction ranges.
    /// 
    /// The default value of 0.95 means:
    /// - You want to be 95% confident that the true value falls within your prediction range
    /// - Only 5% of the time should the actual value fall outside your predicted range
    /// 
    /// Think of it like setting the width of a safety net:
    /// - A higher confidence level (like 0.99) is a wider net - you're more likely to catch the true value, but your predictions are less precise
    /// - A lower confidence level (like 0.80) is a narrower net - you get more precise ranges, but are more likely to miss the true value
    /// 
    /// You might want a higher confidence level (like 0.99):
    /// - For critical applications where missing the true value is costly
    /// - In medical, financial, or safety-critical predictions
    /// - When you need to be very certain about the potential range of outcomes
    /// 
    /// You might want a lower confidence level (like 0.90 or 0.80):
    /// - When narrower, more precise prediction ranges are more valuable
    /// - In exploratory analysis where approximate ranges are sufficient
    /// - When communicating results to audiences who prefer precision over certainty
    /// 
    /// In statistical terms, this is equivalent to the significance level a = 1 - ConfidenceLevel
    /// (e.g., 95% confidence = 5% significance level).
    /// </para>
    /// </remarks>
    public double ConfidenceLevel { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets the number of steps used when generating learning curves.
    /// </summary>
    /// <value>The number of learning curve steps, defaulting to 10.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls the number of data points in the learning curve analysis. A learning curve
    /// shows model performance as a function of training set size, by training the model on progressively
    /// larger subsets of the training data. The LearningCurveSteps value determines how many different
    /// training set sizes will be evaluated. For example, with 10 steps and 1000 training examples,
    /// the model would be trained on approximately 100, 200, 300, ..., 1000 examples. More steps provide
    /// a more detailed curve but increase computation time. Fewer steps generate learning curves more
    /// quickly but may miss important trends in model improvement.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how many data points are used to create your learning curve.
    /// 
    /// The default value of 10 means:
    /// - Your training data will be divided into 10 progressively larger subsets
    /// - The model is trained on each subset (10%, 20%, 30%, ... 100% of your data)
    /// - Performance is measured for each subset to create the learning curve
    /// 
    /// Think of it like tracking your progress learning a new skill:
    /// - You could measure your skill after 1 week, 2 weeks, 3 weeks, etc.
    /// - More frequent measurements (more steps) give you a more detailed picture of your improvement
    /// - Fewer measurements are quicker but might miss important improvement patterns
    /// 
    /// You might want more steps (like 20 or 50):
    /// - When you have a large dataset and want detailed insight into learning patterns
    /// - When you suspect the learning process has interesting dynamics you want to capture
    /// - When you need a smooth, detailed curve for publication or presentation
    /// 
    /// You might want fewer steps (like 5):
    /// - When you have limited computational resources
    /// - When you only need a rough idea of the learning trend
    /// - When your dataset is small and more granular steps wouldn't be meaningful
    /// 
    /// The computational cost increases with more steps, as the model must be retrained
    /// multiple times on different data subsets.
    /// </para>
    /// </remarks>
    public int LearningCurveSteps { get; set; } = 10;
}
