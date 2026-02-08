namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for detecting overfitting, underfitting, and good fitting in machine learning models
/// using cross-validation techniques.
/// </summary>
/// <remarks>
/// <para>
/// The CrossValidationFitDetectorOptions class provides threshold settings that help determine whether a model
/// is overfitting (performing well on training data but poorly on validation data), underfitting (performing
/// poorly on both training and validation data), or has a good fit (performing well on both).
/// </para>
/// <para><b>For Beginners:</b> Think of these settings as the criteria for judging how well your AI model has learned.
/// Just like when learning a new skill, an AI can learn too little (underfit), memorize without understanding (overfit),
/// or learn just right (good fit). These thresholds help determine which category your model falls into by comparing
/// how it performs on data it has seen before (training data) versus new data (validation data).</para>
/// </remarks>
public class CrossValidationFitDetectorOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the threshold for detecting overfitting in a model.
    /// </summary>
    /// <value>The overfit threshold as a decimal between 0 and 1, defaulting to 0.1 (10%).</value>
    /// <remarks>
    /// <para>
    /// Overfitting is detected when the difference between training performance and validation performance
    /// exceeds this threshold. A smaller value makes the detector more sensitive to overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps detect when your model is "memorizing" the training data
    /// instead of learning general patterns. With the default value of 0.1, if your model performs more than 10%
    /// better on training data than on validation data, it's considered to be overfitting. This is like a student
    /// who memorizes answers for a practice test but can't apply the knowledge to slightly different questions
    /// on the real test.</para>
    /// </remarks>
    public double OverfitThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the threshold for detecting underfitting in a model.
    /// </summary>
    /// <value>The underfit threshold as a decimal between 0 and 1, defaulting to 0.7 (70%).</value>
    /// <remarks>
    /// <para>
    /// Underfitting is detected when the training performance is below this threshold. A higher value
    /// makes the detector more sensitive to underfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps detect when your model hasn't learned enough from the
    /// training data. With the default value of 0.7, if your model's performance on training data is below 70%
    /// of the maximum possible performance, it's considered to be underfitting. This is like a student who
    /// hasn't studied enough and performs poorly even on practice questions they've seen before.</para>
    /// </remarks>
    public double UnderfitThreshold { get; set; } = 0.7;

    /// <summary>
    /// Gets or sets the threshold for identifying a good fit in a model.
    /// </summary>
    /// <value>The good fit threshold as a decimal between 0 and 1, defaulting to 0.9 (90%).</value>
    /// <remarks>
    /// <para>
    /// A good fit is identified when both training and validation performance exceed this threshold,
    /// and the difference between them is less than the overfit threshold.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps identify when your model has learned well. With the default
    /// value of 0.9, if your model performs above 90% of the maximum possible performance on both training and
    /// validation data, and the difference between them is small, it's considered to have a good fit. This is like
    /// a student who understands the material deeply and performs well on both practice tests and new questions,
    /// showing they've truly learned the subject rather than just memorizing answers.</para>
    /// </remarks>
    public double GoodFitThreshold { get; set; } = 0.9;
}
