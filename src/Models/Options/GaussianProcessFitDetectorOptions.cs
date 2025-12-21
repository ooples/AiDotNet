namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Gaussian Process Fit Detector, which analyzes model fit quality
/// using Gaussian Process regression to detect overfitting, underfitting, and uncertainty issues.
/// </summary>
/// <remarks>
/// <para>
/// Gaussian Process Fit Detector uses Gaussian Process regression to analyze the residuals (differences
/// between predicted and actual values) of a model. By examining patterns in these residuals, it can
/// detect issues like overfitting (model captures noise rather than true patterns), underfitting
/// (model fails to capture important patterns), and areas of high uncertainty.
/// </para>
/// <para><b>For Beginners:</b> Think of this as a quality control tool that examines how well your
/// model's predictions match the actual data. It uses a special technique called Gaussian Process
/// regression that's particularly good at detecting patterns and measuring uncertainty. This detector
/// looks at the errors your model makes and determines whether they seem random (which is good) or
/// show patterns (which suggests problems). It can tell you if your model is too complex and memorizing
/// the data instead of learning general patterns (overfitting), or if it's too simple and missing
/// important patterns (underfitting). It can also identify areas where your model is particularly
/// uncertain about its predictions.</para>
/// </remarks>
public class GaussianProcessFitDetectorOptions
{
    /// <summary>
    /// Gets or sets the threshold for considering model fit as good based on normalized residual patterns.
    /// </summary>
    /// <value>The good fit threshold, defaulting to 0.1 (10%).</value>
    /// <remarks>
    /// <para>
    /// When the normalized measure of pattern strength in residuals is below this threshold, the model
    /// is considered to have a good fit. Lower values indicate more random residuals, suggesting the
    /// model has captured the true patterns in the data without fitting to noise.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines when your model is considered to be "just right" -
    /// not overfitting or underfitting. With the default value of 0.1, if the detector finds that the
    /// patterns in your model's errors are less than 10% of what would be expected by chance, it considers
    /// your model to have a good fit. Think of it like checking if the mistakes your model makes appear
    /// random rather than systematic. Random errors suggest your model has learned the true patterns in
    /// the data without memorizing the noise. If you want to be more strict about what counts as a good fit,
    /// you could lower this threshold (e.g., to 0.05).</para>
    /// </remarks>
    public double GoodFitThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the threshold for detecting overfitting based on residual patterns.
    /// </summary>
    /// <value>The overfit threshold, defaulting to 0.2 (20%).</value>
    /// <remarks>
    /// <para>
    /// When the normalized measure of high-frequency pattern strength in residuals exceeds this threshold,
    /// the model is considered to be overfitting. Overfitting occurs when the model captures noise in the
    /// training data rather than just the underlying patterns, leading to poor generalization.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps identify when your model is too complex and has started
    /// memorizing the training data instead of learning general patterns. With the default value of 0.2,
    /// if the detector finds strong high-frequency patterns in your model's errors that exceed 20% of a
    /// reference level, it flags potential overfitting. Think of it like a student who memorizes specific
    /// test questions instead of understanding the concepts - they'll do well on questions they've seen before
    /// but poorly on new questions. An overfitting model shows distinctive error patterns because it's trying
    /// too hard to fit every little fluctuation in the training data. If you want to be more sensitive to
    /// overfitting, you could lower this threshold.</para>
    /// </remarks>
    public double OverfitThreshold { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets the threshold for detecting underfitting based on residual patterns.
    /// </summary>
    /// <value>The underfit threshold, defaulting to 0.3 (30%).</value>
    /// <remarks>
    /// <para>
    /// When the normalized measure of low-frequency pattern strength in residuals exceeds this threshold,
    /// the model is considered to be underfitting. Underfitting occurs when the model is too simple to
    /// capture important patterns in the data, leading to systematic errors.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps identify when your model is too simple and missing
    /// important patterns in the data. With the default value of 0.3, if the detector finds strong
    /// low-frequency patterns in your model's errors that exceed 30% of a reference level, it flags
    /// potential underfitting. Think of it like using a straight line to approximate a curve - the line
    /// will systematically overestimate in some regions and underestimate in others, creating visible
    /// patterns in the errors. An underfitting model makes systematic mistakes because it lacks the
    /// complexity to capture the true relationships in the data. If you want to be more sensitive to
    /// underfitting, you could lower this threshold.</para>
    /// </remarks>
    public double UnderfitThreshold { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets the threshold for considering prediction uncertainty as low.
    /// </summary>
    /// <value>The low uncertainty threshold, defaulting to 0.1 (10%).</value>
    /// <remarks>
    /// <para>
    /// When the normalized uncertainty measure is below this threshold, the model's predictions are
    /// considered to have low uncertainty. Low uncertainty suggests the model is confident in its
    /// predictions, which is desirable if the predictions are also accurate.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines when your model's predictions are considered
    /// highly confident. With the default value of 0.1, if the uncertainty in predictions is less than
    /// 10% of a reference level, the model is considered to have low uncertainty. Think of it like a
    /// weather forecast that confidently predicts exactly 78°F tomorrow - it's making a very specific
    /// prediction with little hedging. Low uncertainty is generally good if the predictions are also
    /// accurate, but can be problematic if the model is confidently wrong. If you want your model to be
    /// more cautious about claiming high confidence, you could lower this threshold.</para>
    /// </remarks>
    public double LowUncertaintyThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the threshold for considering prediction uncertainty as high.
    /// </summary>
    /// <value>The high uncertainty threshold, defaulting to 0.5 (50%).</value>
    /// <remarks>
    /// <para>
    /// When the normalized uncertainty measure exceeds this threshold, the model's predictions are
    /// considered to have high uncertainty. High uncertainty suggests the model lacks confidence in
    /// its predictions, which may indicate insufficient training data in certain regions or inherent
    /// complexity in the relationship being modeled.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines when your model's predictions are considered
    /// highly uncertain. With the default value of 0.5, if the uncertainty in predictions exceeds 50%
    /// of a reference level, the model is considered to have high uncertainty. Think of it like a weather
    /// forecast that says "temperatures between 65-90°F tomorrow" - it's giving a very wide range because
    /// it's not confident in a specific prediction. High uncertainty often occurs in regions where you have
    /// little training data or where the relationship is inherently complex and variable. If you want to be
    /// more sensitive to detecting uncertain predictions, you could lower this threshold.</para>
    /// </remarks>
    public double HighUncertaintyThreshold { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the length scale parameter for the Gaussian Process kernel.
    /// </summary>
    /// <value>The length scale, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// The length scale controls how rapidly the correlation between points decreases with distance.
    /// Smaller values make the Gaussian Process more sensitive to small-scale variations, while larger
    /// values make it focus on broader trends. This parameter affects how the detector interprets
    /// patterns in the residuals.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how far the detector looks when identifying
    /// patterns in your model's errors. With the default value of 1.0, the detector uses a standard
    /// distance for determining whether nearby points should be related. Think of it like adjusting
    /// the focus on a camera - a smaller length scale (like 0.5) focuses on fine details and local
    /// patterns, while a larger length scale (like 2.0) blurs the details and focuses on broader trends.
    /// If your data has rapid changes or fine structure, a smaller length scale might be appropriate.
    /// If your data changes more gradually, a larger length scale might work better.</para>
    /// </remarks>
    public double LengthScale { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the assumed noise variance in the data.
    /// </summary>
    /// <value>The noise variance, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// This parameter represents the expected level of random noise in the data. Higher values assume
    /// more inherent randomness, making the detector more tolerant of small residuals. Lower values
    /// assume less noise, making the detector more sensitive to even small patterns in the residuals.
    /// </para>
    /// <para><b>For Beginners:</b> This setting tells the detector how much random noise to expect in
    /// your data. With the default value of 0.1, the detector assumes a moderate level of random
    /// fluctuation in your measurements. Think of it like listening for a conversation in a noisy room -
    /// if you expect a lot of background noise (high noise variance), you won't be concerned by small
    /// variations in what you hear. If you expect silence (low noise variance), even tiny sounds will
    /// seem significant. If your data comes from precise measurements with little random error, you
    /// might lower this value. If your data is naturally noisy with lots of random fluctuation, you
    /// might increase it.</para>
    /// </remarks>
    public double NoiseVariance { get; set; } = 0.1;
}
