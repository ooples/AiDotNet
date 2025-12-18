namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Bootstrap Fit Detector, which evaluates model fit quality using bootstrap resampling.
/// </summary>
/// <remarks>
/// <para>
/// Bootstrap resampling is a statistical technique that creates multiple datasets by randomly sampling with replacement
/// from the original dataset. By training models on these resampled datasets and evaluating their performance,
/// we can assess how well a model generalizes and detect issues like overfitting or underfitting.
/// </para>
/// <para><b>For Beginners:</b> This class contains settings for a tool that helps you determine if your AI model is 
/// working well. It uses a technique called "bootstrapping" - imagine randomly picking data points from your dataset 
/// (sometimes picking the same point multiple times) to create many similar-but-different datasets. By training your 
/// model on these different datasets and seeing how consistent the results are, we can tell if your model is learning 
/// real patterns or just memorizing the training data.</para>
/// </remarks>
public class BootstrapFitDetectorOptions
{
    /// <summary>
    /// Gets or sets the number of bootstrap samples to generate for evaluation.
    /// </summary>
    /// <value>The number of bootstrap samples, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines how many resampled datasets will be created and evaluated.
    /// A higher number provides more reliable estimates but increases computation time.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many different random samples we'll create to test your model. 
    /// The default (1000) gives reliable results for most cases. Think of it like flipping a coin - if you flip it only 
    /// 10 times, you might get 7 heads and think the coin is biased. But if you flip it 1000 times, you'll get a much 
    /// more accurate picture of whether the coin is fair. More samples give more reliable results but take longer to process.</para>
    /// </remarks>
    public int NumberOfBootstraps { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the confidence interval used for statistical assessments.
    /// </summary>
    /// <value>The confidence interval as a decimal between 0 and 1, defaulting to 0.95 (95%).</value>
    /// <remarks>
    /// <para>
    /// The confidence interval represents the probability that the true parameter value falls within the estimated range.
    /// A 95% confidence interval (the default) means we are 95% confident that the true value lies within our estimated range.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how certain we want to be about our conclusions. The default 
    /// value of 0.95 means we want to be 95% confident. Think of it like a weather forecast - saying there's a 95% chance 
    /// of rain means you should probably bring an umbrella. Higher values (like 0.99) mean we want to be even more certain 
    /// before making a conclusion, while lower values (like 0.90) mean we're willing to be less certain.</para>
    /// </remarks>
    public double ConfidenceInterval { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets the threshold for detecting overfitting.
    /// </summary>
    /// <value>The overfit threshold as a decimal between 0 and 1, defaulting to 0.1 (10%).</value>
    /// <remarks>
    /// <para>
    /// If the difference between training performance and testing performance exceeds this threshold,
    /// the model is considered to be overfitting. Lower values make the detector more sensitive to overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> Overfitting happens when your model performs great on training data but poorly 
    /// on new data - it's like memorizing exam questions instead of understanding the subject. This setting determines 
    /// how big the difference between training and testing performance needs to be before we warn you about overfitting. 
    /// The default value (0.1) means if your model is more than 10% better on training data than on test data, we'll 
    /// flag it as potentially overfitting.</para>
    /// </remarks>
    public double OverfitThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the threshold for detecting underfitting.
    /// </summary>
    /// <value>The underfit threshold as a decimal between 0 and 1, defaulting to 0.7 (70%).</value>
    /// <remarks>
    /// <para>
    /// If the model's performance (on both training and testing data) is below this threshold,
    /// the model is considered to be underfitting. Higher values make the detector more sensitive to underfitting.
    /// </para>
    /// <para><b>For Beginners:</b> Underfitting happens when your model is too simple to capture the patterns in your data - 
    /// like trying to draw a circle using only straight lines. This setting determines the minimum performance level 
    /// your model needs to achieve to avoid being flagged as underfitting. The default value (0.7) means if your model's 
    /// accuracy is below 70% on both training and test data, we'll suggest that it might be underfitting and needs to be 
    /// more complex.</para>
    /// </remarks>
    public double UnderfitThreshold { get; set; } = 0.7;

    /// <summary>
    /// Gets or sets the threshold for identifying a good model fit.
    /// </summary>
    /// <value>The good fit threshold as a decimal between 0 and 1, defaulting to 0.9 (90%).</value>
    /// <remarks>
    /// <para>
    /// If the model's performance exceeds this threshold and there is no significant difference between
    /// training and testing performance, the model is considered to have a good fit.
    /// </para>
    /// <para><b>For Beginners:</b> This setting defines what we consider "good performance" for your model. The default 
    /// value (0.9) means we'll consider your model to be performing well if it achieves at least 90% accuracy and doesn't 
    /// show signs of overfitting. Think of it as setting the bar for what counts as success. You might lower this if your 
    /// problem is particularly difficult, or raise it if you need extremely high accuracy.</para>
    /// </remarks>
    public double GoodFitThreshold { get; set; } = 0.9;
}
