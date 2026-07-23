using AiDotNet.Models.Options;

namespace AiDotNet.Models;

/// <summary>
/// Configuration options for the Residual Bootstrap Fit Detector, which uses bootstrap resampling
/// of residuals to assess model fit quality and detect overfitting or underfitting.
/// </summary>
/// <remarks>
/// <para>
/// Bootstrap resampling is a statistical technique that involves repeatedly sampling with replacement 
/// from the original dataset to estimate the sampling distribution of a statistic. The Residual Bootstrap 
/// Fit Detector applies this technique to model residuals (the differences between predicted and actual values) 
/// to assess whether a model is overfitting or underfitting the data. Overfitting occurs when a model learns 
/// the training data too well, including its noise, resulting in poor generalization to new data. Underfitting 
/// occurs when a model is too simple to capture the underlying patterns in the data. This class provides 
/// configuration options for the bootstrap process, including the number of bootstrap samples to generate, 
/// minimum sample size requirements, thresholds for detecting overfitting and underfitting, and an optional 
/// seed for reproducibility.
/// </para>
/// <para><b>For Beginners:</b> This class helps detect if your model is learning too much or too little from your data.
/// 
/// Two common problems in machine learning are:
/// - Overfitting: When your model learns the training data too well, including random noise
/// - Underfitting: When your model is too simple and misses important patterns in the data
/// 
/// This detector uses a technique called "bootstrap resampling" to check for these problems:
/// - It creates many random samples from your original data (with replacement)
/// - For each sample, it calculates how well the model performs
/// - By comparing performance across these samples, it can detect overfitting or underfitting
/// 
/// Think of it like testing a student:
/// - Overfitting is like memorizing the textbook but failing to understand the concepts
/// - Underfitting is like not studying enough and missing basic information
/// - This detector helps identify which problem your model might have
/// 
/// The settings in this class control how thoroughly and strictly this testing process works.
/// </para>
/// </remarks>
public class ResidualBootstrapFitDetectorOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the number of bootstrap samples to generate for the analysis.
    /// </summary>
    /// <value>A positive integer, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// This property determines how many bootstrap samples will be generated during the analysis. Each bootstrap 
    /// sample is created by randomly sampling with replacement from the original set of residuals. A larger number 
    /// of bootstrap samples provides more stable and reliable estimates of the sampling distribution, but requires 
    /// more computational resources. The default value of 1000 is generally sufficient for most applications to 
    /// provide stable estimates while maintaining reasonable computational efficiency. For very critical applications 
    /// or when extreme precision is required, this value might be increased, while for exploratory analysis or when 
    /// computational resources are limited, a smaller value might be used.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many random samples to create for testing your model.
    /// 
    /// The bootstrap method works by creating many random samples from your data:
    /// - Each sample is created by randomly selecting data points (with replacement)
    /// - "With replacement" means the same data point can be selected multiple times
    /// - More samples give more reliable results but take longer to compute
    /// 
    /// The default value of 1000 means:
    /// - The detector will create 1000 different random samples
    /// - It will test your model's performance on each sample
    /// - This gives a robust distribution of performance metrics
    /// 
    /// When to adjust this value:
    /// - Increase it (e.g., to 2000 or 5000) when you need very precise detection of overfitting/underfitting
    /// - Decrease it (e.g., to 500 or 200) when you need faster results and can accept slightly less precision
    /// - For most applications, the default of 1000 provides a good balance
    /// 
    /// This is similar to how polling works - the more people you survey, the more confident you can be
    /// in your results, but at some point, adding more responses gives diminishing returns.
    /// </para>
    /// </remarks>
    public int NumBootstrapSamples { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the minimum sample size required for bootstrap analysis.
    /// </summary>
    /// <value>A positive integer, defaulting to 30.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the minimum number of data points required to perform the bootstrap analysis. 
    /// Bootstrap methods rely on the Central Limit Theorem, which generally requires a sufficient sample size 
    /// to ensure that the sampling distribution of the mean approximates a normal distribution. The default 
    /// value of 30 is commonly used in statistics as a rule of thumb for when the Central Limit Theorem begins 
    /// to apply. If the available data has fewer points than this threshold, the detector may not perform the 
    /// analysis or may issue warnings about the reliability of the results. For very small datasets, this value 
    /// might be reduced, but with the understanding that the statistical validity of the results may be compromised.
    /// </para>
    /// <para><b>For Beginners:</b> This setting defines how much data you need for the bootstrap method to work reliably.
    /// 
    /// Statistical methods like bootstrapping work best with enough data:
    /// - Too few data points can lead to unreliable conclusions
    /// - The default minimum of 30 is based on a statistical rule of thumb
    /// - This ensures the results follow expected statistical properties
    /// 
    /// What this means in practice:
    /// - If your dataset has fewer than 30 points, the detector might not run
    /// - Or it might run but warn you that the results may not be reliable
    /// 
    /// When to adjust this value:
    /// - Decrease it (e.g., to 20 or 15) when you have limited data and understand the risks
    /// - Increase it (e.g., to 50 or 100) when you want more conservative, reliable results
    /// - Keep the default of 30 when you're unsure, as it's a widely accepted minimum
    /// 
    /// Think of it like sample size in a scientific study - larger samples give more reliable results,
    /// and there's a minimum size needed for the results to be meaningful.
    /// </para>
    /// </remarks>
    public int MinSampleSize { get; set; } = 30;

    /// <summary>
    /// Gets or sets the threshold for detecting overfitting in the model.
    /// </summary>
    /// <value>A double value, defaulting to 1.96 (approximately 95% confidence level).</value>
    /// <remarks>
    /// <para>
    /// This property defines the threshold used to detect overfitting in the model. The detector compares 
    /// standardized performance metrics between the training and validation datasets. If the standardized 
    /// difference exceeds this threshold, the model is considered to be overfitting. The default value of 1.96 
    /// corresponds to approximately the 95% confidence level in a two-tailed normal distribution, which is a 
    /// commonly used statistical threshold. A higher threshold makes the detector less sensitive to overfitting 
    /// (requiring stronger evidence), while a lower threshold makes it more sensitive (potentially flagging 
    /// overfitting more frequently, including false positives). The appropriate threshold may depend on the 
    /// specific application and the relative costs of false positives versus false negatives in overfitting detection.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how sensitive the detector is to overfitting.
    /// 
    /// Overfitting happens when your model performs much better on training data than on new data:
    /// - The detector compares performance between training and validation data
    /// - It calculates how significant the difference is (in statistical terms)
    /// - If this difference exceeds the threshold, it flags overfitting
    /// 
    /// The default value of 1.96:
    /// - Comes from statistics and represents a 95% confidence level
    /// - This means there's only a 5% chance of falsely detecting overfitting when it's not present
    /// - It's a balanced threshold that works well for most applications
    /// 
    /// When to adjust this value:
    /// - Increase it (e.g., to 2.58 for 99% confidence) when false alarms are costly
    /// - Decrease it (e.g., to 1.65 for 90% confidence) when you want to be more cautious about overfitting
    /// 
    /// For example, in a medical diagnosis model where overfitting could lead to missed diagnoses,
    /// you might use a lower threshold to be more sensitive to potential overfitting.
    /// </para>
    /// </remarks>
    public double OverfitThreshold { get; set; } = 1.96; // ~95% confidence level

    /// <summary>
    /// Gets or sets the threshold for detecting underfitting in the model.
    /// </summary>
    /// <value>A double value, defaulting to -1.96 (approximately 95% confidence level).</value>
    /// <remarks>
    /// <para>
    /// This property defines the threshold used to detect underfitting in the model. The detector compares 
    /// standardized performance metrics between the training and validation datasets. If the standardized 
    /// difference is below this threshold, the model is considered to be underfitting. The default value of -1.96 
    /// corresponds to approximately the 5% tail of a two-tailed normal distribution, which is a commonly used 
    /// statistical threshold. A lower (more negative) threshold makes the detector less sensitive to underfitting 
    /// (requiring stronger evidence), while a higher (less negative) threshold makes it more sensitive (potentially 
    /// flagging underfitting more frequently, including false positives). The appropriate threshold may depend on 
    /// the specific application and the relative costs of false positives versus false negatives in underfitting detection.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how sensitive the detector is to underfitting.
    /// 
    /// Underfitting happens when your model is too simple to capture important patterns:
    /// - The detector looks for cases where performance is poor on both training and validation data
    /// - It calculates how significant this poor performance is (in statistical terms)
    /// - If this significance exceeds the threshold, it flags underfitting
    /// 
    /// The default value of -1.96:
    /// - Is the negative version of the overfitting threshold
    /// - Also represents a 95% confidence level
    /// - The negative sign is because we're looking for performance that's significantly worse
    /// 
    /// When to adjust this value:
    /// - Decrease it (e.g., to -2.58) when you want to be less sensitive to underfitting
    /// - Increase it (e.g., to -1.65) when you want to be more sensitive to underfitting
    /// 
    /// For example, in a recommendation system where missing patterns means missed opportunities,
    /// you might use a higher (less negative) threshold to catch potential underfitting more aggressively.
    /// </para>
    /// </remarks>
    public double UnderfitThreshold { get; set; } = -1.96; // ~95% confidence level

    /// <summary>
    /// Gets or sets the random seed for the bootstrap sampling process.
    /// </summary>
    /// <value>A nullable integer, defaulting to null (no fixed seed).</value>
    /// <remarks>
    /// <para>
    /// This property allows setting a specific seed value for the random number generator used in the bootstrap 
    /// sampling process. Setting a fixed seed ensures that the random sampling is reproducible across different 
    /// runs, which can be important for debugging, validation, or when comparing different models under identical 
    /// conditions. When set to null (the default), the system will use a non-deterministic seed, resulting in 
    /// different random samples each time the detector is run. This is generally preferred for production use to 
    /// avoid any potential biases that might occur with a fixed seed, but makes exact reproduction of results 
    /// more difficult.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls whether the random sampling is reproducible.
    /// 
    /// The bootstrap method relies on random sampling, but sometimes you want consistent results:
    /// - Setting a seed value (any integer) makes the "random" sampling predictable
    /// - The same seed will always produce the same sequence of random samples
    /// - This is useful for debugging or when you need to reproduce exact results
    /// 
    /// The default value of null means:
    /// - No fixed seed is used
    /// - Each time you run the detector, you'll get slightly different results
    /// - This is generally good for production use as it avoids any potential bias
    /// 
    /// When to set a specific seed value:
    /// - During development and testing to get consistent results
    /// - When comparing different models to ensure they're tested on the same samples
    /// - When you need to reproduce exact results for verification
    /// 
    /// For example, if you're writing a research paper or technical report about your model,
    /// you might set a seed value so others can reproduce your exact findings.
    /// </para>
    /// </remarks>
    public new int? Seed { get; set; } = null;
}
