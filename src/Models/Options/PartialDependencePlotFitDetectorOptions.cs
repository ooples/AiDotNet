namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Partial Dependence Plot Fit Detector, which uses partial dependence plots
/// to evaluate model fit quality and detect overfitting or underfitting in machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// Partial Dependence Plots (PDPs) visualize the relationship between a target variable and a set of input 
/// features of interest, marginalizing over the values of all other input features. The Partial Dependence 
/// Plot Fit Detector leverages these plots to assess model quality by comparing the complexity and smoothness 
/// of the learned relationships. This approach provides valuable insights into whether a model has captured 
/// meaningful patterns (good fit), is too simplistic (underfit), or has learned noise in the training data 
/// (overfit). By analyzing the characteristics of these plots across different features, the detector can 
/// provide early warnings of potential modeling issues that might not be apparent from aggregate metrics alone.
/// </para>
/// <para><b>For Beginners:</b> The Partial Dependence Plot Fit Detector helps identify if your model is learning properly
/// by examining how it responds to changes in individual input features.
/// 
/// Imagine you built a model to predict house prices based on features like square footage, location, and age:
/// - A partial dependence plot shows how the predicted price changes when you vary just one feature (like square footage)
///   while keeping all other features at their average values
/// 
/// These plots help reveal three common problems:
/// - Overfitting: When your model learns patterns that are too specific to your training data
///   (like a jagged, noisy relationship between square footage and price)
/// - Underfitting: When your model is too simple and misses important patterns
///   (like a flat line that shows no relationship between square footage and price)
/// - Good fit: When your model captures meaningful patterns without excess complexity
///   (like a smooth curve showing prices increasing with square footage, but at a decreasing rate)
/// 
/// This class lets you configure how the detector analyzes these plots to automatically
/// identify potential fitting problems in your models.
/// </para>
/// </remarks>
public class PartialDependencePlotFitDetectorOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the threshold for detecting overfitting based on the variability in partial dependence plots.
    /// </summary>
    /// <value>The overfit threshold, defaulting to 0.8.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines the threshold above which a model is considered to be overfitting based on
    /// the analysis of its partial dependence plots. Overfitting is detected by measuring the irregularity,
    /// jaggedness, and noise in the partial dependence curves. Higher variability between adjacent points
    /// in the plot suggests the model may be fitting to noise rather than true patterns in the data. The value
    /// represents a normalized score where higher values correspond to stricter detection of overfitting.
    /// Appropriate values depend on the domain and the expected smoothness of the true underlying relationships.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how "jagged" or "noisy" a partial
    /// dependence plot can be before the model is flagged as overfitting.
    /// 
    /// The default value of 0.8 means:
    /// - If the variability score in the plot exceeds 0.8 (on a scale from 0 to 1)
    /// - The detector will flag the model as potentially overfitting
    /// 
    /// Think of it like looking at a line representing house prices vs. square footage:
    /// - A smooth line suggests the model has found a general pattern
    /// - A very jagged, zig-zagging line suggests the model is trying to fit every small fluctuation in the data
    /// 
    /// You might want a higher value (like 0.9) if:
    /// - Your data naturally has some irregular patterns that aren't noise
    /// - You're working in a domain where relationships are inherently complex
    /// - You want to be more conservative about flagging potential overfitting
    /// 
    /// You might want a lower value (like 0.7) if:
    /// - You strongly prefer simpler, more generalizable models
    /// - Your domain typically has smooth underlying relationships
    /// - You want to be more aggressive in detecting potential overfitting
    /// 
    /// This threshold helps automate the process of reviewing partial dependence plots,
    /// which data scientists often examine visually to assess model quality.
    /// </para>
    /// </remarks>
    public double OverfitThreshold { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets the threshold for detecting underfitting based on the flatness in partial dependence plots.
    /// </summary>
    /// <value>The underfit threshold, defaulting to 0.2.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines the threshold below which a model is considered to be underfitting based on
    /// the analysis of its partial dependence plots. Underfitting is detected by measuring the overall variation
    /// and responsiveness of the model to changes in the input features. Flatter, less responsive curves suggest
    /// the model may be too simple and failing to capture important relationships in the data. The value
    /// represents a normalized score where lower values correspond to stricter detection of underfitting.
    /// The appropriate threshold depends on how much feature influence is expected in the domain.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how "flat" or "unresponsive" a partial
    /// dependence plot can be before the model is flagged as underfitting.
    /// 
    /// The default value of 0.2 means:
    /// - If the variability score in the plot is below 0.2 (on a scale from 0 to 1)
    /// - The detector will flag the model as potentially underfitting
    /// 
    /// Continuing with our house price example:
    /// - If changing the square footage barely affects the predicted price (flat line)
    /// - When we know square footage should significantly impact house prices
    /// - This suggests the model is underfitting and not capturing important relationships
    /// 
    /// You might want a lower value (like 0.1) if:
    /// - Some features genuinely have minimal impact on your target variable
    /// - You're working with many potentially irrelevant features
    /// - You want to be more conservative about flagging potential underfitting
    /// 
    /// You might want a higher value (like 0.3) if:
    /// - You expect all features to have meaningful impact on your target
    /// - You've already performed feature selection to remove irrelevant variables
    /// - You want to be more aggressive in detecting potential underfitting
    /// 
    /// This threshold helps identify when your model might be too simple or failing to learn
    /// the patterns that exist in your data.
    /// </para>
    /// </remarks>
    public double UnderfitThreshold { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets the number of points to sample for generating each partial dependence plot.
    /// </summary>
    /// <value>The number of points to sample, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines the resolution of the partial dependence plots used for fit detection.
    /// Each plot is generated by evaluating the model at a specified number of points across the range
    /// of each feature of interest. More points provide higher resolution for detecting patterns and
    /// irregularities but require more computation. The appropriate number depends on the complexity
    /// of the relationships being modeled and the computational resources available. For most applications,
    /// 100 points provides a good balance between resolution and computational efficiency.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many sample points are used
    /// to create each partial dependence plot.
    /// 
    /// The default value of 100 means:
    /// - The feature range is divided into 100 equally spaced points
    /// - The model is evaluated at each of these points to create the plot
    /// 
    /// Think of it like photograph resolution:
    /// - More points (higher resolution) show more detail but require more processing
    /// - Fewer points (lower resolution) are faster but might miss some details
    /// 
    /// You might want more points (like 200) if:
    /// - You're analyzing complex models with potentially subtle or highly nonlinear relationships
    /// - You need very precise detection of overfitting or underfitting
    /// - You have sufficient computational resources
    /// 
    /// You might want fewer points (like 50) if:
    /// - You need faster processing
    /// - You're working with many features or a very large dataset
    /// - You're doing preliminary analysis
    /// 
    /// The right number of points balances detail with computational efficiency.
    /// 100 points is sufficient for most applications, capturing meaningful patterns
    /// without excessive computation.
    /// </para>
    /// </remarks>
    public int NumPoints { get; set; } = 100;
}
