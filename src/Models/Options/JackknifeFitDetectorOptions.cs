namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Jackknife Fit Detector, which uses the jackknife resampling technique
/// to evaluate model stability and detect overfitting or underfitting.
/// </summary>
/// <remarks>
/// <para>
/// The jackknife technique (also known as "leave-one-out") involves systematically leaving out one
/// observation at a time from the dataset, retraining the model on the remaining data, and evaluating
/// its performance. By analyzing how model performance varies when different observations are excluded,
/// the detector can assess model stability and identify potential overfitting or underfitting issues.
/// </para>
/// <para><b>For Beginners:</b> The Jackknife Fit Detector helps you understand if your model is reliable
/// by testing how it performs when small parts of your data are removed.
/// 
/// Imagine you're building a recipe and want to know if it's robust. You might try leaving out one
/// ingredient at a time to see how much the taste changes. If leaving out any single ingredient drastically
/// changes the taste, your recipe isn't very stable. Similarly, if your model's predictions change
/// dramatically when you leave out just one data point, that's a sign your model might not be reliable.
/// 
/// This detector runs your model many times, each time leaving out a different data point, and analyzes
/// how consistent the results are. This helps identify:
/// - Overfitting: If your model performs much worse when certain points are left out
/// - Underfitting: If your model performs consistently poorly regardless of which points are left out
/// - Influential outliers: Individual data points that have an unusually large impact on your model
/// 
/// The jackknife approach is particularly useful for smaller datasets where you can't afford to set
/// aside a large validation set.</para>
/// </remarks>
public class JackknifeFitDetectorOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the minimum sample size required to perform jackknife analysis.
    /// </summary>
    /// <value>The minimum sample size, defaulting to 30.</value>
    /// <remarks>
    /// <para>
    /// This threshold ensures that the jackknife analysis is only performed when there is sufficient data.
    /// If the dataset has fewer observations than this threshold, the detector will issue a warning and
    /// may fall back to alternative methods or provide less confident assessments.
    /// </para>
    /// <para><b>For Beginners:</b> This setting specifies how many data points you need at minimum before
    /// the jackknife analysis can be reliably performed. With the default value of 30, the detector will
    /// only run if you have at least 30 data points in your dataset.
    /// 
    /// The jackknife technique works by removing one data point at a time and seeing how that affects your
    /// model. If you start with too few data points, removing even one could significantly distort your
    /// model, making the results of the jackknife analysis less reliable.
    /// 
    /// Think of it like taste-testing a pot of soup: if you have a large pot, removing one spoonful for
    /// tasting won't significantly change the soup. But if you only have a very small amount of soup to
    /// begin with, removing a spoonful might noticeably alter what's left.
    /// 
    /// The value of 30 is a common statistical rule of thumb for when a sample is considered "large enough"
    /// for many analyses. If your dataset is smaller than this, you might want to consider other validation
    /// techniques or gather more data.</para>
    /// </remarks>
    public int MinSampleSize { get; set; } = 30;

    /// <summary>
    /// Gets or sets the threshold for detecting overfitting based on the jackknife analysis results.
    /// </summary>
    /// <value>The overfit threshold, defaulting to 0.1 (10%).</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when a model is considered to be overfitting based on the jackknife analysis.
    /// If the relative standard deviation of model performance across jackknife samples exceeds this threshold,
    /// or if the average drop in performance when points are left out exceeds this threshold, the model is
    /// likely overfitting to specific data points.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps identify when your model is too sensitive to specific
    /// data points, which is a sign of overfitting. With the default value of 0.1, if your model's performance
    /// varies by more than 10% when different individual data points are left out, it's flagged as potentially
    /// overfitting.
    /// 
    /// For example, if your model achieves 90% accuracy on the full dataset, but when you leave out certain
    /// individual points, the accuracy drops to 80% or below, that's a sign that your model is relying too
    /// heavily on those specific points rather than learning general patterns.
    /// 
    /// Think of it like a student who memorizes specific test questions instead of understanding the underlying
    /// concepts. If they can't answer questions that are slightly different from what they memorized, they
    /// haven't truly learned the material. Similarly, a model that can't maintain its performance when small
    /// parts of the data are removed hasn't truly learned the underlying patterns.
    /// 
    /// When overfitting is detected, you might want to:
    /// - Simplify your model
    /// - Add regularization
    /// - Get more training data
    /// - Remove or investigate influential outliers</para>
    /// </remarks>
    public double OverfitThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the threshold for detecting underfitting based on the jackknife analysis results.
    /// </summary>
    /// <value>The underfit threshold, defaulting to 0.1 (10%).</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when a model is considered to be underfitting based on the jackknife analysis.
    /// If the average performance across jackknife samples is below the expected performance by more than this
    /// threshold, or if the performance is consistently poor regardless of which points are left out, the model
    /// is likely underfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps identify when your model is too simple and isn't capturing
    /// important patterns in your data. With the default value of 0.1, if your model's performance is consistently
    /// at least 10% worse than expected across different jackknife samples, it's flagged as potentially underfitting.
    /// 
    /// For example, if similar models or previous versions achieved 80% accuracy on this type of data, but your
    /// current model consistently achieves only 70% or less regardless of which data points are included or excluded,
    /// that's a sign that your model is underfitting.
    /// 
    /// Think of it like using a linear equation to model a clearly curved relationship - no matter which data
    /// points you include, a straight line will never fit a curve well. Similarly, if your model performs
    /// consistently poorly across all jackknife samples, it's likely too simple for the problem at hand.
    /// 
    /// When underfitting is detected, you might want to:
    /// - Increase model complexity
    /// - Add more features
    /// - Reduce regularization
    /// - Train for more iterations
    /// - Try a different type of model</para>
    /// </remarks>
    public double UnderfitThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the maximum number of jackknife iterations to perform.
    /// </summary>
    /// <value>The maximum number of iterations, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// This setting limits the computational cost of the jackknife analysis by capping the number of iterations.
    /// For large datasets, performing a complete jackknife (leaving out each observation once) could be
    /// prohibitively expensive. This setting allows the detector to use a random subset of possible leave-one-out
    /// combinations while still providing statistically valid results.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many times the detector will retrain your model
    /// with different data points left out. With the default value of 1000, the detector will run up to 1000
    /// different variations of your model, each missing one data point.
    /// 
    /// In a true jackknife analysis, you would retrain your model once for each data point you have (leaving
    /// out a different point each time). However, if you have a very large dataset, this could mean thousands
    /// or millions of retrainings, which would take too long. This setting puts a practical limit on how many
    /// iterations to run.
    /// 
    /// For example, if you have 10,000 data points, a complete jackknife would require 10,000 model retrainings.
    /// With MaxIterations set to 1000, the detector would instead randomly select 1,000 data points to leave out
    /// (one at a time), giving you a good statistical approximation without the full computational cost.
    /// 
    /// For smaller datasets (fewer than 1000 points), the detector will still perform a complete jackknife,
    /// leaving out each point exactly once. The MaxIterations setting only comes into play for larger datasets.
    /// 
    /// You might want to decrease this value if:
    /// - You have limited computational resources
    /// - You need results more quickly
    /// - Your model takes a long time to train
    /// 
    /// You might want to increase this value if:
    /// - You have a very large dataset and want more thorough analysis
    /// - You need more precise estimates of model stability
    /// - You have ample computational resources</para>
    /// </remarks>
    public int MaxIterations { get; set; } = 1000;
}
