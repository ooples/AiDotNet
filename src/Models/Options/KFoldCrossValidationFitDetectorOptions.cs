namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the K-Fold Cross Validation Fit Detector, which evaluates model quality
/// by analyzing performance across multiple data partitions.
/// </summary>
/// <remarks>
/// <para>
/// K-Fold Cross Validation is a technique that divides the dataset into K equal parts (folds), then
/// trains and evaluates the model K times, each time using a different fold as the validation set and
/// the remaining folds as the training set. This provides a more robust assessment of model performance
/// than a single train-test split, especially for smaller datasets. The Fit Detector analyzes the
/// patterns in performance across these folds to identify overfitting, underfitting, and other model
/// quality issues.
/// </para>
/// <para><b>For Beginners:</b> K-Fold Cross Validation is like testing a recipe multiple times with
/// slightly different ingredients each time to make sure it's consistently good.
/// 
/// Instead of splitting your data just once into training and testing sets, K-Fold Cross Validation
/// divides your data into K equal parts (typically 5 or 10). It then runs K separate experiments:
/// 
/// - Experiment 1: Train on parts 2-K, test on part 1
/// - Experiment 2: Train on parts 1 and 3-K, test on part 2
/// - And so on...
/// 
/// This gives you K different performance measurements, which helps you understand:
/// - How consistent your model's performance is (does it work well on all parts of your data?)
/// - Whether your model is overfitting (working well on training data but poorly on test data)
/// - Whether your model is underfitting (working poorly on all data)
/// 
/// The Fit Detector analyzes these results automatically and tells you if there are any problems with
/// your model. This class lets you configure how sensitive the detector should be to different types
/// of problems.</para>
/// </remarks>
public class KFoldCrossValidationFitDetectorOptions
{
    /// <summary>
    /// Gets or sets the threshold for detecting overfitting based on the difference between training
    /// and validation performance across folds.
    /// </summary>
    /// <value>The overfit threshold, defaulting to 0.1 (10%).</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when a model is considered to be overfitting based on the average gap
    /// between training and validation performance across all folds. If the relative difference exceeds
    /// this threshold, the model is likely overfitting to the training data.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps identify when your model is "memorizing" the
    /// training data instead of learning general patterns. With the default value of 0.1, if your model
    /// performs more than 10% better on the training data than on the validation data (averaged across
    /// all folds), it's flagged as potentially overfitting.
    /// 
    /// For example, if your model achieves 95% accuracy on training data but only 84% on validation data,
    /// that's an 11% difference, which exceeds the threshold and suggests overfitting.
    /// 
    /// Think of it like a student who memorizes test questions instead of understanding the subject.
    /// They'll do great on practice tests they've seen before, but poorly on new questions. The gap
    /// between their performance on familiar versus new material reveals they haven't truly learned
    /// the subject.
    /// 
    /// When overfitting is detected, you might want to:
    /// - Simplify your model
    /// - Add regularization
    /// - Get more training data
    /// - Use techniques like early stopping or dropout</para>
    /// </remarks>
    public double OverfitThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the threshold for detecting underfitting based on the absolute performance level
    /// across folds.
    /// </summary>
    /// <value>The underfit threshold, defaulting to 0.5 (50%).</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when a model is considered to be underfitting based on its absolute
    /// performance level. If the average performance across all folds (on both training and validation data)
    /// is below this threshold relative to a perfect score, the model is likely underfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps identify when your model is too simple and isn't
    /// capturing important patterns in your data. With the default value of 0.5, if your model's overall
    /// performance is below 50% of what would be considered perfect performance, it's flagged as
    /// potentially underfitting.
    /// 
    /// For example, if perfect performance would be an error of 0 or accuracy of 100%, and your model
    /// achieves an error of 0.6 or accuracy of 40% (averaged across all folds), that would be flagged
    /// as underfitting.
    /// 
    /// Think of it like using a bicycle to travel across the country - it's too simple a solution for
    /// such a complex journey, and no matter how well you use the bicycle, it won't be adequate for
    /// the task.
    /// 
    /// When underfitting is detected, you might want to:
    /// - Increase model complexity
    /// - Add more features
    /// - Reduce regularization
    /// - Train for more iterations
    /// - Try a different type of model</para>
    /// </remarks>
    public double UnderfitThreshold { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the threshold for detecting high variance (inconsistent performance) across different
    /// folds.
    /// </summary>
    /// <value>The high variance threshold, defaulting to 0.1 (10%).</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when a model is considered to have high variance based on how much its
    /// performance varies across different folds. If the coefficient of variation (standard deviation
    /// divided by mean) of performance across folds exceeds this threshold, the model likely has high
    /// variance and may be sensitive to the specific data used for training.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps identify when your model's performance is too
    /// inconsistent across different subsets of your data. With the default value of 0.1, if your model's
    /// performance varies by more than 10% across different folds, it's flagged as having high variance.
    /// 
    /// For example, if your model achieves accuracies of 85%, 92%, 78%, 88%, and 81% on five different
    /// folds, the variation might exceed 10% of the average, indicating high variance.
    /// 
    /// Think of it like a chef whose dishes taste great sometimes but terrible other times - the
    /// inconsistency suggests they haven't mastered the recipe. Similarly, if your model performs very
    /// differently depending on which data it sees, it hasn't truly learned robust patterns.
    /// 
    /// High variance often indicates that:
    /// - Your model is too complex for the amount of data you have
    /// - There are outliers or unusual subgroups in your data
    /// - Your data might not be representative of the full problem space
    /// 
    /// When high variance is detected, you might want to:
    /// - Simplify your model
    /// - Get more training data
    /// - Use ensemble methods
    /// - Apply regularization techniques</para>
    /// </remarks>
    public double HighVarianceThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the threshold for confirming good fit based on performance and consistency across folds.
    /// </summary>
    /// <value>The good fit threshold, defaulting to 0.7 (70%).</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when a model is considered to have a good fit based on its overall performance
    /// level. If the average performance across all folds (on validation data) exceeds this threshold relative
    /// to a perfect score, and other issues like overfitting and high variance are not detected, the model is
    /// considered to have a good fit.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps identify when your model has achieved a good balance -
    /// performing well without overfitting or showing high variance. With the default value of 0.7, if your
    /// model's overall performance is at least 70% of what would be considered perfect performance, and it
    /// doesn't show signs of overfitting or high variance, it's considered to have a good fit.
    /// 
    /// For example, if perfect performance would be an error of 0 or accuracy of 100%, and your model
    /// achieves an error of 0.25 or accuracy of 75% (averaged across all validation folds), that would
    /// meet the good fit threshold.
    /// 
    /// Think of it like a well-balanced recipe that consistently produces tasty results with ingredients
    /// that are readily available - it's practical, reliable, and satisfying.
    /// 
    /// A good fit means your model has struck the right balance between underfitting and overfitting - it's
    /// complex enough to learn from the data but not so complex that it just memorizes it. This is the
    /// "Goldilocks zone" we aim for in machine learning.</para>
    /// </remarks>
    public double GoodFitThreshold { get; set; } = 0.7;

    /// <summary>
    /// Gets or sets the threshold for assessing model stability based on performance consistency across
    /// multiple cross-validation runs.
    /// </summary>
    /// <value>The stability threshold, defaulting to 0.05 (5%).</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when a model is considered stable based on how much its average performance
    /// varies when the entire cross-validation process is repeated with different random fold assignments.
    /// If the variation in average performance across multiple cross-validation runs exceeds this threshold,
    /// the model may be unstable or the dataset may have problematic characteristics.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps identify whether your model's performance depends too
    /// much on exactly how the data is split into folds. With the default value of 0.05, if your model's
    /// average performance varies by more than 5% when you run K-fold cross-validation multiple times with
    /// different random splits, it's flagged as potentially unstable.
    /// 
    /// For example, if you run 5-fold cross-validation three times and get average accuracies of 82%, 87%,
    /// and 78%, the variation exceeds 5%, suggesting instability.
    /// 
    /// Think of it like testing a car on different roads - if it performs well on some road tests but poorly
    /// on others, it's not a reliable vehicle. Similarly, if your model's performance changes significantly
    /// depending on exactly how you split the data, it's not a reliable model.
    /// 
    /// Instability often indicates:
    /// - Your dataset might be too small
    /// - There might be significant outliers or unusual subgroups in your data
    /// - Your model might be too sensitive to specific data points
    /// 
    /// When instability is detected, you might want to:
    /// - Get more training data
    /// - Use stratified sampling to ensure folds are representative
    /// - Apply ensemble methods
    /// - Investigate and potentially address outliers
    /// - Use more robust model types</para>
    /// </remarks>
    public double StabilityThreshold { get; set; } = 0.05;
}
