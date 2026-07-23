namespace AiDotNet.Enums;

/// <summary>
/// Defines strategies for setting the importance threshold in feature selection.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> When selecting features based on importance scores from a model,
/// you need to decide which features are "important enough" to keep. This enum provides
/// different strategies for making that decision.
/// </para>
/// <para>
/// Think of it like deciding which students make the honor roll:
/// - Mean: Keep students who score above the class average
/// - Median: Keep the top 50% of students
/// </para>
/// <para>
/// Note: You can also use a custom threshold by calling the SelectFromModel constructor
/// that accepts a specific threshold value instead of a strategy.
/// </para>
/// </remarks>
public enum ImportanceThresholdStrategy
{
    /// <summary>
    /// Keep features with importance greater than or equal to the mean importance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This strategy calculates the average importance of all features
    /// and keeps only the features that are above average.
    /// </para>
    /// <para>
    /// For example, if you have 10 features with importances: [0.01, 0.05, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30],
    /// the mean is 0.158, so it would keep the 6 features with importance >= 0.158.
    /// </para>
    /// <para>
    /// Advantages:
    /// - Simple and intuitive
    /// - Automatically adapts to your data
    /// - Tends to keep roughly half the features (assuming symmetric distribution)
    /// </para>
    /// <para>
    /// Disadvantages:
    /// - May keep too many features if importances are skewed
    /// - Doesn't guarantee a specific number of features
    /// </para>
    /// </remarks>
    Mean,

    /// <summary>
    /// Keep features with importance greater than or equal to the median importance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This strategy keeps the top 50% of features by importance.
    /// The median is the middle value when all importances are sorted.
    /// </para>
    /// <para>
    /// This is like a class where exactly half the students are above the median score.
    /// </para>
    /// <para>
    /// Advantages:
    /// - Always keeps approximately 50% of features
    /// - More robust to outliers than mean
    /// - Predictable number of features
    /// </para>
    /// <para>
    /// Disadvantages:
    /// - May keep too many features if most are unimportant
    /// - May keep too few features if many are equally important
    /// </para>
    /// </remarks>
    Median
}
