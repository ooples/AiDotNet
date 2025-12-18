namespace AiDotNet.Enums;

/// <summary>
/// Specifies the criterion used to determine the best way to split data in decision trees and other tree-based models.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Split criteria are like the "rules" that help decision trees decide how to divide data.
/// 
/// Imagine you're organizing books on shelves. You could sort them by:
/// - Size (big books vs. small books)
/// - Color (red books vs. blue books)
/// - Topic (fiction vs. non-fiction)
/// 
/// But which way of sorting is best? Split criteria help the AI decide which way of dividing
/// the data will lead to the most accurate predictions. Different criteria measure "best" in
/// different ways, each with their own advantages.
/// 
/// These criteria are primarily used for regression problems (predicting numeric values like
/// house prices or temperatures) rather than classification problems (predicting categories).
/// </remarks>
public enum SplitCriterion
{
    /// <summary>
    /// Selects splits that maximize the reduction in variance of the target variable.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This criterion tries to group similar values together.
    /// 
    /// Variance measures how spread out numbers are. High variance means the numbers are all over the place,
    /// while low variance means they're clustered together.
    /// 
    /// Example: If we have house prices [100K, 105K, 110K] in one group and [500K, 510K, 490K] in another,
    /// each group has low variance (the prices are similar within each group).
    /// 
    /// This criterion:
    /// - Tries to create groups where the values within each group are as similar as possible
    /// - Is the traditional approach for regression trees
    /// - Works well for most regression problems
    /// 
    /// Think of it like sorting books by height, so each shelf has books of similar height.
    /// </remarks>
    VarianceReduction,

    /// <summary>
    /// Selects splits that minimize the mean squared error between the actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This criterion focuses on reducing the squared difference between predictions and actual values.
    /// 
    /// Mean Squared Error (MSE) works by:
    /// 1. Taking the difference between each actual value and the predicted value
    /// 2. Squaring each difference (to make negatives positive and emphasize large errors)
    /// 3. Calculating the average of these squared differences
    /// 
    /// This criterion:
    /// - Penalizes large errors more heavily than small ones
    /// - Is sensitive to outliers (unusual values far from the average)
    /// - Often produces similar results to VarianceReduction
    /// 
    /// Think of it like trying to minimize the "penalty points" you get when your guess is wrong,
    /// with bigger mistakes costing exponentially more points.
    /// </remarks>
    MeanSquaredError,

    /// <summary>
    /// Selects splits that minimize the mean absolute error between the actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This criterion focuses on reducing the absolute (positive) difference between predictions and actual values.
    /// 
    /// Mean Absolute Error (MAE) works by:
    /// 1. Taking the difference between each actual value and the predicted value
    /// 2. Converting each difference to a positive number (absolute value)
    /// 3. Calculating the average of these absolute differences
    /// 
    /// This criterion:
    /// - Treats all sizes of errors more equally than MSE
    /// - Is more robust to outliers (unusual values)
    /// - May be better when your data has some extreme values
    /// 
    /// Think of it like measuring the average distance between your guesses and the correct answers,
    /// regardless of whether you guessed too high or too low.
    /// </remarks>
    MeanAbsoluteError,

    /// <summary>
    /// Selects splits using Friedman's improvement to MSE, which accounts for the potential improvement from further splits.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is an advanced version of MSE that looks ahead to future possibilities.
    /// 
    /// Named after statistician Jerome Friedman, this criterion:
    /// - Uses the standard MSE calculation but with an additional factor
    /// - Considers not just the current split but potential future improvements
    /// - Often leads to better tree structures in gradient boosting models
    /// 
    /// Think of it like a chess player who doesn't just consider the current move,
    /// but also thinks about how that move sets up future opportunities.
    /// 
    /// This criterion is particularly useful in gradient boosting decision trees (like XGBoost or LightGBM)
    /// and can lead to better overall model performance.
    /// </remarks>
    FriedmanMSE
}
