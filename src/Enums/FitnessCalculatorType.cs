namespace AiDotNet.Enums;

/// <summary>
/// Specifies different methods for calculating how well a model fits the data.
/// </summary>
/// <remarks>
/// <para>
/// For Beginners: Fitness calculators measure how well your AI model's predictions match the actual data.
/// 
/// Think of fitness metrics like grades on a test:
/// - They tell you how well your model is performing
/// - Different metrics focus on different aspects of performance
/// - Lower error values (like MSE, MAE) mean better performance
/// - Higher R-squared values mean better performance
/// 
/// When building AI models, you need ways to:
/// - Compare different models to choose the best one
/// - Know when to stop training your model
/// - Understand if your model is actually learning useful patterns
/// - Detect if your model is overfitting (memorizing data instead of learning)
/// 
/// Different metrics are better for different situations, so it's common to look at multiple metrics
/// when evaluating a model.
/// </para>
/// </remarks>
public enum FitnessCalculatorType
{
    /// <summary>
    /// Calculates the average of the squared differences between predicted and actual values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Mean Squared Error (MSE) measures the average squared difference between the predicted values 
    /// and the actual values.
    /// 
    /// Formula: MSE = (1/n) * Σ(actual - predicted)²
    /// 
    /// Think of it as:
    /// - Finding the difference between each prediction and actual value
    /// - Squaring each difference (to make negatives positive and emphasize larger errors)
    /// - Taking the average of all these squared differences
    /// 
    /// Key characteristics:
    /// - Always positive (0 is perfect)
    /// - Heavily penalizes large errors due to squaring
    /// - Uses the same units as your data, but squared
    /// 
    /// Best used for:
    /// - When large errors are particularly undesirable
    /// - Regression problems
    /// - When outliers should have a significant impact
    /// 
    /// Example: If predicting house prices, MSE would heavily penalize being off by $100,000 
    /// much more than being off by $1,000 on multiple houses.
    /// </para>
    /// </remarks>
    MeanSquaredError,

    /// <summary>
    /// Calculates the average of the absolute differences between predicted and actual values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Mean Absolute Error (MAE) measures the average absolute difference between the predicted values 
    /// and the actual values.
    /// 
    /// Formula: MAE = (1/n) * Σ|actual - predicted|
    /// 
    /// Think of it as:
    /// - Finding the difference between each prediction and actual value
    /// - Taking the absolute value (to make negatives positive)
    /// - Taking the average of all these absolute differences
    /// 
    /// Key characteristics:
    /// - Always positive (0 is perfect)
    /// - Treats all sizes of errors linearly (no extra penalty for large errors)
    /// - Uses the same units as your data
    /// - More robust to outliers than MSE
    /// 
    /// Best used for:
    /// - When you want errors to be treated equally regardless of size
    /// - When outliers should not have outsized influence
    /// - When you want error in the same units as your data
    /// 
    /// Example: If predicting daily temperatures, MAE tells you on average how many degrees 
    /// your prediction is off by.
    /// </para>
    /// </remarks>
    MeanAbsoluteError,

    /// <summary>
    /// Measures the proportion of variance in the dependent variable explained by the independent variables.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For Beginners: R-Squared (R²) measures how well your model explains the variation in your data.
    /// 
    /// Formula: R² = 1 - (Sum of Squared Residuals / Total Sum of Squares)
    /// 
    /// Think of it as:
    /// - A percentage that tells you how much of the data's variation your model explains
    /// - 100% (or 1.0) means your model perfectly predicts every point
    /// - 0% (or 0.0) means your model does no better than just guessing the average value
    /// - Negative values are possible and mean your model is worse than just predicting the average
    /// 
    /// Key characteristics:
    /// - Usually ranges from 0 to 1 (higher is better)
    /// - Easy to interpret: "This model explains X% of the variation in the data"
    /// - Not affected by the scale of the data
    /// 
    /// Best used for:
    /// - Regression problems
    /// - When you want an easy-to-understand measure of fit
    /// - Comparing models on the same dataset
    /// 
    /// Example: An R² of 0.75 means your model explains 75% of the variation in your data.
    /// 
    /// Caution: R² will always increase when you add more variables to your model, even if those 
    /// variables aren't actually helpful. This is why Adjusted R² exists.
    /// </para>
    /// </remarks>
    RSquared,

    /// <summary>
    /// A modified version of R-squared that adjusts for the number of predictors in the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For Beginners: Adjusted R-Squared is a modified version of R-Squared that accounts for the number 
    /// of variables in your model.
    /// 
    /// Formula: Adjusted R² = 1 - [(1 - R²) * (n - 1) / (n - p - 1)]
    /// where n is the number of data points and p is the number of predictors
    /// 
    /// Think of it as:
    /// - R-Squared with a penalty for adding too many variables
    /// - A way to prevent "kitchen sink" models that throw in every possible variable
    /// - A more honest assessment of your model's performance
    /// 
    /// Key characteristics:
    /// - Always less than or equal to R²
    /// - Can decrease when you add variables that don't help much
    /// - Better for comparing models with different numbers of variables
    /// 
    /// Best used for:
    /// - When comparing models with different numbers of variables
    /// - When you want to avoid overfitting
    /// - Feature selection (deciding which variables to include)
    /// 
    /// Example: If adding a new variable increases R² slightly but decreases Adjusted R², 
    /// that variable probably isn't worth including in your model.
    /// </para>
    /// </remarks>
    AdjustedRSquared
}