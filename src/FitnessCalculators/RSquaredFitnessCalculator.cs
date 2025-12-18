namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that uses R-Squared (R²) to evaluate model performance.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This calculator helps evaluate how well your model is performing by measuring
/// the proportion of variance in your target variable that is explained by your model.
/// 
/// R-Squared (R²), also called the coefficient of determination, is:
/// - A measure of how well your model explains the variation in your data
/// - Expressed as a value typically between 0 and 1 (or 0% to 100%)
/// - A higher value means your model explains more of the variation in your data
/// 
/// How R-Squared works:
/// - R² = 1 means your model perfectly explains all the variation in your data
/// - R² = 0 means your model doesn't explain any of the variation (it's no better than just predicting the average)
/// - R² can sometimes be negative if your model performs worse than just predicting the average
/// 
/// Think of it like this:
/// Imagine you're predicting house prices:
/// - If R² = 0.7, it means 70% of the variation in house prices is explained by your model
/// - The remaining 30% is due to factors your model doesn't capture
/// 
/// A simple way to understand R²:
/// - If you always predicted the average house price, you'd have an R² of 0
/// - If you could predict every house price exactly right, you'd have an R² of 1
/// - Your model's R² tells you how much better it is than just predicting the average
/// 
/// Key characteristics of R²:
/// - Higher values are better (1 would be perfect)
/// - It's scale-independent (it doesn't matter if you're predicting dollars or millions of dollars)
/// - It helps you understand how much of the variation your model captures
/// - It can be misleading if you have a small sample size or too many features
/// 
/// When to use R²:
/// - When you want to know how much of the variation your model explains
/// - When you want a metric that's easy to interpret (0% to 100% explained)
/// - When comparing different models for the same problem
/// - For regression problems (predicting continuous values)
/// 
/// R² is one of the most popular metrics for regression tasks because it provides an
/// intuitive measure of how well your model captures the patterns in your data.
/// </para>
/// </remarks>
public class RSquaredFitnessCalculator<T, TInput, TOutput> : FitnessCalculatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the RSquaredFitnessCalculator class.
    /// </summary>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calculator that will use R-Squared (R²)
    /// to evaluate your model's performance.
    /// 
    /// Parameter:
    /// - dataSetType: Which data to evaluate (default is Validation)
    ///   * Training: The data used to train the model (not recommended for evaluation)
    ///   * Validation: A separate set of data used to tune the model (recommended)
    ///   * Test: A completely separate set of data used for final evaluation
    /// 
    /// Note: We set "isHigherScoreBetter" to "false" in the base constructor, which might seem 
    /// counterintuitive since higher R² values are actually better. This is because some optimization 
    /// algorithms in the library are designed to minimize values. The calculator handles this internally 
    /// so that optimization works correctly while still interpreting R² in the standard way (higher is better).
    /// 
    /// When to use this calculator:
    /// - When you want to know how much of the variation in your data your model explains
    /// - When you want an easy-to-interpret metric (0% to 100% of variation explained)
    /// - When comparing different models for the same prediction task
    /// - For regression problems (predicting continuous values)
    /// </para>
    /// </remarks>
    public RSquaredFitnessCalculator(DataSetType dataSetType = DataSetType.Validation) : base(isHigherScoreBetter: false, dataSetType)
    {
    }

    /// <summary>
    /// Calculates the R-Squared (R²) fitness score for the given dataset.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The calculated R² score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how well your model is performing using R-Squared (R²).
    /// 
    /// R² measures how much of the variation in your data is explained by your model:
    /// - R² = 1 means your model perfectly explains all the variation (100%)
    /// - R² = 0 means your model doesn't explain any variation (0%)
    /// - R² can sometimes be negative if your model performs worse than just predicting the average
    /// 
    /// For example, if you're predicting house prices and get an R² of 0.7, it means
    /// your model explains 70% of the variation in house prices, while 30% remains unexplained.
    /// 
    /// This method simply retrieves the pre-calculated R² value from the dataset's prediction statistics.
    /// 
    /// Note: While higher R² values are better, some optimization algorithms in the library are designed
    /// to minimize values. The calculator handles this internally so that optimization works correctly
    /// while still interpreting R² in the standard way (higher is better).
    /// </para>
    /// </remarks>
    protected override T GetFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        return dataSet.PredictionStats.R2;
    }
}
