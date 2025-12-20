namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that uses Root Mean Squared Error (RMSE) to evaluate model performance.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This calculator helps evaluate how well your model is performing by measuring
/// the average size of the errors in your predictions, with a special emphasis on larger errors.
/// 
/// Root Mean Squared Error (RMSE) is one of the most commonly used metrics in machine learning and is:
/// - A measure of how far off your predictions are from the actual values
/// - Calculated by taking the square root of the average of squared differences between predictions and actual values
/// - Expressed in the same units as your target variable (which makes it easy to interpret)
/// 
/// How RMSE works:
/// 1. Calculate the difference between each predicted value and the actual value
/// 2. Square each of these differences (to make all values positive and emphasize larger errors)
/// 3. Calculate the average (mean) of these squared differences
/// 4. Take the square root of this average to get back to the original units
/// 
/// Think of it like this:
/// Imagine you're predicting house prices in dollars:
/// - If your RMSE is $50,000, it means your predictions are off by about $50,000 on average
/// - But because of the squaring step, being off by $100,000 on one house is penalized more than
///   being off by $50,000 on two houses
/// 
/// Key characteristics of RMSE:
/// - Lower values are better (0 would be perfect predictions)
/// - It penalizes larger errors more heavily than smaller ones
/// - It's sensitive to outliers (a few very bad predictions can significantly increase RMSE)
/// - It's in the same units as your target variable (making it easy to understand)
/// 
/// When to use RMSE:
/// - When you want to heavily penalize large errors
/// - When outliers in your predictions should be considered important
/// - When you want a metric that's in the same units as your target variable
/// - For regression problems (predicting continuous values like prices, temperatures, etc.)
/// 
/// RMSE is one of the most popular metrics for regression tasks because it provides a clear,
/// interpretable measure of prediction error in the original units of the target variable.
/// </para>
/// </remarks>
public class RootMeanSquaredErrorFitnessCalculator<T, TInput, TOutput> : FitnessCalculatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the RootMeanSquaredErrorFitnessCalculator class.
    /// </summary>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calculator that will use Root Mean Squared Error (RMSE)
    /// to evaluate your model's performance.
    /// 
    /// Parameter:
    /// - dataSetType: Which data to evaluate (default is Validation)
    ///   * Training: The data used to train the model (not recommended for evaluation)
    ///   * Validation: A separate set of data used to tune the model (recommended)
    ///   * Test: A completely separate set of data used for final evaluation
    /// 
    /// We set "isHigherScoreBetter" to "false" in the base constructor because with RMSE,
    /// lower values indicate better performance (0 would be perfect).
    /// 
    /// When to use this calculator:
    /// - When you want a common, well-understood error metric
    /// - When larger errors should be penalized more heavily than smaller ones
    /// - When you want an error metric in the same units as your target variable
    /// - For regression problems (predicting continuous values)
    /// </para>
    /// </remarks>
    public RootMeanSquaredErrorFitnessCalculator(DataSetType dataSetType = DataSetType.Validation) : base(isHigherScoreBetter: false, dataSetType)
    {
    }

    /// <summary>
    /// Calculates the Root Mean Squared Error (RMSE) fitness score for the given dataset.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The calculated RMSE score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how well your model is performing using Root Mean Squared Error.
    /// 
    /// It works by:
    /// 1. Taking the difference between each predicted value and the actual value
    /// 2. Squaring each difference (to make all values positive and emphasize larger errors)
    /// 3. Calculating the average (mean) of these squared differences
    /// 4. Taking the square root to get back to the original units
    /// 
    /// A lower score means better performance (0 would be perfect).
    /// 
    /// For example, if you're predicting house prices and get an RMSE of $50,000, it means
    /// your predictions are off by about $50,000 on average, but with a greater penalty
    /// for predictions that are very far off.
    /// 
    /// This method simply retrieves the pre-calculated RMSE value from the dataset's error statistics.
    /// </para>
    /// </remarks>
    protected override T GetFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        return dataSet.ErrorStats.RMSE;
    }
}
