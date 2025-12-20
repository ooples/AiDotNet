namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that uses Mean Absolute Error (MAE) to evaluate model performance, particularly for regression tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This calculator helps evaluate how well your model is performing on regression tasks
/// (where you're predicting continuous values like prices, temperatures, etc.) by measuring the average
/// size of the errors in your predictions without considering their direction.
/// 
/// Mean Absolute Error (MAE) is one of the simplest and most intuitive ways to measure prediction errors:
/// - It calculates the absolute difference between each prediction and the actual value
/// - It then takes the average of all these differences
/// - The result tells you, on average, how far off your predictions are
/// 
/// How MAE works:
/// - Take the difference between each predicted value and actual value
/// - Convert all differences to positive numbers (take the absolute value)
/// - Calculate the average of these absolute differences
/// 
/// Think of it like this:
/// Imagine you're predicting house prices:
/// - For one house, you predict $200,000 but the actual price is $220,000 (error of $20,000)
/// - For another house, you predict $350,000 but the actual price is $330,000 (error of $20,000)
/// - The MAE would be $20,000, telling you that on average, your predictions are off by $20,000
/// 
/// Key characteristics of MAE:
/// - It treats all errors equally (unlike Mean Squared Error which penalizes large errors more)
/// - It's measured in the same units as your original data (dollars, degrees, etc.)
/// - It's less sensitive to outliers than Mean Squared Error
/// - Lower values are better (0 would be perfect predictions)
/// 
/// Common applications include:
/// - Price prediction
/// - Temperature forecasting
/// - Any regression task where you want errors measured in the original units
/// - Situations where outliers should not have outsized influence
/// </para>
/// </remarks>
public class MeanAbsoluteErrorFitnessCalculator<T, TInput, TOutput> : FitnessCalculatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the MeanAbsoluteErrorFitnessCalculator class.
    /// </summary>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calculator that will use Mean Absolute Error (MAE)
    /// to evaluate your model's performance, which is especially good for regression problems
    /// where you want to understand the typical size of your prediction errors in the original units.
    /// 
    /// Parameter:
    /// - dataSetType: Which data to evaluate (default is Validation)
    ///   * Training: The data used to train the model (not recommended for evaluation)
    ///   * Validation: A separate set of data used to tune the model (recommended)
    ///   * Test: A completely separate set of data used for final evaluation
    /// 
    /// We set isHigherScoreBetter to "false" in the base constructor because in MAE,
    /// lower values indicate better performance (0 would be a perfect model).
    /// 
    /// Unlike some other loss functions, MAE doesn't have additional parameters to tune,
    /// making it simpler to use.
    /// </para>
    /// </remarks>
    public MeanAbsoluteErrorFitnessCalculator(DataSetType dataSetType = DataSetType.Validation) : base(isHigherScoreBetter: false, dataSetType)
    {
    }

    /// <summary>
    /// Calculates the Mean Absolute Error (MAE) fitness score for the given dataset.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The calculated MAE score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how well your model is performing using Mean Absolute Error.
    /// 
    /// It works by:
    /// 1. Taking the predictions your model made
    /// 2. Comparing them to the actual correct answers
    /// 3. Finding the absolute difference for each prediction (how far off, regardless of direction)
    /// 4. Taking the average of these absolute differences
    /// 
    /// A lower score means better performance (0 would be perfect).
    /// 
    /// The formula is: MAE = (1/n) * S|predicted - actual|
    /// where n is the number of predictions and S means "sum of".
    /// 
    /// This method simply retrieves the pre-calculated MAE from the dataSet's ErrorStats property,
    /// which contains various error metrics that have already been computed.
    /// 
    /// MAE is particularly useful when:
    /// - You want to understand errors in the same units as your original data
    /// - You want a simple, intuitive measure of model performance
    /// - You don't want outliers to have an outsized influence on your error metric
    /// </para>
    /// </remarks>
    protected override T GetFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        return dataSet.ErrorStats.MAE;
    }
}
