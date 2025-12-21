namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that uses Mean Squared Error (MSE) to evaluate model performance, particularly for regression tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This calculator helps evaluate how well your model is performing on regression tasks
/// (where you're predicting continuous values like prices, temperatures, etc.) by measuring the average
/// of the squared differences between predictions and actual values.
/// 
/// Mean Squared Error (MSE) is one of the most commonly used metrics in machine learning:
/// - It calculates the square of the difference between each prediction and the actual value
/// - It then takes the average of all these squared differences
/// - The result tells you how large your errors are, with larger errors being penalized more heavily
/// 
/// How MSE works:
/// - Take the difference between each predicted value and actual value
/// - Square each difference (multiply it by itself)
/// - Calculate the average of these squared differences
/// 
/// Think of it like this:
/// Imagine you're predicting house prices:
/// - For one house, you predict $200,000 but the actual price is $220,000 (error of $20,000)
/// - For another house, you predict $350,000 but the actual price is $330,000 (error of $20,000)
/// - When squared, both errors become 400,000,000
/// - The MSE would be 400,000,000, which is much larger than the original error
/// 
/// Key characteristics of MSE:
/// - It penalizes larger errors more heavily than smaller ones (due to squaring)
/// - It's measured in squared units (e.g., squared dollars, squared degrees)
/// - It's more sensitive to outliers than Mean Absolute Error
/// - Lower values are better (0 would be perfect predictions)
/// 
/// Common applications include:
/// - Training many types of regression models
/// - Situations where large errors should be penalized more heavily
/// - When you want a differentiable loss function for optimization
/// </para>
/// </remarks>
public class MeanSquaredErrorFitnessCalculator<T, TInput, TOutput> : FitnessCalculatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the MeanSquaredErrorFitnessCalculator class.
    /// </summary>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calculator that will use Mean Squared Error (MSE)
    /// to evaluate your model's performance, which is especially good for regression problems
    /// where you want to penalize larger errors more heavily than smaller ones.
    /// 
    /// Parameter:
    /// - dataSetType: Which data to evaluate (default is Validation)
    ///   * Training: The data used to train the model (not recommended for evaluation)
    ///   * Validation: A separate set of data used to tune the model (recommended)
    ///   * Test: A completely separate set of data used for final evaluation
    /// 
    /// We set isHigherScoreBetter to "false" in the base constructor because in MSE,
    /// lower values indicate better performance (0 would be a perfect model).
    /// 
    /// MSE is one of the most common loss functions in machine learning because:
    /// - It's mathematically convenient for optimization
    /// - It heavily penalizes outliers, which can be desirable in many applications
    /// - It's differentiable everywhere, which helps with gradient-based optimization
    /// </para>
    /// </remarks>
    public MeanSquaredErrorFitnessCalculator(DataSetType dataSetType = DataSetType.Validation) :
        base(isHigherScoreBetter: false, dataSetType)
    {
    }

    /// <summary>
    /// Calculates the Mean Squared Error (MSE) fitness score for the given dataset.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The calculated MSE score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how well your model is performing using Mean Squared Error.
    /// 
    /// It works by:
    /// 1. Taking the predictions your model made
    /// 2. Comparing them to the actual correct answers
    /// 3. Finding the difference for each prediction
    /// 4. Squaring each difference (which makes all values positive and emphasizes larger errors)
    /// 5. Taking the average of these squared differences
    /// 
    /// A lower score means better performance (0 would be perfect).
    /// 
    /// The formula is: MSE = (1/n) * S(predicted - actual)²
    /// where n is the number of predictions and S means "sum of".
    /// 
    /// This method simply retrieves the pre-calculated MSE from the dataSet's ErrorStats property,
    /// which contains various error metrics that have already been computed.
    /// 
    /// MSE is particularly useful when:
    /// - You want to penalize larger errors more heavily
    /// - You're using gradient-based optimization methods
    /// - You're working with models that assume normally distributed errors
    /// 
    /// Note that because of squaring, MSE is measured in squared units (e.g., if you're predicting
    /// dollars, MSE is in squared dollars). If you want an error metric in the original units,
    /// you might consider Root Mean Squared Error (RMSE) or Mean Absolute Error (MAE).
    /// </para>
    /// </remarks>
    protected override T GetFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        return dataSet.ErrorStats.MSE;
    }
}
