namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that uses Log-Cosh Loss to evaluate model performance, particularly for regression tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This calculator helps evaluate how well your model is performing on regression tasks
/// (where you're predicting continuous values like prices, temperatures, etc.) while being less
/// sensitive to outliers than some other methods.
/// 
/// Log-Cosh Loss is a smooth approximation that combines the best features of two common loss functions:
/// - Mean Squared Error (MSE): Good for most predictions but very sensitive to outliers
/// - Mean Absolute Error (MAE): Less sensitive to outliers but has mathematical limitations
/// 
/// How Log-Cosh Loss works:
/// - For small errors: It behaves almost like Mean Squared Error
/// - For large errors: It behaves more like Mean Absolute Error
/// - It uses the natural logarithm of the hyperbolic cosine function to achieve this balance
/// 
/// Think of it like this:
/// Imagine you're measuring how far off your predictions are:
/// - For small mistakes, Log-Cosh Loss increases quickly (like MSE)
/// - For large mistakes, it increases more slowly (like MAE)
/// - But unlike MAE, it's smooth everywhere (which helps with optimization)
/// 
/// Common applications include:
/// - Price prediction
/// - Any regression task where outliers might be present
/// - Situations where you want a loss function that's both smooth and robust
/// 
/// The main advantage of Log-Cosh Loss is that it's less sensitive to outliers than MSE
/// but still has nice mathematical properties that make it easier to optimize than MAE.
/// </para>
/// </remarks>
public class LogCoshLossFitnessCalculator<T, TInput, TOutput> : FitnessCalculatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the LogCoshLossFitnessCalculator class.
    /// </summary>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calculator that will use Log-Cosh Loss
    /// to evaluate your model's performance, which is especially good for regression problems
    /// where you want a balance between sensitivity to errors and robustness against outliers.
    /// 
    /// Parameter:
    /// - dataSetType: Which data to evaluate (default is Validation)
    ///   * Training: The data used to train the model (not recommended for evaluation)
    ///   * Validation: A separate set of data used to tune the model (recommended)
    ///   * Test: A completely separate set of data used for final evaluation
    /// 
    /// We set the first parameter to "false" in the base constructor because in Log-Cosh Loss,
    /// lower values indicate better performance (0 would be a perfect model).
    /// 
    /// Unlike some other loss functions, Log-Cosh Loss doesn't have additional parameters to tune,
    /// making it simpler to use.
    /// </para>
    /// </remarks>
    public LogCoshLossFitnessCalculator(DataSetType dataSetType = DataSetType.Validation)
        : base(false, dataSetType)
    {
    }

    /// <summary>
    /// Calculates the Log-Cosh Loss fitness score for the given dataset.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The calculated Log-Cosh Loss score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how well your model is performing using Log-Cosh Loss.
    /// 
    /// It works by:
    /// 1. Taking the predictions your model made
    /// 2. Comparing them to the actual correct answers
    /// 3. For each prediction, calculating log(cosh(prediction - actual))
    /// 4. Taking the average of these values
    /// 
    /// A lower score means better performance (0 would be perfect).
    /// 
    /// The formula is: log(cosh(x)) where x is the difference between predicted and actual values.
    /// This creates a function that:
    /// - Is approximately x²/2 for small x (like MSE)
    /// - Is approximately |x| - log(2) for large x (like MAE)
    /// 
    /// This method uses the NeuralNetworkHelper to do the actual Log-Cosh Loss calculation,
    /// passing in your model's predictions and the actual values.
    /// 
    /// Log-Cosh Loss is particularly useful when you want a loss function that's:
    /// - Smooth everywhere (unlike MAE)
    /// - Not overly sensitive to outliers (unlike MSE)
    /// - Has good mathematical properties for optimization
    /// </para>
    /// </remarks>
    protected override T GetFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        return new LogCoshLoss<T>().CalculateLoss(ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Predicted),
            ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Actual));
    }
}
