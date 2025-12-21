namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that uses Huber Loss to evaluate model performance, combining the best aspects of Mean Squared Error and Mean Absolute Error.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This calculator helps evaluate how well your model is performing, especially for regression tasks
/// (where you're predicting continuous values like prices, temperatures, etc.).
/// 
/// Huber Loss is a special type of loss function that combines two popular approaches:
/// - Mean Squared Error (MSE): Good for most predictions but very sensitive to outliers
/// - Mean Absolute Error (MAE): Less sensitive to outliers but doesn't penalize large errors as strongly
/// 
/// How Huber Loss works:
/// - For small errors (less than delta): It behaves like Mean Squared Error
/// - For large errors (greater than delta): It behaves like Mean Absolute Error
/// 
/// Think of it like this:
/// Imagine you're a teacher grading papers:
/// - For small mistakes (typos, minor calculation errors), you take off points proportional to the mistake (like MSE)
/// - For huge mistakes (completely wrong answers), you cap the penalty at a certain level (like MAE)
/// - The "delta" parameter is where you draw the line between "small" and "huge" mistakes
/// 
/// This makes Huber Loss more robust against outliers (unusual data points) while still
/// maintaining the benefits of Mean Squared Error for normal cases.
/// 
/// Note: Huber Loss is also sometimes called "Smooth L1 Loss" in some frameworks like PyTorch,
/// but they refer to the same loss function with slightly different parameterizations.
/// 
/// Common applications include:
/// - Price prediction
/// - Any regression task where outliers might be present
/// - Situations where both small and large errors need to be handled appropriately
/// </para>
/// </remarks>
public class HuberLossFitnessCalculator<T, TInput, TOutput> : FitnessCalculatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// The threshold parameter that determines the transition point between quadratic and linear loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This parameter (delta) controls where Huber Loss switches from behaving like
    /// Mean Squared Error to behaving like Mean Absolute Error.
    /// 
    /// - For errors smaller than delta: The loss function is quadratic (like MSE)
    /// - For errors larger than delta: The loss function is linear (like MAE)
    /// 
    /// Think of delta as a sensitivity control:
    /// - Small delta values: More robust to outliers, but might not learn as well from small differences
    /// - Large delta values: More sensitive to all errors, but might be overly influenced by outliers
    /// 
    /// The default value of 1.0 works well for many problems, but you might adjust it:
    /// - Decrease it if your data has many outliers
    /// - Increase it if outliers are rare and you want to focus more on precision
    /// 
    /// In practical terms, delta represents the error value beyond which you consider a prediction
    /// to be an "outlier" that shouldn't influence your model too strongly.
    /// </para>
    /// </remarks>
    private readonly T _delta;

    /// <summary>
    /// Initializes a new instance of the HuberLossFitnessCalculator class.
    /// </summary>
    /// <param name="delta">The threshold parameter that determines the transition point between quadratic and linear loss (default is 1.0).</param>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calculator that will use Huber Loss
    /// to evaluate your model's performance, which is especially good for regression problems
    /// where you might have some unusual data points (outliers).
    /// 
    /// Parameters:
    /// - delta: Controls where the loss function switches behavior (default 1.0)
    ///   * Smaller values make the function more robust to outliers
    ///   * Larger values make the function more sensitive to all errors
    /// 
    /// - dataSetType: Which data to evaluate (default is Validation)
    ///   * Training: The data used to train the model (not recommended for evaluation)
    ///   * Validation: A separate set of data used to tune the model (recommended)
    ///   * Test: A completely separate set of data used for final evaluation
    /// 
    /// We set the first parameter to "false" in the base constructor because in Huber Loss,
    /// lower values indicate better performance (0 would be a perfect model).
    /// </para>
    /// </remarks>
    public HuberLossFitnessCalculator(T? delta = default, DataSetType dataSetType = DataSetType.Validation)
        : base(false, dataSetType)
    {
        _delta = delta ?? _numOps.FromDouble(1.0);
    }

    /// <summary>
    /// Calculates the Huber Loss fitness score for the given dataset.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The calculated Huber Loss score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how well your model is performing using Huber Loss.
    /// 
    /// It works by:
    /// 1. Taking the predictions your model made
    /// 2. Comparing them to the actual correct answers
    /// 3. Calculating a score that represents how wrong the model was, with special handling:
    ///    - For small errors (less than delta): Uses squared error (like MSE)
    ///    - For large errors (greater than delta): Uses absolute error (like MAE)
    /// 
    /// A lower score means better performance (0 would be perfect).
    /// 
    /// This method uses the NeuralNetworkHelper to do the actual Huber Loss calculation,
    /// passing in your model's predictions, the actual values, and the delta parameter.
    /// 
    /// Huber Loss is particularly useful when your data might contain some unusual values
    /// that could throw off a standard Mean Squared Error calculation.
    /// </para>
    /// </remarks>
    protected override T GetFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        return new HuberLoss<T>(Convert.ToDouble(_delta)).CalculateLoss(ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Predicted),
            ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Actual));
    }
}
