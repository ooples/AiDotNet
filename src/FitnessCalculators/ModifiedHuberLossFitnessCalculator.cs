namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that uses Modified Huber Loss to evaluate model performance, particularly for classification tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This calculator helps evaluate how well your model is performing on classification tasks
/// (where you're predicting categories or classes) by using a special loss function that's more robust
/// to outliers and noisy data than some alternatives.
/// 
/// Modified Huber Loss is a sophisticated loss function that combines the benefits of different approaches:
/// - For predictions that are very wrong, it increases linearly (like absolute error)
/// - For predictions that are moderately wrong, it increases quadratically (like squared error)
/// - This makes it less sensitive to outliers than squared error alone
/// 
/// How Modified Huber Loss works:
/// - It looks at how confident your model is in its predictions
/// - For predictions where the model is very wrong (far from the true value), it applies a gentler penalty
/// - For predictions where the model is somewhat wrong, it applies a stronger penalty
/// 
/// Think of it like this:
/// Imagine you're grading a test:
/// - If a student is completely wrong (guessing randomly), you don't want to penalize them too harshly
/// - If a student is somewhat wrong (they understood the concept but made a mistake), you want to provide stronger feedback
/// - Modified Huber Loss follows this intuition by adjusting the penalty based on how wrong the prediction is
/// 
/// Key characteristics of Modified Huber Loss:
/// - It's more robust to outliers and noisy data than squared loss
/// - It's smoother than Hinge Loss (another popular classification loss)
/// - It's particularly useful for binary classification problems
/// - Lower values are better (0 would be perfect predictions)
/// 
/// Common applications include:
/// - Binary classification tasks (yes/no predictions)
/// - Situations with potentially noisy or mislabeled data
/// - When you want a balance between robustness and mathematical convenience
/// </para>
/// </remarks>
public class ModifiedHuberLossFitnessCalculator<T, TInput, TOutput> : FitnessCalculatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the ModifiedHuberLossFitnessCalculator class.
    /// </summary>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calculator that will use Modified Huber Loss
    /// to evaluate your model's performance, which is especially good for classification problems
    /// where you want a balance between robustness to outliers and mathematical convenience.
    /// 
    /// Parameter:
    /// - dataSetType: Which data to evaluate (default is Validation)
    ///   * Training: The data used to train the model (not recommended for evaluation)
    ///   * Validation: A separate set of data used to tune the model (recommended)
    ///   * Test: A completely separate set of data used for final evaluation
    /// 
    /// We set the first parameter to "false" in the base constructor because in Modified Huber Loss,
    /// lower values indicate better performance (0 would be a perfect model).
    /// 
    /// Modified Huber Loss is particularly useful when:
    /// - You're working with binary classification (yes/no predictions)
    /// - Your data might contain some mislabeled examples
    /// - You want a loss function that's both robust and mathematically convenient
    /// </para>
    /// </remarks>
    public ModifiedHuberLossFitnessCalculator(DataSetType dataSetType = DataSetType.Validation)
        : base(false, dataSetType)
    {
    }

    /// <summary>
    /// Calculates the Modified Huber Loss fitness score for the given dataset.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The calculated Modified Huber Loss score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how well your model is performing using Modified Huber Loss.
    /// 
    /// It works by:
    /// 1. Taking the predictions your model made
    /// 2. Comparing them to the actual correct answers
    /// 3. Applying a special formula that changes based on how wrong each prediction is:
    ///    - For predictions that are very wrong: applies a linear penalty
    ///    - For predictions that are moderately wrong: applies a quadratic penalty
    /// 4. Taking the average of these penalties
    /// 
    /// A lower score means better performance (0 would be perfect).
    /// 
    /// The formula is:
    /// - If z * y = -1: loss = max(0, 1 - z * y)²
    /// - If z * y < -1: loss = -4 * z * y
    /// where z is the prediction and y is the actual value.
    /// 
    /// This method calls the NeuralNetworkHelper's ModifiedHuberLoss method to perform the actual calculation,
    /// passing in the predicted and actual values from the dataset.
    /// 
    /// Modified Huber Loss is particularly useful when:
    /// - You're working with binary classification problems
    /// - You want a loss function that's less sensitive to outliers
    /// - You need a function that's differentiable everywhere (for optimization)
    /// - Your data might contain some noise or mislabeled examples
    /// </para>
    /// </remarks>
    protected override T GetFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        return new ModifiedHuberLoss<T>().CalculateLoss(ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Predicted),
            ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Actual));
    }
}
