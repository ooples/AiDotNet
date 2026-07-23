namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that uses Exponential Loss to evaluate model performance, particularly for classification tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This calculator helps you evaluate how well your model is performing on classification tasks,
/// with a special focus on heavily penalizing predictions that are both wrong and confident.
/// 
/// Exponential Loss works by:
/// - Giving a small penalty for minor mistakes (when your model is slightly unsure about the right answer)
/// - Giving a MUCH larger penalty for confident mistakes (when your model is very sure about a wrong answer)
/// 
/// Think of it like a teacher grading a test:
/// - If you answer "I'm not sure, but maybe A" and the answer is B, you get a small penalty
/// - If you answer "I'm 100% certain it's A!" and the answer is B, you get a huge penalty
/// 
/// Some common applications include:
/// - Fraud detection (where confidently missing a fraud case is very costly)
/// - Medical diagnosis (where confidently giving the wrong diagnosis could be dangerous)
/// - Any situation where being wrong AND confident is much worse than being unsure
/// 
/// Exponential Loss is used in algorithms like AdaBoost and can help your model focus on the examples
/// it's getting wrong, especially those where it's making confident mistakes.
/// </para>
/// </remarks>
public class ExponentialLossFitnessCalculator<T, TInput, TOutput> : FitnessCalculatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the ExponentialLossFitnessCalculator class.
    /// </summary>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calculator that will use Exponential Loss
    /// to evaluate your model's performance, with a focus on heavily penalizing confident mistakes.
    /// 
    /// The "dataSetType" parameter lets you choose which data to evaluate:
    /// - Training: The data used to train the model (not recommended for evaluation)
    /// - Validation: A separate set of data used to tune the model (default and recommended)
    /// - Test: A completely separate set of data used for final evaluation
    /// 
    /// We set the first parameter to "false" because in Exponential Loss, lower values indicate
    /// better performance (0 would be a perfect model). This tells the system that smaller numbers are better.
    /// </para>
    /// </remarks>
    public ExponentialLossFitnessCalculator(DataSetType dataSetType = DataSetType.Validation)
        : base(false, dataSetType)
    {
    }

    /// <summary>
    /// Calculates the Exponential Loss between predicted and actual values.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The Exponential Loss value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how well your model's predictions match the actual values,
    /// with a special emphasis on penalizing confident mistakes.
    /// 
    /// The method works by:
    /// 1. Taking the predicted values from your model (e.g., confidence scores for each class)
    /// 2. Comparing these with the actual correct values
    /// 3. Calculating a loss that grows exponentially larger as mistakes become more confident
    /// 
    /// For example:
    /// - If your model correctly predicts a class with high confidence, the loss is very small
    /// - If your model is unsure about its prediction (even if wrong), the loss is moderate
    /// - If your model confidently predicts the wrong class, the loss is very large
    /// 
    /// This encourages your model to be cautious when it's not sure, rather than making confident mistakes.
    /// 
    /// The method is called internally when you evaluate your model's fitness, so you
    /// typically won't need to call it directly.
    /// </para>
    /// </remarks>
    protected override T GetFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        return new ExponentialLoss<T>().CalculateLoss(ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Predicted),
            ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Actual));
    }
}
