namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that uses Squared Hinge Loss to evaluate model performance, particularly for binary classification tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This calculator helps evaluate how well your model is performing on classification tasks,
/// especially when you're trying to decide if something belongs to a category or not (binary classification).
/// 
/// Squared Hinge Loss is a variation of Hinge Loss that puts even more emphasis on getting predictions right.
/// 
/// How Squared Hinge Loss works:
/// - It expects predictions to be either -1 (for negative class) or +1 (for positive class)
/// - It wants predictions to be not just correct, but confident (with a margin of safety)
/// - It penalizes incorrect or uncertain predictions by squaring the error, which makes larger errors much more significant
/// 
/// Think of it like this:
/// Imagine you're a teacher grading true/false questions:
/// - Students get full credit for being confidently correct
/// - Students lose points for being wrong or uncertain
/// - The more confidently wrong they are, the exponentially more points they lose
/// 
/// Compared to regular Hinge Loss:
/// - Regular Hinge Loss increases penalties linearly for wrong predictions
/// - Squared Hinge Loss increases penalties quadratically (much faster) for wrong predictions
/// - This makes Squared Hinge Loss more sensitive to outliers and large errors
/// 
/// Common applications include:
/// - Email spam detection
/// - Sentiment analysis (positive/negative)
/// - Medical diagnosis (presence/absence of a condition)
/// - Any binary classification problem where you want to strongly penalize misclassifications
/// </para>
/// </remarks>
public class SquaredHingeLossFitnessCalculator<T, TInput, TOutput> : FitnessCalculatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the SquaredHingeLossFitnessCalculator class.
    /// </summary>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calculator that will use Squared Hinge Loss
    /// to evaluate your model's performance on classification tasks.
    /// 
    /// The parameter:
    /// - dataSetType: Which data to evaluate (default is Validation)
    ///   * Training: The data used to train the model (not recommended for evaluation)
    ///   * Validation: A separate set of data used to tune the model (recommended)
    ///   * Test: A completely separate set of data used for final evaluation
    /// 
    /// We set the first parameter to "false" in the base constructor because in Squared Hinge Loss,
    /// lower values indicate better performance (0 would be a perfect model).
    /// </para>
    /// </remarks>
    public SquaredHingeLossFitnessCalculator(DataSetType dataSetType = DataSetType.Validation)
        : base(false, dataSetType)
    {
    }

    /// <summary>
    /// Calculates the Squared Hinge Loss fitness score for the given dataset.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The calculated Squared Hinge Loss score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how well your model is performing using Squared Hinge Loss.
    /// 
    /// It works by:
    /// 1. Taking the predictions your model made
    /// 2. Comparing them to the actual correct answers
    /// 3. For each prediction:
    ///    - If the prediction is correct and confident: No penalty
    ///    - If the prediction is wrong or not confident enough: Apply a squared penalty
    /// 4. Average all these penalties to get the final score
    /// 
    /// A lower score means better performance (0 would be perfect).
    /// 
    /// This method uses the NeuralNetworkHelper to do the actual Squared Hinge Loss calculation,
    /// passing in your model's predictions and the actual values.
    /// 
    /// Squared Hinge Loss is particularly useful when you want your model to be very confident
    /// in its predictions and you want to strongly penalize any mistakes.
    /// </para>
    /// </remarks>
    protected override T GetFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        return new SquaredHingeLoss<T>().CalculateLoss(ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Predicted),
            ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Actual));
    }
}
