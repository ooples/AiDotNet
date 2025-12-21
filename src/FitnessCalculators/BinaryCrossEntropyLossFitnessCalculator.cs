namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that uses Binary Cross-Entropy Loss to evaluate model performance for binary classification problems.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This calculator helps you evaluate how well your model is performing on binary classification tasks
/// (problems where you're predicting one of two possible outcomes, like "yes/no" or "spam/not spam").
/// 
/// Binary Cross-Entropy Loss measures the difference between your model's predicted probabilities and the actual
/// outcomes. Here's how to understand it:
/// 
/// - It's specifically designed for binary classification problems (where the answer is one of two possibilities)
/// - Lower values are better (0 would be a perfect model)
/// - It heavily penalizes confident but wrong predictions (e.g., if your model is 99% sure something is spam when it's not)
/// - It's commonly used in neural networks and logistic regression
/// 
/// Think of it like a scoring system that gives your model a harsh penalty when it's very confident but wrong,
/// and a small penalty when it's uncertain. This encourages your model to be confident only when it has good reason to be.
/// 
/// Unlike accuracy (which just counts right vs. wrong), Binary Cross-Entropy Loss takes into account how confident
/// your model was in its predictions.
/// </para>
/// </remarks>
public class BinaryCrossEntropyLossFitnessCalculator<T, TInput, TOutput> : FitnessCalculatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the BinaryCrossEntropyLossFitnessCalculator class.
    /// </summary>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calculator that will use Binary Cross-Entropy Loss
    /// to evaluate your model's performance on binary classification tasks.
    /// 
    /// The "dataSetType" parameter lets you choose which data to evaluate:
    /// - Training: The data used to train the model (not recommended for evaluation)
    /// - Validation: A separate set of data used to tune the model (default and recommended)
    /// - Test: A completely separate set of data used for final evaluation
    /// 
    /// We set the first parameter to "false" because in Binary Cross-Entropy Loss, lower values indicate
    /// better performance (0 would be a perfect model). This tells the system that smaller numbers are better.
    /// </para>
    /// </remarks>
    public BinaryCrossEntropyLossFitnessCalculator(DataSetType dataSetType = DataSetType.Validation)
        : base(false, dataSetType)
    {
    }

    /// <summary>
    /// Calculates the Binary Cross-Entropy Loss between predicted and actual values.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The Binary Cross-Entropy Loss value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how well your model's predictions match the actual outcomes
    /// using the Binary Cross-Entropy Loss formula.
    /// 
    /// The formula looks at each prediction your model made, compares it to the actual outcome,
    /// and calculates a penalty based on how wrong the prediction was. These penalties are then
    /// averaged to give you a single number representing your model's performance.
    /// 
    /// The method is called internally when you evaluate your model's fitness, so you
    /// typically won't need to call it directly.
    /// 
    /// Remember:
    /// - A value of 0 means perfect predictions
    /// - Higher values mean worse performance
    /// - The maximum value can be very large if predictions are completely wrong
    /// </para>
    /// </remarks>
    protected override T GetFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        return new BinaryCrossEntropyLoss<T>().CalculateLoss(ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Predicted),
            ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Actual));
    }
}
