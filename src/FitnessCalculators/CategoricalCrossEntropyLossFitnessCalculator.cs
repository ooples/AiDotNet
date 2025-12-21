namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that uses Categorical Cross-Entropy Loss to evaluate model performance for multi-class classification problems.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This calculator helps you evaluate how well your model is performing on multi-class classification tasks
/// (problems where you're predicting one of several possible categories, like "dog/cat/bird" or "red/green/blue/yellow").
/// 
/// Categorical Cross-Entropy Loss measures how well your model's predicted probabilities match the actual categories.
/// Here's how to understand it:
/// 
/// - It's designed for problems with multiple possible categories (3 or more classes)
/// - Lower values are better (0 would be a perfect model)
/// - It heavily penalizes confident but wrong predictions (e.g., if your model is 95% sure an image is a dog when it's actually a cat)
/// - It's commonly used in neural networks for image classification, text categorization, and other multi-class problems
/// 
/// While Binary Cross-Entropy Loss works with two categories (like yes/no questions), Categorical Cross-Entropy Loss
/// handles multiple categories (like multiple-choice questions).
/// 
/// Think of it like a teacher grading a multiple-choice test: your model gets more points when it's confident about
/// the right answer and loses points when it's confident about the wrong answer.
/// </para>
/// </remarks>
public class CategoricalCrossEntropyLossFitnessCalculator<T, TInput, TOutput> : FitnessCalculatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the CategoricalCrossEntropyLossFitnessCalculator class.
    /// </summary>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calculator that will use Categorical Cross-Entropy Loss
    /// to evaluate your model's performance on multi-class classification tasks.
    /// 
    /// The "dataSetType" parameter lets you choose which data to evaluate:
    /// - Training: The data used to train the model (not recommended for evaluation)
    /// - Validation: A separate set of data used to tune the model (default and recommended)
    /// - Test: A completely separate set of data used for final evaluation
    /// 
    /// We set the first parameter to "false" because in Categorical Cross-Entropy Loss, lower values indicate
    /// better performance (0 would be a perfect model). This tells the system that smaller numbers are better.
    /// </para>
    /// </remarks>
    public CategoricalCrossEntropyLossFitnessCalculator(DataSetType dataSetType = DataSetType.Validation)
        : base(false, dataSetType)
    {
    }

    /// <summary>
    /// Calculates the Categorical Cross-Entropy Loss between predicted and actual values.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The Categorical Cross-Entropy Loss value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how well your model's predictions match the actual categories
    /// using the Categorical Cross-Entropy Loss formula.
    /// 
    /// The method works by:
    /// 1. Converting the predicted and actual values into matrices (organized tables of numbers)
    /// 2. Comparing the predicted probabilities for each category with the actual categories
    /// 3. Calculating a penalty based on how wrong the predictions were
    /// 
    /// For example, if your model is predicting animal types:
    /// - Actual: Cat (represented as [0,1,0] for [Dog,Cat,Bird])
    /// - Prediction: [0.1,0.7,0.2] (10% Dog, 70% Cat, 20% Bird)
    /// The loss would be relatively low because the highest probability matches the correct category.
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
        return new CategoricalCrossEntropyLoss<T>().CalculateLoss(
            ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Predicted),
            ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Actual)
        );
    }
}
