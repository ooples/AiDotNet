namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that uses Cross Entropy Loss to evaluate model performance for classification tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This calculator helps you evaluate how well your model is performing on classification tasks,
/// where you need to assign items to specific categories or classes.
/// 
/// Cross Entropy Loss is one of the most common ways to measure how well a model is doing when it needs to
/// choose between multiple options (like identifying if an image contains a cat, dog, or bird).
/// 
/// Think of it like a test where:
/// - Your model gives a confidence score for each possible answer (e.g., "I'm 80% sure this is a cat")
/// - The correct answer is known (e.g., "This is actually a cat")
/// - The loss measures how far off your model's confidence was from being perfectly correct
/// 
/// Some common applications include:
/// - Image classification (identifying objects in images)
/// - Sentiment analysis (determining if text is positive, negative, or neutral)
/// - Medical diagnosis (classifying medical conditions)
/// - Spam detection (determining if an email is spam or not)
/// 
/// Lower values mean your model is more confident about the correct answers and less confident about wrong answers.
/// A perfect model would have a loss of 0, while a completely wrong model would have a very high loss.
/// </para>
/// </remarks>
public class CrossEntropyLossFitnessCalculator<T, TInput, TOutput> : FitnessCalculatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the CrossEntropyLossFitnessCalculator class.
    /// </summary>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calculator that will use Cross Entropy Loss
    /// to evaluate your model's performance on classification tasks.
    /// 
    /// The "dataSetType" parameter lets you choose which data to evaluate:
    /// - Training: The data used to train the model (not recommended for evaluation)
    /// - Validation: A separate set of data used to tune the model (default and recommended)
    /// - Test: A completely separate set of data used for final evaluation
    /// 
    /// We set the first parameter to "false" because in Cross Entropy Loss, lower values indicate
    /// better performance (0 would be a perfect model). This tells the system that smaller numbers are better.
    /// </para>
    /// </remarks>
    public CrossEntropyLossFitnessCalculator(DataSetType dataSetType = DataSetType.Validation)
        : base(false, dataSetType)
    {
    }

    /// <summary>
    /// Calculates the Cross Entropy Loss between predicted and actual values.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The Cross Entropy Loss value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how well your model's predictions match the actual categories.
    /// 
    /// The method works by:
    /// 1. Taking the predicted probabilities from your model (e.g., "70% chance it's a cat, 20% dog, 10% bird")
    /// 2. Comparing these with the actual correct answers (e.g., "It's actually a cat")
    /// 3. Calculating a score that penalizes the model for being uncertain about correct answers
    ///    or confident about wrong answers
    /// 
    /// For example:
    /// - If your model says "99% cat" and it's actually a cat, the loss is very low (good)
    /// - If your model says "51% cat" and it's actually a cat, the loss is higher (less good)
    /// - If your model says "10% cat" and it's actually a cat, the loss is very high (bad)
    /// 
    /// The method is called internally when you evaluate your model's fitness, so you
    /// typically won't need to call it directly.
    /// </para>
    /// </remarks>
    protected override T GetFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        return new CrossEntropyLoss<T>().CalculateLoss(ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Predicted),
            ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Actual));
    }
}
