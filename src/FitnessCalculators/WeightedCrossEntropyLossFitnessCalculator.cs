namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that uses Weighted Cross Entropy Loss to evaluate model performance, particularly for classification problems with imbalanced classes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This calculator helps evaluate how well your model is performing on classification tasks,
/// especially when some classes are more important than others or appear less frequently in your data.
/// 
/// Cross Entropy Loss is a common way to measure how well a classification model is performing.
/// The "weighted" part means we can give more importance to certain classes.
/// 
/// Think of it like grading a test:
/// - Regular Cross Entropy treats all questions equally
/// - Weighted Cross Entropy lets you assign more points to harder or more important questions
/// 
/// This is particularly useful when:
/// - Some classes appear much less frequently than others (like rare diseases in medical diagnosis)
/// - Some mistakes are more costly than others (like falsely classifying a tumor as benign)
/// - You want to focus the model's attention on specific classes
/// 
/// By providing weights, you can:
/// - Increase the penalty for misclassifying minority classes
/// - Balance the learning process when your data is imbalanced
/// - Prioritize certain types of predictions based on their importance
/// 
/// Common applications include:
/// - Medical diagnosis (where false negatives might be dangerous)
/// - Fraud detection (where fraud is rare but important to catch)
/// - Any classification problem with imbalanced classes
/// </para>
/// </remarks>
public class WeightedCrossEntropyLossFitnessCalculator<T, TInput, TOutput> : FitnessCalculatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// The weights to apply to each class when calculating the cross entropy loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These weights determine how important each class is when evaluating your model.
    /// 
    /// - Higher weight = More important class (errors on this class are penalized more)
    /// - Lower weight = Less important class (errors on this class are penalized less)
    /// 
    /// For example, in a medical diagnosis system:
    /// - You might set a higher weight for "has disease" class
    /// - And a lower weight for "no disease" class
    /// 
    /// This would make the model work harder to correctly identify people with the disease,
    /// even if it means occasionally misclassifying healthy people.
    /// 
    /// If no weights are provided, the system will create default weights that treat all classes equally.
    /// </para>
    /// </remarks>
    private Vector<T>? _weights;

    /// <summary>
    /// Initializes a new instance of the WeightedCrossEntropyLossFitnessCalculator class.
    /// </summary>
    /// <param name="weights">Optional vector of weights for each class. If not provided, equal weights will be used.</param>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calculator that will use Weighted Cross Entropy Loss
    /// to evaluate your model's performance on classification tasks.
    /// 
    /// The parameters:
    /// - weights: How much importance to give to each class (higher = more important)
    /// - dataSetType: Which data to evaluate (default is Validation)
    ///   * Training: The data used to train the model (not recommended for evaluation)
    ///   * Validation: A separate set of data used to tune the model (recommended)
    ///   * Test: A completely separate set of data used for final evaluation
    /// 
    /// We set the first parameter to "false" in the base constructor because in Cross Entropy Loss,
    /// lower values indicate better performance (0 would be a perfect model).
    /// 
    /// If you don't provide weights, the system will create default weights that treat all classes equally.
    /// </para>
    /// </remarks>
    public WeightedCrossEntropyLossFitnessCalculator(Vector<T>? weights = default, DataSetType dataSetType = DataSetType.Validation)
        : base(false, dataSetType)
    {
        _weights = weights;
    }

    /// <summary>
    /// Calculates the Weighted Cross Entropy Loss fitness score for the given dataset.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The calculated Weighted Cross Entropy Loss score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how well your model is performing using Weighted Cross Entropy Loss.
    /// 
    /// It works by:
    /// 1. Checking if the weights are properly set up (creating default weights if needed)
    /// 2. Comparing the model's predictions with the actual correct values
    /// 3. Applying the weights to give more importance to certain classes
    /// 4. Calculating an overall score that represents how well the model is doing
    /// 
    /// A lower score means better performance (0 would be perfect).
    /// 
    /// The method ensures that the weights vector has the correct length to match your data.
    /// If you didn't provide weights or if the weights don't match the data size, it creates
    /// a new weights vector with equal weights for all classes.
    /// </para>
    /// </remarks>
    protected override T GetFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        var actual = ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Actual);
        if (_weights == null || _weights.Length != actual.Length)
        {
            _weights = new Vector<T>(actual.Length);
        }

        return new WeightedCrossEntropyLoss<T>(_weights).CalculateLoss(ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Predicted),
            ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Actual));
    }
}
