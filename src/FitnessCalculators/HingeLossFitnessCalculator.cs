namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that uses Hinge Loss to evaluate model performance, particularly for binary classification and support vector machines.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This calculator helps evaluate how well your model is performing on classification tasks,
/// especially when you're trying to clearly separate two classes from each other.
/// 
/// Hinge Loss is commonly used with Support Vector Machines (SVMs), which are models that try to find
/// the best dividing line (or "hyperplane") between different classes of data.
/// 
/// How Hinge Loss works:
/// - It penalizes predictions that are both incorrect AND confident
/// - It doesn't care how correct your correct predictions are, only that they're on the right side of the boundary
/// - It creates a "margin" around the decision boundary and wants predictions to be clearly on one side or the other
/// 
/// Think of it like this:
/// Imagine you're trying to separate apples and oranges on a table with a stick.
/// - Hinge Loss doesn't just want the stick to separate them
/// - It wants all apples to be at least a certain distance from the stick on one side
/// - And all oranges to be at least a certain distance from the stick on the other side
/// 
/// This creates a "safety margin" that makes the model more robust.
/// 
/// Common applications include:
/// - Text classification (like spam detection)
/// - Image classification
/// - Any binary classification problem where you want a clear separation between classes
/// </para>
/// </remarks>
public class HingeLossFitnessCalculator<T, TInput, TOutput> : FitnessCalculatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the HingeLossFitnessCalculator class.
    /// </summary>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calculator that will use Hinge Loss
    /// to evaluate your model's performance, which is especially good for binary classification
    /// problems (where you're deciding between two options).
    /// 
    /// Parameter:
    /// - dataSetType: Which data to evaluate (default is Validation)
    ///   * Training: The data used to train the model (not recommended for evaluation)
    ///   * Validation: A separate set of data used to tune the model (recommended)
    ///   * Test: A completely separate set of data used for final evaluation
    /// 
    /// We set the first parameter to "false" in the base constructor because in Hinge Loss,
    /// lower values indicate better performance (0 would be a perfect model).
    /// 
    /// Unlike some other loss functions, Hinge Loss doesn't have additional parameters to tune,
    /// making it simpler to use.
    /// </para>
    /// </remarks>
    public HingeLossFitnessCalculator(DataSetType dataSetType = DataSetType.Validation)
        : base(false, dataSetType)
    {
    }

    /// <summary>
    /// Calculates the Hinge Loss fitness score for the given dataset.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The calculated Hinge Loss score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how well your model is performing using Hinge Loss.
    /// 
    /// It works by:
    /// 1. Taking the predictions your model made
    /// 2. Comparing them to the actual correct answers
    /// 3. Calculating a score that represents how wrong the model was, with special attention to:
    ///    - Whether predictions are on the correct side of the decision boundary
    ///    - Whether they're far enough from the boundary (in the "margin")
    /// 
    /// A lower score means better performance (0 would be perfect).
    /// 
    /// For binary classification:
    /// - Predictions should be in the range [-1, 1] or [0, 1]
    /// - Actual values should be either -1/1 or 0/1 depending on the format
    /// 
    /// This method uses the NeuralNetworkHelper to do the actual Hinge Loss calculation,
    /// passing in your model's predictions and the actual values.
    /// </para>
    /// </remarks>
    protected override T GetFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        return new HingeLoss<T>().CalculateLoss(ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Predicted),
            ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Actual));
    }
}
