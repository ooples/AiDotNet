namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that uses Cosine Similarity Loss to evaluate model performance for tasks where the direction of vectors matters more than their magnitude.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This calculator helps you evaluate how well your model is learning to predict vectors that point in the same direction.
/// 
/// Cosine Similarity measures the angle between two vectors (sets of numbers), ignoring their length or magnitude.
/// It's like comparing the direction two people are facing, without caring how far away they are standing.
/// 
/// Some common applications include:
/// - Text similarity (comparing document topics)
/// - Recommendation systems (finding similar products or content)
/// - Image retrieval (finding images with similar content)
/// - Natural language processing (comparing word meanings)
/// 
/// The loss ranges from:
/// - 0 (best): Vectors point in exactly the same direction
/// - 2 (worst): Vectors point in exactly opposite directions
/// 
/// For example, if your model is trying to learn word meanings, cosine similarity would help determine if
/// two words are related in meaning (pointing in similar directions) regardless of how common or rare they are.
/// </para>
/// </remarks>
public class CosineSimilarityLossFitnessCalculator<T, TInput, TOutput> : FitnessCalculatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the CosineSimilarityLossFitnessCalculator class.
    /// </summary>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calculator that will use Cosine Similarity Loss
    /// to evaluate your model's performance on tasks where the direction of vectors is important.
    /// 
    /// The "dataSetType" parameter lets you choose which data to evaluate:
    /// - Training: The data used to train the model (not recommended for evaluation)
    /// - Validation: A separate set of data used to tune the model (default and recommended)
    /// - Test: A completely separate set of data used for final evaluation
    /// 
    /// We set the first parameter to "false" because in Cosine Similarity Loss, lower values indicate
    /// better performance (0 would be a perfect model). This tells the system that smaller numbers are better.
    /// </para>
    /// </remarks>
    public CosineSimilarityLossFitnessCalculator(DataSetType dataSetType = DataSetType.Validation)
        : base(false, dataSetType)
    {
    }

    /// <summary>
    /// Calculates the Cosine Similarity Loss between predicted and actual values.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The Cosine Similarity Loss value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how well your model's predictions align with the actual values
    /// in terms of direction, regardless of magnitude.
    /// 
    /// The method works by:
    /// 1. Taking the predicted values from your model and the actual target values
    /// 2. Calculating the cosine of the angle between these two sets of values
    /// 3. Converting this to a loss value (where 0 is perfect alignment and 2 is complete opposition)
    /// 
    /// Imagine two arrows:
    /// - If both arrows point in the same direction, the loss is 0 (perfect)
    /// - If they point in somewhat similar directions, the loss is between 0 and 1
    /// - If they point in perpendicular directions (90° angle), the loss is 1
    /// - If they point in opposite directions, the loss is 2 (worst case)
    /// 
    /// The method is called internally when you evaluate your model's fitness, so you
    /// typically won't need to call it directly.
    /// </para>
    /// </remarks>
    protected override T GetFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        return new CosineSimilarityLoss<T>().CalculateLoss(ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Predicted),
            ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Actual));
    }
}
