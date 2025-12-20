namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that uses Jaccard Loss to evaluate model performance, particularly for segmentation and classification tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This calculator helps evaluate how well your model is performing on tasks where
/// you need to identify regions or categories, especially when you care about the overlap between
/// your predictions and the actual answers.
/// 
/// Jaccard Loss is based on the Jaccard Index (also called Intersection over Union or IoU),
/// which measures the similarity between two sets by comparing what they have in common
/// versus their combined elements.
/// 
/// How Jaccard Loss works:
/// - It calculates how much your prediction and the actual answer overlap
/// - It divides this overlap by the total area covered by both
/// - It then converts this similarity score into a loss (by subtracting from 1)
/// 
/// Think of it like this:
/// Imagine you and a friend are each drawing circles on a piece of paper:
/// - The Jaccard Index measures how much your circles overlap compared to their total area
/// - A score of 1 means perfect overlap (identical circles)
/// - A score of 0 means no overlap at all
/// - Jaccard Loss is simply 1 minus this score (so 0 is perfect, 1 is terrible)
/// 
/// Common applications include:
/// - Image segmentation (identifying regions in images)
/// - Object detection
/// - Multi-class classification
/// - Any task where you care about the overlap between predicted and actual regions
/// </para>
/// </remarks>
public class JaccardLossFitnessCalculator<T, TInput, TOutput> : FitnessCalculatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the JaccardLossFitnessCalculator class.
    /// </summary>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calculator that will use Jaccard Loss
    /// to evaluate your model's performance, which is especially good for tasks where you
    /// need to measure how well your predictions overlap with the actual answers.
    /// 
    /// Parameter:
    /// - dataSetType: Which data to evaluate (default is Validation)
    ///   * Training: The data used to train the model (not recommended for evaluation)
    ///   * Validation: A separate set of data used to tune the model (recommended)
    ///   * Test: A completely separate set of data used for final evaluation
    /// 
    /// We set the first parameter to "false" in the base constructor because in Jaccard Loss,
    /// lower values indicate better performance (0 would be a perfect model).
    /// 
    /// Unlike some other loss functions, Jaccard Loss doesn't have additional parameters to tune,
    /// making it simpler to use.
    /// </para>
    /// </remarks>
    public JaccardLossFitnessCalculator(DataSetType dataSetType = DataSetType.Validation)
        : base(false, dataSetType)
    {
    }

    /// <summary>
    /// Calculates the Jaccard Loss fitness score for the given dataset.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The calculated Jaccard Loss score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how well your model is performing using Jaccard Loss.
    /// 
    /// It works by:
    /// 1. Taking the predictions your model made
    /// 2. Comparing them to the actual correct answers
    /// 3. Calculating how much they overlap divided by their combined area
    /// 4. Converting this to a loss score (1 - overlap ratio)
    /// 
    /// A lower score means better performance (0 would be perfect).
    /// 
    /// For classification tasks:
    /// - Predictions should typically be probabilities (values between 0 and 1)
    /// - Actual values should be binary (0 or 1)
    /// 
    /// For segmentation tasks:
    /// - Both predictions and actual values represent regions (often as binary masks)
    /// 
    /// This method uses the NeuralNetworkHelper to do the actual Jaccard Loss calculation,
    /// passing in your model's predictions and the actual values.
    /// 
    /// Jaccard Loss is particularly useful when you care about the exact regions or categories
    /// your model is identifying, rather than just whether it's generally correct.
    /// </para>
    /// </remarks>
    protected override T GetFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        return new JaccardLoss<T>().CalculateLoss(ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Predicted),
            ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Actual));
    }
}
