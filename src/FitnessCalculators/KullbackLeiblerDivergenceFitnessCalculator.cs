namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that uses Kullback-Leibler Divergence to evaluate model performance, particularly for probability distributions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This calculator helps evaluate how well your model is performing when you're
/// trying to predict probability distributions (like when your model needs to assign probabilities
/// to different possible outcomes).
/// 
/// Kullback-Leibler Divergence (often called KL Divergence) measures how different two probability
/// distributions are from each other. In machine learning, we use it to compare:
/// - The distribution your model predicted
/// - The actual distribution from your data
/// 
/// How KL Divergence works:
/// - It measures the "extra information" needed to represent the actual distribution using your predicted distribution
/// - It's always non-negative (0 or greater)
/// - A value of 0 means the distributions are identical (perfect prediction)
/// - Higher values mean the distributions are more different (worse prediction)
/// 
/// Think of it like this:
/// Imagine you're trying to guess the weather forecast:
/// - The actual forecast says: 70% chance of rain, 30% chance of sun
/// - Your guess is: 60% chance of rain, 40% chance of sun
/// - KL Divergence measures how "surprised" you would be when you see the actual weather,
///   given that you were expecting your guessed probabilities
/// 
/// Common applications include:
/// - Training generative models (like GANs or VAEs)
/// - Multi-class classification problems
/// - Natural language processing
/// - Any task where your model outputs probabilities across multiple categories
/// </para>
/// </remarks>
public class KullbackLeiblerDivergenceFitnessCalculator<T, TInput, TOutput> : FitnessCalculatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the KullbackLeiblerDivergenceFitnessCalculator class.
    /// </summary>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calculator that will use KL Divergence
    /// to evaluate your model's performance, which is especially good for tasks where your model
    /// is predicting probabilities across multiple categories.
    /// 
    /// Parameter:
    /// - dataSetType: Which data to evaluate (default is Validation)
    ///   * Training: The data used to train the model (not recommended for evaluation)
    ///   * Validation: A separate set of data used to tune the model (recommended)
    ///   * Test: A completely separate set of data used for final evaluation
    /// 
    /// We set the first parameter to "false" in the base constructor because in KL Divergence,
    /// lower values indicate better performance (0 would be a perfect model).
    /// 
    /// Unlike some other loss functions, KL Divergence doesn't have additional parameters to tune,
    /// making it simpler to use.
    /// </para>
    /// </remarks>
    public KullbackLeiblerDivergenceFitnessCalculator(DataSetType dataSetType = DataSetType.Validation)
        : base(false, dataSetType)
    {
    }

    /// <summary>
    /// Calculates the Kullback-Leibler Divergence fitness score for the given dataset.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The calculated KL Divergence score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how well your model is performing using KL Divergence.
    /// 
    /// It works by:
    /// 1. Taking the probability distributions your model predicted
    /// 2. Comparing them to the actual probability distributions
    /// 3. Calculating how much "extra information" would be needed to represent the actual distribution
    ///    using your predicted distribution
    /// 
    /// A lower score means better performance (0 would be perfect).
    /// 
    /// Important requirements:
    /// - Both predictions and actual values should be probability distributions (they should sum to 1)
    /// - Values should be between 0 and 1
    /// - The actual values should not contain zeros where the predicted values are non-zero
    ///   (this would cause division by zero)
    /// 
    /// This method uses the NeuralNetworkHelper to do the actual KL Divergence calculation,
    /// passing in your model's predicted probabilities and the actual probabilities.
    /// 
    /// KL Divergence is particularly useful when the exact probabilities matter, not just
    /// which category has the highest probability.
    /// </para>
    /// </remarks>
    protected override T GetFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        return new KullbackLeiblerDivergence<T>().CalculateLoss(ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Predicted),
            ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Actual));
    }
}
