namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that uses the Adjusted R-Squared metric to evaluate model performance.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This calculator helps you evaluate how well your model fits the data using 
/// a metric called "Adjusted R-Squared."
/// 
/// Regular R-Squared (also called the coefficient of determination) measures how well your model 
/// explains the variation in your data, ranging from 0 to 1:
/// - 0 means your model doesn't explain any of the variation
/// - 1 means your model perfectly explains all variation
/// 
/// However, regular R-Squared has a problem: it always increases when you add more features to your model,
/// even if those features don't actually help with predictions.
/// 
/// Adjusted R-Squared fixes this issue by penalizing models that use too many features. It's like
/// R-Squared with a built-in protection against overly complex models. This helps you find the
/// right balance between model complexity and accuracy.
/// 
/// Unlike regular R-Squared, Adjusted R-Squared can be negative if your model performs very poorly.
/// Generally, you want this value to be as close to 1 as possible.
/// </para>
/// </remarks>
public class AdjustedRSquaredFitnessCalculator<T, TInput, TOutput> : FitnessCalculatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the AdjustedRSquaredFitnessCalculator class.
    /// </summary>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calculator that will use Adjusted R-Squared
    /// to evaluate your model.
    /// 
    /// The "dataSetType" parameter lets you choose which data to evaluate:
    /// - Training: The data used to train the model (not recommended for evaluation)
    /// - Validation: A separate set of data used to tune the model (default and recommended)
    /// - Test: A completely separate set of data used for final evaluation
    /// 
    /// We set "isHigherScoreBetter" to false because in AiDotNet, fitness scores are used for
    /// optimization where lower values are considered better. This is just an internal convention -
    /// you should still interpret Adjusted R-Squared in the normal way (higher is better).
    /// </para>
    /// </remarks>
    public AdjustedRSquaredFitnessCalculator(DataSetType dataSetType = DataSetType.Validation) : base(isHigherScoreBetter: false, dataSetType)
    {
    }

    /// <summary>
    /// Retrieves the Adjusted R-Squared value from the dataset statistics.
    /// </summary>
    /// <param name="dataSet">The dataset containing prediction statistics.</param>
    /// <returns>The Adjusted R-Squared value for the specified dataset.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method extracts the Adjusted R-Squared value from your model's
    /// prediction statistics. This value tells you how well your model fits the data while
    /// accounting for the number of features used.
    /// 
    /// The method is called internally when you evaluate your model's fitness, so you
    /// typically won't need to call it directly.
    /// </para>
    /// </remarks>
    protected override T GetFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        return dataSet.PredictionStats.AdjustedR2;
    }
}
