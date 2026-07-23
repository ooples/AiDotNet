namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that uses Poisson Loss to evaluate model performance, particularly for count-based prediction tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This calculator helps evaluate how well your model is performing on count-based prediction tasks,
/// which are problems where you're predicting the number of times something happens in a fixed interval.
/// 
/// Poisson Loss is designed specifically for problems where:
/// - You're predicting counts or frequencies (whole numbers, 0 or greater)
/// - The events occur independently of each other
/// - The average rate of events is constant
/// 
/// Examples of count-based prediction problems:
/// - Number of customer arrivals per hour
/// - Number of goals scored in a soccer match
/// - Number of website visits per day
/// - Number of defects in a manufacturing process
/// - Number of calls to a call center per hour
/// 
/// How Poisson Loss works:
/// - It's based on the Poisson distribution, which models random events occurring over a fixed interval
/// - It's particularly good for data where most values are small counts (0, 1, 2, etc.) but occasionally have larger values
/// - It penalizes both overestimation and underestimation, but in a way that's appropriate for count data
/// 
/// Think of it like this:
/// Imagine you're predicting how many customers will visit a store each hour:
/// - Some hours might have 0 customers
/// - Most hours might have 5-10 customers
/// - Occasionally, there might be 20+ customers
/// - Poisson Loss is designed to handle this kind of pattern well
/// 
/// Key characteristics:
/// - It's specifically designed for count data
/// - It works well when the variance of your data increases with the mean
/// - Lower values are better (0 would be perfect predictions)
/// - It assumes non-negative values (counts can't be negative)
/// 
/// When to use this calculator:
/// - When your target values are counts (whole numbers, 0 or greater)
/// - When your data follows a Poisson-like distribution (many small values, fewer large values)
/// - When you're predicting the frequency of events in a fixed time or space interval
/// </para>
/// </remarks>
public class PoissonLossFitnessCalculator<T, TInput, TOutput> : FitnessCalculatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the PoissonLossFitnessCalculator class.
    /// </summary>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calculator that will use Poisson Loss
    /// to evaluate your model's performance on count-based prediction problems.
    /// 
    /// Parameter:
    /// - dataSetType: Which data to evaluate (default is Validation)
    ///   * Training: The data used to train the model (not recommended for evaluation)
    ///   * Validation: A separate set of data used to tune the model (recommended)
    ///   * Test: A completely separate set of data used for final evaluation
    /// 
    /// We set the first parameter to "false" in the base constructor because in Poisson Loss,
    /// lower values indicate better performance (0 would be perfect).
    /// 
    /// When to use this calculator:
    /// - When you're predicting counts or frequencies (like number of events per time period)
    /// - When your data consists of non-negative integers (0, 1, 2, 3, etc.)
    /// - When your data might have occasional large values but mostly smaller values
    /// </para>
    /// </remarks>
    public PoissonLossFitnessCalculator(DataSetType dataSetType = DataSetType.Validation)
        : base(false, dataSetType)
    {
    }

    /// <summary>
    /// Calculates the Poisson Loss fitness score for the given dataset.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The calculated Poisson Loss score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how well your model is performing using Poisson Loss.
    /// 
    /// It works by comparing your model's predictions to the actual values and calculating
    /// a score that represents how far off your predictions are. The calculation is specifically
    /// designed for count data (like predicting the number of events).
    /// 
    /// A lower score means better performance (0 would be perfect).
    /// 
    /// The Poisson Loss takes into account that:
    /// - Count data often follows a Poisson distribution
    /// - The variance of count data often increases with the mean
    /// - Predictions should be non-negative (you can't have negative counts)
    /// 
    /// This loss function is particularly useful when:
    /// - Your target values are counts or frequencies
    /// - Your data has many small values and fewer large values
    /// - You're predicting events that occur independently at a constant average rate
    /// </para>
    /// </remarks>
    protected override T GetFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        return new PoissonLoss<T>().CalculateLoss(ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Predicted),
            ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Actual));
    }
}
