namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that uses Quantile Loss to evaluate model performance, particularly for prediction tasks 
/// where you want to estimate specific quantiles of the target distribution.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This calculator helps evaluate how well your model is performing when you want to predict
/// a specific percentile of your data rather than just the average value.
/// 
/// Quantile Loss is designed specifically for problems where:
/// - You're interested in predicting a specific percentile (like the median or 90th percentile)
/// - You want to account for asymmetric risks (where over-prediction and under-prediction have different costs)
/// - You need to make predictions that reflect uncertainty in your data
/// 
/// What is a quantile?
/// A quantile divides your data into equal portions:
/// - The 0.5 quantile (50th percentile) is the median - half the values are above it, half below
/// - The 0.9 quantile (90th percentile) is the value below which 90% of observations fall
/// - The 0.1 quantile (10th percentile) is the value below which 10% of observations fall
/// 
/// Examples of when to use Quantile Loss:
/// - Predicting delivery times (where late deliveries might be more costly than early ones)
/// - Estimating sales (where underestimating inventory needs might be worse than overestimating)
/// - Forecasting energy demand (where underestimating could lead to blackouts)
/// - Predicting financial risk (where you want to know the worst-case scenario at a certain confidence level)
/// 
/// How Quantile Loss works:
/// - It penalizes under-predictions and over-predictions differently based on the quantile you choose
/// - For the median (0.5 quantile), it penalizes both equally (this is also called the Mean Absolute Error)
/// - For higher quantiles (>0.5), it penalizes under-predictions more heavily
/// - For lower quantiles (<0.5), it penalizes over-predictions more heavily
/// 
/// Think of it like this:
/// Imagine you're predicting package delivery times:
/// - If you use the 0.5 quantile (median), you're predicting when half the packages will arrive
/// - If you use the 0.9 quantile, you're predicting a time by which 90% of packages will arrive
///   (useful for giving customers a "delivered by" guarantee)
/// - If you use the 0.1 quantile, you're predicting when the earliest 10% of packages will arrive
/// 
/// Key characteristics:
/// - It allows for asymmetric penalties on errors
/// - Lower values are better (0 would be perfect predictions)
/// - The default quantile is 0.5 (the median)
/// - Different quantiles give you different insights about your predictions
/// </para>
/// </remarks>
public class QuantileLossFitnessCalculator<T, TInput, TOutput> : FitnessCalculatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// The quantile value to use for the loss calculation, between 0 and 1.
    /// </summary>
    /// <remarks>
    /// Common values include:
    /// - 0.5 for median (default)
    /// - 0.9 for 90th percentile
    /// - 0.1 for 10th percentile
    /// </remarks>
    private readonly T _quantile;

    /// <summary>
    /// Initializes a new instance of the QuantileLossFitnessCalculator class.
    /// </summary>
    /// <param name="quantile">The quantile to use for loss calculation (between 0 and 1, default is 0.5).</param>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calculator that will use Quantile Loss
    /// to evaluate your model's performance when predicting specific percentiles.
    /// 
    /// Parameters:
    /// - quantile: Which percentile you want to predict (between 0 and 1)
    ///   * 0.5 means the median (middle value) - this is the default
    ///   * 0.9 means the 90th percentile (value below which 90% of data falls)
    ///   * 0.1 means the 10th percentile (value below which 10% of data falls)
    /// 
    /// - dataSetType: Which data to evaluate (default is Validation)
    ///   * Training: The data used to train the model (not recommended for evaluation)
    ///   * Validation: A separate set of data used to tune the model (recommended)
    ///   * Test: A completely separate set of data used for final evaluation
    /// 
    /// We set the first parameter to "false" in the base constructor because in Quantile Loss,
    /// lower values indicate better performance (0 would be perfect).
    /// 
    /// When to use different quantiles:
    /// - Use 0.5 (median) when over-predictions and under-predictions are equally bad
    /// - Use higher quantiles (like 0.9) when under-predictions are more costly
    ///   (e.g., when underestimating how much inventory you need)
    /// - Use lower quantiles (like 0.1) when over-predictions are more costly
    ///   (e.g., when overestimating revenue could lead to poor business decisions)
    /// </para>
    /// </remarks>
    public QuantileLossFitnessCalculator(T? quantile = default, DataSetType dataSetType = DataSetType.Validation)
        : base(false, dataSetType)
    {
        _quantile = quantile ?? _numOps.FromDouble(0.5);
    }

    /// <summary>
    /// Calculates the Quantile Loss fitness score for the given dataset.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The calculated Quantile Loss score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how well your model is performing using Quantile Loss.
    /// 
    /// It works by comparing your model's predictions to the actual values, but with a twist:
    /// - It penalizes under-predictions and over-predictions differently based on the quantile you chose
    /// - For the median (0.5 quantile), it treats both types of errors equally
    /// - For higher quantiles (>0.5), it penalizes under-predictions more heavily
    /// - For lower quantiles (<0.5), it penalizes over-predictions more heavily
    /// 
    /// A lower score means better performance (0 would be perfect).
    /// 
    /// This is particularly useful when:
    /// - The cost of over-prediction and under-prediction is different
    /// - You need to make predictions at specific percentiles (like the 90th percentile)
    /// - You want to capture the uncertainty in your predictions
    /// 
    /// For example:
    /// - If predicting delivery times, a 0.9 quantile loss would help ensure 90% of deliveries arrive on time
    /// - If predicting sales, a 0.1 quantile loss would give you a conservative estimate to avoid overstocking
    /// </para>
    /// </remarks>
    protected override T GetFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        return new QuantileLoss<T>(_numOps.ToDouble(_quantile)).CalculateLoss(ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Predicted),
            ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Actual));
    }
}
