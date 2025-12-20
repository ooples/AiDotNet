namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that uses Elastic Net Loss to evaluate model performance while encouraging simpler models through regularization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This calculator helps you evaluate how well your model is performing while also encouraging it to be simpler.
/// 
/// Elastic Net Loss combines two approaches to keep your model from becoming too complex:
/// 1. It measures how well your predictions match the actual values (like other loss functions)
/// 2. It adds penalties for having too many or too large parameters in your model
/// 
/// Think of it like building a bridge:
/// - You want the bridge to be strong enough to do its job (make good predictions)
/// - But you also want to use as few materials as possible (keep the model simple)
/// - Elastic Net helps you find this balance
/// 
/// Some common applications include:
/// - Financial predictions where you want to identify only the most important factors
/// - Medical models where you need to know which few symptoms are most predictive
/// - Any situation where you have many potential input features but want to use only the most important ones
/// 
/// Elastic Net is particularly useful when you have many input features that might be related to each other,
/// as it helps select the most important ones while reducing the impact of less important or redundant features.
/// </para>
/// </remarks>
public class ElasticNetLossFitnessCalculator<T, TInput, TOutput> : FitnessCalculatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// The ratio that determines the mix between L1 (absolute value) and L2 (squared value) regularization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This value controls the balance between two different ways of keeping your model simple:
    /// - When closer to 1: Favors removing less important features entirely (making them exactly zero)
    /// - When closer to 0: Favors making all features smaller but keeping more of them
    /// 
    /// The default value of 0.5 provides a balanced approach between these two strategies.
    /// </para>
    /// </remarks>
    private readonly T _l1Ratio;

    /// <summary>
    /// The strength of the regularization penalty applied to the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This value controls how strongly the model is encouraged to be simple:
    /// - Higher values: Strongly encourage simplicity (even if it means slightly worse predictions)
    /// - Lower values: Focus more on making accurate predictions (allowing more complexity)
    /// 
    /// The default value of 1.0 provides a moderate level of regularization.
    /// </para>
    /// </remarks>
    private readonly T _alpha;

    /// <summary>
    /// Initializes a new instance of the ElasticNetLossFitnessCalculator class.
    /// </summary>
    /// <param name="l1Ratio">The ratio between L1 and L2 regularization (default is 0.5).</param>
    /// <param name="alpha">The strength of the regularization penalty (default is 1.0).</param>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calculator that will use Elastic Net Loss
    /// to evaluate your model's performance while encouraging simplicity.
    /// 
    /// Parameters:
    /// - l1Ratio: Controls the balance between removing features entirely (values closer to 1) 
    ///   versus making all features smaller (values closer to 0). Default is 0.5 for a balanced approach.
    /// 
    /// - alpha: Controls how strongly to enforce simplicity. Higher values (like 10.0) strongly push for
    ///   simpler models, while lower values (like 0.1) focus more on prediction accuracy. Default is 1.0.
    /// 
    /// - dataSetType: Lets you choose which data to evaluate:
    ///   * Training: The data used to train the model (not recommended for evaluation)
    ///   * Validation: A separate set of data used to tune the model (default and recommended)
    ///   * Test: A completely separate set of data used for final evaluation
    /// 
    /// We set the first parameter to "false" because in Elastic Net Loss, lower values indicate
    /// better performance (0 would be a perfect model). This tells the system that smaller numbers are better.
    /// </para>
    /// </remarks>
    public ElasticNetLossFitnessCalculator(T? l1Ratio = default, T? alpha = default, DataSetType dataSetType = DataSetType.Validation)
        : base(false, dataSetType)
    {
        _l1Ratio = l1Ratio ?? _numOps.FromDouble(0.5);
        _alpha = alpha ?? _numOps.FromDouble(1.0);
    }

    /// <summary>
    /// Calculates the Elastic Net Loss between predicted and actual values, including regularization penalties.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The Elastic Net Loss value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how well your model's predictions match the actual values,
    /// while also adding penalties for complexity.
    /// 
    /// The method works by:
    /// 1. Calculating how far off your predictions are from the actual values
    /// 2. Adding penalties based on the size and number of parameters in your model
    /// 3. Combining these into a single score where lower values are better
    /// 
    /// The penalties help prevent "overfitting," which is when a model performs well on training data
    /// but poorly on new data because it's too complex and has essentially memorized the training data
    /// rather than learning general patterns.
    /// 
    /// The method is called internally when you evaluate your model's fitness, so you
    /// typically won't need to call it directly.
    /// </para>
    /// </remarks>
    protected override T GetFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        return new ElasticNetLoss<T>(Convert.ToDouble(_l1Ratio), Convert.ToDouble(_alpha)).CalculateLoss(
            ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Predicted), ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Actual));
    }
}
