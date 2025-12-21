namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that uses Focal Loss to evaluate model performance, particularly for imbalanced classification problems.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This calculator helps evaluate how well your model is performing on classification tasks,
/// especially when some classes appear much more frequently than others in your data.
/// 
/// Focal Loss is designed to solve a common problem in AI:
/// When one class is very common (like "normal emails") and another is rare (like "spam emails"),
/// models tend to focus too much on the common class and perform poorly on the rare class.
/// 
/// Focal Loss works by:
/// - Giving more importance to the difficult, misclassified examples
/// - Reducing the importance of the easy, well-classified examples
/// 
/// Think of it like a teacher who spends more time helping students with difficult problems
/// and less time on problems the students already understand well.
/// 
/// The two main parameters that control Focal Loss are:
/// - gamma: Controls how much to focus on hard-to-classify examples (higher values = more focus)
/// - alpha: Helps balance between different classes (adjusts for class imbalance)
/// 
/// Common applications include:
/// - Object detection in images (where most of the image is background)
/// - Medical diagnosis of rare conditions
/// - Fraud detection (where most transactions are legitimate)
/// - Any situation with imbalanced classes
/// </para>
/// </remarks>
public class FocalLossFitnessCalculator<T, TInput, TOutput> : FitnessCalculatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// The focusing parameter that controls how much to down-weight easy examples.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This parameter (gamma) controls how much the model should focus on hard examples.
    /// 
    /// - When gamma = 0: Focal Loss becomes the standard Cross-Entropy Loss
    /// - When gamma increases: The model pays more attention to difficult, misclassified examples
    /// 
    /// Think of it like a "difficulty dial":
    /// - Low gamma: Treats all examples more equally
    /// - High gamma: Focuses training heavily on the examples your model gets wrong
    /// 
    /// The default value of 2.0 works well for many problems, but you might adjust it:
    /// - Increase it if your model is ignoring rare classes
    /// - Decrease it if your model is overfitting to a few difficult examples
    /// </para>
    /// </remarks>
    private readonly T _gamma;

    /// <summary>
    /// The weighting factor that balances the importance of positive/negative examples.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This parameter (alpha) helps balance between different classes when they appear
    /// with different frequencies in your data.
    /// 
    /// - When alpha = 0.5: Both classes are treated equally
    /// - When alpha < 0.5: Gives less weight to the positive class
    /// - When alpha > 0.5: Gives more weight to the positive class
    /// 
    /// Think of it like adjusting the scales of justice:
    /// - If you have 1000 "normal" examples but only 10 "rare" examples, alpha helps
    ///   make sure the model doesn't ignore the rare examples
    /// 
    /// The default value of 0.25 is often a good starting point, but you might need to adjust it
    /// based on how imbalanced your classes are.
    /// </para>
    /// </remarks>
    private readonly T _alpha;

    /// <summary>
    /// Initializes a new instance of the FocalLossFitnessCalculator class.
    /// </summary>
    /// <param name="gamma">The focusing parameter that controls how much to down-weight easy examples (default is 2.0).</param>
    /// <param name="alpha">The weighting factor that balances the importance of positive/negative examples (default is 0.25).</param>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calculator that will use Focal Loss
    /// to evaluate your model's performance, with special attention to difficult examples
    /// and handling class imbalance.
    /// 
    /// Parameters:
    /// - gamma: Controls focus on hard examples (default 2.0)
    ///   * Higher values make the model focus more on misclassified examples
    ///   * Lower values treat all examples more equally
    /// 
    /// - alpha: Balances between classes (default 0.25)
    ///   * Adjusts for situations where one class appears more frequently than others
    ///   * Values closer to 0 give less weight to the positive class
    ///   * Values closer to 1 give more weight to the positive class
    /// 
    /// - dataSetType: Which data to evaluate (default is Validation)
    ///   * Training: The data used to train the model (not recommended for evaluation)
    ///   * Validation: A separate set of data used to tune the model (recommended)
    ///   * Test: A completely separate set of data used for final evaluation
    /// 
    /// We set the first parameter to "false" in the base constructor because in Focal Loss,
    /// lower values indicate better performance (0 would be a perfect model).
    /// </para>
    /// </remarks>
    public FocalLossFitnessCalculator(T? gamma = default, T? alpha = default, DataSetType dataSetType = DataSetType.Validation)
        : base(false, dataSetType)
    {
        _gamma = gamma ?? _numOps.FromDouble(2.0);
        _alpha = alpha ?? _numOps.FromDouble(0.25);
    }

    /// <summary>
    /// Calculates the Focal Loss fitness score for the given dataset.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The calculated Focal Loss score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how well your model is performing using Focal Loss.
    /// 
    /// It works by:
    /// 1. Taking the predictions your model made
    /// 2. Comparing them to the actual correct answers
    /// 3. Calculating a score that represents how wrong the model was, with special attention to:
    ///    - Hard examples (controlled by gamma)
    ///    - Class balance (controlled by alpha)
    /// 
    /// A lower score means better performance (0 would be perfect).
    /// 
    /// This method uses the NeuralNetworkHelper to do the actual Focal Loss calculation,
    /// passing in your model's predictions, the actual values, and the gamma and alpha parameters.
    /// </para>
    /// </remarks>
    protected override T GetFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        return new FocalLoss<T>(Convert.ToDouble(_gamma), Convert.ToDouble(_alpha)).CalculateLoss(ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Predicted),
            ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Actual));
    }
}
