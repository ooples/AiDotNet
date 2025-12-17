namespace AiDotNet.FitnessCalculators;

/// <summary>
/// Base class for all fitness calculators that evaluate how well a model performs.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This is a foundation class that all fitness calculators build upon.
/// 
/// Think of a fitness calculator like a judge in a competition:
/// - It looks at how your AI model performed (its predictions vs. the actual answers)
/// - It gives a score based on specific criteria (like accuracy, error rate, etc.)
/// - It helps you determine if one model is better than another
/// 
/// Different fitness calculators judge models in different ways, just like different 
/// sports have different scoring systems. Some calculators consider higher scores better 
/// (like accuracy), while others consider lower scores better (like error rates).
/// 
/// This base class provides the common functionality that all these different "judges" share.
/// </para>
/// </remarks>
public abstract class FitnessCalculatorBase<T, TInput, TOutput> : IFitnessCalculator<T, TInput, TOutput>
{
    /// <summary>
    /// Indicates whether higher fitness scores represent better performance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This tells us whether bigger numbers mean better performance.
    /// 
    /// For example:
    /// - For accuracy: Higher is better (100% accuracy is perfect)
    /// - For error rates: Lower is better (0% error is perfect)
    /// 
    /// This helps the system know how to compare different models.
    /// </para>
    /// </remarks>
    protected bool _isHigherScoreBetter;

    /// <summary>
    /// Provides mathematical operations for the specific numeric type being used.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a toolkit that helps perform math operations 
    /// regardless of whether we're using integers, decimals, doubles, etc.
    /// 
    /// It allows the calculator to work with different numeric types without 
    /// having to rewrite the math operations for each type.
    /// </para>
    /// </remarks>
    protected readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Specifies which dataset (training, validation, or testing) to use for fitness calculation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This tells the calculator which set of data to use when evaluating your model.
    /// 
    /// There are typically three types of data:
    /// - Training data: Used to teach the model (like study materials)
    /// - Validation data: Used to fine-tune the model (like practice tests)
    /// - Testing data: Used for final evaluation (like the actual exam)
    /// 
    /// Usually, we use validation data to evaluate fitness during training.
    /// </para>
    /// </remarks>
    protected readonly DataSetType _dataSetType;

    /// <summary>
    /// Initializes a new instance of the FitnessCalculatorBase class.
    /// </summary>
    /// <param name="isHigherScoreBetter">Indicates whether higher scores represent better performance.</param>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This sets up the basic properties of the fitness calculator.
    /// 
    /// Parameters:
    /// - isHigherScoreBetter: Tells the system whether bigger numbers mean better performance
    ///   (true for metrics like accuracy, false for metrics like error rates)
    /// 
    /// - dataSetType: Specifies which data to use when evaluating the model:
    ///   * Training: The data used to teach the model (not usually recommended for evaluation)
    ///   * Validation: A separate set of data used to tune the model (default and recommended)
    ///   * Testing: A completely separate set of data used for final evaluation
    /// </para>
    /// </remarks>
    protected FitnessCalculatorBase(bool isHigherScoreBetter, DataSetType dataSetType = DataSetType.Validation)
    {
        _isHigherScoreBetter = isHigherScoreBetter;
        _numOps = MathHelper.GetNumericOperations<T>();
        _dataSetType = dataSetType;
    }

    /// <summary>
    /// Calculates the fitness score for a model using the specified evaluation data.
    /// </summary>
    /// <param name="evaluationData">The data containing model predictions and actual values.</param>
    /// <returns>The calculated fitness score.</returns>
    /// <exception cref="ArgumentNullException">Thrown when evaluationData is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the requested dataset is not available.</exception>
    /// <exception cref="ArgumentException">Thrown when an unsupported DataSetType is specified.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how well your model is performing based on its predictions.
    /// 
    /// It works by:
    /// 1. Taking the evaluation data that contains both predictions and actual values
    /// 2. Selecting the appropriate dataset (training, validation, or testing)
    /// 3. Calculating a score that represents how well the model performed
    /// 
    /// The score's meaning depends on which specific fitness calculator you're using:
    /// - It might be accuracy (higher is better)
    /// - It might be an error rate (lower is better)
    /// - It might be some other measure of performance
    /// 
    /// This method handles the common logic, while the specific scoring method is defined
    /// in each calculator that extends this base class.
    /// </para>
    /// </remarks>
    public T CalculateFitnessScore(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        if (evaluationData == null)
            throw new ArgumentNullException(nameof(evaluationData));

        DataSetStats<T, TInput, TOutput> dataSet = _dataSetType switch
        {
            DataSetType.Training => evaluationData.TrainingSet,
            DataSetType.Validation => evaluationData.ValidationSet,
            DataSetType.Testing => evaluationData.TestSet,
            _ => throw new ArgumentException($"Unsupported DataSetType: {_dataSetType}")
        };

        return dataSet == null
            ? throw new InvalidOperationException($"The {_dataSetType} dataset is not available in the provided ModelEvaluationData.")
            : GetFitnessScore(dataSet);
    }

    /// <summary>
    /// Calculates the fitness score using a specific dataset.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The calculated fitness score.</returns>
    /// <exception cref="ArgumentNullException">Thrown when dataSet is null.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is an alternative way to calculate your model's performance
    /// when you already have a specific dataset you want to use.
    /// 
    /// Instead of providing all the evaluation data and letting the calculator choose
    /// which dataset to use, you directly provide the exact dataset you want to evaluate.
    /// 
    /// This is useful when you want to:
    /// - Calculate fitness on a custom dataset
    /// - Compare performance across different datasets
    /// - Evaluate a model on data that wasn't part of the original training/validation/testing split
    /// </para>
    /// </remarks>
    public T CalculateFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        if (dataSet == null)
            throw new ArgumentNullException(nameof(dataSet));

        return GetFitnessScore(dataSet);
    }

    /// <summary>
    /// Abstract method that must be implemented by derived classes to calculate the specific fitness score.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The calculated fitness score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a placeholder method that each specific calculator must fill in.
    /// 
    /// Think of it like a template that says "here's where you put your specific scoring logic."
    /// Each calculator that extends this base class will provide its own implementation of this method,
    /// defining exactly how it calculates the fitness score.
    /// 
    /// For example:
    /// - An accuracy calculator would implement this to calculate the percentage of correct predictions
    /// - A mean squared error calculator would implement this to calculate the average squared difference
    ///   between predictions and actual values
    /// </para>
    /// </remarks>
    protected abstract T GetFitnessScore(DataSetStats<T, TInput, TOutput> dataSet);

    /// <summary>
    /// Gets a value indicating whether higher fitness scores represent better performance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This property tells you whether bigger numbers mean better performance for this calculator.
    /// 
    /// For example:
    /// - For accuracy: IsHigherScoreBetter is true (100% accuracy is better than 90%)
    /// - For error rates: IsHigherScoreBetter is false (1% error is better than 10%)
    /// 
    /// This helps you interpret the scores correctly when comparing different models.
    /// </para>
    /// </remarks>
    public bool IsHigherScoreBetter => _isHigherScoreBetter;

    /// <summary>
    /// Determines whether a new fitness score is better than the current best score.
    /// </summary>
    /// <param name="newScore">The new fitness score to evaluate.</param>
    /// <param name="currentBestScore">The current best fitness score.</param>
    /// <returns>True if the new score is better than the current best score; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method compares two scores and tells you which one is better.
    /// 
    /// It takes into account whether higher scores are better or lower scores are better:
    /// - If higher scores are better (like accuracy), it returns true when the new score is higher
    /// - If lower scores are better (like error rates), it returns true when the new score is lower
    /// 
    /// This is particularly useful when:
    /// - Selecting the best model from multiple options
    /// - Deciding whether to keep training or stop (if the model isn't improving)
    /// - Saving the best model during training
    /// </para>
    /// </remarks>
    public bool IsBetterFitness(T newScore, T currentBestScore)
    {
        return _isHigherScoreBetter
            ? _numOps.GreaterThan(newScore, currentBestScore)
            : _numOps.LessThan(newScore, currentBestScore);
    }
}
