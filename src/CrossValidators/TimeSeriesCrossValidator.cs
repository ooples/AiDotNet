namespace AiDotNet.CrossValidators;

/// <summary>
/// Implements a time series cross-validation strategy for model evaluation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
/// <typeparam name="TInput">The type of input data (e.g., Matrix&lt;T&gt; for tabular data, Tensor&lt;T&gt; for images).</typeparam>
/// <typeparam name="TOutput">The type of output data (e.g., Vector&lt;T&gt; for predictions, custom types for other formats).</typeparam>
/// <remarks>
/// <para>
/// This class provides a time series cross-validation implementation, which respects the temporal order of the data.
/// It uses an expanding window approach, where the training set grows over time.
/// </para>
/// <para><b>For Beginners:</b> Time series cross-validation is designed for data that has a time component.
///
/// What this class does:
/// - Starts with a small portion of your data for training
/// - Uses the next part for validation
/// - Expands the training set to include the previous validation set
/// - Repeats this process, moving forward in time
/// - Calculates how well your model performs on average across all these tests
///
/// This is useful because:
/// - It respects the time order of your data
/// - It simulates how the model would perform in a real-world scenario where you use past data to predict the future
/// - It helps detect if your model's performance changes over time
/// </para>
/// </remarks>
public class TimeSeriesCrossValidator<T, TInput, TOutput> : CrossValidatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// The initial size of the training set.
    /// </summary>
    /// <remarks>
    /// This determines how many data points are used for the first training set.
    /// For beginners: Think of this as the minimum amount of historical data you need to start making predictions.
    /// </remarks>
    private readonly int _initialTrainSize;

    /// <summary>
    /// The size of the validation set.
    /// </summary>
    /// <remarks>
    /// This determines how many data points are used for validation in each fold.
    /// For beginners: This is like the number of future data points you're trying to predict each time.
    /// </remarks>
    private readonly int _validationSize;

    /// <summary>
    /// The step size for expanding the training set.
    /// </summary>
    /// <remarks>
    /// This determines how many data points are added to the training set in each iteration.
    /// For beginners: This is like how much new data you incorporate before making your next prediction.
    /// </remarks>
    private readonly int _step;

    /// <summary>
    /// Initializes a new instance of the TimeSeriesCrossValidator class.
    /// </summary>
    /// <param name="initialTrainSize">The initial size of the training set.</param>
    /// <param name="validationSize">The size of the validation set.</param>
    /// <param name="step">The step size for expanding the training set.</param>
    /// <param name="options">The options for cross-validation. If null, default options are used.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes the TimeSeriesCrossValidator with the provided parameters and options.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the time series cross-validator with your chosen settings.
    /// 
    /// What it does:
    /// - Takes in your preferences for how to split the data over time
    /// - Prepares the cross-validator to use these time-based splits
    /// - Gets everything ready to start the cross-validation process
    /// 
    /// It's like setting up a series of tests that respect the time order of your data.
    /// </para>
    /// </remarks>
    public TimeSeriesCrossValidator(int initialTrainSize, int validationSize, int step, CrossValidationOptions? options = null)
        : base(options ?? new())
    {
        _initialTrainSize = initialTrainSize;
        _validationSize = validationSize;
        _step = step;
    }

    /// <summary>
    /// Performs the time series cross-validation process on the given model using the provided data and optimizer.
    /// </summary>
    /// <param name="model">The machine learning model to validate.</param>
    /// <param name="X">The feature matrix containing the input data.</param>
    /// <param name="y">The target vector containing the output data.</param>
    /// <param name="optimizer">The optimizer to use for training the model on each fold.</param>
    /// <returns>A CrossValidationResult containing the results of the validation process.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the core time series cross-validation logic. It creates the folds using the CreateFolds method,
    /// respecting the temporal order of the data, then performs the cross-validation using these folds and the provided optimizer.
    /// </para>
    /// <para><b>For Beginners:</b> This method is where the actual time series cross-validation happens.
    ///
    /// What it does:
    /// - Takes your model, your time-ordered data (X and y), and an optimizer for training
    /// - Creates time-based folds using the CreateFolds method
    /// - Runs the PerformCrossValidation method, which:
    ///   - Trains your model using the optimizer multiple times, each time moving forward in time
    ///   - Collects and summarizes the results of all these tests
    ///
    /// The optimizer ensures consistent training across all folds.
    ///
    /// It's like putting your model through a series of tests that simulate how it would perform
    /// if you were using it to make predictions over time, with a standardized training procedure.
    /// </para>
    /// </remarks>
    public override CrossValidationResult<T, TInput, TOutput> Validate(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y,
        IOptimizer<T, TInput, TOutput> optimizer)
    {
        var folds = CreateFolds(X, y);
        return PerformCrossValidation(model, X, y, folds, optimizer);
    }

    /// <summary>
    /// Creates the folds for time series cross-validation based on the provided parameters.
    /// </summary>
    /// <param name="X">The feature matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <returns>An enumerable of tuples containing the train and validation indices for each fold.</returns>
    /// <remarks>
    /// <para>
    /// This method generates the indices for the training and validation sets for each fold of the time series cross-validation process. 
    /// It uses an expanding window approach, where the training set grows over time and the validation set is always ahead of the training set.
    /// </para>
    /// <para><b>For Beginners:</b> This method decides how to split your time-ordered data for cross-validation.
    /// 
    /// What it does:
    /// - Starts with the initial training size
    /// - For each fold:
    ///   - Uses all data up to a certain point for training
    ///   - Uses the next chunk of data for validation
    ///   - Moves forward in time by the step size for the next fold
    /// - Returns these time-based splits so the main method can use them
    /// 
    /// It's like reading through a history book, using more and more of the past to predict
    /// what happens next, and then checking if your prediction was correct.
    /// </para>
    /// </remarks>
    private IEnumerable<(int[] trainIndices, int[] validationIndices)> CreateFolds(TInput X, TOutput y)
    {
        int totalSamples = InputHelper<T, TInput>.GetBatchSize(X);

        for (int trainEnd = _initialTrainSize; trainEnd < totalSamples - _validationSize; trainEnd += _step)
        {
            int[] trainIndices = [.. Enumerable.Range(0, trainEnd)];
            int[] validationIndices = [.. Enumerable.Range(trainEnd, _validationSize)];

            yield return (trainIndices, validationIndices);
        }
    }
}
