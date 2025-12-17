namespace AiDotNet.CrossValidators;

/// <summary>
/// Implements a Leave-One-Out cross-validation strategy for model evaluation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
/// <typeparam name="TInput">The type of input data (e.g., Matrix&lt;T&gt; for tabular data, Tensor&lt;T&gt; for images).</typeparam>
/// <typeparam name="TOutput">The type of output data (e.g., Vector&lt;T&gt; for predictions, custom types for other formats).</typeparam>
/// <remarks>
/// <para>
/// This class provides a Leave-One-Out cross-validation implementation, where each data point is used once as the validation set
/// while the remaining data points form the training set.
/// </para>
/// <para><b>For Beginners:</b> Leave-One-Out cross-validation is like testing your model on each data point individually.
///
/// What this class does:
/// - For each data point in your dataset:
///   - Uses that single point for testing
///   - Uses all other points for training
/// - Repeats this process for every single data point
/// - Calculates how well your model performs on average across all these tests
///
/// This is useful because:
/// - It uses almost all of your data for training in each iteration
/// - It gives you a performance estimate for each individual data point
/// - It's particularly useful for small datasets
///
/// However, it can be computationally expensive for large datasets.
/// </para>
/// </remarks>
public class LeaveOneOutCrossValidator<T, TInput, TOutput> : CrossValidatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the LeaveOneOutCrossValidator class.
    /// </summary>
    /// <param name="options">The options for cross-validation. If null, default options are used.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes the LeaveOneOutCrossValidator with the provided options or default options if none are specified.
    /// It sets up the cross-validator to perform leave-one-out cross-validation.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the leave-one-out cross-validator.
    /// 
    /// What it does:
    /// - Takes in your preferences for how to do the cross-validation (or uses default settings if you don't specify any)
    /// - Prepares the cross-validator to use each data point once as a test set
    /// - Gets everything ready to start the cross-validation process
    /// 
    /// It's like setting up a series of tests where each piece of data gets a chance to be the test case.
    /// </para>
    /// </remarks>
    public LeaveOneOutCrossValidator(CrossValidationOptions? options = null) : base(options ?? new())
    {
    }

    /// <summary>
    /// Performs the leave-one-out cross-validation process on the given model using the provided data and optimizer.
    /// </summary>
    /// <param name="model">The machine learning model to validate.</param>
    /// <param name="X">The feature matrix containing the input data.</param>
    /// <param name="y">The target vector containing the output data.</param>
    /// <param name="optimizer">The optimizer to use for training the model on each fold.</param>
    /// <returns>A CrossValidationResult containing the results of the validation process.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the core leave-one-out cross-validation logic. It creates the folds using the CreateFolds method,
    /// where each fold consists of a single data point for validation and all other points for training.
    /// </para>
    /// <para><b>For Beginners:</b> This method is where the actual leave-one-out cross-validation happens.
    ///
    /// What it does:
    /// - Takes your model, your data (X and y), and an optimizer for training
    /// - Creates folds where each fold is just one data point (using the CreateFolds method)
    /// - Runs the PerformCrossValidation method, which:
    ///   - Trains the model using the optimizer on all but one data point and tests it on that left-out point
    ///   - Repeats this for every single data point
    ///   - Collects and summarizes the results of all these tests
    ///
    /// The optimizer ensures consistent training across all folds.
    ///
    /// It's like putting your model through a series of tests, each focused on a single data point,
    /// using a standardized training procedure, and then giving you a report card that shows how well it performed overall.
    /// </para>
    /// </remarks>
    public override CrossValidationResult<T, TInput, TOutput> Validate(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y,
        IOptimizer<T, TInput, TOutput> optimizer)
    {
        var folds = CreateFolds(X, y);
        return PerformCrossValidation(model, X, y, folds, optimizer);
    }

    /// <summary>
    /// Creates the folds for leave-one-out cross-validation.
    /// </summary>
    /// <param name="X">The input data.</param>
    /// <param name="y">The output data.</param>
    /// <returns>An enumerable of tuples containing the train and validation indices for each fold.</returns>
    /// <remarks>
    /// <para>
    /// This method generates the indices for the training and validation sets for each fold of the leave-one-out cross-validation process.
    /// Each fold consists of a single data point for validation and all other points for training.
    /// </para>
    /// <para><b>For Beginners:</b> This method decides how to split your data for leave-one-out cross-validation.
    ///
    /// What it does:
    /// - Creates a fold for each data point in your dataset
    /// - For each fold:
    ///   - Uses one data point as the validation data
    ///   - Uses all other data points as the training data
    /// - Returns these splits so the main method can use them
    ///
    /// It's like taking each card from a deck one at a time, using it as a test card,
    /// and using all the other cards for training, then repeating this for every card in the deck.
    /// </para>
    /// </remarks>
    private IEnumerable<(int[] trainIndices, int[] validationIndices)> CreateFolds(TInput X, TOutput y)
    {
        int totalSamples = InputHelper<T, TInput>.GetBatchSize(X);
        var allIndices = Enumerable.Range(0, totalSamples).ToArray();

        if (Options.ShuffleData)
        {
            allIndices = [.. allIndices.OrderBy(x => Random.Next())];
        }

        for (int i = 0; i < totalSamples; i++)
        {
            int[] validationIndex = [allIndices[i]];
            int[] trainIndices = [.. allIndices.Where(index => index != allIndices[i])];

            yield return (trainIndices, validationIndex);
        }
    }
}
