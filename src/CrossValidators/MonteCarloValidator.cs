namespace AiDotNet.CrossValidators;

/// <summary>
/// Implements a Monte Carlo cross-validation strategy for model evaluation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
/// <typeparam name="TInput">The type of input data (e.g., Matrix&lt;T&gt; for tabular data, Tensor&lt;T&gt; for images).</typeparam>
/// <typeparam name="TOutput">The type of output data (e.g., Vector&lt;T&gt; for predictions, custom types for other formats).</typeparam>
/// <remarks>
/// <para>
/// This class provides a Monte Carlo cross-validation implementation, which randomly splits the data into training and validation sets multiple times.
/// </para>
/// <para><b>For Beginners:</b> Monte Carlo cross-validation is like repeatedly shuffling and splitting your data to test your model.
///
/// What this class does:
/// - Randomly splits your data into training and validation sets
/// - Repeats this process multiple times (as specified in the options)
/// - For each split:
///   - Trains the model on the training set
///   - Evaluates the model on the validation set
/// - Calculates how well your model performs on average across all splits
///
/// This is useful because:
/// - It provides a robust estimate of model performance
/// - It helps to reduce the impact of how the data is split on the results
/// - It can be more flexible than k-fold cross-validation for certain types of data
///
/// However, it can be computationally expensive for a large number of iterations.
/// </para>
/// </remarks>
public class MonteCarloValidator<T, TInput, TOutput> : CrossValidatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the MonteCarloValidator class.
    /// </summary>
    /// <param name="options">The options for cross-validation. If null, default options are used.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes the MonteCarloValidator with the provided options or default options if none are specified.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the Monte Carlo cross-validator with your chosen settings.
    /// 
    /// What it does:
    /// - Takes in your preferences for how to do the cross-validation (or uses default settings if you don't specify any)
    /// - Prepares the cross-validator to perform Monte Carlo cross-validation based on these settings
    /// 
    /// It's like setting up a series of random tests for your model based on your instructions.
    /// </para>
    /// </remarks>
    private new MonteCarloValidationOptions Options => (MonteCarloValidationOptions)base.Options;

    public MonteCarloValidator(MonteCarloValidationOptions? options = null)
        : base(options ?? new MonteCarloValidationOptions())
    {
    }

    /// <summary>
    /// Performs the Monte Carlo cross-validation process on the given model using the provided data and optimizer.
    /// </summary>
    /// <param name="model">The machine learning model to validate.</param>
    /// <param name="X">The feature matrix containing the input data.</param>
    /// <param name="y">The target vector containing the output data.</param>
    /// <param name="optimizer">The optimizer to use for training the model on each fold.</param>
    /// <returns>A CrossValidationResult containing the results of the validation process.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the core Monte Carlo cross-validation logic. It creates the folds using the CreateFolds method,
    /// where each fold is a random split of the data into training and validation sets.
    /// </para>
    /// <para><b>For Beginners:</b> This method is where the actual Monte Carlo cross-validation happens.
    ///
    /// What it does:
    /// - Takes your model, your data (X and y), and an optimizer for training
    /// - Creates random splits of your data (using the CreateFolds method)
    /// - Runs the PerformCrossValidation method, which:
    ///   - Trains the model using the optimizer on each training set and tests it on the corresponding validation set
    ///   - Repeats this for the number of iterations specified in the options
    ///   - Collects and summarizes the results of all these tests
    ///
    /// The optimizer ensures consistent training across all iterations.
    ///
    /// It's like putting your model through a series of random tests using a standardized training procedure
    /// and then giving you a report card that shows how well it performed overall.
    /// </para>
    /// </remarks>
    public override CrossValidationResult<T, TInput, TOutput> Validate(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y,
        IOptimizer<T, TInput, TOutput> optimizer)
    {
        var folds = CreateFolds(X, y);
        return PerformCrossValidation(model, X, y, folds, optimizer);
    }

    /// <summary>
    /// Creates the folds for Monte Carlo cross-validation.
    /// </summary>
    /// <param name="X">The feature matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <returns>An enumerable of tuples containing the train and validation indices for each fold.</returns>
    /// <remarks>
    /// <para>
    /// This method generates the indices for the training and validation sets for each iteration of the Monte Carlo cross-validation process. 
    /// Each fold is a random split of the data into training and validation sets.
    /// </para>
    /// <para><b>For Beginners:</b> This method decides how to split your data for Monte Carlo cross-validation.
    /// 
    /// What it does:
    /// - Creates a random split for each iteration
    /// - For each split:
    ///   - Randomly selects some data points for validation
    ///   - Uses the remaining data points for training
    /// - Returns these splits so the main method can use them
    /// 
    /// It's like shuffling a deck of cards and dealing them into two piles (training and validation)
    /// multiple times, with each deal creating a new test scenario for your model.
    /// </para>
    /// </remarks>
    private IEnumerable<(int[] trainIndices, int[] validationIndices)> CreateFolds(TInput X, TOutput y)
    {
        int totalSamples = InputHelper<T, TInput>.GetBatchSize(X);
        int validationSize = (int)(totalSamples * Options.ValidationSize);

        for (int i = 0; i < Options.NumberOfFolds; i++)
        {
            var allIndices = Enumerable.Range(0, totalSamples).ToArray();
            var shuffledIndices = allIndices.OrderBy(x => Random.Next()).ToArray();

            var validationIndices = shuffledIndices.Take(validationSize).ToArray();
            var trainIndices = shuffledIndices.Skip(validationSize).ToArray();

            yield return (trainIndices, validationIndices);
        }
    }
}
