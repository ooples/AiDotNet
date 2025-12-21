namespace AiDotNet.CrossValidators;

/// <summary>
/// Implements a k-fold cross-validation strategy for model evaluation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
/// <typeparam name="TInput">The type of input data (e.g., Matrix&lt;T&gt; for tabular data, Tensor&lt;T&gt; for images).</typeparam>
/// <typeparam name="TOutput">The type of output data (e.g., Vector&lt;T&gt; for predictions, custom types for other formats).</typeparam>
/// <remarks>
/// <para>
/// This class provides a k-fold cross-validation implementation, where the data is split into k equal-sized folds.
/// Each fold is used once as a validation set while the k-1 remaining folds form the training set.
/// </para>
/// <para><b>For Beginners:</b> K-fold cross-validation is like dividing your data into k equal parts.
///
/// What this class does:
/// - Splits your data into k parts (folds)
/// - Uses each part once for testing and the rest for training
/// - Repeats this process k times, so each part gets a chance to be the test set
/// - Calculates how well your model performs on average across all these tests
///
/// This is useful because:
/// - It uses all of your data for both training and testing
/// - It gives a more reliable estimate of your model's performance
/// - It helps detect if your model is overfitting to a particular subset of the data
/// </para>
/// </remarks>
public class KFoldCrossValidator<T, TInput, TOutput> : CrossValidatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the KFoldCrossValidator class.
    /// </summary>
    /// <param name="options">The options for cross-validation. If null, default options are used.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes the KFoldCrossValidator with the provided options or default options if none are specified.
    /// It sets up the cross-validator to perform k-fold cross-validation based on the specified parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the k-fold cross-validator with your chosen settings.
    /// 
    /// What it does:
    /// - Takes in your preferences for how to do the cross-validation (or uses default settings if you don't specify any)
    /// - Prepares the cross-validator to split your data into the number of parts you specified
    /// - Gets everything ready to start the cross-validation process
    /// 
    /// It's like setting up a series of tests for your model based on your instructions.
    /// </para>
    /// </remarks>
    public KFoldCrossValidator(CrossValidationOptions? options = null) : base(options ?? new())
    {
    }

    /// <summary>
    /// Performs the k-fold cross-validation process on the given model using the provided data and optimizer.
    /// </summary>
    /// <param name="model">The machine learning model to validate.</param>
    /// <param name="X">The feature matrix containing the input data.</param>
    /// <param name="y">The target vector containing the output data.</param>
    /// <param name="optimizer">The optimizer to use for training the model on each fold.</param>
    /// <returns>A CrossValidationResult containing the results of the validation process.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the core k-fold cross-validation logic. It creates the folds using the CreateFolds method,
    /// then performs the cross-validation using these folds and the provided optimizer.
    /// </para>
    /// <para><b>For Beginners:</b> This method is where the actual k-fold cross-validation happens.
    ///
    /// What it does:
    /// - Takes your model, your data (X and y), and an optimizer for training
    /// - Splits your data into k parts (folds) using the CreateFolds method
    /// - Runs the PerformCrossValidation method, which:
    ///   - Trains your model using the optimizer k times, each time using a different part as the test set
    ///   - Collects and summarizes the results of all these tests
    ///
    /// The optimizer ensures that advanced optimization techniques (like genetic algorithms, Bayesian optimization, etc.)
    /// are applied consistently across all folds.
    ///
    /// It's like putting your model through a series of k tests using a standardized training procedure
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
    /// Creates the folds for k-fold cross-validation based on the provided options.
    /// </summary>
    /// <param name="X">The input data.</param>
    /// <param name="y">The output data.</param>
    /// <returns>An enumerable of tuples containing the train and validation indices for each fold.</returns>
    /// <remarks>
    /// <para>
    /// This method generates the indices for the training and validation sets for each fold of the k-fold cross-validation process.
    /// It supports data shuffling and uses the specified number of folds from the options. The method ensures that each
    /// data point is used exactly once as a validation sample.
    /// </para>
    /// <para><b>For Beginners:</b> This method decides how to split your data for k-fold cross-validation.
    ///
    /// What it does:
    /// - Creates a list of all your data points
    /// - If requested, shuffles this list randomly
    /// - Splits the list into k equal parts (folds)
    /// - For each part:
    ///   - Uses that part as the validation data
    ///   - Uses all other parts as the training data
    /// - Returns these splits so the main method can use them
    ///
    /// It's like dealing a deck of cards into k piles, where each pile will take a turn being the "test" pile,
    /// and the other piles together form the "training" pile.
    /// </para>
    /// </remarks>
    private IEnumerable<(int[] trainIndices, int[] validationIndices)> CreateFolds(TInput X, TOutput y)
    {
        var batchSize = InputHelper<T, TInput>.GetBatchSize(X);
        var indices = Enumerable.Range(0, batchSize).ToArray();
        if (Options.ShuffleData)
        {
            indices = [.. indices.OrderBy(x => Random.Next())];
        }

        int foldSize = batchSize / Options.NumberOfFolds;
        for (int i = 0; i < Options.NumberOfFolds; i++)
        {
            int[] validationIndices = [.. indices.Skip(i * foldSize).Take(foldSize)];
            int[] trainIndices = [.. indices.Except(validationIndices)];

            yield return (trainIndices, validationIndices);
        }
    }
}
