global using System.Diagnostics;

namespace AiDotNet.CrossValidators;

/// <summary>
/// Implements a standard cross-validation strategy for model evaluation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
/// <typeparam name="TInput">The type of input data (e.g., Matrix&lt;T&gt; for tabular data, Tensor&lt;T&gt; for images).</typeparam>
/// <typeparam name="TOutput">The type of output data (e.g., Vector&lt;T&gt; for predictions, custom types for other formats).</typeparam>
/// <remarks>
/// <para>
/// This class provides a standard implementation of cross-validation, a technique used to assess how the results of a
/// statistical analysis will generalize to an independent data set. It is particularly important in contexts where the goal
/// is prediction, and one wants to estimate how accurately a predictive model will perform in practice.
/// </para>
/// <para><b>For Beginners:</b> Cross-validation is like a thorough test for your machine learning model.
///
/// What this class does:
/// - Splits your data into several parts (called folds)
/// - Trains and tests your model multiple times, each time using a different part as the test set
/// - Calculates how well your model performs on average across all these tests
///
/// This is useful because:
/// - It helps you understand how well your model will work on new, unseen data
/// - It can detect if your model is overfitting (memorizing the training data instead of learning general patterns)
/// - It provides a more reliable estimate of your model's performance than a single train-test split
///
/// For example, if you're building a model to predict house prices, cross-validation would test it on different subsets
/// of your house data, giving you a better idea of how well it will predict prices for houses it hasn't seen before.
/// </para>
/// </remarks>
public class StandardCrossValidator<T, TInput, TOutput> : CrossValidatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the StandardCrossValidator class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor initializes the StandardCrossValidator with the appropriate numeric operations for the generic type T.
    /// It uses the MathHelper class to obtain the correct numeric operations, allowing the cross-validator to work with 
    /// different numeric types (e.g., float, double, decimal) without changing the implementation.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the cross-validator so it can work with different types of numbers.
    /// 
    /// What it does:
    /// - Gets the right math operations for the type of numbers you're using (like addition, multiplication, etc.)
    /// - This allows the cross-validator to work with different types of numbers without changing its code
    /// 
    /// It's like setting up a calculator that can work with different number systems (decimals, fractions, etc.) 
    /// without needing to change the calculator itself.
    /// </para>
    /// </remarks>
    public StandardCrossValidator(CrossValidationOptions? options = null) : base(options ?? new())
    {
    }

    /// <summary>
    /// Performs the cross-validation process on the given model using the provided data and optimizer.
    /// </summary>
    /// <param name="model">The machine learning model to validate.</param>
    /// <param name="X">The feature matrix containing the input data.</param>
    /// <param name="y">The target vector containing the output data.</param>
    /// <param name="optimizer">The optimizer to use for training the model on each fold.</param>
    /// <returns>A CrossValidationResult containing the results of the validation process.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the core cross-validation logic. It creates the folds using the CreateFolds method,
    /// then performs the cross-validation using these folds and the provided optimizer.
    /// </para>
    /// <para><b>For Beginners:</b> This method is where the actual cross-validation happens.
    ///
    /// What it does:
    /// - Takes your model, your data (X and y), and an optimizer for training
    /// - Splits your data into parts (folds) using the CreateFolds method
    /// - Runs the PerformCrossValidation method, which:
    ///   - Trains your model using the optimizer multiple times using these different parts
    ///   - Collects and summarizes the results of all these tests
    ///
    /// The optimizer ensures that advanced optimization techniques are applied consistently across all folds.
    ///
    /// It's like putting your model through a series of tests using a standardized training procedure
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
    /// Creates the folds for cross-validation based on the provided options.
    /// </summary>
    /// <param name="X">The input data.</param>
    /// <param name="y">The output data.</param>
    /// <param name="options">The cross-validation options.</param>
    /// <returns>An enumerable of tuples containing the train and test indices for each fold.</returns>
    /// <remarks>
    /// <para>
    /// This method generates the indices for the training and testing sets for each fold of the cross-validation process.
    /// It supports data shuffling and uses the specified number of folds from the options. The method ensures that each
    /// data point is used exactly once as a test sample.
    /// </para>
    /// <para><b>For Beginners:</b> This method decides how to split your data for cross-validation.
    ///
    /// What it does:
    /// - Creates a list of all your data points
    /// - If requested, shuffles this list randomly
    /// - Splits the list into the number of parts (folds) you specified
    /// - For each part:
    ///   - Uses that part as the test data
    ///   - Uses all other parts as the training data
    /// - Returns these splits so the main method can use them
    ///
    /// It's like dealing a deck of cards into several piles, where each pile will take a turn being the "test" pile.
    /// </para>
    /// </remarks>
    private IEnumerable<(int[] trainIndices, int[] testIndices)> CreateFolds(TInput X, TOutput y)
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
            int[] testIndices = [.. indices.Skip(i * foldSize).Take(foldSize)];
            int[] trainIndices = [.. indices.Except(testIndices)];

            yield return (trainIndices, testIndices);
        }
    }
}
