namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for cross-validation implementations in machine learning models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
/// <typeparam name="TInput">The type of input data (e.g., Matrix&lt;T&gt; for tabular data, custom types for other formats).</typeparam>
/// <typeparam name="TOutput">The type of output data (e.g., Vector&lt;T&gt; for predictions, custom types for other formats).</typeparam>
/// <remarks>
/// <para>
/// This interface specifies the method signature for performing cross-validation on machine learning models.
/// Cross-validation is a crucial technique for assessing how the results of a statistical analysis will generalize
/// to an independent data set. It's particularly important in contexts where the goal is prediction, and one wants
/// to estimate how accurately a predictive model will perform in practice.
/// </para>
/// <para><b>For Beginners:</b> This interface is like a blueprint for creating cross-validation tools.
///
/// What it does:
/// - Defines a standard way to perform cross-validation on any machine learning model
/// - Ensures that all cross-validation implementations will work the same way, regardless of the specific details
/// - Works with any data format (matrices, tensors, custom structures) through generic type parameters
///
/// It's like setting a standard recipe that all cross-validation methods must follow, ensuring consistency
/// and ease of use across different types of models and data.
/// </para>
/// </remarks>
public interface ICrossValidator<T, TInput, TOutput>
{
    /// <summary>
    /// Performs cross-validation on the given model using the provided data and optimizer.
    /// </summary>
    /// <param name="model">The machine learning model to validate.</param>
    /// <param name="X">The input data containing the features.</param>
    /// <param name="y">The output data containing the targets.</param>
    /// <param name="optimizer">The optimizer to use for training the model on each fold.</param>
    /// <returns>A CrossValidationResult containing the results of the validation process.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the core cross-validation logic. It typically involves splitting the data into folds,
    /// training and evaluating the model on different combinations of these folds using the provided optimizer,
    /// and aggregating the results. The specific implementation details may vary depending on the type of
    /// cross-validation being performed.
    /// </para>
    /// <para><b>For Beginners:</b> This method is where the actual cross-validation happens.
    ///
    /// What it does:
    /// - Takes your model, your data (X and y), and an optimizer for training
    /// - Splits your data into parts based on your options
    /// - Trains your model using the optimizer multiple times on different parts
    /// - Tests the trained model on held-out data for each fold
    /// - Collects and summarizes the results of all these tests
    ///
    /// The optimizer parameter allows you to use advanced optimization techniques (like genetic algorithms,
    /// Bayesian optimization, etc.) during cross-validation, ensuring consistent training across all folds.
    ///
    /// It's like putting your model through a series of tests using a standardized training procedure
    /// and then giving you a report card that shows how well it performed overall.
    /// </para>
    /// </remarks>
    CrossValidationResult<T, TInput, TOutput> Validate(
        IFullModel<T, TInput, TOutput> model,
        TInput X,
        TOutput y,
        IOptimizer<T, TInput, TOutput> optimizer);
}
