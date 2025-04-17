namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for cross-validation implementations in machine learning models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
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
/// 
/// It's like setting a standard recipe that all cross-validation methods must follow, ensuring consistency 
/// and ease of use across different types of models and data.
/// </para>
/// </remarks>
public interface ICrossValidator<T>
{
    /// <summary>
    /// Performs cross-validation on the given model using the provided data and options.
    /// </summary>
    /// <param name="model">The machine learning model to validate.</param>
    /// <param name="X">The feature matrix containing the input data.</param>
    /// <param name="y">The target vector containing the output data.</param>
    /// <returns>A CrossValidationResult containing the results of the validation process.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the core cross-validation logic. It typically involves splitting the data into folds, 
    /// training and evaluating the model on different combinations of these folds, and aggregating the results.
    /// The specific implementation details may vary depending on the type of cross-validation being performed.
    /// </para>
    /// <para><b>For Beginners:</b> This method is where the actual cross-validation happens.
    /// 
    /// What it does:
    /// - Takes your model, your data (X and y), and your chosen options
    /// - Splits your data into parts based on your options
    /// - Trains and tests your model multiple times using these different parts
    /// - Collects and summarizes the results of all these tests
    /// 
    /// It's like putting your model through a series of tests and then giving you a report card 
    /// that shows how well it performed overall.
    /// </para>
    /// </remarks>
    CrossValidationResult<T> Validate(
        IFullModel<T, Matrix<T>, Vector<T>> model,
        Matrix<T> X,
        Vector<T> y);
}