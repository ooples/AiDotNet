namespace AiDotNet.CrossValidators;

/// <summary>
/// Implements a nested cross-validation strategy for model evaluation and hyperparameter tuning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
/// <remarks>
/// <para>
/// This class provides a nested cross-validation implementation, which consists of an outer loop for model assessment
/// and an inner loop for model selection (hyperparameter tuning).
/// </para>
/// <para><b>For Beginners:</b> Nested cross-validation is like a two-layer testing process for your model.
/// 
/// What this class does:
/// - Splits your data into outer folds for overall model assessment
/// - For each outer fold:
///   - Further splits the training data into inner folds for hyperparameter tuning
///   - Uses the inner folds to find the best hyperparameters
///   - Trains a model with the best hyperparameters on the full outer training set
///   - Evaluates this model on the outer test set
/// - Calculates how well your model performs on average across all outer folds
/// 
/// This is useful because:
/// - It helps you choose the best hyperparameters for your model
/// - It provides an unbiased estimate of your model's performance on new data
/// - It helps prevent overfitting during the model selection process
/// </para>
/// </remarks>
public class NestedCrossValidator<T> : CrossValidatorBase<T>
{
    /// <summary>
    /// The cross-validator used for the outer loop of nested cross-validation.
    /// </summary>
    /// <remarks>
    /// This validator is responsible for creating the main folds used for overall model assessment.
    /// For beginners: Think of this as the main test that your model will go through.
    /// </remarks>
    private readonly ICrossValidator<T> _outerValidator = default!;

    /// <summary>
    /// The cross-validator used for the inner loop of nested cross-validation.
    /// </summary>
    /// <remarks>
    /// This validator is used within each outer fold to tune hyperparameters.
    /// For beginners: This is like a practice test that helps your model improve before the main test.
    /// </remarks>
    private readonly ICrossValidator<T> _innerValidator = default!;

    /// <summary>
    /// A function that selects the best model based on inner cross-validation results.
    /// </summary>
    /// <remarks>
    /// This function takes the results of the inner cross-validation and returns the best model configuration.
    /// For beginners: This is like choosing the best version of your model after the practice tests.
    /// </remarks>
    private readonly Func<CrossValidationResult<T>, IFullModel<T, Matrix<T>, Vector<T>>> _modelSelector = default!;

    /// <summary>
    /// Initializes a new instance of the NestedCrossValidator class.
    /// </summary>
    /// <param name="outerValidator">The cross-validator for the outer loop.</param>
    /// <param name="innerValidator">The cross-validator for the inner loop.</param>
    /// <param name="modelSelector">A function that selects the best model based on inner cross-validation results.</param>
    /// <param name="options">The options for cross-validation. If null, default options are used.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes the NestedCrossValidator with the provided validators, model selector, and options.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the nested cross-validator with your chosen settings.
    /// 
    /// What it does:
    /// - Takes in your preferences for how to do the outer and inner cross-validation
    /// - Takes a function that decides which model is best based on the inner cross-validation results
    /// - Prepares the cross-validator to perform nested cross-validation
    /// 
    /// It's like setting up a complex series of tests that will help you find the best version of your model.
    /// </para>
    /// </remarks>
    public NestedCrossValidator(ICrossValidator<T> outerValidator, ICrossValidator<T> innerValidator, 
        Func<CrossValidationResult<T>, IFullModel<T, Matrix<T>, Vector<T>>> modelSelector, ModelType modelType, CrossValidationOptions? options = null) 
        : base(options ?? new(), modelType)
    {
        _outerValidator = outerValidator;
        _innerValidator = innerValidator;
        _modelSelector = modelSelector;
    }

    /// <summary>
    /// Performs the nested cross-validation process on the given model using the provided data.
    /// </summary>
    /// <param name="model">The machine learning model to validate.</param>
    /// <param name="X">The feature matrix containing the input data.</param>
    /// <param name="y">The target vector containing the output data.</param>
    /// <returns>A CrossValidationResult containing the results of the validation process.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the core nested cross-validation logic. It uses the outer validator to create folds,
    /// then for each fold, it uses the inner validator to tune hyperparameters, and finally evaluates the best model on the outer test set.
    /// </para>
    /// <para><b>For Beginners:</b> This method is where the actual nested cross-validation happens.
    /// 
    /// What it does:
    /// - Takes your model and your data (X and y)
    /// - Uses the outer validator to split your data into training and test sets
    /// - For each split:
    ///   - Uses the inner validator to find the best hyperparameters on the training data
    ///   - Trains a model with these best hyperparameters on all the training data
    ///   - Tests this model on the test data
    /// - Collects and summarizes the results of all these tests
    /// 
    /// It's like putting your model through a series of increasingly challenging tests to find the best version
    /// and accurately estimate how well it will perform on new data.
    /// </para>
    /// </remarks>
    public override CrossValidationResult<T> Validate(IFullModel<T, Matrix<T>, Vector<T>> model, Matrix<T> X, Vector<T> y)
    {
        var nestedResults = new List<FoldResult<T>>();
        var totalTimer = Stopwatch.StartNew();

        // Use the Validate method of the outer validator
        var outerResults = _outerValidator.Validate(model, X, y);

        foreach (var outerFoldResult in outerResults.FoldResults)
        {
            // Extract training data for this outer fold
            var trainIndices = GetTrainingIndices(X, outerFoldResult.ActualValues, y);
            var outerTrainX = X.Submatrix(trainIndices);
            var outerTrainY = y.Subvector(trainIndices);

            // Perform inner cross-validation
            var innerResult = _innerValidator.Validate(model, outerTrainX, outerTrainY);
            var bestModel = _modelSelector(innerResult);

            // Perform outer fold evaluation using the best model
            bestModel.Train(outerTrainX, outerTrainY);
            var validationIndices = GetValidationIndices(X, outerFoldResult.ActualValues, y);
            var validationPredictions = bestModel.Predict(X.Submatrix(validationIndices));

            // Get feature importance from the best model
            var featureImportance = bestModel.GetModelMetadata().FeatureImportance;

            // Create adjusted fold result
            var adjustedFoldResult = new FoldResult<T>(
                outerFoldResult.FoldIndex,
                outerTrainY,
                bestModel.Predict(outerTrainX),
                outerFoldResult.ActualValues,
                validationPredictions,
                featureImportance,
                outerFoldResult.TrainingTime + innerResult.TotalTime,
                outerFoldResult.EvaluationTime,
                X.Columns
            );

            nestedResults.Add(adjustedFoldResult);
        }

        totalTimer.Stop();
        return new CrossValidationResult<T>(nestedResults, totalTimer.Elapsed, ModelType);
    }

    /// <summary>
    /// Gets the indices of the training set based on the validation set.
    /// </summary>
    /// <param name="X">The feature matrix containing the input data.</param>
    /// <param name="validationSet">The validation set vector.</param>
    /// <param name="y">The target vector containing the output data.</param>
    /// <returns>An array of indices representing the training set.</returns>
    /// <remarks>
    /// <para>
    /// This method determines the training set indices by excluding the validation set indices from the full dataset.
    /// </para>
    /// <para><b>For Beginners:</b> This method figures out which data points should be used for training.
    /// 
    /// What it does:
    /// - Looks at all the data points
    /// - Removes the ones that are used for validation
    /// - Returns the remaining ones, which will be used for training
    /// 
    /// It's like separating your study materials into what you'll use for practice (training) 
    /// and what you'll use for the final test (validation).
    /// </para>
    /// </remarks>
    private int[] GetTrainingIndices(Matrix<T> X, Vector<T> validationSet, Vector<T> y)
    {
        return [.. Enumerable.Range(0, X.Rows).Except(GetValidationIndices(X, validationSet, y))];
    }

    /// <summary>
    /// Gets the indices of the validation set.
    /// </summary>
    /// <param name="X">The feature matrix containing the input data.</param>
    /// <param name="validationSet">The validation set vector.</param>
    /// <param name="y">The target vector containing the output data.</param>
    /// <returns>An array of indices representing the validation set.</returns>
    /// <remarks>
    /// <para>
    /// This method determines the validation set indices by finding which elements of y are present in the validationSet.
    /// </para>
    /// <para><b>For Beginners:</b> This method identifies which data points should be used for validation.
    /// 
    /// What it does:
    /// - Looks at all the data points
    /// - Checks which ones match the values in the validation set
    /// - Returns the indices of those matching points
    /// 
    /// It's like picking out specific questions from your study materials to use as a practice test, 
    /// helping you gauge how well you've learned the material.
    /// </para>
    /// </remarks>
    private int[] GetValidationIndices(Matrix<T> X, Vector<T> validationSet, Vector<T> y)
    {
        return [.. Enumerable.Range(0, X.Rows).Where(i => validationSet.Contains(y[i]))];
    }
}