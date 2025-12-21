namespace AiDotNet.CrossValidators;

/// <summary>
/// Implements a nested cross-validation strategy for model evaluation and hyperparameter tuning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
/// <typeparam name="TInput">The type of input data (e.g., Matrix&lt;T&gt; for tabular data, Tensor&lt;T&gt; for images).</typeparam>
/// <typeparam name="TOutput">The type of output data (e.g., Vector&lt;T&gt; for predictions, custom types for other formats).</typeparam>
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
public class NestedCrossValidator<T, TInput, TOutput> : CrossValidatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// The cross-validator used for the outer loop of nested cross-validation.
    /// </summary>
    /// <remarks>
    /// This validator is responsible for creating the main folds used for overall model assessment.
    /// For beginners: Think of this as the main test that your model will go through.
    /// </remarks>
    private readonly ICrossValidator<T, TInput, TOutput> _outerValidator;

    /// <summary>
    /// The cross-validator used for the inner loop of nested cross-validation.
    /// </summary>
    /// <remarks>
    /// This validator is used within each outer fold to tune hyperparameters.
    /// For beginners: This is like a practice test that helps your model improve before the main test.
    /// </remarks>
    private readonly ICrossValidator<T, TInput, TOutput> _innerValidator;

    /// <summary>
    /// A function that selects the best model based on inner cross-validation results.
    /// </summary>
    /// <remarks>
    /// This function takes the results of the inner cross-validation and returns the best model configuration.
    /// For beginners: This is like choosing the best version of your model after the practice tests.
    /// </remarks>
    private readonly Func<CrossValidationResult<T, TInput, TOutput>, IFullModel<T, TInput, TOutput>> _modelSelector;

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
    public NestedCrossValidator(ICrossValidator<T, TInput, TOutput> outerValidator, ICrossValidator<T, TInput, TOutput> innerValidator,
        Func<CrossValidationResult<T, TInput, TOutput>, IFullModel<T, TInput, TOutput>> modelSelector, CrossValidationOptions? options = null)
        : base(options ?? new())
    {
        _outerValidator = outerValidator;
        _innerValidator = innerValidator;
        _modelSelector = modelSelector;
    }

    /// <summary>
    /// Performs the nested cross-validation process on the given model using the provided data and optimizer.
    /// </summary>
    /// <param name="model">The machine learning model to validate.</param>
    /// <param name="X">The feature matrix containing the input data.</param>
    /// <param name="y">The target vector containing the output data.</param>
    /// <param name="optimizer">The optimizer to use for training the model on each fold.</param>
    /// <returns>A CrossValidationResult containing the results of the validation process.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the core nested cross-validation logic. It uses the outer validator to create folds,
    /// then for each fold, it uses the inner validator to tune hyperparameters, and finally evaluates the best model on the outer test set.
    /// </para>
    /// <para><b>For Beginners:</b> This method is where the actual nested cross-validation happens.
    ///
    /// What it does:
    /// - Takes your model, your data (X and y), and an optimizer for training
    /// - Uses the outer validator to split your data into training and test sets
    /// - For each split:
    ///   - Uses the inner validator with the optimizer to find the best hyperparameters on the training data
    ///   - Trains a model with these best hyperparameters using the optimizer on all the training data
    ///   - Tests this model on the test data
    /// - Collects and summarizes the results of all these tests
    ///
    /// The optimizer ensures consistent training across both inner and outer cross-validation loops.
    ///
    /// It's like putting your model through a series of increasingly challenging tests using a standardized training procedure
    /// to find the best version and accurately estimate how well it will perform on new data.
    /// </para>
    /// </remarks>
    public override CrossValidationResult<T, TInput, TOutput> Validate(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y,
        IOptimizer<T, TInput, TOutput> optimizer)
    {
        var nestedResults = new List<FoldResult<T, TInput, TOutput>>();
        var totalTimer = Stopwatch.StartNew();
        int outerFoldIndex = 0;

        // Use the Validate method of the outer validator with the optimizer
        var outerResults = _outerValidator.Validate(model, X, y, optimizer);

        foreach (var outerFoldResult in outerResults.FoldResults)
        {
            // Extract training data for this outer fold using indices from FoldResult
            var validationActualVector = outerFoldResult.ActualValues;
            var trainIndices = outerFoldResult.TrainingIndices ?? throw new InvalidOperationException(
                $"TrainingIndices not available for outer fold {outerFoldResult.FoldIndex}. " +
                "Ensure the outer cross-validator populates TrainingIndices in FoldResult.");
            var outerTrainX = InputHelper<T, TInput>.GetBatch(X, trainIndices);
            var outerTrainY = InputHelper<T, TOutput>.GetBatch(y, trainIndices);

            // Perform inner cross-validation with the optimizer
            var innerResult = _innerValidator.Validate(model, outerTrainX, outerTrainY, optimizer);
            var bestModel = _modelSelector(innerResult);

            // Perform outer fold evaluation using the best model and optimizer
            var (emptyXVal, emptyYVal, _) = ModelHelper<T, TInput, TOutput>.CreateDefaultModelData();
            var optimizationInput = new OptimizationInputData<T, TInput, TOutput>
            {
                XTrain = outerTrainX,
                YTrain = outerTrainY,
                XValidation = emptyXVal,
                YValidation = emptyYVal,
                XTest = emptyXVal,
                YTest = emptyYVal
            };

            var optimizationResult = optimizer.Optimize(optimizationInput);

            // Throw exception if optimization failed to prevent evaluating untrained models
            if (optimizationResult.BestSolution == null)
            {
                throw new InvalidOperationException(
                    $"Optimization failed for outer fold {outerFoldIndex} in nested cross-validation: BestSolution is null. " +
                    "Cannot evaluate an untrained model. " +
                    "This indicates the optimizer was unable to find a valid solution.");
            }

            bestModel.SetParameters(optimizationResult.BestSolution.GetParameters());

            var validationIndices = outerFoldResult.ValidationIndices ?? throw new InvalidOperationException(
                $"ValidationIndices not available for outer fold {outerFoldResult.FoldIndex}. " +
                "Ensure the outer cross-validator populates ValidationIndices in FoldResult.");
            var outerValidationX = InputHelper<T, TInput>.GetBatch(X, validationIndices);
            var validationPredictions = bestModel.Predict(outerValidationX);

            // Get feature importance from the best model
            var featureImportance = bestModel.GetModelMetadata().FeatureImportance;

            // Convert predictions to Vector<T> for metrics calculation
            var trainingPredictionsVector = ConversionsHelper.ConvertToVector<T, TOutput>(bestModel.Predict(outerTrainX));
            var trainingActualVector = ConversionsHelper.ConvertToVector<T, TOutput>(outerTrainY);
            var validationPredictionsVector = ConversionsHelper.ConvertToVector<T, TOutput>(validationPredictions);

            var featureCount = InputHelper<T, TInput>.GetInputSize(X);

            // Create adjusted fold result
            var adjustedFoldResult = new FoldResult<T, TInput, TOutput>(
                outerFoldResult.FoldIndex,
                trainingActualVector,
                trainingPredictionsVector,
                validationActualVector,
                validationPredictionsVector,
                featureImportance,
                outerFoldResult.TrainingTime + innerResult.TotalTime,
                outerFoldResult.EvaluationTime,
                featureCount,
                bestModel,  // Pass the trained model for this fold
                null,  // clusteringMetrics
                trainIndices,  // Pass the training indices from outer fold
                validationIndices  // Pass the validation indices from outer fold
            );

            nestedResults.Add(adjustedFoldResult);
            outerFoldIndex++;
        }

        totalTimer.Stop();
        return new CrossValidationResult<T, TInput, TOutput>(nestedResults, totalTimer.Elapsed);
    }
}
