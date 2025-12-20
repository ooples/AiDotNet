namespace AiDotNet.CrossValidators;


/// <summary>
/// Provides a base implementation for cross-validation strategies in machine learning models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
/// <typeparam name="TInput">The type of input data (e.g., Matrix&lt;T&gt; for tabular data, Tensor&lt;T&gt; for images).</typeparam>
/// <typeparam name="TOutput">The type of output data (e.g., Vector&lt;T&gt; for predictions, custom types for other formats).</typeparam>
/// <remarks>
/// <para>
/// This abstract class serves as a foundation for implementing various cross-validation strategies.
/// It encapsulates common functionality such as numeric operations, random number generation,
/// and the core cross-validation process. Specific cross-validation types can be implemented
/// by deriving from this class and providing their own fold creation logic.
/// </para>
/// <para>
/// <b>For Beginners:</b> Cross-validation is a technique used to assess how well a machine learning
/// model will perform on new, unseen data. This base class provides the common structure and
/// functionality that all cross-validation methods share. Think of it as a blueprint for
/// creating different types of cross-validation strategies.
/// </para>
/// <para>
/// Key features:
/// - Manages numeric operations and random number generation.
/// - Provides a common method for performing cross-validation once folds are created.
/// - Allows for easy implementation of various cross-validation strategies by extending this class.
/// - Supports generic input and output types for flexibility with different data formats.
/// </para>
/// </remarks>
public abstract class CrossValidatorBase<T, TInput, TOutput> : ICrossValidator<T, TInput, TOutput>
{
    /// <summary>
    /// Provides operations for numeric calculations specific to type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Random number generator for operations that require randomness.
    /// </summary>
    protected readonly Random Random;

    /// <summary>
    /// Holds configuration options for cross-validation.
    /// </summary>
    protected readonly CrossValidationOptions Options;

    /// <summary>
    /// Initializes a new instance of the CrossValidationBase class.
    /// </summary>
    /// <param name="seed">Optional seed for the random number generator to ensure reproducibility.</param>
    /// <remarks>
    /// <para>
    /// This constructor sets up the necessary components for cross-validation:
    /// - It initializes the numeric operations for type T.
    /// - It creates a random number generator, optionally with a specified seed for reproducibility.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like setting up the tools we need before we start the actual
    /// cross-validation process. The seed parameter allows us to get the same "random" results
    /// each time we run the code, which is useful for testing and reproducibility.
    /// </para>
    /// </remarks>
    protected CrossValidatorBase(CrossValidationOptions options)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        Options = options;
        Random = options.RandomSeed.HasValue ? RandomHelper.CreateSeededRandom(options.RandomSeed.Value) : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Performs cross-validation on the given model using the provided data, options, and optimizer.
    /// </summary>
    /// <param name="model">The machine learning model to validate.</param>
    /// <param name="X">The input data containing the features.</param>
    /// <param name="y">The output data containing the targets.</param>
    /// <param name="optimizer">The optimizer to use for training the model on each fold.</param>
    /// <returns>A CrossValidationResult containing the results of the validation process.</returns>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to define how folds are created
    /// for a specific cross-validation strategy. The actual cross-validation process is then
    /// performed using these folds and the provided optimizer.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method is like a placeholder that says "each specific type of
    /// cross-validation needs to decide how to split the data into folds". The actual splitting
    /// logic will be implemented in the classes that inherit from this base class. The optimizer
    /// parameter ensures that the same training procedure is used consistently across all folds.
    /// </para>
    /// </remarks>
    public abstract CrossValidationResult<T, TInput, TOutput> Validate(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y,
        IOptimizer<T, TInput, TOutput> optimizer);

    /// <summary>
    /// Executes the cross-validation process using the provided model, data, folds, and optimizer.
    /// </summary>
    /// <param name="model">The machine learning model to validate.</param>
    /// <param name="X">The input data containing the features.</param>
    /// <param name="y">The output data containing the targets.</param>
    /// <param name="folds">The pre-computed folds for cross-validation.</param>
    /// <param name="optimizer">The optimizer to use for training the model on each fold.</param>
    /// <returns>A CrossValidationResult containing the results of the validation process.</returns>
    /// <remarks>
    /// <para>
    /// This method performs the actual cross-validation process:
    /// - It iterates through each fold.
    /// - For each fold, it creates an independent copy of the model to prevent state leakage.
    /// - It trains the model using the optimizer on the training data and evaluates it on the validation data.
    /// - It collects performance metrics, timing information, feature importance, and the trained model for each fold.
    /// - Finally, it aggregates the results from all folds into a single CrossValidationResult.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method is like running a series of experiments. For each fold:
    /// 1. We create a fresh copy of the model to ensure independence between folds.
    /// 2. We train the model using the optimizer on most of the data (training set).
    /// 3. We test the model on the remaining data (validation set).
    /// 4. We record how well the model did, how long it took, and save the trained model.
    /// 5. At the end, we combine all these mini-experiments into one big result.
    /// This helps us understand how well our model performs on different subsets of the data
    /// and ensures that the optimizer's configuration is applied consistently across all folds.
    /// </para>
    /// </remarks>
    protected CrossValidationResult<T, TInput, TOutput> PerformCrossValidation(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y,
        IEnumerable<(int[] trainIndices, int[] validationIndices)> folds,
        IOptimizer<T, TInput, TOutput> optimizer)
    {
        var foldResults = new List<FoldResult<T, TInput, TOutput>>();
        var totalTimer = Stopwatch.StartNew();
        int foldIndex = 0;

        foreach (var (trainIndices, validationIndices) in folds)
        {
            // Reset optimizer state before each fold to ensure independent evaluations
            // This prevents state contamination (accumulated fitness lists, cache, learning rates)
            optimizer.Reset();

            // Create a deep copy of the model for this fold to prevent state leakage
            var foldModel = model.DeepCopy();

            // Use InputHelper to subset data generically
            var XTrain = InputHelper<T, TInput>.GetBatch(X, trainIndices);
            var yTrain = InputHelper<T, TOutput>.GetBatch(y, trainIndices);
            var XValidation = InputHelper<T, TInput>.GetBatch(X, validationIndices);
            var yValidation = InputHelper<T, TOutput>.GetBatch(y, validationIndices);

            var trainingTimer = Stopwatch.StartNew();

            // Use optimizer.Optimize() instead of model.Train()
            // Create empty test data using ModelHelper
            var (emptyXTest, emptyYTest, _) = ModelHelper<T, TInput, TOutput>.CreateDefaultModelData();

            var optimizationInput = new OptimizationInputData<T, TInput, TOutput>
            {
                XTrain = XTrain,
                YTrain = yTrain,
                XValidation = XValidation,
                YValidation = yValidation,
                // Use empty test data for cross-validation
                XTest = emptyXTest,
                YTest = emptyYTest
            };

            var optimizationResult = optimizer.Optimize(optimizationInput);

            // Update the fold model with optimized parameters
            // Throw exception if optimization failed to prevent evaluating untrained models
            if (optimizationResult.BestSolution == null)
            {
                throw new InvalidOperationException(
                    $"Optimization failed for fold {foldIndex}: BestSolution is null. " +
                    "Cannot evaluate an untrained model in cross-validation. " +
                    "This indicates the optimizer was unable to find a valid solution.");
            }

            foldModel.SetParameters(optimizationResult.BestSolution.GetParameters());

            trainingTimer.Stop();
            var trainingTime = trainingTimer.Elapsed;

            var evaluationTimer = Stopwatch.StartNew();
            var trainingPredictions = foldModel.Predict(XTrain);
            var validationPredictions = foldModel.Predict(XValidation);
            evaluationTimer.Stop();
            var evaluationTime = evaluationTimer.Elapsed;

            var featureImportance = foldModel.GetModelMetadata().FeatureImportance;

            // Convert predictions to Vector<T> for metrics calculation
            var trainingPredictionsVector = ConversionsHelper.ConvertToVector<T, TOutput>(trainingPredictions);
            var trainingActualVector = ConversionsHelper.ConvertToVector<T, TOutput>(yTrain);
            var validationPredictionsVector = ConversionsHelper.ConvertToVector<T, TOutput>(validationPredictions);
            var validationActualVector = ConversionsHelper.ConvertToVector<T, TOutput>(yValidation);

            var featureCount = InputHelper<T, TInput>.GetInputSize(X);

            var foldResult = new FoldResult<T, TInput, TOutput>(
                foldIndex,
                trainingActualVector,
                trainingPredictionsVector,
                validationActualVector,
                validationPredictionsVector,
                featureImportance,
                trainingTime,
                evaluationTime,
                featureCount,
                foldModel,  // Pass the trained model for this fold
                null,  // clusteringMetrics
                trainIndices,  // Pass the training indices for this fold
                validationIndices  // Pass the validation indices for this fold
            );

            foldResults.Add(foldResult);
            foldIndex++;
        }

        totalTimer.Stop();

        return new CrossValidationResult<T, TInput, TOutput>(foldResults, totalTimer.Elapsed);
    }
}
