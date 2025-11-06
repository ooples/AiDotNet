namespace AiDotNet.CrossValidators;


/// <summary>
/// Provides a base implementation for cross-validation strategies in machine learning models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
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
/// </para>
/// </remarks>
public abstract class CrossValidatorBase<T> : ICrossValidator<T>
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
        Random = options.RandomSeed.HasValue ? new Random(options.RandomSeed.Value) : new Random();
    }

    /// <summary>
    /// Performs cross-validation on the given model using the provided data, options, and optimizer.
    /// </summary>
    /// <param name="model">The machine learning model to validate.</param>
    /// <param name="X">The feature matrix containing the input data.</param>
    /// <param name="y">The target vector containing the output data.</param>
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
    public abstract CrossValidationResult<T> Validate(IFullModel<T, Matrix<T>, Vector<T>> model, Matrix<T> X, Vector<T> y,
        IOptimizer<T, Matrix<T>, Vector<T>> optimizer);

    /// <summary>
    /// Executes the cross-validation process using the provided model, data, folds, and optimizer.
    /// </summary>
    /// <param name="model">The machine learning model to validate.</param>
    /// <param name="X">The feature matrix containing the input data.</param>
    /// <param name="y">The target vector containing the output data.</param>
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
    protected CrossValidationResult<T> PerformCrossValidation(IFullModel<T, Matrix<T>, Vector<T>> model, Matrix<T> X, Vector<T> y,
        IEnumerable<(int[] trainIndices, int[] validationIndices)> folds,
        IOptimizer<T, Matrix<T>, Vector<T>> optimizer)
    {
        var foldResults = new List<FoldResult<T>>();
        var totalTimer = Stopwatch.StartNew();
        int foldIndex = 0;

        foreach (var (trainIndices, validationIndices) in folds)
        {
            // Create a deep copy of the model for this fold to prevent state leakage
            var foldModel = model.DeepCopy();

            var XTrain = X.Submatrix(trainIndices);
            var yTrain = y.Subvector(trainIndices);
            var XValidation = X.Submatrix(validationIndices);
            var yValidation = y.Subvector(validationIndices);

            var trainingTimer = Stopwatch.StartNew();

            // Use optimizer.Optimize() instead of model.Train()
            var optimizationInput = new OptimizationInputData<T, Matrix<T>, Vector<T>>
            {
                XTrain = XTrain,
                YTrain = yTrain,
                XValidation = XValidation,
                YValidation = yValidation,
                // Use empty test data for cross-validation
                XTest = new Matrix<T>(0, XTrain.Columns),
                YTest = new Vector<T>(0)
            };

            var optimizationResult = optimizer.Optimize(optimizationInput);

            // Update the fold model with optimized parameters
            if (optimizationResult.BestSolution != null)
            {
                foldModel.SetParameters(optimizationResult.BestSolution.GetParameters());
            }

            trainingTimer.Stop();
            var trainingTime = trainingTimer.Elapsed;

            var evaluationTimer = Stopwatch.StartNew();
            var trainingPredictions = foldModel.Predict(XTrain);
            var validationPredictions = foldModel.Predict(XValidation);
            evaluationTimer.Stop();
            var evaluationTime = evaluationTimer.Elapsed;

            var featureImportance = foldModel.GetModelMetadata().FeatureImportance;

            var foldResult = new FoldResult<T>(
                foldIndex,
                yTrain,
                trainingPredictions,
                yValidation,
                validationPredictions,
                featureImportance,
                trainingTime,
                evaluationTime,
                X.Columns,
                foldModel  // Pass the trained model for this fold
            );

            foldResults.Add(foldResult);
            foldIndex++;
        }

        totalTimer.Stop();

        return new CrossValidationResult<T>(foldResults, totalTimer.Elapsed);
    }
}