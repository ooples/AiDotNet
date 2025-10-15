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

    protected readonly ModelType ModelType;

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
    protected CrossValidatorBase(CrossValidationOptions options, ModelType modelType)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        Options = options;
        Random = options.RandomSeed.HasValue ? new Random(options.RandomSeed.Value) : new Random();
        ModelType = modelType;
    }

    /// <summary>
    /// Performs cross-validation on the given model using the provided data and options.
    /// </summary>
    /// <param name="model">The machine learning model to validate.</param>
    /// <param name="X">The feature matrix containing the input data.</param>
    /// <param name="y">The target vector containing the output data.</param>
    /// <param name="options">The options specifying how to perform the cross-validation.</param>
    /// <returns>A CrossValidationResult containing the results of the validation process.</returns>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to define how folds are created
    /// for a specific cross-validation strategy. The actual cross-validation process is then
    /// performed using these folds.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method is like a placeholder that says "each specific type of
    /// cross-validation needs to decide how to split the data into folds". The actual splitting
    /// logic will be implemented in the classes that inherit from this base class.
    /// </para>
    /// </remarks>
    public abstract CrossValidationResult<T> Validate(IFullModel<T, Matrix<T>, Vector<T>> model, Matrix<T> X, Vector<T> y);

    /// <summary>
    /// Executes the cross-validation process using the provided model, data, and folds.
    /// </summary>
    /// <param name="model">The machine learning model to validate.</param>
    /// <param name="X">The feature matrix containing the input data.</param>
    /// <param name="y">The target vector containing the output data.</param>
    /// <param name="folds">The pre-computed folds for cross-validation.</param>
    /// <param name="options">The options specifying how to perform the cross-validation.</param>
    /// <returns>A CrossValidationResult containing the results of the validation process.</returns>
    /// <remarks>
    /// <para>
    /// This method performs the actual cross-validation process:
    /// - It iterates through each fold.
    /// - For each fold, it trains the model on the training data and evaluates it on the validation data.
    /// - It collects performance metrics, timing information, and feature importance for each fold.
    /// - Finally, it aggregates the results from all folds into a single CrossValidationResult.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method is like running a series of experiments. For each fold:
    /// 1. We train the model on most of the data (training set).
    /// 2. We test the model on the remaining data (validation set).
    /// 3. We record how well the model did and how long it took.
    /// 4. At the end, we combine all these mini-experiments into one big result.
    /// This helps us understand how well our model performs on different subsets of the data.
    /// </para>
    /// </remarks>
    protected CrossValidationResult<T> PerformCrossValidation(IFullModel<T, Matrix<T>, Vector<T>> model, Matrix<T> X, Vector<T> y, 
        IEnumerable<(int[] trainIndices, int[] validationIndices)> folds)
    {
        var foldResults = new List<FoldResult<T>>();
        var totalTimer = Stopwatch.StartNew();
        int foldIndex = 0;

        foreach (var fold in folds)
        {
            var trainIndices = fold.trainIndices;
            var validationIndices = fold.validationIndices;
            var XTrain = X.Submatrix(trainIndices);
            var yTrain = y.Subvector(trainIndices);
            var XValidation = X.Submatrix(validationIndices);
            var yValidation = y.Subvector(validationIndices);

            var trainingTimer = Stopwatch.StartNew();
            model.Train(XTrain, yTrain);
            trainingTimer.Stop();
            var trainingTime = trainingTimer.Elapsed;

            var evaluationTimer = Stopwatch.StartNew();
            var trainingPredictions = model.Predict(XTrain);
            var validationPredictions = model.Predict(XValidation);
            evaluationTimer.Stop();
            var evaluationTime = evaluationTimer.Elapsed;

            var featureImportance = model.GetModelMetadata().FeatureImportance;

            var foldResult = new FoldResult<T>(
                foldIndex,
                yTrain,
                trainingPredictions,
                yValidation,
                validationPredictions,
                featureImportance,
                trainingTime,
                evaluationTime,
                X.Columns
            );

            foldResults.Add(foldResult);
            foldIndex++;
        }

        totalTimer.Stop();

        return new CrossValidationResult<T>(foldResults, totalTimer.Elapsed, ModelType);
    }
}