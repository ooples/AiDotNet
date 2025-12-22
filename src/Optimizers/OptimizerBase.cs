global using AiDotNet.Caching;
global using AiDotNet.Enums;
global using AiDotNet.Evaluation;
global using AiDotNet.Models.Inputs;
using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Represents the base class for all optimization algorithms, providing common functionality and interfaces.
/// </summary>
/// <remarks>
/// <para>
/// OptimizerBase is an abstract class that serves as the foundation for all optimization algorithms. It defines 
/// the common structure and functionality that all optimizers must implement, such as solution evaluation, 
/// caching, and adaptive parameter management. This class handles the core mechanics of optimization processes, 
/// allowing derived classes to focus on their specific optimization strategies.
/// </para>
/// <para><b>For Beginners:</b> This is the blueprint that all optimization algorithms follow.
/// 
/// Think of OptimizerBase as the common foundation that all optimizers are built upon:
/// - It defines what every optimizer must be able to do (evaluate solutions, manage caching)
/// - It provides shared tools that all optimizers can use (like adaptive learning rates and early stopping)
/// - It manages the evaluation of solutions and tracks the optimization progress
/// - It handles saving and loading optimizer states
/// 
/// All specific optimizer types (like genetic algorithms, particle swarm, etc.) inherit from this class,
/// which ensures they all work together consistently in the optimization process.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The type of input data for the model.</typeparam>
/// <typeparam name="TOutput">The type of output data for the model.</typeparam>
public abstract class OptimizerBase<T, TInput, TOutput> : IOptimizer<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the global execution engine for vector operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Provides numeric operations for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Provides random number generation for all derived classes.
    /// </summary>
    protected readonly Random Random;

    /// <summary>
    /// Contains the configuration options for the optimization algorithm.
    /// </summary>
    protected readonly OptimizationAlgorithmOptions<T, TInput, TOutput> Options;

    /// <summary>
    /// Options for prediction statistics calculations.
    /// </summary>
    protected readonly PredictionStatsOptions PredictionOptions;

    /// <summary>
    /// Options for model statistics calculations.
    /// </summary>
    protected readonly ModelStatsOptions ModelStatsOptions;

    /// <summary>
    /// Evaluates the performance of models.
    /// </summary>
    protected readonly IModelEvaluator<T, TInput, TOutput> ModelEvaluator;

    /// <summary>
    /// Detects the quality of fit for models.
    /// </summary>
    protected readonly IFitDetector<T, TInput, TOutput> FitDetector;

    /// <summary>
    /// Calculates the fitness score of models.
    /// </summary>
    protected readonly IFitnessCalculator<T, TInput, TOutput> FitnessCalculator;

    /// <summary>
    /// Stores the fitness scores of evaluated models.
    /// </summary>
    protected readonly List<T> FitnessList;

    /// <summary>
    /// Stores information about each optimization iteration.
    /// </summary>
    protected readonly List<OptimizationIterationInfo<T>> IterationHistoryList;

    /// <summary>
    /// Caches evaluated models to avoid redundant calculations.
    /// </summary>
    protected readonly IModelCache<T, TInput, TOutput> ModelCache;

    /// <summary>
    /// The current learning rate used in the optimization process.
    /// </summary>
    protected T CurrentLearningRate;

    /// <summary>
    /// The current momentum used in the optimization process.
    /// </summary>
    protected T CurrentMomentum;

    /// <summary>
    /// Counts the number of consecutive iterations without improvement.
    /// </summary>
    protected int IterationsWithoutImprovement;

    /// <summary>
    /// Counts the number of consecutive iterations with improvement.
    /// </summary>
    protected int IterationsWithImprovement;

    /// <summary>
    /// Gets the model that this optimizer is configured to optimize.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property provides access to the model that the optimizer is working with.
    /// It implements the IOptimizer interface property to expose the protected Model field.
    /// </para>
    /// <para><b>For Beginners:</b> This property lets external code see which model
    /// the optimizer is currently working with, without being able to change it.
    /// It's like a window that lets you look at the model but not touch it.
    /// </para>
    /// </remarks>
    public IFullModel<T, TInput, TOutput>? Model => _model;

    /// <summary>
    /// The model that this optimizer is configured to optimize.
    /// </summary>
    private IFullModel<T, TInput, TOutput>? _model;

    /// <summary>
    /// Initializes a new instance of the OptimizerBase class.
    /// </summary>
    /// <param name="model">The model to be optimized (can be null if set later).</param>
    /// <param name="options">The optimization algorithm options.</param>
    protected OptimizerBase(IFullModel<T, TInput, TOutput>? model,
        OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        _model = model;
        Random = new();
        NumOps = MathHelper.GetNumericOperations<T>();
        Options = options ?? new OptimizationAlgorithmOptions<T, TInput, TOutput>();
        PredictionOptions = Options.PredictionOptions;
        ModelStatsOptions = Options.ModelStatsOptions;
        ModelEvaluator = Options.ModelEvaluator;
        FitDetector = Options.FitDetector;
        FitnessCalculator = Options.FitnessCalculator;
        FitnessList = new List<T>();
        IterationHistoryList = new List<OptimizationIterationInfo<T>>();
        ModelCache = Options.ModelCache;
        CurrentLearningRate = NumOps.Zero;
        CurrentMomentum = NumOps.Zero;
    }

    /// <summary>
    /// Performs the optimization process.
    /// </summary>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    public abstract OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData);

    /// <summary>
    /// Retrieves cached step data for a given solution.
    /// </summary>
    /// <param name="key">The cache key for the solution.</param>
    /// <returns>The cached step data, if available; otherwise, null.</returns>
    protected OptimizationStepData<T, TInput, TOutput>? GetCachedStepData(string key)
    {
        return ModelCache.GetCachedStepData(key);
    }

    /// <summary>
    /// Caches step data for a given solution.
    /// </summary>
    /// <param name="key">The cache key for the solution.</param>
    /// <param name="stepData">The step data to cache.</param>
    protected void CacheStepData(string key, OptimizationStepData<T, TInput, TOutput> stepData)
    {
        ModelCache.CacheStepData(key, stepData);
    }

    /// <summary>
    /// Adjusts the parameters (weights) of a model.
    /// </summary>
    /// <param name="model">The model whose parameters should be adjusted.</param>
    /// <param name="adjustmentScale">Scale factor for parameter adjustments.</param>
    /// <param name="signFlipProbability">Probability of flipping a parameter's sign.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This is like adjusting the quantities of ingredients in your recipe.
    /// While keeping the same ingredients, you're changing how much of each one you use to
    /// find the perfect balance.
    /// </remarks>
    protected virtual void AdjustModelParameters(
        IFullModel<T, TInput, TOutput> model,
        double adjustmentScale = 0.1,
        double signFlipProbability = 0.05)
    {
        // Get current parameters
        var currentParameters = model.GetParameters();

        // Create new parameters by applying random adjustments
        var newParameters = AdjustParameters(
            currentParameters,
            adjustmentScale * Options.ExplorationRate,
            signFlipProbability);

        // Apply the new parameters to the model
        var updatedModel = model.WithParameters(newParameters);
    }

    /// <summary>
    /// Randomly selects a subset of features to use in a model.
    /// </summary>
    /// <param name="totalFeatures">The total number of available features.</param>
    /// <param name="minFeatures">The minimum number of features to select.</param>
    /// <param name="maxFeatures">The maximum number of features to select.</param>
    /// <returns>A list of selected feature indices.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This is like randomly selecting a subset of ingredients from
    /// your pantry to include in your recipe experiment.
    /// </remarks>
    protected virtual List<int> RandomlySelectFeatures(
        int totalFeatures,
        int? minFeatures = null,
        int? maxFeatures = null)
    {
        int min = minFeatures ?? Options.MinimumFeatures;
        int max = maxFeatures ?? Math.Min(Options.MaximumFeatures, totalFeatures);

        // Ensure min/max values are valid
        min = Math.Max(1, Math.Min(min, totalFeatures));
        max = Math.Min(max, totalFeatures);

        // If min > max (due to constraints), set them equal
        if (min > max)
        {
            max = min;
        }

        var selectedFeatures = new List<int>();
        int numFeatures = Random.Next(min, max + 1);

        while (selectedFeatures.Count < numFeatures)
        {
            int feature = Random.Next(totalFeatures);
            if (!selectedFeatures.Contains(feature))
            {
                selectedFeatures.Add(feature);
            }
        }

        return selectedFeatures;
    }

    /// <summary>
    /// Applies the selected features to a model.
    /// </summary>
    /// <param name="model">The model to apply feature selection to.</param>
    /// <param name="selectedFeatures">The list of selected feature indices.</param>
    protected virtual void ApplyFeatureSelection(IFullModel<T, TInput, TOutput> model, List<int> selectedFeatures)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        if (selectedFeatures == null || selectedFeatures.Count == 0)
            throw new ArgumentException("At least one feature must be selected.", nameof(selectedFeatures));

        // Apply features if model supports it
        if (model is IFeatureAware featureAwareModel)
        {
            featureAwareModel.SetActiveFeatureIndices(selectedFeatures);
        }
    }

    /// <summary>
    /// Adjusts a vector of parameters by applying random modifications.
    /// </summary>
    /// <param name="parameters">The original parameters.</param>
    /// <param name="adjustmentScale">Scale factor for parameter adjustments.</param>
    /// <param name="signFlipProbability">Probability of flipping a parameter's sign.</param>
    /// <returns>A new vector with adjusted parameters.</returns>
    protected virtual Vector<T> AdjustParameters(
        Vector<T> parameters,
        double adjustmentScale,
        double signFlipProbability)
    {
        var newParameters = new Vector<T>(parameters.Length);

        for (int i = 0; i < parameters.Length; i++)
        {
            // Generate a random adjustment factor
            double factor = 1.0 + ((Random.NextDouble() * 2.0 - 1.0) * adjustmentScale);

            // Apply the adjustment
            T originalValue = parameters[i];
            T newValue = NumOps.Multiply(originalValue, NumOps.FromDouble(factor));

            // Add some probability of the parameter flipping sign
            if (Random.NextDouble() < signFlipProbability)
            {
                newValue = NumOps.Negate(newValue);
            }

            newParameters[i] = newValue;
        }

        return newParameters;
    }

    /// <summary>
    /// Evaluates a solution, using cached results if available.
    /// </summary>
    /// <param name="solution">The solution to evaluate.</param>
    /// <param name="inputData">The input data for evaluation.</param>
    /// <returns>The evaluation results for the solution.</returns>
    protected virtual OptimizationStepData<T, TInput, TOutput> EvaluateSolution(
        IFullModel<T, TInput, TOutput> solution,
        OptimizationInputData<T, TInput, TOutput> inputData)
    {
        string cacheKey = GenerateCacheKey(solution, inputData);
        var cachedStepData = ModelCache.GetCachedStepData(cacheKey);

        if (cachedStepData != null)
        {
            return cachedStepData;
        }

        var stepData = PrepareAndEvaluateSolution(solution, inputData);
        ModelCache.CacheStepData(cacheKey, stepData);

        return stepData;
    }

    /// <summary>
    /// Prepares and evaluates a solution, applying feature selection before checking the cache.
    /// </summary>
    /// <param name="solution">The solution to evaluate.</param>
    /// <param name="inputData">The input data for evaluation.</param>
    /// <returns>The evaluation results for the solution.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method prepares a model with a specific set of features,
    /// checks if we've already trained this exact configuration before, and if not,
    /// trains and evaluates the model with the selected features.
    /// </remarks>
    protected OptimizationStepData<T, TInput, TOutput> PrepareAndEvaluateSolution(
        IFullModel<T, TInput, TOutput> solution,
        OptimizationInputData<T, TInput, TOutput> inputData)
    {
        // Step 1: Generate random feature selection independent of model state
        var selectedFeaturesIndices = RandomlySelectFeatures(
            InputHelper<T, TInput>.GetInputSize(inputData.XTrain),
            Options.MinimumFeatures,
            Options.MaximumFeatures);

        // Step 2: Apply feature selection to the model BEFORE we check the cache
        ApplyFeatureSelection(solution, selectedFeaturesIndices);

        // Step 3: Generate cache key based on the selected features and check cache
        string cacheKey = GenerateCacheKey(solution, inputData);
        var cachedStepData = ModelCache.GetCachedStepData(cacheKey);

        if (cachedStepData != null)
        {
            return cachedStepData;
        }

        // Step 4: Apply feature selection to input data
        var selectedFeatures = ModelHelper<T, TInput, TOutput>.GetColumnVectors(
            inputData.XTrain, [.. selectedFeaturesIndices]);

        var XTrainSubset = OptimizerHelper<T, TInput, TOutput>.SelectFeatures(
            inputData.XTrain, selectedFeaturesIndices);
        var XValSubset = OptimizerHelper<T, TInput, TOutput>.SelectFeatures(
            inputData.XValidation, selectedFeaturesIndices);
        var XTestSubset = OptimizerHelper<T, TInput, TOutput>.SelectFeatures(
            inputData.XTest, selectedFeaturesIndices);

        // Step 5: Create input data with selected features
        var subsetInputData = new OptimizationInputData<T, TInput, TOutput>
        {
            XTrain = XTrainSubset,
            YTrain = inputData.YTrain,
            XValidation = XValSubset,
            YValidation = inputData.YValidation,
            XTest = XTestSubset,
            YTest = inputData.YTest
        };

        // Step 6: Create evaluation input
        var input = new ModelEvaluationInput<T, TInput, TOutput>
        {
            Model = solution,
            InputData = subsetInputData
        };

        // Step 7: Train and evaluate
        var (currentFitnessScore, fitDetectionResult, evaluationData) =
            TrainAndEvaluateSolution(input);

        FitnessList.Add(currentFitnessScore);

        // Step 8: Create and store step data
        var stepData = new OptimizationStepData<T, TInput, TOutput>
        {
            Solution = solution.DeepCopy(),  // Now trained, so DeepCopy works
            SelectedFeatures = selectedFeatures,
            XTrainSubset = XTrainSubset,
            XValSubset = XValSubset,
            XTestSubset = XTestSubset,
            FitnessScore = currentFitnessScore,
            FitDetectionResult = fitDetectionResult,
            EvaluationData = evaluationData
        };

        // Step 9: Cache the results
        ModelCache.CacheStepData(cacheKey, stepData);

        return stepData;
    }

    /// <summary>
    /// Evaluates a solution using the model evaluator, fit detector, and fitness calculator.
    /// </summary>
    /// <param name="input">The input data for model evaluation.</param>
    /// <returns>A tuple containing the fitness score, fit detection result, and evaluation data.</returns>
    private (T CurrentFitnessScore, FitDetectorResult<T> FitDetectionResult, ModelEvaluationData<T, TInput, TOutput> EvaluationData)
        TrainAndEvaluateSolution(ModelEvaluationInput<T, TInput, TOutput> input)
    {
        // Train the model
        input.Model?.Train(input.InputData.XTrain, input.InputData.YTrain);

        // Evaluate the trained model
        var evaluationData = ModelEvaluator.EvaluateModel(input);
        var fitDetectionResult = FitDetector.DetectFit(evaluationData);
        var currentFitnessScore = FitnessCalculator.CalculateFitnessScore(evaluationData);

        return (currentFitnessScore, fitDetectionResult, evaluationData);
    }

    /// <summary>
    /// Calculates the loss for a given solution.
    /// </summary>
    /// <param name="solution">The solution to evaluate.</param>
    /// <param name="inputData">The input data for evaluation.</param>
    /// <returns>The calculated loss value.</returns>
    protected virtual T CalculateLoss(
        IFullModel<T, TInput, TOutput> solution,
        OptimizationInputData<T, TInput, TOutput> inputData)
    {
        var stepData = EvaluateSolution(solution, inputData);
        return FitnessCalculator.CalculateFitnessScore(stepData.EvaluationData);
    }

    /// <summary>
    /// Creates a new optimization result based on the best step data found during optimization.
    /// </summary>
    /// <param name="bestStepData">The data from the best optimization step.</param>
    /// <param name="input">The original input data used for optimization.</param>
    /// <returns>A structured optimization result containing all relevant information.</returns>
    /// <remarks>
    /// <para>
    /// This method packages all the optimization results into a single structured object that can be returned to the caller.
    /// It includes the best solution found, its fitness score, training metrics, validation metrics, and test metrics.
    /// </para>
    /// <para><b>For Beginners:</b> Think of this method as packaging up all the results from your optimization process
    /// into one neat container.
    /// 
    /// It's like finishing a science experiment and organizing all your findings into a clear report:
    /// - It includes the best solution found
    /// - It shows how well that solution performed (fitness score)
    /// - It contains detailed statistics about how the solution performed on different datasets
    /// - It records other important information like selected features and iteration count
    /// 
    /// This makes it easy for you or other code to work with the results.
    /// </para>
    /// </remarks>
    protected OptimizationResult<T, TInput, TOutput> CreateOptimizationResult(OptimizationStepData<T, TInput, TOutput> bestStepData, OptimizationInputData<T, TInput, TOutput> input)
    {
        return OptimizerHelper<T, TInput, TOutput>.CreateOptimizationResult(
            bestStepData.Solution,
            bestStepData.FitnessScore,
            FitnessList,
            bestStepData.SelectedFeatures,
            new OptimizationResult<T, TInput, TOutput>.DatasetResult
            {
                X = bestStepData.XTrainSubset,
                Y = input.YTrain,
                Predictions = bestStepData.EvaluationData.TrainingSet.Predicted,
                ErrorStats = bestStepData.EvaluationData.TrainingSet.ErrorStats,
                ActualBasicStats = bestStepData.EvaluationData.TrainingSet.ActualBasicStats,
                PredictedBasicStats = bestStepData.EvaluationData.TrainingSet.PredictedBasicStats,
                PredictionStats = bestStepData.EvaluationData.TrainingSet.PredictionStats
            },
            new OptimizationResult<T, TInput, TOutput>.DatasetResult
            {
                X = bestStepData.XValSubset,
                Y = input.YValidation,
                Predictions = bestStepData.EvaluationData.ValidationSet.Predicted,
                ErrorStats = bestStepData.EvaluationData.ValidationSet.ErrorStats,
                ActualBasicStats = bestStepData.EvaluationData.ValidationSet.ActualBasicStats,
                PredictedBasicStats = bestStepData.EvaluationData.ValidationSet.PredictedBasicStats,
                PredictionStats = bestStepData.EvaluationData.ValidationSet.PredictionStats
            },
            new OptimizationResult<T, TInput, TOutput>.DatasetResult
            {
                X = bestStepData.XTestSubset,
                Y = input.YTest,
                Predictions = bestStepData.EvaluationData.TestSet.Predicted,
                ErrorStats = bestStepData.EvaluationData.TestSet.ErrorStats,
                ActualBasicStats = bestStepData.EvaluationData.TestSet.ActualBasicStats,
                PredictedBasicStats = bestStepData.EvaluationData.TestSet.PredictedBasicStats,
                PredictionStats = bestStepData.EvaluationData.TestSet.PredictionStats
            },
            bestStepData.FitDetectionResult,
            IterationHistoryList.Count
        );
    }

    /// <summary>
    /// Applies feature selection to a model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method selects a subset of features to be used by the model, potentially
    /// improving its performance by focusing on the most relevant data dimensions.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like deciding which ingredients to include in your recipe. Some ingredients
    /// might not be necessary or might even make the dish worse, so you're experimenting
    /// with different combinations to find which ones are truly important.
    /// </para>
    /// </remarks>
    /// <param name="model">The model to apply feature selection to.</param>
    /// <param name="totalFeatures">The total number of available features.</param>
    protected virtual void ApplyFeatureSelection(IFullModel<T, TInput, TOutput> model, int totalFeatures)
    {
        // Randomly select features
        var selectedFeatures = RandomlySelectFeatures(
            totalFeatures,
            Options.MinimumFeatures,
            Options.MaximumFeatures);

        // Apply the selected features to the model using the base class method
        ApplyFeatureSelection(model, selectedFeatures);
    }

    /// <summary>
    /// Creates a potential solution based on the optimization mode.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method creates a new model variant by either selecting features, adjusting parameters,
    /// or both, depending on the optimization mode.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating a new version of the recipe. Depending on what you're focusing on,
    /// you might change which ingredients you use, how much of each ingredient you add,
    /// or both aspects at once.
    /// </para>
    /// </remarks>
    /// <param name="xTrain">Training data used to determine data dimensions.</param>
    /// <returns>A new potential solution (model variant).</returns>
    protected virtual IFullModel<T, TInput, TOutput> CreateSolution(TInput xTrain)
    {
        // Create a deep copy of the model to avoid modifying the original
        var solution = Model!.DeepCopy();

        // Return the deep copy - subclasses can override to add custom solution creation logic
        return solution;
    }

    /// <summary>
    /// Generates a cache key for the given solution and input data.
    /// </summary>
    /// <param name="solution">The solution model.</param>
    /// <param name="inputData">The optimization input data.</param>
    /// <returns>A unique cache key string.</returns>
    protected virtual string GenerateCacheKey(IFullModel<T, TInput, TOutput> solution, OptimizationInputData<T, TInput, TOutput> inputData)
    {
        // Generate a simple cache key based on parameter values
        var parameters = solution.GetParameters();
        var paramHash = parameters.GetHashCode();
        return $"{solution.GetType().Name}_{paramHash}";
    }

    /// <summary>
    /// Compares the current model result with the best result found so far and updates the best result if the current one is better.
    /// </summary>
    /// <param name="currentResult">The most recent model result.</param>
    /// <param name="bestResult">The best model result found so far, passed by reference so it can be updated.</param>
    /// <remarks>
    /// <para>
    /// This method compares fitness scores between the current and best results using the fitness calculator.
    /// If the current result has a better fitness score, all properties of the best result are updated
    /// with the values from the current result.
    /// </para>
    /// <para><b>For Beginners:</b> This method is like keeping track of a high score in a game.
    ///
    /// When you get a new score (currentResult), it checks:
    /// - Is this score better than the previous best score?
    /// - If yes, it updates the high score (bestResult) with this new score
    /// - It also saves all the details about how that high score was achieved
    ///
    /// The "ref" keyword means it directly updates the bestResult that was passed in,
    /// so the caller immediately sees the updated values.
    /// </para>
    /// </remarks>
    private void UpdateAndApplyBestSolution(ModelResult<T, TInput, TOutput> currentResult, ref ModelResult<T, TInput, TOutput> bestResult)
    {
        if (FitnessCalculator.IsBetterFitness(currentResult.Fitness, bestResult.Fitness))
        {
            bestResult.Solution = currentResult.Solution;
            bestResult.Fitness = currentResult.Fitness;
            bestResult.FitDetectionResult = currentResult.FitDetectionResult;
            bestResult.EvaluationData = currentResult.EvaluationData;
            bestResult.SelectedFeatures = currentResult.SelectedFeatures;
        }
    }

    /// <summary>
    /// Updates the best step data if the current step data has a better solution.
    /// </summary>
    /// <param name="currentStepData">The current optimization step data.</param>
    /// <param name="bestStepData">The best optimization step data found so far, passed by reference to be updated.</param>
    /// <remarks>
    /// <para>
    /// This method wraps the current and best step data in ModelResult objects and calls UpdateAndApplyBestSolution
    /// to determine if the current step data is better. If it is, the bestStepData reference is updated with values
    /// from the current step data.
    /// </para>
    /// <para><b>For Beginners:</b> This method compares two sets of optimization results to keep the better one.
    /// 
    /// Think of it like a talent competition:
    /// - You have a current contestant (currentStepData)
    /// - You have the current champion (bestStepData)
    /// - This method compares them to see who performs better
    /// - If the contestant wins, they become the new champion
    /// 
    /// This is a key part of optimization - always keeping track of the best solution found so far.
    /// </para>
    /// </remarks>
    protected void UpdateBestSolution(OptimizationStepData<T, TInput, TOutput> currentStepData, ref OptimizationStepData<T, TInput, TOutput> bestStepData)
    {
        var currentResult = new ModelResult<T, TInput, TOutput>
        {
            Solution = currentStepData.Solution,
            Fitness = currentStepData.FitnessScore,
            FitDetectionResult = currentStepData.FitDetectionResult,
            EvaluationData = currentStepData.EvaluationData,
            SelectedFeatures = currentStepData.SelectedFeatures
        };

        var bestResult = new ModelResult<T, TInput, TOutput>
        {
            Solution = bestStepData.Solution,
            Fitness = bestStepData.FitnessScore,
            FitDetectionResult = bestStepData.FitDetectionResult,
            EvaluationData = bestStepData.EvaluationData,
            SelectedFeatures = bestStepData.SelectedFeatures
        };

        UpdateAndApplyBestSolution(currentResult, ref bestResult);

        // Update the bestStepData with the new best values
        bestStepData.Solution = bestResult.Solution;
        bestStepData.FitnessScore = bestResult.Fitness;
        bestStepData.FitDetectionResult = bestResult.FitDetectionResult;
        bestStepData.EvaluationData = bestResult.EvaluationData;
        bestStepData.SelectedFeatures = bestResult.SelectedFeatures;
    }

    /// <summary>
    /// Initializes the adaptive parameters used during optimization to their starting values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets the initial values for parameters that can adapt during optimization, such as
    /// learning rate and momentum. It also resets tracking variables for monitoring improvement.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the starting values for parameters that will change
    /// during the optimization process.
    /// 
    /// Think of it like setting up a car before a long journey:
    /// - Setting the initial speed (learning rate)
    /// - Setting the initial acceleration (momentum)
    /// - Resetting the trip counters (improvement trackers)
    /// 
    /// These values will adjust automatically during optimization to help find the best solution efficiently.
    /// The learning rate controls how big each step is, while momentum helps move through flat areas.
    /// </para>
    /// </remarks>
    protected virtual void InitializeAdaptiveParameters()
    {
        CurrentLearningRate = NumOps.FromDouble(Options.InitialLearningRate);
        CurrentMomentum = NumOps.FromDouble(Options.InitialMomentum);
        IterationsWithoutImprovement = 0;
        IterationsWithImprovement = 0;
    }

    /// <summary>
    /// Resets the optimizer state, clearing the model cache.
    /// </summary>
    public virtual void Reset()
    {
        ModelCache.ClearCache();
        ResetAdaptiveParameters();
    }

    /// <summary>
    /// Resets the adaptive parameters back to their initial values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method calls InitializeAdaptiveParameters to reset all adaptive parameters to their starting values.
    /// This can be useful when the optimization process needs to be restarted or when a significant change occurs.
    /// </para>
    /// <para><b>For Beginners:</b> This method resets the optimization process parameters to start fresh.
    /// 
    /// It's like restarting a navigation system when you've gone off course:
    /// - You reset your speed back to the initial value
    /// - You reset your direction and momentum
    /// - You clear any history of previous attempts
    /// 
    /// This gives the optimization a clean slate to try again from the beginning.
    /// </para>
    /// </remarks>
    protected virtual void ResetAdaptiveParameters()
    {
        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Updates the adaptive parameters based on the progress of optimization.
    /// </summary>
    /// <param name="currentStepData">The current optimization step data.</param>
    /// <param name="previousStepData">The previous optimization step data.</param>
    /// <remarks>
    /// <para>
    /// This method adjusts the learning rate and momentum based on whether the optimization is improving.
    /// If the fitness is improving, it may increase certain parameters to move faster.
    /// If the fitness is not improving, it may decrease certain parameters to explore more carefully.
    /// </para>
    /// <para><b>For Beginners:</b> This method changes how the optimization behaves based on its progress.
    /// 
    /// Think of it like adjusting your driving based on the road conditions:
    /// - If you're making good progress, you might speed up (decrease learning rate, increase momentum)
    /// - If you're not improving, you might slow down and try different directions (increase learning rate, decrease momentum)
    /// 
    /// These automatic adjustments help the optimizer find better solutions by being more efficient:
    /// - When close to a good solution, it takes smaller, more precise steps
    /// - When stuck in a difficult area, it tries different approaches
    /// 
    /// The learning rate controls how big each step is in the optimization process.
    /// The momentum helps maintain direction through flat or noisy areas.
    /// </para>
    /// </remarks>
    protected virtual void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        if (Options.UseAdaptiveLearningRate)
        {
            if (FitnessCalculator.IsBetterFitness(currentStepData.FitnessScore, previousStepData.FitnessScore))
            {
                CurrentLearningRate = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(Options.LearningRateDecay));
                IterationsWithoutImprovement++;
                IterationsWithImprovement = 0;
            }
            else
            {
                CurrentLearningRate = NumOps.Divide(CurrentLearningRate, NumOps.FromDouble(Options.LearningRateDecay));
                IterationsWithImprovement++;
                IterationsWithoutImprovement = 0;
            }

            CurrentLearningRate = MathHelper.Max(NumOps.FromDouble(Options.MinLearningRate),
                MathHelper.Min(NumOps.FromDouble(Options.MaxLearningRate), CurrentLearningRate));
        }

        if (Options.UseAdaptiveMomentum)
        {
            if (IterationsWithImprovement > 2)
            {
                CurrentMomentum = NumOps.Multiply(CurrentMomentum, NumOps.FromDouble(Options.MomentumIncreaseFactor));
            }
            else if (IterationsWithoutImprovement > 2)
            {
                CurrentMomentum = NumOps.Multiply(CurrentMomentum, NumOps.FromDouble(Options.MomentumDecreaseFactor));
            }

            CurrentMomentum = MathHelper.Max(NumOps.FromDouble(Options.MinMomentum),
                MathHelper.Min(NumOps.FromDouble(Options.MaxMomentum), CurrentMomentum));
        }
    }

    /// <summary>
    /// Updates the iteration history with the current step data and checks if early stopping should be applied.
    /// </summary>
    /// <param name="iteration">The current iteration number.</param>
    /// <param name="stepData">The current step data.</param>
    /// <returns>True if optimization should stop early, false if it should continue.</returns>
    /// <remarks>
    /// <para>
    /// This method adds the current iteration's data to the history list and then checks if early stopping
    /// criteria have been met. Early stopping helps prevent overfitting by stopping the optimization process
    /// when progress stagnates for a number of iterations.
    /// </para>
    /// <para><b>For Beginners:</b> This method keeps track of progress and decides if it's time to stop trying.
    /// 
    /// Imagine you're trying to climb a hill to find the highest point:
    /// - You keep a record of your altitude at each step (the iteration history)
    /// - If you haven't gone any higher after walking for a while, you might decide to stop
    /// - This saves time and prevents you from wandering too far
    /// 
    /// Early stopping is important because:
    /// - It saves computation time when further optimization isn't helping
    /// - It can prevent overfitting (when a model works too well on training data but poorly on new data)
    /// - It tells you when you've found a good enough solution
    /// </para>
    /// </remarks>
    protected bool UpdateIterationHistoryAndCheckEarlyStopping(int iteration, OptimizationStepData<T, TInput, TOutput> stepData)
    {
        IterationHistoryList.Add(new OptimizationIterationInfo<T>
        {
            Iteration = iteration,
            Fitness = stepData.FitnessScore,
            FitDetectionResult = stepData.FitDetectionResult
        });

        // Check for early stopping
        if (Options.UseEarlyStopping && ShouldEarlyStop())
        {
            return true; // Signal to stop the optimization
        }

        return false; // Continue optimization
    }

    /// <summary>
    /// Determines whether the optimization process should stop early based on the recent history of fitness scores.
    /// </summary>
    /// <returns>True if early stopping criteria are met, false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// This method checks if the fitness score has not improved significantly over a specified number of iterations.
    /// If the improvement is below a threshold for a consecutive number of iterations, it suggests stopping early.
    /// </para>
    /// <para><b>For Beginners:</b> This method decides if it's time to stop trying to improve the solution.
    /// 
    /// Imagine you're trying to beat your personal best in a game:
    /// - You keep playing and tracking your scores
    /// - If your score hasn't improved much after several attempts, you might decide to stop
    /// - This method does that for the optimization process
    /// 
    /// It's useful because:
    /// - It prevents wasting time when the solution isn't getting much better
    /// - It helps avoid overfitting, where the model becomes too specific to the training data
    /// - It can save computational resources by stopping when further improvement is unlikely
    /// </para>
    /// </remarks>
    public virtual bool ShouldEarlyStop()
    {
        if (IterationHistoryList.Count < Options.EarlyStoppingPatience)
        {
            return false;
        }

        var recentIterations = IterationHistoryList.Skip(Math.Max(0, IterationHistoryList.Count - Options.EarlyStoppingPatience)).ToList();

        // Find the best fitness score
        T bestFitness = IterationHistoryList[0].Fitness;
        foreach (var iteration in IterationHistoryList)
        {
            if (FitnessCalculator.IsBetterFitness(iteration.Fitness, bestFitness))
            {
                bestFitness = iteration.Fitness;
            }
        }

        // Check for improvement in recent iterations
        bool noImprovement = true;
        foreach (var iteration in recentIterations)
        {
            if (FitnessCalculator.IsBetterFitness(iteration.Fitness, bestFitness))
            {
                noImprovement = false;
                break;
            }
        }

        // Check for consecutive bad fits
        int consecutiveBadFits = 0;
        foreach (var iteration in recentIterations.Reverse<OptimizationIterationInfo<T>>())
        {
            if (iteration.FitDetectionResult.FitType != FitType.GoodFit)
            {
                consecutiveBadFits++;
            }
            else
            {
                break;
            }
        }

        return noImprovement || consecutiveBadFits >= Options.BadFitPatience;
    }

    /// <summary>
    /// Serializes the optimizer state to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized optimizer state.</returns>
    /// <remarks>
    /// <para>
    /// This method saves the current state of the optimizer, including its options and any derived class-specific data.
    /// The serialized data can be used to reconstruct the optimizer's state later or to transfer it between processes.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves all the important information about the optimizer's current state.
    /// 
    /// Think of it like taking a snapshot of the optimizer:
    /// - It captures all the current settings and progress
    /// - This snapshot can be saved to a file or sent to another computer
    /// - Later, you can use this snapshot to continue from where you left off
    /// 
    /// This is useful for:
    /// - Saving your progress in case the program crashes
    /// - Sharing your optimizer's state with others
    /// - Continuing a long optimization process after a break
    /// </para>
    /// </remarks>
    public virtual byte[] Serialize()
    {
        using var memoryStream = new MemoryStream();
        using var writer = new BinaryWriter(memoryStream);

        // Write the type name
        writer.Write(GetType().AssemblyQualifiedName ?? string.Empty);

        // Serialize options
        var optionsJson = JsonConvert.SerializeObject(Options);
        writer.Write(optionsJson);

        // Allow derived classes to serialize additional data
        SerializeAdditionalData(writer);

        return memoryStream.ToArray();
    }

    /// <summary>
    /// Reconstructs the optimizer from a serialized byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized optimizer.</param>
    /// <exception cref="InvalidOperationException">Thrown when the serialized type doesn't match the current type.</exception>
    /// <remarks>
    /// <para>
    /// This method rebuilds the optimizer's state from a serialized byte array. It verifies that the
    /// serialized type matches the current type, restores configuration options, and calls into derived
    /// classes to restore any additional data.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved optimizer state.
    /// 
    /// It's like restoring a snapshot:
    /// - It takes a byte array that was previously created with Serialize()
    /// - It checks that the type matches (you can't load settings from a different type of optimizer)
    /// - It reconstructs all the settings and values
    /// 
    /// This allows you to:
    /// - Continue working with an optimizer that you previously saved
    /// - Use an optimizer that someone else created and shared
    /// - Recover from backups if needed
    /// </para>
    /// </remarks>
    public virtual void Deserialize(byte[] data)
    {
        using MemoryStream ms = new(data);
        using BinaryReader reader = new(ms);

        // Read and verify the type
        string typeName = reader.ReadString();
        if (typeName != this.GetType().AssemblyQualifiedName)
        {
            throw new InvalidOperationException("Mismatched optimizer type during deserialization.");
        }

        // Deserialize options
        string optionsJson = reader.ReadString();
        Type optionsType = Options?.GetType() ?? typeof(OptimizationAlgorithmOptions<T, TInput, TOutput>);
        object? deserializedOptions = JsonConvert.DeserializeObject(optionsJson, optionsType);
        var options = deserializedOptions as OptimizationAlgorithmOptions<T, TInput, TOutput>;

        // Update the options
        if (options != null)
        {
            UpdateOptions(options);
        }

        // Allow derived classes to deserialize additional data
        DeserializeAdditionalData(reader);
    }

    /// <summary>
    /// Serializes additional data specific to derived optimizer classes.
    /// </summary>
    /// <param name="writer">The binary writer to use for serialization.</param>
    /// <remarks>
    /// <para>
    /// This protected virtual method allows derived optimizer classes to serialize additional data
    /// beyond what is handled by the base implementation. The base implementation does nothing.
    /// </para>
    /// <para><b>For Beginners:</b> This method allows specialized optimizers to save their unique settings.
    /// 
    /// Think of it as adding extra details to the snapshot:
    /// - The base optimizer saves the common information
    /// - This method lets specialized optimizers save their specific settings
    /// - It's empty in the base class since it doesn't have any special data to save
    /// 
    /// This is part of the extensible design that allows different types of optimizers
    /// to all use the same saving and loading system.
    /// </para>
    /// </remarks>
    protected virtual void SerializeAdditionalData(BinaryWriter writer)
    {
        // Base implementation does nothing
    }

    /// <summary>
    /// Deserializes additional data specific to derived optimizer classes.
    /// </summary>
    /// <param name="reader">The binary reader to use for deserialization.</param>
    /// <remarks>
    /// <para>
    /// This protected virtual method allows derived optimizer classes to deserialize additional data
    /// beyond what is handled by the base implementation. The base implementation does nothing.
    /// </para>
    /// <para><b>For Beginners:</b> This method allows specialized optimizers to load their unique settings.
    /// 
    /// Think of it as reading extra details from the snapshot:
    /// - The base optimizer loads the common information
    /// - This method lets specialized optimizers load their specific settings
    /// - It's empty in the base class since it doesn't have any special data to load
    /// 
    /// This complements the SerializeAdditionalData method to allow different optimizers
    /// to save and load their specialized settings.
    /// </para>
    /// </remarks>
    protected virtual void DeserializeAdditionalData(BinaryReader reader)
    {
        // Base implementation does nothing
    }

    /// <summary>
    /// Updates the optimizer's options with the provided options.
    /// </summary>
    /// <param name="options">The options to apply to this optimizer.</param>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to update their specific options
    /// based on the provided generic optimization options.
    /// </para>
    /// <para><b>For Beginners:</b> This method configures the optimizer with specific settings.
    /// 
    /// Think of it like updating the settings on your device:
    /// - You provide a set of options (settings)
    /// - This method applies those options to the optimizer
    /// - Each type of optimizer will implement this differently based on what options it supports
    /// 
    /// This is important because different optimizers might interpret the same options differently,
    /// or might have additional specialized options.
    /// </para>
    /// </remarks>
    protected abstract void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options);

    /// <summary>
    /// Performs a single optimization step, updating the model parameters based on gradients.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method performs one iteration of parameter updates. The default implementation
    /// throws a NotImplementedException, and gradient-based optimizers should override this method
    /// to implement their specific parameter update logic.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like taking one small step toward a better model.
    /// After calculating how wrong the model is (gradients), this method adjusts the
    /// model's parameters slightly to make it more accurate.
    ///
    /// Think of it like adjusting a recipe:
    /// 1. You taste the dish (check model performance)
    /// 2. You determine what needs changing (calculate gradients)
    /// 3. You adjust the ingredients (this Step method updates parameters)
    /// 4. Repeat until the dish tastes good (model is accurate)
    ///
    /// Most training loops call this method many times, each time making the model
    /// a little bit better.
    /// </para>
    /// </remarks>
    public virtual void Step()
    {
        throw new NotImplementedException(
            "Step() method is not implemented for this optimizer type. " +
            "This optimizer may be a non-gradient-based optimizer that uses the Optimize() method instead, " +
            "or the derived class needs to implement the Step() method.");
    }

    /// <summary>
    /// Calculates the parameter update based on the provided gradients.
    /// </summary>
    /// <param name="gradients">The gradients used to compute the parameter updates.</param>
    /// <returns>The calculated parameter updates as a dictionary mapping parameter names to their update vectors.</returns>
    public virtual Dictionary<string, Vector<T>> CalculateUpdate(Dictionary<string, Vector<T>> gradients)
    {
        throw new NotImplementedException(
            "CalculateUpdate() method is not implemented for this optimizer type. " +
            "This optimizer may be a non-gradient-based optimizer that uses the Optimize() method instead, " +
            "or the derived class needs to implement the CalculateUpdate() method.");
    }

    /// <summary>
    /// Gets the current options for this optimizer.
    /// </summary>
    /// <returns>The current optimization algorithm options.</returns>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to return their current
    /// configuration options.
    /// </para>
    /// <para><b>For Beginners:</b> This method retrieves the current settings of the optimizer.
    ///
    /// It's like checking the current configuration of your device:
    /// - It returns all the settings that control how the optimizer behaves
    /// - Each type of optimizer will implement this differently to return its specific settings
    ///
    /// This is useful for:
    /// - Seeing what settings are currently active
    /// - Making a copy of settings to modify and apply later
    /// - Comparing settings between different optimizers
    /// </para>
    /// </remarks>
    public abstract OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions();

    /// <summary>
    /// Calculates the parameter updates based on the gradients.
    /// </summary>
    /// <param name="gradients">The gradients of the loss function with respect to the parameters.</param>
    /// <param name="parameters">The current parameter values.</param>
    /// <returns>The updates to be applied to the parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This base implementation returns the gradients as-is,
    /// which represents vanilla gradient descent. Derived classes should override this
    /// to implement specific optimization algorithms (like Adam, SGD with momentum, etc.).</para>
    /// </remarks>
    public virtual Vector<T> CalculateUpdate(Vector<T> gradients, Vector<T> parameters)
    {
        // Base implementation: return gradients as-is (vanilla gradient descent)
        // Derived classes should override this for specific optimization algorithms
        return gradients;
    }

    /// <summary>
    /// Initializes a random solution within the given bounds.
    /// </summary>
    /// <param name="lowerBounds">Lower bounds for each parameter.</param>
    /// <param name="upperBounds">Upper bounds for each parameter.</param>
    /// <returns>A vector representing a random solution.</returns>
    protected virtual Vector<T> InitializeRandomSolution(Vector<T> lowerBounds, Vector<T> upperBounds)
    {
        if (lowerBounds == null) throw new ArgumentNullException(nameof(lowerBounds));
        if (upperBounds == null) throw new ArgumentNullException(nameof(upperBounds));
        if (lowerBounds.Length != upperBounds.Length)
            throw new ArgumentException("Lower and upper bounds must have the same length");

        // Validate bounds
        for (int i = 0; i < lowerBounds.Length; i++)
        {
            if (NumOps.GreaterThan(lowerBounds[i], upperBounds[i]))
            {
                throw new ArgumentException(
                    $"Lower bound ({lowerBounds[i]}) is greater than upper bound ({upperBounds[i]}) at dimension {i}.",
                    nameof(lowerBounds));
            }
        }

        var solution = new Vector<T>(lowerBounds.Length);
        for (int i = 0; i < lowerBounds.Length; i++)
        {
            // Generate random value between lower and upper bounds
            var range = NumOps.Subtract(upperBounds[i], lowerBounds[i]);
            var randomFraction = NumOps.FromDouble(Random.NextDouble());
            var randomValue = NumOps.Add(lowerBounds[i], NumOps.Multiply(range, randomFraction));
            solution[i] = randomValue;
        }
        return solution;
    }

    /// <summary>
    /// Initializes a random solution by computing lower and upper bounds from training data.
    /// </summary>
    /// <param name="trainingData">The training data used to compute bounds.</param>
    /// <returns>A model with randomly initialized parameters within the computed bounds.</returns>
    /// <remarks>
    /// <para>
    /// This method computes proper lower and upper bounds from the training data:
    /// - Lower bounds: minimum value for each feature across all training samples
    /// - Upper bounds: maximum value for each feature across all training samples
    /// </para>
    /// <para>
    /// This ensures valid random initialization where lower < upper for proper random solution generation.
    /// </para>
    /// <para><b>For Beginners:</b> Instead of you having to manually specify lower and upper bounds,
    /// this method analyzes your training data to find the minimum and maximum values for each feature,
    /// then creates random parameters somewhere within those ranges.</para>
    /// </remarks>
    protected virtual IFullModel<T, TInput, TOutput> InitializeRandomSolution(TInput trainingData)
    {
        if (trainingData == null) throw new ArgumentNullException(nameof(trainingData));

        // Compute lower and upper bounds from the training data
        // Following GitHub Copilot's suggestion: compute min/max from the data
        Vector<T> lowerBounds;
        Vector<T> upperBounds;

        if (trainingData is Matrix<T> matrix)
        {
            // Validate non-empty matrix before accessing elements
            if (matrix.Rows == 0)
            {
                throw new ArgumentException("Training data matrix cannot be empty", nameof(trainingData));
            }

            // Validate matrix has columns
            if (matrix.Columns == 0)
            {
                throw new ArgumentException("Training data matrix must have at least one column", nameof(trainingData));
            }

            // For Matrix input: compute min and max of each column (feature)
            int features = matrix.Columns;
            int paramCount = _model!.ParameterCount;

            lowerBounds = new Vector<T>(paramCount);
            upperBounds = new Vector<T>(paramCount);

            // Compute min/max for each feature column
            var featureMins = new T[features];
            var featureMaxs = new T[features];
            for (int col = 0; col < features; col++)
            {
                // Safe to access matrix[0, col] after validation above
                if (matrix.Rows == 0)
                {
                    throw new ArgumentException("Matrix cannot be empty", nameof(matrix));
                }
                T initialValue = matrix[0, col];
                T min = initialValue;
                T max = initialValue;
                for (int row = 1; row < matrix.Rows; row++)
                {
                    T value = matrix[row, col];
                    if (NumOps.LessThan(value, min)) min = value;
                    if (NumOps.GreaterThan(value, max)) max = value;
                }
                featureMins[col] = min;
                featureMaxs[col] = max;
            }

            // Fill bounds for each parameter using feature min/max, repeating or defaulting as needed
            for (int i = 0; i < paramCount; i++)
            {
                if (i < features)
                {
                    lowerBounds[i] = featureMins[i];
                    upperBounds[i] = featureMaxs[i];
                }
                else
                {
                    // If more parameters than features, use global min/max from all features
                    T min = featureMins[0];
                    T max = featureMaxs[0];
                    for (int j = 1; j < featureMins.Length; j++)
                    {
                        if (NumOps.LessThan(featureMins[j], min)) min = featureMins[j];
                        if (NumOps.GreaterThan(featureMaxs[j], max)) max = featureMaxs[j];
                    }
                    lowerBounds[i] = min;
                    upperBounds[i] = max;
                }
            }
        }
        else if (trainingData is Vector<T> vector)
        {
            // For Vector input: compute min and max of the vector
            if (vector.Length == 0)
            {
                throw new ArgumentException("Training data vector cannot be empty", nameof(trainingData));
            }

            T initialValue = vector[0];
            T min = initialValue;
            T max = initialValue;

            for (int i = 1; i < vector.Length; i++)
            {
                T value = vector[i];
                if (NumOps.LessThan(value, min)) min = value;
                if (NumOps.GreaterThan(value, max)) max = value;
            }

            // Bounds should match parameter count, not input dimensionality
            int paramCount = _model!.ParameterCount;
            lowerBounds = new Vector<T>(paramCount);
            upperBounds = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                lowerBounds[i] = min;
                upperBounds[i] = max;
            }
        }
        else
        {
            // Fallback: create reasonable default bounds based on parameter count
            int paramCount = _model!.ParameterCount;
            lowerBounds = new Vector<T>(paramCount);
            upperBounds = new Vector<T>(paramCount);

            for (int i = 0; i < paramCount; i++)
            {
                lowerBounds[i] = NumOps.FromDouble(-10.0);
                upperBounds[i] = NumOps.FromDouble(10.0);
            }
        }

        // Generate random parameters within the computed bounds
        // Note: InitializeRandomSolution(Vector<T>, Vector<T>) returns Vector<T> (verified at line 1184)
        // This is the correct return type for SetParameters() below
        var randomParams = InitializeRandomSolution(lowerBounds, upperBounds);

        // Create a new model with these random parameters
        var randomModel = _model.Clone();
        randomModel.SetParameters(randomParams);
        return randomModel;
    }

    /// <summary>
    /// Saves the optimizer state to a file.
    /// </summary>
    /// <param name="filePath">The path where the optimizer should be saved.</param>
    /// <remarks>
    /// <para>
    /// This method saves the complete state of the optimizer, including all configuration options
    /// and any optimizer-specific data, to a file.
    /// </para>
    /// <para><b>For Beginners:</b> This saves your optimizer's current settings and state to a file.
    ///
    /// Think of it like saving your progress:
    /// - It captures all the optimizer's settings and current state
    /// - This can be loaded later to resume optimization or reuse the same settings
    /// - It's useful for checkpointing long-running optimizations
    /// </para>
    /// </remarks>
    public virtual void SaveModel(string filePath)
    {
        byte[] serializedData = Serialize();
        File.WriteAllBytes(filePath, serializedData);
    }

    /// <summary>
    /// Loads the optimizer state from a file.
    /// </summary>
    /// <param name="filePath">The path to the file containing the saved optimizer.</param>
    /// <remarks>
    /// <para>
    /// This method loads the complete state of the optimizer from a file, including all configuration
    /// options and any optimizer-specific data.
    /// </para>
    /// <para><b>For Beginners:</b> This loads a previously saved optimizer from a file.
    ///
    /// It's like loading a saved game:
    /// - It restores all the optimizer's settings and state
    /// - You can continue optimization from where you left off
    /// - You can reuse optimizer configurations that worked well previously
    /// </para>
    /// </remarks>
    public virtual void LoadModel(string filePath)
    {
        byte[] serializedData = File.ReadAllBytes(filePath);
        Deserialize(serializedData);
    }
}
