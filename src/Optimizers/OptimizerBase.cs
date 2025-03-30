global using AiDotNet.Models.Inputs;
global using AiDotNet.Evaluation;
global using AiDotNet.Caching;

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
public abstract class OptimizerBase<T> : IOptimizer<T>
{
    /// <summary>
    /// Provides numeric operations for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Contains the configuration options for the optimization algorithm.
    /// </summary>
    protected readonly OptimizationAlgorithmOptions Options;

    /// <summary>
    /// Options for prediction statistics calculations.
    /// </summary>
    protected readonly PredictionStatsOptions _predictionOptions;

    /// <summary>
    /// Options for model statistics calculations.
    /// </summary>
    protected readonly ModelStatsOptions _modelStatsOptions;

    /// <summary>
    /// Evaluates the performance of models.
    /// </summary>
    protected readonly IModelEvaluator<T> _modelEvaluator;

    /// <summary>
    /// Detects the quality of fit for models.
    /// </summary>
    protected readonly IFitDetector<T> _fitDetector;

    /// <summary>
    /// Calculates the fitness score of models.
    /// </summary>
    protected readonly IFitnessCalculator<T> _fitnessCalculator;

    /// <summary>
    /// Stores the fitness scores of evaluated models.
    /// </summary>
    protected readonly List<T> _fitnessList;

    /// <summary>
    /// Stores information about each optimization iteration.
    /// </summary>
    protected readonly List<OptimizationIterationInfo<T>> _iterationHistoryList;

    /// <summary>
    /// Caches evaluated models to avoid redundant calculations.
    /// </summary>
    protected readonly IModelCache<T> ModelCache;

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
    /// Initializes a new instance of the OptimizerBase class.
    /// </summary>
    /// <param name="options">The optimization algorithm options.</param>
    /// <param name="predictionOptions">Options for prediction statistics.</param>
    /// <param name="modelOptions">Options for model statistics.</param>
    /// <param name="modelEvaluator">The model evaluator to use.</param>
    /// <param name="fitDetector">The fit detector to use.</param>
    /// <param name="fitnessCalculator">The fitness calculator to use.</param>
    /// <param name="modelCache">The model cache to use.</param>
    protected OptimizerBase(
        OptimizationAlgorithmOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        Options = options ?? new OptimizationAlgorithmOptions();
        _predictionOptions = predictionOptions ?? new PredictionStatsOptions();
        _modelStatsOptions = modelOptions ?? new ModelStatsOptions();
        _modelEvaluator = modelEvaluator ?? new ModelEvaluator<T>(_predictionOptions);
        _fitDetector = fitDetector ?? new DefaultFitDetector<T>();
        _fitnessCalculator = fitnessCalculator ?? new MeanSquaredErrorFitnessCalculator<T>();
        _fitnessList = [];
        _iterationHistoryList = [];
        ModelCache = modelCache ?? new DefaultModelCache<T>();
        CurrentLearningRate = NumOps.Zero;
        CurrentMomentum = NumOps.Zero;
    }

    /// <summary>
    /// Performs the optimization process.
    /// </summary>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    public abstract OptimizationResult<T> Optimize(OptimizationInputData<T> inputData);

    /// <summary>
    /// Retrieves cached step data for a given solution.
    /// </summary>
    /// <param name="key">The cache key for the solution.</param>
    /// <returns>The cached step data, if available; otherwise, null.</returns>
    protected OptimizationStepData<T>? GetCachedStepData(string key)
    {
        return ModelCache.GetCachedStepData(key);
    }

    /// <summary>
    /// Caches step data for a given solution.
    /// </summary>
    /// <param name="key">The cache key for the solution.</param>
    /// <param name="stepData">The step data to cache.</param>
    protected void CacheStepData(string key, OptimizationStepData<T> stepData)
    {
        ModelCache.CacheStepData(key, stepData);
    }

    /// <summary>
    /// Resets the optimizer state, clearing the model cache.
    /// </summary>
    public virtual void Reset()
    {
        ModelCache.ClearCache();
    }

    /// <summary>
    /// Evaluates a solution, using cached results if available.
    /// </summary>
    /// <param name="solution">The solution to evaluate.</param>
    /// <param name="inputData">The input data for evaluation.</param>
    /// <returns>The evaluation results for the solution.</returns>
    protected virtual OptimizationStepData<T> EvaluateSolution(ISymbolicModel<T> solution, OptimizationInputData<T> inputData)
    {
        string cacheKey = solution.GetHashCode().ToString();
        var cachedStepData = GetCachedStepData(cacheKey);
    
        if (cachedStepData != null)
        {
            return cachedStepData;
        }

        var stepData = PrepareAndEvaluateSolution(solution, inputData);
        CacheStepData(cacheKey, stepData);

        return stepData;
    }

    /// <summary>
    /// Prepares and evaluates a solution, creating subsets of data based on selected features.
    /// </summary>
    /// <param name="solution">The solution to evaluate.</param>
    /// <param name="inputData">The input data for evaluation.</param>
    /// <returns>The evaluation results for the solution.</returns>
    protected OptimizationStepData<T> PrepareAndEvaluateSolution(ISymbolicModel<T> solution, OptimizationInputData<T> inputData)
    {
        string cacheKey = solution.GetHashCode().ToString();
        var cachedStepData = GetCachedStepData(cacheKey);
    
        if (cachedStepData != null)
        {
            return cachedStepData;
        }

        var selectedFeatures = inputData.XTrain.GetColumnVectors(OptimizerHelper<T>.GetSelectedFeatures(solution));
        var XTrainSubset = OptimizerHelper<T>.SelectFeatures(inputData.XTrain, selectedFeatures);
        var XValSubset = OptimizerHelper<T>.SelectFeatures(inputData.XVal, selectedFeatures);
        var XTestSubset = OptimizerHelper<T>.SelectFeatures(inputData.XTest, selectedFeatures);

        var input = new ModelEvaluationInput<T>
        {
            InputData = inputData
        };

        var (currentFitnessScore, fitDetectionResult, evaluationData) = EvaluateSolution(input);
        _fitnessList.Add(currentFitnessScore);

        var stepData = new OptimizationStepData<T>
        {
            Solution = solution,
            SelectedFeatures = selectedFeatures,
            XTrainSubset = XTrainSubset,
            XValSubset = XValSubset,
            XTestSubset = XTestSubset,
            FitnessScore = currentFitnessScore,
            FitDetectionResult = fitDetectionResult,
            EvaluationData = evaluationData
        };

        CacheStepData(cacheKey, stepData);

        return stepData;
    }

    /// <summary>
    /// Evaluates a solution using the model evaluator, fit detector, and fitness calculator.
    /// </summary>
    /// <param name="input">The input data for model evaluation.</param>
    /// <returns>A tuple containing the fitness score, fit detection result, and evaluation data.</returns>
    private (T CurrentFitnessScore, FitDetectorResult<T> FitDetectionResult, ModelEvaluationData<T> EvaluationData)
    EvaluateSolution(ModelEvaluationInput<T> input)
    {
        var evaluationData = _modelEvaluator.EvaluateModel(input);
        var fitDetectionResult = _fitDetector.DetectFit(evaluationData);
        var currentFitnessScore = _fitnessCalculator.CalculateFitnessScore(evaluationData);

        return (currentFitnessScore, fitDetectionResult, evaluationData);
    }

    /// <summary>
    /// Calculates the loss for a given solution.
    /// </summary>
    /// <param name="solution">The solution to evaluate.</param>
    /// <param name="inputData">The input data for evaluation.</param>
    /// <returns>The calculated loss value.</returns>
    protected virtual T CalculateLoss(ISymbolicModel<T> solution, OptimizationInputData<T> inputData)
    {
        var stepData = EvaluateSolution(solution, inputData);
        return _fitnessCalculator!.CalculateFitnessScore(stepData.EvaluationData);
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
    protected OptimizationResult<T> CreateOptimizationResult(OptimizationStepData<T> bestStepData, OptimizationInputData<T> input)
    {
        return OptimizerHelper<T>.CreateOptimizationResult(
            bestStepData.Solution,
            bestStepData.FitnessScore,
            _fitnessList,
            bestStepData.SelectedFeatures,
            new OptimizationResult<T>.DatasetResult
            {
                X = bestStepData.XTrainSubset,
                Y = input.YTrain,
                Predictions = bestStepData.EvaluationData.TrainingSet.Predicted,
                ErrorStats = bestStepData.EvaluationData.TrainingSet.ErrorStats,
                ActualBasicStats = bestStepData.EvaluationData.TrainingSet.ActualBasicStats,
                PredictedBasicStats = bestStepData.EvaluationData.TrainingSet.PredictedBasicStats,
                PredictionStats = bestStepData.EvaluationData.TrainingSet.PredictionStats
            },
            new OptimizationResult<T>.DatasetResult
            {
                X = bestStepData.XValSubset,
                Y = input.YVal,
                Predictions = bestStepData.EvaluationData.ValidationSet.Predicted,
                ErrorStats = bestStepData.EvaluationData.ValidationSet.ErrorStats,
                ActualBasicStats = bestStepData.EvaluationData.ValidationSet.ActualBasicStats,
                PredictedBasicStats = bestStepData.EvaluationData.ValidationSet.PredictedBasicStats,
                PredictionStats = bestStepData.EvaluationData.ValidationSet.PredictionStats
            },
            new OptimizationResult<T>.DatasetResult
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
            _iterationHistoryList.Count
        );
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
    private void UpdateAndApplyBestSolution(ModelResult<T> currentResult, ref ModelResult<T> bestResult)
    {
        if (_fitnessCalculator.IsBetterFitness(currentResult.Fitness, bestResult.Fitness))
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
    protected void UpdateBestSolution(OptimizationStepData<T> currentStepData, ref OptimizationStepData<T> bestStepData)
    {
        var currentResult = new ModelResult<T>
        {
            Solution = currentStepData.Solution ?? new VectorModel<T>(Vector<T>.Empty()),
            Fitness = currentStepData.FitnessScore,
            FitDetectionResult = currentStepData.FitDetectionResult,
            EvaluationData = currentStepData.EvaluationData,
            SelectedFeatures = currentStepData.SelectedFeatures
        };

        var bestResult = new ModelResult<T>
        {
            Solution = bestStepData.Solution ?? new VectorModel<T>(Vector<T>.Empty()),
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
    protected virtual void UpdateAdaptiveParameters(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
    {
        if (Options.UseAdaptiveLearningRate)
        {
            if (NumOps.GreaterThan(currentStepData.FitnessScore, previousStepData.FitnessScore))
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
    protected bool UpdateIterationHistoryAndCheckEarlyStopping(int iteration, OptimizationStepData<T> stepData)
    {
        _iterationHistoryList.Add(new OptimizationIterationInfo<T>
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
    /// Determines whether the optimization process should terminate early based on progress metrics.
    /// </summary>
    /// <returns>True if optimization should stop early, false if it should continue.</returns>
    /// <remarks>
    /// <para>
    /// This method evaluates whether the optimization process should terminate early by analyzing recent performance.
    /// It checks for lack of improvement over a specified number of iterations (patience) and for consecutive
    /// bad fits. Early stopping helps prevent overfitting and saves computation time.
    /// </para>
    /// <para><b>For Beginners:</b> This method decides if it's worth continuing the optimization or if it's time to stop.
    /// 
    /// It's like deciding whether to keep shopping for a better price:
    /// - If you haven't found a better price after visiting several stores (patience), you might decide to stop
    /// - If the last few stores had poor quality items (bad fits), you might also decide to stop
    /// 
    /// The method looks at:
    /// - Whether there's been any improvement in recent iterations
    /// - Whether recent solutions have been "good fits" or not
    /// - How many iterations to check before making a decision (patience)
    /// 
    /// Early stopping is a way to be efficient with computing resources and avoid overfitting.
    /// </para>
    /// </remarks>
    public virtual bool ShouldEarlyStop()
    {
        if (_iterationHistoryList.Count < Options.EarlyStoppingPatience)
        {
            return false;
        }

        var recentIterations = _iterationHistoryList.Skip(Math.Max(0, _iterationHistoryList.Count - Options.EarlyStoppingPatience)).ToList();

        // Find the best fitness score
        T bestFitness = _iterationHistoryList[0].Fitness;
        foreach (var iteration in _iterationHistoryList)
        {
            if (_fitnessCalculator.IsBetterFitness(iteration.Fitness, bestFitness))
            {
                bestFitness = iteration.Fitness;
            }
        }

        // Check for improvement in recent iterations
        bool noImprovement = true;
        foreach (var iteration in recentIterations)
        {
            if (_fitnessCalculator.IsBetterFitness(iteration.Fitness, bestFitness))
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
    /// Creates a random initial solution for the optimization process.
    /// </summary>
    /// <param name="numberOfFeatures">The number of features in the input data.</param>
    /// <returns>A randomly initialized symbolic model.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a random starting point for the optimization process by generating a symbolic model
    /// with random parameters. The model can be either an expression tree or another symbolic representation,
    /// depending on the algorithm's configuration.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a random starting point for the optimization.
    /// 
    /// Think of it like starting a treasure hunt:
    /// - You need a place to begin your search
    /// - This method gives you a random starting location
    /// - From there, the optimizer will try to find better and better solutions
    /// 
    /// Starting with a random solution is important because:
    /// - It gives the optimizer a place to begin exploring
    /// - Different random starts can help find different good solutions
    /// - It prevents bias that might come from always starting in the same place
    /// </para>
    /// </remarks>
    protected ISymbolicModel<T> InitializeRandomSolution(int numberOfFeatures)
    {
        return SymbolicModelFactory<T>.CreateRandomModel(Options.UseExpressionTrees, numberOfFeatures + 1);
    }

    /// <summary>
    /// Serializes the optimizer to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized optimizer.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the optimizer's state into a byte array that can be stored or transmitted.
    /// It includes the type information, configuration options, and any additional data specific to
    /// derived optimizer classes.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the current state of the optimizer to a format that can be stored.
    /// 
    /// It's like taking a snapshot of the optimizer:
    /// - It captures all the important settings and values
    /// - It converts them into a format (byte array) that can be saved to disk or sent over a network
    /// - It includes information about what type of optimizer it is
    /// 
    /// This is useful for:
    /// - Saving your progress to continue later
    /// - Sharing a trained optimizer with others
    /// - Creating backups of your work
    /// </para>
    /// </remarks>
    public virtual byte[] Serialize()
    {
        using MemoryStream ms = new();
        using (BinaryWriter writer = new(ms))
        {
            // Write the type of the optimizer
            writer.Write(this.GetType()?.AssemblyQualifiedName ?? string.Empty);

            // Serialize options
           string optionsJson = JsonConvert.SerializeObject(Options);
            writer.Write(optionsJson);

            // Allow derived classes to serialize additional data
            SerializeAdditionalData(writer);
        }

        return ms.ToArray();
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
        var options = JsonConvert.DeserializeObject<OptimizationAlgorithmOptions>(optionsJson);

        // Update the options
        UpdateOptions(options ?? new());

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
    protected abstract void UpdateOptions(OptimizationAlgorithmOptions options);

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
    public abstract OptimizationAlgorithmOptions GetOptions();
}