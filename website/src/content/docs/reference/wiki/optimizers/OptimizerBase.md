---
title: "OptimizerBase<T, TInput, TOutput>"
description: "Represents the base class for all optimization algorithms, providing common functionality and interfaces."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Optimizers`

Represents the base class for all optimization algorithms, providing common functionality and interfaces.

## For Beginners

This is the blueprint that all optimization algorithms follow.

Think of OptimizerBase as the common foundation that all optimizers are built upon:

- It defines what every optimizer must be able to do (evaluate solutions, manage caching)
- It provides shared tools that all optimizers can use (like adaptive learning rates and early stopping)
- It manages the evaluation of solutions and tracks the optimization progress
- It handles saving and loading optimizer states

All specific optimizer types (like genetic algorithms, particle swarm, etc.) inherit from this class,
which ensures they all work together consistently in the optimization process.

## How It Works

OptimizerBase is an abstract class that serves as the foundation for all optimization algorithms. It defines 
the common structure and functionality that all optimizers must implement, such as solution evaluation, 
caching, and adaptive parameter management. This class handles the core mechanics of optimization processes, 
allowing derived classes to focus on their specific optimization strategies.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OptimizerBase(IFullModel<,,>,OptimizationAlgorithmOptions<,,>)` | Initializes a new instance of the OptimizerBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Gets the global execution engine for vector operations. |
| `EvaluationBatchSize` | Per-axis-0 chunk size used when the optimizer's evaluator path (`PredictionType)` / `PredictionType)`) routes its `model.Predict(X)` call through `Int32)`. |
| `Model` | Gets the model that this optimizer is configured to optimize. |
| `SkipTrainingInEvaluation` | When true, TrainAndEvaluateSolution skips model.Train() during evaluation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdjustModelParameters(IFullModel<,,>,Double,Double)` | Adjusts the parameters (weights) of a model. |
| `AdjustParameters(Vector<>,Double,Double)` | Adjusts a vector of parameters by applying random modifications. |
| `ApplyFeatureSelection(IFullModel<,,>,Int32)` | Applies feature selection to a model. |
| `ApplyFeatureSelection(IFullModel<,,>,List<Int32>)` | Applies the selected features to a model. |
| `BuildDataDerivedRandomParameters(,Int64)` | Computes data-derived random parameter bounds and samples a vector of length `parameterCount` uniformly in those bounds. |
| `CacheStepData(String,OptimizationStepData<,,>)` | Caches step data for a given solution. |
| `CalculateLoss(IFullModel<,,>,OptimizationInputData<,,>)` | Calculates the loss for a given solution. |
| `CalculateR2OnlyStatsForOptimizer(IFullModel<,,>,,,PredictionType)` | Lightweight stats helper: computes R² only (the metric FitDetector reads) for a dataset, skipping the expensive PredictionStats / ErrorStats / BasicStats construction. |
| `CalculateUpdate(Dictionary<String,Vector<>>)` | Calculates the parameter update based on the provided gradients. |
| `CalculateUpdate(Vector<>,Vector<>)` | Calculates the parameter updates based on the gradients. |
| `CreateOptimizationResult(OptimizationStepData<,,>,OptimizationInputData<,,>)` | Creates a new optimization result based on the best step data found during optimization. |
| `CreateSolution()` | Creates a potential solution based on the optimization mode. |
| `Deserialize(Byte[])` | Reconstructs the optimizer from a serialized byte array. |
| `DeserializeAdditionalData(BinaryReader)` | Deserializes additional data specific to derived optimizer classes. |
| `EvaluateModelDirectly(ModelEvaluationInput<,,>)` | Directly evaluates a model without external evaluator dependency. |
| `EvaluateSolution(IFullModel<,,>,OptimizationInputData<,,>)` | Evaluates a solution, using cached results if available. |
| `GenerateCacheKey(IFullModel<,,>,OptimizationInputData<,,>)` | Generates a cache key for the given solution and input data. |
| `GetCachedStepData(String)` | Retrieves cached step data for a given solution. |
| `GetDynamicShapeInfo` |  |
| `GetInputShape` | Saves the optimizer state to a file. |
| `GetOptions` | Gets the current options for this optimizer. |
| `GetOutputShape` |  |
| `HasNonFlatNeuralInput(IFullModel<,,>)` | Determines whether `model` takes a non-flat (multi-dimensional) input — i.e. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters used during optimization to their starting values. |
| `InitializeRandomSolution()` | Legacy entry point. |
| `InitializeRandomSolution(Vector<>,Vector<>)` | Initializes a random solution within the given bounds. |
| `IsEmbeddingBasedModel(IFullModel<,,>)` | Checks whether a model uses embedding-based input (Transformers, RNNs with token input, etc.) and therefore does not support feature selection. |
| `LoadModel(String)` | Loads the optimizer state from a file. |
| `OnInitialTrainingCompleted` | Called after the first TrainAndEvaluateSolution completes successfully. |
| `OnModelChanged(IFullModel<,,>,IFullModel<,,>)` | Called whenever the optimizer's model is changed via `IFullModel{`. |
| `Optimize(OptimizationInputData<,,>)` | Performs the optimization process. |
| `PredictForEvaluation(IFullModel<,,>,)` | Routes the evaluator's `model.Predict(X)` through `Int32)` at the optimizer's `EvaluationBatchSize` chunk size, so the per-epoch full-dataset forward inside `EvaluateModelDirectly` is bounded for neural-network models while staying identica… |
| `PrepareAndEvaluateSolution(IFullModel<,,>,OptimizationInputData<,,>)` | Prepares and evaluates a solution, applying feature selection before checking the cache. |
| `RandomlySelectFeatures(Int32,Nullable<Int32>,Nullable<Int32>)` | Randomly selects a subset of features to use in a model. |
| `RequireModel` | Returns the current model or throws if none has been set via `IFullModel{`. |
| `Reset` | Resets the optimizer state, clearing the model cache. |
| `ResetAdaptiveParameters` | Resets the adaptive parameters back to their initial values. |
| `Serialize` | Serializes the optimizer state to a byte array. |
| `SerializeAdditionalData(BinaryWriter)` | Serializes additional data specific to derived optimizer classes. |
| `SetModel(IFullModel<,,>)` |  |
| `ShouldEarlyStop` | Determines whether the optimization process should stop early based on the recent history of fitness scores. |
| `SpawnIndividual()` | Spawns a fresh independent solution that population optimizers (PSO / Bayesian / CMAES / GA / DE / SA / AntColony / …) add to their swarm or population. |
| `Step` | Performs a single optimization step, updating the model parameters based on gradients. |
| `TrainAndEvaluateSolution(ModelEvaluationInput<,,>,Boolean)` | Evaluates a solution using the fit detector and fitness calculator. |
| `UpdateAdaptiveParameters(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates the adaptive parameters based on the progress of optimization. |
| `UpdateAndApplyBestSolution(ModelResult<,,>,ModelResult<,,>)` | Compares the current model result with the best result found so far and updates the best result if the current one is better. |
| `UpdateBestSolution(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates the best step data if the current step data has a better solution. |
| `UpdateIterationHistoryAndCheckEarlyStopping(Int32,OptimizationStepData<,,>)` | Updates the iteration history with the current step data and checks if early stopping should be applied. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the optimizer's options with the provided options. |

## Fields

| Field | Summary |
|:-----|:--------|
| `CurrentLearningRate` | The current learning rate used in the optimization process. |
| `CurrentMomentum` | The current momentum used in the optimization process. |
| `FitDetector` | Detects the quality of fit for models. |
| `FitnessCalculator` | Calculates the fitness score of models. |
| `FitnessList` | Stores the fitness scores of evaluated models. |
| `IterationHistoryList` | Stores information about each optimization iteration. |
| `IterationsWithImprovement` | Counts the number of consecutive iterations with improvement. |
| `IterationsWithoutImprovement` | Counts the number of consecutive iterations without improvement. |
| `ModelCache` | Caches evaluated models to avoid redundant calculations. |
| `ModelStatsOptions` | Options for model statistics calculations. |
| `NumOps` | Provides numeric operations for type T. |
| `Options` | Contains the configuration options for the optimization algorithm. |
| `PredictionOptions` | Options for prediction statistics calculations. |
| `Random` | Provides random number generation for all derived classes. |
| `_model` | The model that this optimizer is configured to optimize. |

