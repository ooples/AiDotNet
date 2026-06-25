---
title: "AutoMLModelBase<T, TInput, TOutput>"
description: "Base class for AutoML models that automatically search for optimal model configurations"
section: "API Reference"
---

`Base Classes` · `AiDotNet.AutoML`

Base class for AutoML models that automatically search for optimal model configurations

## Properties

| Property | Summary |
|:-----|:--------|
| `BestModel` | Gets the best model found so far |
| `BestScore` | Gets the best score achieved |
| `DefaultLossFunction` | Gets the default loss function for gradient computation. |
| `Engine` | Hardware-accelerated engine for vector/tensor operations. |
| `FeatureNames` | Gets the feature names |
| `MaximizeOptimizationMetric` | Gets a value indicating whether higher metric values are better. |
| `OptimizationMetric` | Gets the optimization metric used to rank trials. |
| `ParameterCount` | Gets the number of parameters |
| `Status` | Gets the model type |
| `SupportsParameterInitialization` |  |
| `TimeLimit` | Gets or sets the time limit for the AutoML search |
| `TrialLimit` | Gets or sets the maximum number of trials to run |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Applies gradients by delegating to the best model. |
| `Clone` | Creates a memberwise clone of the AutoML model using MemberwiseClone(). |
| `ComputeGradients(,,ILossFunction<>)` | Computes gradients by delegating to the best model. |
| `ConfigureSearchSpace(Dictionary<String,ParameterRange>)` | Configures the search space for hyperparameter optimization |
| `CreateInstanceForCopy` | Factory method for creating a new instance for deep copy. |
| `CreateModelAsync(Type,Dictionary<String,Object>)` | Creates a model instance for the given type and parameters |
| `CreateModelWithHookAsync(Type,Dictionary<String,Object>)` | Calls `Object})` and then fires `OnCandidateCreated` for the resulting candidate. |
| `DeepCopy` | Creates a deep copy of the AutoML model |
| `Deserialize(Byte[])` | Deserializes the model from bytes |
| `Dispose` |  |
| `Dispose(Boolean)` | Releases resources held by this AutoML model. |
| `EnableEarlyStopping(Int32,Double)` | Enables early stopping |
| `EnableNAS(Boolean)` | Enables Neural Architecture Search (NAS) for automatic network design |
| `EvaluateModelAsync(IFullModel<,,>,,)` | Evaluates a model on the validation set |
| `EvaluateModelDirectly(IFullModel<,,>,,)` | Directly evaluates a model without external evaluator dependency. |
| `ExtractMetricFromEvaluation(ModelEvaluationData<,,>)` | Extracts the appropriate metric value from the evaluation results |
| `GetActiveFeatureIndices` | Gets the indices of active features |
| `GetDefaultSearchSpace(Type)` | Gets the default search space for a model type |
| `GetDynamicShapeInfo` |  |
| `GetFeatureImportance` | Gets the feature importance scores |
| `GetFeatureImportanceAsync` | Gets feature importance from the best model |
| `GetInputShape` | Saves the model to a file |
| `GetModelMetadata` | Gets model metadata |
| `GetOutputShape` |  |
| `GetParameters` | Gets the model parameters |
| `GetResults` | Gets the results of all trials performed during search |
| `GetTrialHistory` | Gets the history of all trials |
| `IsFeatureUsed(Int32)` | Checks if a feature is used |
| `LoadModel(String)` | Loads the model from a file |
| `LoadState(Stream)` | Loads the AutoML model's state from a stream. |
| `Predict()` | Makes predictions using the best model found |
| `Predict(Double[][])` | Makes predictions using the best model (legacy method) |
| `ReportTrialFailureAsync(Dictionary<String,Object>,Exception,TimeSpan)` | Reports a failed trial result without terminating the full AutoML run. |
| `ReportTrialResultAsync(Dictionary<String,Object>,Double,TimeSpan)` | Reports the result of a trial |
| `Run(,,,)` | Runs the AutoML optimization process (alternative name for Search) |
| `SanitizeParameters(Vector<>)` |  |
| `SaveState(Stream)` | Saves the AutoML model's current state to a stream. |
| `Search(,,,)` | Performs the AutoML search process (synchronous version) |
| `SearchAsync(,,,,TimeSpan,CancellationToken)` | Searches for the best model configuration |
| `SearchBestModel(,,,)` | Searches for the best model configuration (synchronous version) |
| `Serialize` | Serializes the model to bytes |
| `SetActiveFeatureIndices(IEnumerable<Int32>)` | Sets the active feature indices |
| `SetCandidateModels(List<Type>)` | Sets the models to consider in the search |
| `SetConstraints(List<SearchConstraint>)` | Sets constraints for the search |
| `SetModelsToTry(List<Type>)` | Sets which model types should be considered during the search |
| `SetOptimizationMetric(MetricType,Boolean)` | Sets the optimization metric |
| `SetParameters(Vector<>)` | Sets the model parameters |
| `SetSearchSpace(Dictionary<String,ParameterRange>)` | Sets the search space for hyperparameters |
| `SetTimeLimit(TimeSpan)` | Sets the time limit for the AutoML search process |
| `SetTrialLimit(Int32)` | Sets the maximum number of trials to execute during search |
| `ShouldStop` | Checks if early stopping criteria is met |
| `SuggestNextTrialAsync` | Suggests the next hyperparameters to try |
| `Train(,)` | Trains the AutoML model by searching for the best configuration |
| `Train(Double[][],Double[])` | Trains the model (legacy method - use SearchAsync instead) |
| `ValidateConstraints(Dictionary<String,Object>,IFullModel<,,>)` | Validates constraints for a given configuration |
| `WithParameters(Vector<>)` | Creates a new instance with the given parameters |

## Fields

| Field | Summary |
|:-----|:--------|
| `ModelTypeKey` | Standard key used in trial parameter dictionaries to store the model `Type`. |

## Events

| Event | Summary |
|:-----|:--------|
| `OnCandidateCreated` | Fired immediately after each candidate model is instantiated, before it is trained or evaluated. |

