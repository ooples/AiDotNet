---
title: "IAutoMLModel<T, TInput, TOutput>"
description: "Defines the contract for AutoML models that automatically search for optimal model configurations."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for AutoML models that automatically search for optimal model configurations.

## How It Works

AutoML (Automated Machine Learning) models automatically search through different model types,
hyperparameters, and architectures to find the best configuration for a given dataset.
This interface extends IFullModel to provide AutoML-specific functionality like search space configuration,
trial management, and optimization settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `BestModel` | Gets the best model found so far |
| `BestScore` | Gets the best score achieved |
| `Status` | Gets the current optimization status |
| `TimeLimit` | Gets or sets the time limit for the AutoML search |
| `TrialLimit` | Gets or sets the maximum number of trials to run |

## Methods

| Method | Summary |
|:-----|:--------|
| `ConfigureSearchSpace(Dictionary<String,ParameterRange>)` | Configures the search space for hyperparameter optimization |
| `EnableEarlyStopping(Int32,Double)` | Enables early stopping |
| `EnableNAS(Boolean)` | Enables Neural Architecture Search (NAS) for automatic network design |
| `GetFeatureImportanceAsync` | Gets feature importance from the best model |
| `GetResults` | Gets the results of all trials performed during search |
| `GetTrialHistory` | Gets the history of all trials |
| `ReportTrialResultAsync(Dictionary<String,Object>,Double,TimeSpan)` | Reports the result of a trial |
| `Run(,,,)` | Runs the AutoML optimization process (alternative name for Search) |
| `Search(,,,)` | Performs the AutoML search process (synchronous version) |
| `SearchAsync(,,,,TimeSpan,CancellationToken)` | Searches for the best model configuration asynchronously |
| `SearchBestModel(,,,)` | Searches for the best model configuration (synchronous version) |
| `SetCandidateModels(List<Type>)` | Sets the models to consider in the search |
| `SetConstraints(List<SearchConstraint>)` | Sets constraints for the search |
| `SetModelsToTry(List<Type>)` | Sets which model types should be considered during the search |
| `SetOptimizationMetric(MetricType,Boolean)` | Sets the optimization metric |
| `SetSearchSpace(Dictionary<String,ParameterRange>)` | Sets the search space for hyperparameters |
| `SetTimeLimit(TimeSpan)` | Sets the time limit for the AutoML search process |
| `SetTrialLimit(Int32)` | Sets the maximum number of trials to execute during search |
| `SuggestNextTrialAsync` | Suggests the next hyperparameters to try |

## Events

| Event | Summary |
|:-----|:--------|
| `OnCandidateCreated` | Fired immediately after each candidate model is instantiated, before it is trained or evaluated. |

