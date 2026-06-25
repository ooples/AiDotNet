---
title: "PartialDependencePlotFitDetector<T, TInput, TOutput>"
description: "A fit detector that uses Partial Dependence Plots to analyze model fit and detect overfitting or underfitting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A fit detector that uses Partial Dependence Plots to analyze model fit and detect overfitting or underfitting.

## For Beginners

A Partial Dependence Plot (PDP) helps you understand how each feature in your data
affects your model's predictions. It shows the relationship between a feature and the predicted outcome
while accounting for the effects of all other features. This detector uses these plots to determine if
your model is learning appropriate patterns from your data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PartialDependencePlotFitDetector(PartialDependencePlotFitDetectorOptions)` | Initializes a new instance of the `PartialDependencePlotFitDetector<T>` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates the confidence level of the fit detection. |
| `CalculateNonlinearity(Vector<>)` | Calculates the nonlinearity of a partial dependence plot. |
| `CalculateNonlinearityScores(Dictionary<String,Vector<>>)` | Calculates nonlinearity scores for each feature based on their partial dependence plots. |
| `CalculatePartialDependencePlot(ModelEvaluationData<,,>,String,Vector<>)` | Calculates a partial dependence plot for a specific feature. |
| `CalculatePartialDependencePlots(ModelEvaluationData<,,>)` | Calculates partial dependence plots for all features in the model. |
| `CreateModifiedDataset(ModelEvaluationData<,,>,String,)` | Creates a modified dataset where a specific feature has the same value across all samples. |
| `DetectFit(ModelEvaluationData<,,>)` | Analyzes a model's fit using Partial Dependence Plots. |
| `DetermineFitType(Dictionary<String,Vector<>>)` | Determines the type of fit based on the calculated partial dependence plots. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the type of fit (overfit, underfit, or good fit) based on model evaluation data. |
| `GenerateRecommendations(FitType,Dictionary<String,Vector<>>)` | Generates detailed recommendations based on the fit type and partial dependence plots. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates recommendations for improving the model based on the detected fit type. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the Partial Dependence Plot fit detector. |

