---
title: "ShapleyValueFitDetector<T, TInput, TOutput>"
description: "A detector that evaluates model fit quality using Shapley values to determine feature importance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A detector that evaluates model fit quality using Shapley values to determine feature importance.

## For Beginners

This class helps you understand which features (input variables) in your model are most 
important for making predictions. It uses something called "Shapley values" from game theory to 
fairly distribute the "credit" for predictions among all your features.

Think of it like figuring out which players on a sports team contributed most to winning a game.
Shapley values help determine if your model is:

- Overfitting: Relying too much on just a few features (like a team depending only on one star player)
- Underfitting: Not using features effectively (like a team not using any player's strengths)
- Good fit: Using features in a balanced way (like a well-coordinated team)

This detector will give you recommendations on how to improve your model based on this analysis.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ShapleyValueFitDetector(ShapleyValueFitDetectorOptions)` | Initializes a new instance of the ShapleyValueFitDetector class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates the confidence level of the fit detection. |
| `CalculatePerformance(ModelEvaluationData<,,>,HashSet<String>)` | Calculates the performance of the model using only a specific subset of features. |
| `CalculateShapleyValues(ModelEvaluationData<,,>,List<String>)` | Calculates Shapley values for each feature in the model. |
| `CreateFeatures(Dictionary<String,Vector<>>)` | Creates a matrix from feature vectors for model prediction. |
| `DetectFit(ModelEvaluationData<,,>)` | Analyzes the model's fit using Shapley values and returns detailed results. |
| `DetermineFitType(Dictionary<String,>)` | Determines the fit type based on calculated Shapley values. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the fit type of the model based on evaluation data. |
| `GenerateRecommendations(FitType,Dictionary<String,>)` | Generates specific recommendations based on the fit type and feature importance. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates recommendations for improving the model based on its fit type. |
| `GetFeatures(ModelEvaluationData<,,>)` | Retrieves the list of feature names from the evaluation data. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the Shapley value fit detector. |
| `_random` | Random number generator used for Monte Carlo sampling. |

