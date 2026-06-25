---
title: "PermutationTestFitDetector<T, TInput, TOutput>"
description: "A detector that uses permutation testing to evaluate model fit quality."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A detector that uses permutation testing to evaluate model fit quality.

## For Beginners

This class helps you determine if your machine learning model is performing well
or if it has common problems like overfitting (memorizing data instead of learning patterns) or
underfitting (being too simple to capture important patterns).

It works by using a technique called "permutation testing" which compares your model's actual 
performance against what would happen if the relationship between your inputs and outputs was random.
This gives us confidence that your model has truly learned meaningful patterns.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PermutationTestFitDetector(PermutationTestFitDetectorOptions)` | Initializes a new instance of the PermutationTestFitDetector class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates the confidence level in the fit detection result. |
| `DetectFit(ModelEvaluationData<,,>)` | Analyzes model performance data to determine the quality of fit. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the type of fit (good fit, overfit, underfit, etc.) based on permutation test results. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates specific recommendations for improving the model based on the detected fit type. |
| `PerformPermutationTest(PredictionStats<>)` | Performs a permutation test on the prediction statistics to determine statistical significance. |
| `SimulatePermutedR2(Double)` | Simulates a permuted R2 value by adding random noise to the original R2. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the permutation test detector. |
| `_random` | Random number generator used for permutation simulations. |

