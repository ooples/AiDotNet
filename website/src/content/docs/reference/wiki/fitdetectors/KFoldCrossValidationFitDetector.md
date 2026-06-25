---
title: "KFoldCrossValidationFitDetector<T, TInput, TOutput>"
description: "A detector that evaluates model fit using K-Fold Cross-Validation technique."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A detector that evaluates model fit using K-Fold Cross-Validation technique.

## For Beginners

This class helps determine if your machine learning model is a good fit for your data.
It uses a technique called "K-Fold Cross-Validation" which:

1. Splits your data into K equal parts (or "folds")
2. Trains the model K times, each time using a different fold as a validation set
3. Analyzes how consistently your model performs across these different splits

This approach helps identify common problems like:

- Overfitting: When your model performs great on training data but poorly on new data (it "memorized" instead of "learned")
- Underfitting: When your model is too simple to capture important patterns in your data
- High Variance: When your model's performance changes dramatically with different data splits
- Instability: When your model doesn't consistently perform well across different data arrangements

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KFoldCrossValidationFitDetector(KFoldCrossValidationFitDetectorOptions)` | Initializes a new instance of the K-Fold Cross-Validation fit detector. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates the confidence level of the fit detection. |
| `DetectFit(ModelEvaluationData<,,>)` | Analyzes model performance data to detect the type of fit and provide recommendations. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the type of fit based on model performance metrics. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates specific recommendations based on the detected fit type. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the K-Fold Cross-Validation fit detector. |

