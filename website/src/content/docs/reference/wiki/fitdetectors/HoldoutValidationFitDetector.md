---
title: "HoldoutValidationFitDetector<T, TInput, TOutput>"
description: "A detector that evaluates model fit quality using holdout validation techniques."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A detector that evaluates model fit quality using holdout validation techniques.

## For Beginners

This class helps you determine if your machine learning model is performing well
by comparing how it performs on different subsets of your data:

- Training data: The data used to build the model
- Validation data: A separate set of data used to tune the model
- Test data: A final set of data used to evaluate the model's performance

By comparing performance across these sets, the detector can identify common problems like:

- Overfitting: When your model performs very well on training data but poorly on new data
- Underfitting: When your model performs poorly on all data sets
- High variance: When your model's performance varies significantly between different data sets

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HoldoutValidationFitDetector(HoldoutValidationFitDetectorOptions)` | Initializes a new instance of the `HoldoutValidationFitDetector<T>` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates the confidence level in the fit type determination. |
| `DetectFit(ModelEvaluationData<,,>)` | Analyzes model performance data to detect the quality of fit. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the type of fit based on model performance metrics. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates practical recommendations for improving the model based on its fit type. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the holdout validation fit detector. |

