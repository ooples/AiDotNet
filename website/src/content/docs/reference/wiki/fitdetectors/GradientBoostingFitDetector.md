---
title: "GradientBoostingFitDetector<T, TInput, TOutput>"
description: "A specialized detector that evaluates how well a gradient boosting model fits the data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A specialized detector that evaluates how well a gradient boosting model fits the data.

## For Beginners

Gradient Boosting is a machine learning technique that builds multiple simple models 
(usually decision trees) sequentially, with each new model trying to correct the errors made by 
previous models. This detector helps you understand if your gradient boosting model is:

- Learning the data well (good fit)
- Not learning enough (underfit)
- Learning too much from the training data and not generalizing well (overfit)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GradientBoostingFitDetector(GradientBoostingFitDetectorOptions)` | Initializes a new instance of the `GradientBoostingFitDetector<T>` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates a confidence level for the fit assessment based on the relative difference between training and validation errors. |
| `DetectFit(ModelEvaluationData<,,>)` | Analyzes model performance data to determine how well the model fits the data. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the type of fit based on the difference between training and validation errors. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates specific recommendations for improving the model based on the detected fit type. |
| `GetPerformanceMetrics(ModelEvaluationData<,,>)` | Extracts key performance metrics from the evaluation data. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options that control how the detector evaluates model fit. |

