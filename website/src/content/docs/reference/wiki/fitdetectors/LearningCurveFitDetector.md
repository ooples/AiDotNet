---
title: "LearningCurveFitDetector<T, TInput, TOutput>"
description: "A detector that evaluates model fit by analyzing learning curves from training and validation data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A detector that evaluates model fit by analyzing learning curves from training and validation data.

## For Beginners

This class helps determine if your machine learning model is a good fit for your data
by looking at "learning curves." Learning curves show how your model's performance improves as it
sees more training examples.

By comparing the trends in training and validation performance, this detector can identify common problems:

- Overfitting: When your model performs great on training data but poorly on new data
- Underfitting: When your model is too simple to capture important patterns in your data
- Good Fit: When your model has learned the underlying patterns without memorizing the training data
- Unstable: When your model's performance is inconsistent or unpredictable

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LearningCurveFitDetector(LearningCurveFitDetectorOptions)` | Initializes a new instance of the Learning Curve fit detector. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates the confidence level of the fit detection. |
| `CalculateSlope(List<>)` | Calculates the slope of a learning curve using linear regression. |
| `CalculateVariance(List<>)` | Calculates the statistical variance of a learning curve. |
| `DetectFit(ModelEvaluationData<,,>)` | Analyzes model performance data to detect the type of fit and provide recommendations. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the type of fit based on learning curve analysis. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the Learning Curve fit detector. |

