---
title: "ConfusionMatrixFitDetector<T, TInput, TOutput>"
description: "A fit detector that analyzes confusion matrix metrics to assess classification model fit."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A fit detector that analyzes confusion matrix metrics to assess classification model fit.

## For Beginners

A confusion matrix is a table that summarizes the performance of a classification 
model by showing the counts of true positives, false positives, true negatives, and false negatives. 
This detector uses metrics derived from the confusion matrix to evaluate how well a model is performing.

## How It Works

Unlike other fit detectors that focus on underfitting and overfitting, this detector primarily assesses 
the overall quality of the model's predictions and can identify issues like class imbalance that might 
affect performance.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConfusionMatrixFitDetector(ConfusionMatrixFitDetectorOptions)` | Initializes a new instance of the ConfusionMatrixFitDetector class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates the confidence level of the confusion matrix-based fit detection. |
| `CalculateConfusionMatrix(Vector<>,Vector<>)` | Calculates a confusion matrix from actual and predicted values. |
| `CalculatePrimaryMetric(ConfusionMatrix<>)` | Calculates the primary metric from a confusion matrix. |
| `DetectFit(ModelEvaluationData<,,>)` | Detects the fit type of a model based on confusion matrix analysis. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the fit type based on confusion matrix metrics. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates recommendations based on the detected fit type and confusion matrix analysis. |
| `IsClassImbalanced(ConfusionMatrix<>)` | Determines if the dataset has class imbalance. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the confusion matrix fit detector. |

