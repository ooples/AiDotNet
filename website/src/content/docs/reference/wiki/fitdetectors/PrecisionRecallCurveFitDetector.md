---
title: "PrecisionRecallCurveFitDetector<T, TInput, TOutput>"
description: "A detector that evaluates model fit quality using precision-recall curve metrics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A detector that evaluates model fit quality using precision-recall curve metrics.

## For Beginners

This class helps you understand how well your classification model is performing
by analyzing two important metrics:

1. AUC (Area Under the Curve): A number between 0 and 1 that tells you how well your model can

distinguish between classes. Higher is better, with 1.0 being perfect.

2. F1 Score: A number between 0 and 1 that balances precision (how many of your positive 

predictions were correct) and recall (how many actual positives your model found).
Higher is better, with 1.0 being perfect.

This detector is especially useful for classification problems where you're trying to identify
specific categories or classes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PrecisionRecallCurveFitDetector(PrecisionRecallCurveFitDetectorOptions)` | Initializes a new instance of the PrecisionRecallCurveFitDetector class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Auc` | Gets or sets the Area Under the Curve value for the precision-recall curve. |
| `F1Score` | Gets or sets the F1 Score, which is the harmonic mean of precision and recall. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates the confidence level in the fit detection result. |
| `DetectFit(ModelEvaluationData<,,>)` | Analyzes the model's performance data and determines the quality of fit. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the type of fit based on AUC and F1 Score values. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates specific recommendations for improving the model based on the detected fit type. |

