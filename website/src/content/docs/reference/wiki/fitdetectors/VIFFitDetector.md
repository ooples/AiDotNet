---
title: "VIFFitDetector<T, TInput, TOutput>"
description: "A detector that evaluates model fit by analyzing Variance Inflation Factor (VIF) to identify multicollinearity issues."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A detector that evaluates model fit by analyzing Variance Inflation Factor (VIF) to identify multicollinearity issues.

## For Beginners

This class helps you identify if your model has a problem called "multicollinearity" - 
which happens when your input features (variables) are too closely related to each other.

Think of multicollinearity like this: if you're trying to predict house prices using both 
"square footage" and "number of rooms" as inputs, these two features might be strongly related 
(bigger houses tend to have more rooms). This relationship can confuse your model and make it 
less reliable.

The VIF (Variance Inflation Factor) is like a warning system that measures how much each feature 
is related to other features. Higher VIF values mean stronger relationships:

- VIF = 1: No relationship with other features (ideal)
- VIF = 5-10: Moderate relationship (concerning)
- VIF > 10: Strong relationship (problematic)

This detector helps you identify these issues and suggests ways to fix them.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VIFFitDetector(VIFFitDetectorOptions,ModelStatsOptions)` | Initializes a new instance of the VIFFitDetector class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates the confidence level of the fit detection based on VIF values and performance metrics. |
| `DetectFit(ModelEvaluationData<,,>)` | Analyzes the model's performance data to detect fit issues related to multicollinearity. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the type of fit based on VIF values and performance metrics. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates specific recommendations for improving the model based on the detected fit type. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_modelStatsOptions` | Configuration options for model statistics calculations. |
| `_options` | Configuration options for the VIF fit detector. |

