---
title: "HeteroscedasticityFitDetector<T, TInput, TOutput>"
description: "A detector that evaluates whether a model's errors have consistent variance across all predictions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A detector that evaluates whether a model's errors have consistent variance across all predictions.

## For Beginners

Heteroscedasticity is a statistical term that means "uneven spread" of errors. 
In a good model, the errors (differences between predictions and actual values) should be 
roughly the same size regardless of what you're predicting. If errors get much larger or smaller 
for certain predictions (like having more accurate predictions for small values but less accurate 
for large values), that's called heteroscedasticity, and it can make your model less reliable.

This detector helps you identify if your model has this problem and suggests ways to fix it.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HeteroscedasticityFitDetector(HeteroscedasticityFitDetectorOptions)` | Initializes a new instance of the `HeteroscedasticityFitDetector<T>` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateBreuschPaganTestStatistic(ModelEvaluationData<,,>)` | Calculates the Breusch-Pagan test statistic to detect heteroscedasticity. |
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates how confident the detector is in its assessment of the model fit. |
| `CalculateWhiteTestStatistic(ModelEvaluationData<,,>)` | Calculates the White test statistic to detect heteroscedasticity. |
| `DetectFit(ModelEvaluationData<,,>)` | Analyzes model performance data to determine if the model has consistent error variance. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the type of fit based on statistical tests for heteroscedasticity. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates specific recommendations based on the detected fit type of the model. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options that control how the detector evaluates heteroscedasticity. |

