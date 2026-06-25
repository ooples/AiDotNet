---
title: "TimeSeriesCrossValidationFitDetector<T, TInput, TOutput>"
description: "A specialized detector that evaluates how well a model fits time series data using cross-validation techniques."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A specialized detector that evaluates how well a model fits time series data using cross-validation techniques.

## For Beginners

This class helps you understand if your time series model is learning correctly from your data.

Time series data is information collected over time in sequence (like daily temperatures, monthly sales, 
or hourly website traffic). When working with time series data, we need special techniques because:

1. The order of data matters (unlike regular data where order doesn't matter)
2. Recent data is often more important than older data
3. There might be patterns that repeat over time (like seasonal patterns)

This detector analyzes your model's performance and tells you if it's:

- Learning too much detail from your data (overfitting)
- Not learning enough patterns (underfitting)
- Performing inconsistently (high variance)
- Working well (good fit)
- Behaving unpredictably (unstable)

It also provides specific recommendations to improve your model based on these findings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeSeriesCrossValidationFitDetector(TimeSeriesCrossValidationFitDetectorOptions)` | Initializes a new instance of the TimeSeriesCrossValidationFitDetector class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates how confident the detector is in its fit type determination. |
| `DetectFit(ModelEvaluationData<,,>)` | Analyzes model performance data and determines how well the model fits the time series data. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the type of fit for the model based on error metrics and thresholds. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates specific recommendations for improving the model based on the detected fit type. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the time series cross-validation fit detector. |

