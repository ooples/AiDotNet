---
title: "ResidualAnalysisFitDetector<T, TInput, TOutput>"
description: "A detector that evaluates model fit quality by analyzing the residuals (errors) of the model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A detector that evaluates model fit quality by analyzing the residuals (errors) of the model.

## For Beginners

This class helps you understand how well your model is performing by looking at 
the "residuals" - the differences between what your model predicted and the actual values.

Think of residuals like the errors your model makes. By analyzing these errors in different ways,
we can tell if your model:

- Is generally accurate (good fit)
- Consistently makes errors in the same direction (bias)
- Makes wildly different errors each time (high variance)
- Works well on training data but poorly on new data (overfitting)
- Doesn't capture the complexity of your data (underfitting)

This detector examines these patterns to give you a clear picture of your model's performance
and how to improve it.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ResidualAnalysisFitDetector(ResidualAnalysisFitDetectorOptions)` | Initializes a new instance of the ResidualAnalysisFitDetector class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates how confident the detector is in its assessment of the model's fit type. |
| `DetectFit(ModelEvaluationData<,,>)` | Analyzes the model's performance data and determines the quality of fit based on residual analysis. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the type of fit by analyzing residual patterns and statistical measures. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the residual analysis detector. |

