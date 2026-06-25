---
title: "CalibratedProbabilityFitDetector<T, TInput, TOutput>"
description: "A fit detector that analyzes the calibration of probability predictions to assess model fit."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A fit detector that analyzes the calibration of probability predictions to assess model fit.

## For Beginners

Probability calibration refers to how well a model's predicted probabilities 
match the actual frequencies of events. A well-calibrated model should predict probabilities that 
match the true likelihood of events.

## How It Works

For example, if a model predicts a 70% probability for 100 different samples, approximately 70 of 
those samples should actually belong to the positive class. This detector analyzes how well your 
model's probability predictions match the actual outcomes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CalibratedProbabilityFitDetector(CalibratedProbabilityFitDetectorOptions)` | Initializes a new instance of the CalibratedProbabilityFitDetector class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateCalibration(ModelEvaluationData<,,>)` | Calculates the expected and observed calibration values. |
| `CalculateCalibrationError(Vector<>,Vector<>)` | Calculates the calibration error between expected and observed calibration values. |
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates the confidence level of the calibration-based fit detection. |
| `DetectFit(ModelEvaluationData<,,>)` | Detects the fit type of a model based on probability calibration analysis. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the fit type based on probability calibration analysis. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates recommendations based on the detected fit type. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the calibrated probability fit detector. |

