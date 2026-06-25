---
title: "ROCCurveFitDetector<T, TInput, TOutput>"
description: "A detector that evaluates model fit quality using ROC (Receiver Operating Characteristic) curve analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A detector that evaluates model fit quality using ROC (Receiver Operating Characteristic) curve analysis.

## For Beginners

This class helps determine how well your classification model performs.
It uses something called an "ROC curve" which is a way to visualize how good your model is at 
distinguishing between positive cases (like "yes, this email is spam") and negative cases 
(like "no, this email is not spam").

The key metric used is called "AUC" (Area Under the Curve), which gives a single number between 0 and 1:

- AUC near 1.0: Your model is excellent at classification
- AUC near 0.5: Your model is no better than random guessing
- AUC near 0.0: Your model is consistently wrong (which means it could be fixed by inverting its predictions)

This detector will tell you if your model is good, moderate, poor, or very poor based on this AUC value.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ROCCurveFitDetector(ROCCurveFitDetectorOptions)` | Creates a new instance of the ROC curve fit detector with optional custom configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Auc` | The Area Under the Curve (AUC) value calculated from the ROC curve. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates how confident the detector is in its assessment of the model's fit type. |
| `DetectFit(ModelEvaluationData<,,>)` | Analyzes the model's performance data and determines how well the model fits the data. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the type of fit based on the AUC value. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates practical recommendations for improving the model based on the detected fit type. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the ROC curve fit detector. |

