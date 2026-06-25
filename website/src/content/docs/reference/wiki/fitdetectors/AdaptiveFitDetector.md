---
title: "AdaptiveFitDetector<T, TInput, TOutput>"
description: "An adaptive fit detector that dynamically selects the most appropriate detection method based on data characteristics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

An adaptive fit detector that dynamically selects the most appropriate detection method based on data characteristics.

## For Beginners

A fit detector helps determine whether a machine learning model is underfitting 
(too simple to capture the patterns in the data) or overfitting (too complex, capturing noise instead 
of true patterns).

## How It Works

This adaptive detector analyzes your data and model performance to automatically choose the most 
appropriate detection method. Think of it like a doctor who selects different diagnostic tools 
based on your symptoms - it uses the right approach for your specific situation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdaptiveFitDetector(AdaptiveFitDetectorOptions)` | Initializes a new instance of the AdaptiveFitDetector class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AssessDataComplexity(ModelEvaluationData<,,>)` | Assesses the complexity of the data based on variance. |
| `AssessModelPerformance(ModelEvaluationData<,,>)` | Assesses the performance of the model based on R² values. |
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates the confidence level of the fit detection. |
| `DetectFit(ModelEvaluationData<,,>)` | Detects the fit type of a model based on evaluation data. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the fit type of a model based on evaluation data. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates recommendations based on the detected fit type and evaluation data. |
| `GetAdaptiveRecommendation(DataComplexity,ModelPerformance)` | Gets an adaptive recommendation based on data complexity and model performance. |
| `GetAdditionalRecommendation(DataComplexity,ModelPerformance)` | Gets additional recommendations based on data complexity and model performance. |
| `GetUsedDetectorName(DataComplexity,ModelPerformance)` | Gets the name of the detector used based on data complexity and model performance. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_hybridDetector` | A detector that combines multiple detection methods. |
| `_learningCurveDetector` | A detector that analyzes learning curves to determine fit. |
| `_options` | Configuration options for the adaptive fit detector. |
| `_residualAnalyzer` | A detector that analyzes model residuals (prediction errors) to determine fit. |

