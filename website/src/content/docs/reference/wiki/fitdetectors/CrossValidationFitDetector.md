---
title: "CrossValidationFitDetector<T, TInput, TOutput>"
description: "A fit detector that analyzes model performance across training, validation, and test datasets to assess model fit."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A fit detector that analyzes model performance across training, validation, and test datasets to assess model fit.

## For Beginners

Cross-validation is a technique for evaluating how well a model will generalize 
to independent data by comparing its performance on different subsets of the data. This detector 
analyzes the model's performance metrics across training, validation, and test datasets to determine 
if it's underfitting, overfitting, or has a good fit.

## How It Works

By comparing metrics like R² (coefficient of determination) across different datasets, the detector 
can identify issues like overfitting (performing much better on training than validation data) or 
underfitting (performing poorly across all datasets).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CrossValidationFitDetector(CrossValidationFitDetectorOptions)` | Initializes a new instance of the CrossValidationFitDetector class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates the confidence level of the cross-validation-based fit detection. |
| `DetectFit(ModelEvaluationData<,,>)` | Detects the fit type of a model based on cross-validation analysis. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the fit type based on cross-validation analysis. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates recommendations based on the detected fit type and cross-validation analysis. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_goodFitThreshold` | Threshold for determining if a model has a good fit. |
| `_options` | Configuration options for the cross-validation fit detector. |
| `_overfitThreshold` | Threshold for determining if a model is overfitting. |
| `_underfitThreshold` | Threshold for determining if a model is underfitting. |

