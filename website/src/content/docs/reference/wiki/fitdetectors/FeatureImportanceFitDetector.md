---
title: "FeatureImportanceFitDetector<T, TInput, TOutput>"
description: "A fit detector that analyzes feature importances and correlations to assess model fit."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A fit detector that analyzes feature importances and correlations to assess model fit.

## For Beginners

Feature importance measures how much each input variable (feature) contributes 
to a model's predictions. This detector uses permutation importance, which works by randomly shuffling 
each feature and measuring how much the model's performance degrades as a result.

## How It Works

By analyzing both feature importances and correlations between features, this detector can identify 
issues like overfitting (relying too heavily on specific features) or underfitting (not effectively 
using the available features).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FeatureImportanceFitDetector(FeatureImportanceFitDetectorOptions)` | Initializes a new instance of the FeatureImportanceFitDetector class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AreFeaturesMostlyUncorrelated(Matrix<>)` | Determines if most features are uncorrelated with each other. |
| `AverageAbsoluteCorrelation(Matrix<>)` | Calculates the average absolute correlation across all feature pairs. |
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates the confidence level of the feature importance-based fit detection. |
| `CalculateError(Vector<>,Vector<>)` | Calculates the error between actual and predicted values. |
| `CalculateFeatureCorrelations(Matrix<>)` | Calculates the correlation matrix for all features. |
| `CalculateFeatureImportances(ModelEvaluationData<,,>)` | Calculates feature importances using permutation importance. |
| `DetectFit(ModelEvaluationData<,,>)` | Detects the fit type of a model based on feature importance analysis. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the fit type based on feature importance analysis. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates recommendations based on the detected fit type and feature importance analysis. |
| `PermuteFeature(Vector<>)` | Randomly shuffles (permutes) the values in a feature vector. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the feature importance fit detector. |

