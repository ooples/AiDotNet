---
title: "CookDistanceFitDetector<T, TInput, TOutput>"
description: "A fit detector that uses Cook's distance to identify influential data points and assess model fit."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A fit detector that uses Cook's distance to identify influential data points and assess model fit.

## For Beginners

Cook's distance is a statistical measure that identifies influential data points 
in a regression analysis. An influential point is one that, if removed, would significantly change the 
model's parameters or predictions.

## How It Works

This detector analyzes the distribution of Cook's distances across all data points to determine if 
the model is overfitting (too sensitive to individual points) or underfitting (not capturing important 
patterns in the data).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CookDistanceFitDetector(CookDistanceFitDetectorOptions)` | Initializes a new instance of the CookDistanceFitDetector class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates the confidence level of the Cook's distance-based fit detection. |
| `CalculateCookDistances(ModelEvaluationData<,,>)` | Calculates Cook's distances for all data points. |
| `DetectFit(ModelEvaluationData<,,>)` | Detects the fit type of a model based on Cook's distance analysis. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the fit type based on Cook's distance analysis. |
| `DetermineFitType(Vector<>)` | Determines the fit type based on a vector of Cook's distances. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates recommendations based on the detected fit type and Cook's distance analysis. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the Cook's distance fit detector. |

