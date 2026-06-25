---
title: "FitDetectorBase<T, TInput, TOutput>"
description: "Base class for all fit detectors that provides common functionality and defines the required interface."
section: "API Reference"
---

`Base Classes` · `AiDotNet.FitDetectors`

Base class for all fit detectors that provides common functionality and defines the required interface.

## For Beginners

This abstract class serves as a template for all fit detectors in the library. 
It defines the common structure and behavior that all fit detectors should have, while allowing 
specific implementations to customize how they detect different types of model fit.

## How It Works

A fit detector analyzes a machine learning model's performance to determine if it's underfitting 
(too simple), overfitting (too complex), or has a good fit (just right). This helps you understand 
how to improve your model.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FitDetectorBase` | Initializes a new instance of the FitDetectorBase class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates the confidence level in the fit type determination. |
| `DetectFit(ModelEvaluationData<,,>)` | Analyzes model performance data and determines the type of fit, confidence level, and recommendations. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the type of fit based on model performance metrics. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates recommendations for improving the model based on the detected fit type. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides numeric operations for the specific type T. |
| `Random` | Random number generator used for feature permutation. |

