---
title: "ModelResult<T, TInput, TOutput>"
description: "Represents the complete results of a model-building process, including the model solution, fitness metrics, fit detection results, evaluation data, and selected features."
section: "API Reference"
---

`Structs` · `AiDotNet.Models.Results`

Represents the complete results of a model-building process, including the model solution, fitness metrics,
fit detection results, evaluation data, and selected features.

## For Beginners

This struct is like a container that holds everything about a model's performance.

When building machine learning or statistical models:

- You need to track many different aspects of model performance
- You want to compare different models to choose the best one
- You need to understand not just how well a model performs, but why

This struct stores:

- The actual model solution (the equation or algorithm)
- How well the model fits the data (fitness score)
- Analysis of potential issues like overfitting or underfitting
- Detailed performance metrics on different datasets
- Which input features were used in the model

Having all this information in one place makes it easier to evaluate, compare,
and document your models.

## How It Works

This struct encapsulates all the important information produced during the model-building and evaluation process. 
It includes the symbolic model itself, a fitness score indicating how well the model performs, detailed fit 
detection results that analyze potential issues like overfitting or underfitting, comprehensive evaluation data 
with various performance metrics, and information about which features were selected for the model. This 
comprehensive package of information allows for thorough analysis and comparison of different models.

## Properties

| Property | Summary |
|:-----|:--------|
| `EvaluationData` | Gets or sets the detailed evaluation data for the model. |
| `FitDetectionResult` | Gets or sets the results of fit detection analysis. |
| `Fitness` | Gets or sets the fitness score of the model. |
| `SelectedFeatureIndices` | Gets or sets the zero-based column indices of features selected during optimization. |
| `SelectedFeatures` | Gets or sets the list of feature vectors selected for the model. |

