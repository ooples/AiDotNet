---
title: "DefaultFitDetector<T, TInput, TOutput>"
description: "A default implementation of a fit detector that analyzes model performance and provides recommendations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A default implementation of a fit detector that analyzes model performance and provides recommendations.

## How It Works

This class evaluates how well a machine learning model fits the data by comparing performance
metrics across training, validation, and test datasets. It can detect common issues like
overfitting and underfitting, and provide appropriate recommendations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DefaultFitDetector` | Initializes a new instance of the DefaultFitDetector class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates the overall confidence level in the model's performance. |
| `DetectFit(ModelEvaluationData<,,>)` | Analyzes model performance data and determines the type of fit, confidence level, and recommendations. |
| `GenerateRecommendations(FitType)` | Generates practical recommendations for improving the model based on the detected fit type. |

