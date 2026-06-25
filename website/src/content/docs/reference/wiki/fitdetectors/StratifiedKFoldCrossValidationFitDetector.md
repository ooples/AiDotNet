---
title: "StratifiedKFoldCrossValidationFitDetector<T, TInput, TOutput>"
description: "A detector that evaluates model fit using Stratified K-Fold Cross-Validation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A detector that evaluates model fit using Stratified K-Fold Cross-Validation.

## For Beginners

This class helps you determine if your machine learning model is performing well
or if it has common problems like overfitting or underfitting.

Stratified K-Fold Cross-Validation is a technique that:

1. Splits your data into K equal parts (folds)
2. Makes sure each fold has a similar distribution of your target variable
3. Trains K different models, each using K-1 folds for training and 1 fold for validation
4. Averages the results to get a more reliable estimate of model performance

This approach helps ensure your model works well on different subsets of your data,
which is important for real-world applications.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StratifiedKFoldCrossValidationFitDetector(StratifiedKFoldCrossValidationFitDetectorOptions)` | Initializes a new instance of the StratifiedKFoldCrossValidationFitDetector class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates the confidence level in the fit type determination. |
| `DetectFit(ModelEvaluationData<,,>)` | Analyzes the model's performance data and determines the type of fit. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the type of fit based on the model's performance metrics. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates specific recommendations for improving the model based on the detected fit type. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the Stratified K-Fold Cross-Validation fit detector. |

