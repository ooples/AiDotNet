---
title: "FitType"
description: "Represents different types of model fit quality and common issues in machine learning models."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Represents different types of model fit quality and common issues in machine learning models.

## For Beginners

Model fit describes how well your AI model matches the data it's trying to learn from.

Think of model fit like trying on clothes:

- A good fit means the model captures the true patterns in your data
- A poor fit means the model doesn't match the data well
- Different types of poor fits have different causes and solutions

Common fit problems include:

- Overfitting: The model memorizes the training data instead of learning general patterns
- Underfitting: The model is too simple to capture important patterns in the data
- Bias and variance issues: Different types of errors that affect how your model performs
- Multicollinearity: When input variables are too closely related to each other
- Autocorrelation: When data points are related to previous data points in a sequence

Understanding the type of fit helps you diagnose problems with your model and make improvements.

## Fields

| Field | Summary |
|:-----|:--------|
| `GoodFit` | Indicates that the model fits the data well, capturing the underlying patterns without memorizing noise. |
| `HighBias` | Indicates that the model consistently misses the true relationship in the data. |
| `HighVariance` | Indicates that the model is too sensitive to small fluctuations in the training data. |
| `Moderate` | Indicates a moderate level of effect or relationship in the data. |
| `ModerateMulticollinearity` | Indicates that input variables have some correlation, potentially affecting coefficient stability. |
| `NoAutocorrelation` | Indicates that data points are not correlated with previous data points. |
| `Overfit` | Indicates that the model has memorized the training data too closely, including its noise and outliers. |
| `PoorFit` | Indicates that the model does not fit the data well but is not completely useless. |
| `SevereMulticollinearity` | Indicates that input variables are highly correlated, causing unreliable coefficient estimates. |
| `StrongNegativeAutocorrelation` | Indicates that data points are strongly correlated with previous data points in a negative direction. |
| `StrongPositiveAutocorrelation` | Indicates that data points are strongly correlated with previous data points in a positive direction. |
| `Underfit` | Indicates that the model is too simple to capture the important patterns in the data. |
| `Unstable` | Indicates that small changes in the input data cause large, unpredictable changes in the model's predictions. |
| `VeryPoorFit` | Indicates that the model performs extremely poorly and fails to capture meaningful patterns in the data. |
| `WeakAutocorrelation` | Indicates that data points have some correlation with previous data points, but the relationship is not strong. |

