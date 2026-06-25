---
title: "HeteroscedasticityFitDetectorOptions"
description: "Configuration options for the Heteroscedasticity Fit Detector, which analyzes whether a model's prediction errors have constant variance across all prediction values."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Heteroscedasticity Fit Detector, which analyzes whether a model's
prediction errors have constant variance across all prediction values.

## For Beginners

This detector checks if your model's errors (the differences between
predicted and actual values) are consistent across all predictions or if they vary in a systematic way.

Imagine you have a model predicting house prices. If your model tends to make small errors for
low-priced houses but much larger errors for expensive houses, that's heteroscedasticity. It means
your model's accuracy depends on what it's predicting, which is usually not ideal.

Think of it like a weather forecaster who's very accurate when predicting mild temperatures but
wildly inaccurate when predicting extreme temperatures. You'd want to know about this inconsistency
because it affects how much you can trust different predictions.

When heteroscedasticity is detected, it often suggests that:

- Your model might be missing important features
- You might need to transform your target variable (e.g., use log of price instead of price)
- You might need a different type of model altogether
- You should be more cautious about the model's predictions in certain ranges

## How It Works

Heteroscedasticity refers to the situation where the variance of errors (residuals) varies across
the range of predicted values. In regression analysis, one of the key assumptions is homoscedasticity,
which means the error variance should be constant across all levels of the predicted values. When this
assumption is violated (heteroscedasticity), it can lead to inefficient parameter estimates and
unreliable confidence intervals.

## Properties

| Property | Summary |
|:-----|:--------|
| `HeteroscedasticityThreshold` | Gets or sets the p-value threshold for detecting heteroscedasticity in model residuals. |
| `HomoscedasticityThreshold` | Gets or sets the p-value threshold for confirming homoscedasticity in model residuals. |

