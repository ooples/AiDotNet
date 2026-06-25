---
title: "RidgeRegressionOptions<T>"
description: "Configuration options for Ridge Regression (L2 regularized linear regression)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Ridge Regression (L2 regularized linear regression).

## For Beginners

Ridge Regression is like standard linear regression with a "shrinkage" effect.

Imagine you're fitting a line to data points, but some of your features are noisy or redundant.
Without regularization, the model might give these noisy features large coefficients, leading to
poor predictions on new data (overfitting).

Ridge Regression solves this by:

- Adding a penalty for large coefficient values
- Shrinking all coefficients toward zero (but never exactly to zero)
- Making the model more stable and generalizable

When to use Ridge Regression:

- When you have many features that might be correlated
- When you want to prevent overfitting without removing features
- When all features are expected to contribute to the prediction

The "alpha" parameter controls how much shrinkage is applied:

- Higher alpha = more shrinkage = simpler model (might underfit)
- Lower alpha = less shrinkage = more complex model (might overfit)

Note: If your features are on different scales, consider normalizing your data
before training using INormalizer implementations like ZScoreNormalizer or MinMaxNormalizer.

## How It Works

Ridge Regression extends ordinary least squares regression by adding an L2 penalty term
to the loss function. This penalty shrinks the coefficients toward zero, helping to prevent
overfitting, especially when dealing with multicollinearity (highly correlated features) or
when the number of features is large relative to the number of samples.

The objective function minimized is: ||y - Xw||^2 + alpha * ||w||^2
where alpha controls the strength of the regularization.

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets or sets the regularization strength (alpha). |
| `DecompositionType` | Gets or sets the type of matrix decomposition used to solve the ridge regression equations. |

