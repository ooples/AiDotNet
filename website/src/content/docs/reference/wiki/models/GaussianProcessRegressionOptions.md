---
title: "GaussianProcessRegressionOptions"
description: "Configuration options for Gaussian Process Regression, a flexible non-parametric approach to regression that provides uncertainty estimates along with predictions."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Gaussian Process Regression, a flexible non-parametric approach to regression
that provides uncertainty estimates along with predictions.

## For Beginners

Think of Gaussian Process Regression as drawing a smooth curve through your data
points, but instead of just giving you one "best" curve, it gives you a range of possible curves with
information about which ones are more likely. This is like a weather forecast that says "70°F with a 90%
chance of being between 65-75°F" rather than just "70°F." This ability to express uncertainty makes GPR
especially useful when you need to know not just what your model predicts, but how confident it is in those
predictions. GPR works well with small to medium datasets and can capture complex patterns without requiring
you to specify the exact form of the relationship beforehand.

## How It Works

Gaussian Process Regression (GPR) is a powerful machine learning technique that models the target function
as a sample from a Gaussian process. Unlike many other regression methods, GPR not only provides point
predictions but also uncertainty estimates, making it valuable for applications where understanding
prediction confidence is important.

## Properties

| Property | Summary |
|:-----|:--------|
| `DecompositionType` | Gets or sets the type of matrix decomposition to use for numerical stability. |
| `LengthScale` | Gets or sets the length scale parameter for the Gaussian Process kernel. |
| `NoiseLevel` | Gets or sets the assumed noise level in the observations. |
| `OptimizeHyperparameters` | Gets or sets whether to automatically optimize the hyperparameters (length scale and signal variance) based on the training data. |
| `SignalVariance` | Gets or sets the signal variance parameter for the Gaussian Process kernel. |

