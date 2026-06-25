---
title: "GAMLSSOptions"
description: "Configuration options for GAMLSS (Generalized Additive Models for Location, Scale, and Shape)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for GAMLSS (Generalized Additive Models for Location, Scale, and Shape).

## For Beginners

In traditional regression, you predict the average (mean) value
of your target variable. But what if the spread (variance) of predictions also depends
on the input features? GAMLSS solves this!

For example, when predicting income:

- Traditional regression: "The predicted income is $60,000"
- GAMLSS: "The predicted income is $60,000 ± $10,000 for young workers, but

$60,000 ± $30,000 for self-employed people (higher uncertainty)"

This is useful when:

- Prediction uncertainty varies based on features (heteroskedasticity)
- You need to model skewness or other distributional properties
- You want probabilistic predictions with feature-dependent confidence intervals

## How It Works

GAMLSS extends traditional regression by allowing you to model all parameters of a
probability distribution as functions of explanatory variables, not just the mean.

## Properties

| Property | Summary |
|:-----|:--------|
| `DistributionFamily` | Gets or sets the type of distribution family to use. |
| `LearningRate` | Gets or sets the learning rate (step size) for the IRLS / Fisher-scoring parameter updates. |
| `LocationModelType` | Gets or sets the type of model to use for the location (mean) parameter. |
| `MaxInnerIterations` | Gets or sets the maximum number of inner iterations for fitting each parameter. |
| `MaxOuterIterations` | Gets or sets the maximum number of outer iterations for fitting all parameters. |
| `RegularizationStrength` | Gets or sets the regularization strength. |
| `ScaleModelType` | Gets or sets the type of model to use for the scale parameter. |
| `Seed` | Gets or sets the random seed for reproducibility. |
| `ShapeModelType` | Gets or sets the type of model to use for the shape parameters (if applicable). |
| `Tolerance` | Gets or sets the convergence tolerance. |
| `UseRegularization` | Gets or sets whether to use regularization for parameter estimation. |

