---
title: "BetaRegressionOptions"
description: "Configuration options for Beta Regression models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Beta Regression models.

## For Beginners

Regular regression can predict any number, but what if you're
predicting something that must be between 0 and 1, like:

- Percentage of students passing an exam
- Proportion of defective products
- Probability estimates
- Market share percentages

Beta Regression handles this naturally by modeling the data as following a Beta
distribution, which is naturally bounded between 0 and 1.

Key benefits:

- Predictions are always in valid range (0,1)
- Can model varying precision (some predictions more certain than others)
- Handles skewed proportions well

## How It Works

Beta Regression is used when your target variable is a proportion or rate bounded
between 0 and 1 (exclusive). Examples include percentages, rates, and proportions.

## Properties

| Property | Summary |
|:-----|:--------|
| `LearningRate` | Gets or sets the learning rate for optimization. |
| `LinkFunction` | Gets or sets the link function for the mean model. |
| `MaxIterations` | Gets or sets the maximum number of iterations for optimization. |
| `ModelVariablePrecision` | Gets or sets whether to model the precision parameter as a function of features. |
| `RegularizationStrength` | Gets or sets the regularization strength. |
| `Seed` | Gets or sets the random seed for reproducibility. |
| `Tolerance` | Gets or sets the convergence tolerance. |
| `UseRegularization` | Gets or sets whether to use regularization. |

