---
title: "ZeroInflatedRegressionOptions"
description: "Configuration options for Zero-Inflated regression models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Zero-Inflated regression models.

## For Beginners

Sometimes you're counting things, but there are lots of zeros:

- Number of insurance claims (most people have zero claims)
- Number of fish caught (many fishers catch nothing)
- Number of cigarettes smoked (many people don't smoke)

Regular Poisson regression doesn't handle this well because it assumes a specific
relationship between the mean and variance, and can't account for "excess zeros."

Zero-Inflated models solve this by saying:
"There are two types of zeros - some come from a zero-generating process
(people who NEVER smoke), and some come from the count process
(smokers who happened to smoke 0 cigarettes today)."

This gives more accurate predictions and proper uncertainty estimates.

## How It Works

Zero-Inflated models handle count data that has more zeros than a standard count
distribution (Poisson, Negative Binomial) would predict. They model the data as
a mixture of a point mass at zero and a count distribution.

## Properties

| Property | Summary |
|:-----|:--------|
| `CountLink` | Gets or sets the link function for the count model. |
| `DistributionFamily` | Gets or sets the base count distribution family. |
| `MaxIterations` | Gets or sets the maximum number of iterations for optimization. |
| `ModelZeroInflation` | Gets or sets whether to model the zero-inflation probability as a function of features. |
| `RegularizationStrength` | Gets or sets the regularization strength. |
| `Seed` | Gets or sets the random seed for reproducibility. |
| `Tolerance` | Gets or sets the convergence tolerance. |
| `UseRegularization` | Gets or sets whether to use regularization. |
| `ZeroLink` | Gets or sets the link function for the zero-inflation model. |

