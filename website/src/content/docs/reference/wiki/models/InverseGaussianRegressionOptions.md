---
title: "InverseGaussianRegressionOptions<T>"
description: "Configuration options for Inverse Gaussian Regression, a generalized linear model for positive continuous data with variance proportional to the cube of the mean."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Inverse Gaussian Regression, a generalized linear model for positive continuous data
with variance proportional to the cube of the mean.

## For Beginners

Inverse Gaussian regression is designed for predicting positive continuous
values where the data has a heavy right tail (extreme large values are possible).

It's useful when:

- Your target values are always positive (never zero or negative)
- Larger values tend to be much more variable than smaller values
- The data has a heavier tail than Gamma distribution

Examples:

- Response times in cognitive experiments
- Time until failure for certain mechanical systems
- First passage times in physics
- Waiting times in queuing systems

Compared to Gamma regression, Inverse Gaussian assumes even more variability for large values.

## How It Works

The Inverse Gaussian distribution (also known as Wald distribution) is appropriate for modeling
positive continuous response variables, particularly those with heavy right tails. The variance
is proportional to μ³, making it suitable when larger values have much more variability than
smaller values.

## Properties

| Property | Summary |
|:-----|:--------|
| `DecompositionType` | Gets or sets the matrix decomposition type for solving the linear system. |
| `InitialDispersion` | Gets or sets the initial dispersion parameter estimate. |
| `LinkFunction` | Gets or sets the link function type for the Inverse Gaussian GLM. |
| `MaxIterations` | Gets or sets the maximum number of iterations for the IRLS algorithm. |
| `Tolerance` | Gets or sets the convergence tolerance for the IRLS algorithm. |

