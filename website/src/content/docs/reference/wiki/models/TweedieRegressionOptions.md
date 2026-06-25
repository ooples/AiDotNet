---
title: "TweedieRegressionOptions<T>"
description: "Configuration options for Tweedie Regression, a flexible generalized linear model that encompasses several distributions (Poisson, Gamma, Inverse Gaussian) as special cases."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Tweedie Regression, a flexible generalized linear model that encompasses
several distributions (Poisson, Gamma, Inverse Gaussian) as special cases.

## For Beginners

Tweedie regression is like having a "dial" that lets you choose
how the variability in your data relates to the average.

It's especially useful for:

- Insurance claims: Many zeros (no claim) plus positive amounts (actual claims)
- Rainfall data: Many dry days (zero) plus positive rainfall amounts
- Any situation where you have both exact zeros and positive continuous values
- When you're not sure whether Poisson or Gamma is the right choice

The key advantage is that Tweedie with p between 1 and 2 naturally handles data that has:

- Exact zeros (like Poisson)
- Positive continuous values (like Gamma)

This makes it ideal for insurance, healthcare costs, and many business applications.

## How It Works

Tweedie regression is a powerful family of distributions where variance is proportional to a power
of the mean: Var(Y) = φ × μ^p. The power parameter p determines which distribution family is used:

- p = 0: Normal/Gaussian (variance independent of mean)
- p = 1: Poisson (variance = mean)
- 1 < p < 2: Compound Poisson-Gamma (excellent for data with exact zeros and positive continuous values)
- p = 2: Gamma (variance = mean²)
- p = 3: Inverse Gaussian (variance = mean³)

## Properties

| Property | Summary |
|:-----|:--------|
| `DecompositionType` | Gets or sets the matrix decomposition type for solving the linear system. |
| `InitialDispersion` | Gets or sets the initial dispersion parameter estimate. |
| `LinkFunction` | Gets or sets the link function type for the Tweedie GLM. |
| `MaxIterations` | Gets or sets the maximum number of iterations for the IRLS algorithm. |
| `PowerParameter` | Gets or sets the power parameter (p) that determines the variance-mean relationship. |
| `Tolerance` | Gets or sets the convergence tolerance for the IRLS algorithm. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the options, ensuring the power parameter is valid. |

