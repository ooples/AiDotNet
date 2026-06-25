---
title: "GammaRegressionOptions<T>"
description: "Configuration options for Gamma Regression, a generalized linear model for positive continuous data."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Gamma Regression, a generalized linear model for positive continuous data.

## For Beginners

Gamma Regression is designed for predicting positive continuous values
where the data tends to be skewed with a long right tail.

It's useful when:

- Your target values are always positive (never zero or negative)
- Larger values tend to be more variable than smaller values
- The data is right-skewed (most values are small, but some are very large)

Examples:

- Insurance claim amounts (can't be negative, large claims are more variable)
- Time until an event occurs (always positive)
- Income (always positive, highly variable for higher earners)
- Costs and prices

Unlike linear regression which can predict negative values, Gamma regression
naturally ensures predictions are always positive.

## How It Works

Gamma Regression is suited for modeling positive continuous response variables, especially those that
are right-skewed and where variance increases with the mean. It uses either a log link function or
inverse link function. Common applications include insurance claims, hospital lengths of stay,
income modeling, and any situation where the response must be strictly positive.

## Properties

| Property | Summary |
|:-----|:--------|
| `DecompositionType` | Gets or sets the matrix decomposition type for solving the linear system. |
| `InitialDispersion` | Gets or sets the initial dispersion parameter estimate. |
| `LinkFunction` | Gets or sets the link function type for the Gamma GLM. |
| `MaxIterations` | Gets or sets the maximum number of iterations for the IRLS algorithm. |
| `Tolerance` | Gets or sets the convergence tolerance for the IRLS algorithm. |

