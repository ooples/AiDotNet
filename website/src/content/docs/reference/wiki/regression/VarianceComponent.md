---
title: "VarianceComponent<T>"
description: "Represents variance components in a mixed-effects model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression.MixedEffects`

Represents variance components in a mixed-effects model.

## For Beginners

Variance components tell you "how much variation comes from where".

For students in schools:

- Random effect variance: How much do schools differ on average?
- Residual variance: How much do individual students vary within schools?

This decomposition is important for:

- Understanding your data structure
- Computing Intraclass Correlation (ICC)
- Assessing if random effects are needed

## How It Works

Variance components partition the total variance in the response into parts
attributable to different sources (fixed effects, random effects, residual).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VarianceComponent(String)` | Initializes a new variance component. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ConfidenceIntervalLower` | Gets or sets the lower bound of the confidence interval. |
| `ConfidenceIntervalUpper` | Gets or sets the upper bound of the confidence interval. |
| `CorrelationMatrix` | Gets or sets the correlation matrix for this variance component. |
| `CovarianceMatrix` | Gets or sets the covariance matrix for this variance component. |
| `Name` | Gets or sets the name of this variance component. |
| `StandardDeviation` | Gets the standard deviation (square root of variance). |
| `StandardError` | Gets or sets the standard error of the variance estimate. |
| `Variance` | Gets or sets the estimated variance for this component. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetVarianceProportion()` | Computes the proportion of total variance explained by this component. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations for type T. |

