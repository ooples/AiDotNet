---
title: "ProbitLink<T>"
description: "Probit link function: g(μ) = Φ⁻¹(μ), where Φ is the standard normal CDF."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LinkFunctions`

Probit link function: g(μ) = Φ⁻¹(μ), where Φ is the standard normal CDF.

## For Beginners

Probit is similar to logit but uses the normal distribution
instead of the logistic distribution. Key differences:

- Probit: Based on normal distribution (bell curve tails)
- Logit: Based on logistic distribution (slightly heavier tails)

In practice, results are usually very similar. Probit is sometimes preferred when:

- The underlying process is modeled as a latent normal variable
- You want consistency with other normal-distribution-based methods

Interpretation: The coefficient represents the change in z-score for
a one-unit change in the predictor.

## How It Works

The probit link maps probabilities to z-scores from the standard normal distribution.
It's an alternative to logit for binary classification.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsCanonical` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Erf(Double)` | Error function using a hybrid approach for full precision. |
| `InverseLink()` |  |
| `InverseLinkDerivative()` |  |
| `Link()` |  |
| `LinkDerivative()` |  |
| `NormalCDF(Double)` | Standard normal CDF using the error function. |
| `NormalInverseCDF(Double)` | Inverse normal CDF (quantile function) using rational approximation. |
| `NormalPDF(Double)` | Standard normal PDF. |
| `Variance()` |  |

