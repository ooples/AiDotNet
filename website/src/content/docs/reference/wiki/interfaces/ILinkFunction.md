---
title: "ILinkFunction<T>"
description: "Interface for link functions used in Generalized Linear Models (GLMs)."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for link functions used in Generalized Linear Models (GLMs).

## For Beginners

In regular linear regression, predictions can be any real number.
But many real-world quantities have constraints:

- Probabilities must be between 0 and 1
- Counts must be non-negative
- Positive quantities (like income) can't be negative

Link functions solve this by transforming predictions to the appropriate range:

- Logit link maps linear predictions to (0,1) for probabilities
- Log link maps linear predictions to (0,∞) for counts/positive values
- Identity link makes no transformation (standard regression)

**Example:** For logistic regression:

- Linear predictor: η = β₀ + β₁x₁ + β₂x₂ (can be any real number)
- Link function (logit): η = log(p/(1-p))
- Inverse link: p = exp(η)/(1+exp(η)) (always between 0 and 1)

## How It Works

A link function connects the linear predictor (Xβ) to the expected value of the response
variable (μ). The link function g satisfies: g(μ) = η = Xβ, where η is the linear predictor.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsCanonical` | Gets whether this is the canonical link for a distribution family. |
| `Name` | Gets the name of the link function. |

## Methods

| Method | Summary |
|:-----|:--------|
| `InverseLink()` | Applies the inverse link function: g⁻¹(η) = μ. |
| `InverseLinkDerivative()` | Computes the derivative of the inverse link function: dg⁻¹/dη. |
| `Link()` | Applies the link function: g(μ) = η. |
| `LinkDerivative()` | Computes the derivative of the link function: dg/dμ. |
| `Variance()` | Computes the variance function: Var(Y) as a function of μ. |

