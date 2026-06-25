---
title: "GLMMOptions<T>"
description: "Configuration options for Generalized Linear Mixed-Effects Models (GLMM)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Generalized Linear Mixed-Effects Models (GLMM).

## For Beginners

GLMMs let you model grouped/nested data when your outcome isn't
continuous and normally distributed.

Common scenarios:

- Binary outcome (pass/fail): Use Binomial family with Logit link
- Count data (# of events): Use Poisson family with Log link
- Overdispersed counts: Use NegativeBinomial family
- Continuous positive: Use Gamma family with Log link

Key settings:

- Family: The distribution of your response variable
- LinkFunction: How to connect predictors to the mean response
- EstimationMethod: PQL is faster, Laplace is more accurate

## How It Works

GLMMs combine generalized linear models with random effects for hierarchical data
with non-Gaussian responses (binary, count, etc.).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GLMMOptions` | Initializes a new instance of GLMMOptions with sensible defaults. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BoundVarianceComponents` | Gets or sets whether to bound variance components to be non-negative. |
| `ComputeConfidenceIntervals` | Gets or sets whether to compute confidence intervals for fixed effects. |
| `ConfidenceLevel` | Gets or sets the confidence level. |
| `EstimationMethod` | Gets or sets the estimation method. |
| `Family` | Gets or sets the response distribution family. |
| `LinkFunction` | Gets or sets the link function. |
| `MaxIterations` | Gets or sets the maximum number of iterations. |
| `NegBinomialTheta` | Gets or sets the theta parameter for Negative Binomial family. |
| `Tolerance` | Gets or sets the convergence tolerance. |
| `Verbose` | Gets or sets whether to print verbose output. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForBinaryClassification` | Creates options for logistic mixed-effects model (binary outcomes). |
| `ForCountData` | Creates options for Poisson mixed-effects model (count data). |
| `ForPositiveContinuous` | Creates options for Gamma mixed-effects model (positive continuous). |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations for type T. |

