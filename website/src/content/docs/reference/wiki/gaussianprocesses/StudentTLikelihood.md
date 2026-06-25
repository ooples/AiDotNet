---
title: "StudentTLikelihood<T>"
description: "Implements the Student-t likelihood for robust regression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.GaussianProcesses`

Implements the Student-t likelihood for robust regression.

## For Beginners

The Student-t likelihood provides robust regression that is
less sensitive to outliers than Gaussian likelihood.

y = f + ε, where ε ~ Student-t(0, σ², ν)

The degrees of freedom ν controls "heaviness" of tails:

- ν → ∞: Approaches Gaussian (not robust)
- ν = 4-5: Moderately robust
- ν = 1: Very robust (Cauchy distribution)

When to use:

- Data has outliers
- Errors might not be Gaussian
- You want predictions to not be pulled by extreme values

Note: Student-t likelihood requires approximate inference.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StudentTLikelihood(Double,Double)` | Initializes a new Student-t likelihood. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DegreesOfFreedom` | Gets the degrees of freedom. |
| `Name` | Gets the name of this likelihood. |
| `NoiseScale` | Gets the noise scale. |

## Methods

| Method | Summary |
|:-----|:--------|
| `LogGamma(Double)` | Computes the log of the gamma function using Stirling's approximation. |
| `LogLikelihood(Vector<>,Vector<>)` | Computes the log-likelihood of observations. |
| `LogLikelihoodGradient(Vector<>,Vector<>)` | Computes the gradient of the log-likelihood. |
| `LogLikelihoodHessianDiag(Vector<>,Vector<>)` | Computes the Hessian diagonal of the log-likelihood. |
| `PredictiveVariance(,)` | Computes predictive variance. |
| `TransformMean()` | Transforms latent function value (identity for Student-t). |

