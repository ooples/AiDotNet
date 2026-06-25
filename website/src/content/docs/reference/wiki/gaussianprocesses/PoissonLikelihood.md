---
title: "PoissonLikelihood<T>"
description: "Implements the Poisson likelihood for count data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.GaussianProcesses`

Implements the Poisson likelihood for count data.

## For Beginners

The Poisson likelihood is used for modeling count data
(number of events, customer arrivals, defects, etc.).

The latent function f is transformed to a rate using exp:
y ~ Poisson(λ), where λ = exp(f)

This means:

- f → +∞ gives high expected counts
- f → -∞ gives low expected counts (near 0)
- f = 0 gives expected count of 1

Properties of Poisson:

- E[y] = λ = exp(f)
- Var[y] = λ = exp(f) (variance equals mean)

Use when:

- Counting discrete events
- Events are independent
- Rate of events can vary smoothly over inputs

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PoissonLikelihood` | Initializes a new Poisson likelihood. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this likelihood. |

## Methods

| Method | Summary |
|:-----|:--------|
| `LogFactorial(Int32)` | Computes log(n!) using Stirling's approximation for large n. |
| `LogLikelihood(Vector<>,Vector<>)` | Computes the log-likelihood of count observations. |
| `LogLikelihoodGradient(Vector<>,Vector<>)` | Computes the gradient of the log-likelihood. |
| `LogLikelihoodHessianDiag(Vector<>,Vector<>)` | Computes the Hessian diagonal of the log-likelihood. |
| `PredictiveVariance(,)` | Computes predictive variance for counts. |
| `TransformMean()` | Transforms latent function value to expected count using exp. |

