---
title: "GaussianLikelihood<T>"
description: "Implements the Gaussian (Normal) likelihood for regression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.GaussianProcesses`

Implements the Gaussian (Normal) likelihood for regression.

## For Beginners

The Gaussian likelihood is the standard choice for regression.
It assumes observations are the true function value plus Gaussian noise:

y = f(x) + ε, where ε ~ N(0, σ²)

This means:

- Errors are normally distributed
- Errors have constant variance (homoscedastic)
- Errors are independent

The noise variance σ² is a hyperparameter that:

- Large σ² → Smoother fit, more uncertainty
- Small σ² → Interpolates data more closely

Gaussian likelihood allows exact GP inference (no approximations needed).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GaussianLikelihood(Double)` | Initializes a new Gaussian likelihood. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this likelihood. |
| `NoiseVariance` | Gets the noise variance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `LogLikelihood(Vector<>,Vector<>)` | Computes the log-likelihood of observations given latent function values. |
| `LogLikelihoodGradient(Vector<>,Vector<>)` | Computes the gradient of the log-likelihood with respect to f. |
| `LogLikelihoodHessianDiag(Vector<>,Vector<>)` | Computes the Hessian diagonal of the log-likelihood. |
| `PredictiveVariance(,)` | Computes predictive variance (adds noise variance to latent variance). |
| `TransformMean()` | For Gaussian likelihood, the mean is just f (identity transform). |

