---
title: "ILikelihood<T>"
description: "Interface for likelihood functions in Gaussian Processes."
section: "API Reference"
---

`Interfaces` · `AiDotNet.GaussianProcesses`

Interface for likelihood functions in Gaussian Processes.

## For Beginners

A likelihood function describes how observed data y relates to
the underlying GP function value f. It models the "noise" or observation process.

Common likelihoods:

- Gaussian: y = f + ε, where ε ~ N(0, σ²) - for regression
- Bernoulli: y ~ Bernoulli(sigmoid(f)) - for binary classification
- Poisson: y ~ Poisson(exp(f)) - for count data

The likelihood affects:

- How we interpret the GP output
- What inference method we need (exact vs approximate)
- What kind of predictions we can make

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this likelihood. |

## Methods

| Method | Summary |
|:-----|:--------|
| `LogLikelihood(Vector<>,Vector<>)` | Computes the log-likelihood of observations given latent function values. |
| `LogLikelihoodGradient(Vector<>,Vector<>)` | Computes the gradient of the log-likelihood with respect to f. |
| `LogLikelihoodHessianDiag(Vector<>,Vector<>)` | Computes the Hessian diagonal of the log-likelihood with respect to f. |
| `PredictiveVariance(,)` | Computes the predictive variance given latent mean and variance. |
| `TransformMean()` | Transforms latent function values to the expected observation value. |

