---
title: "BetaLikelihood<T>"
description: "Beta Likelihood for Gaussian Processes with bounded outputs in [0, 1]."
section: "API Reference"
---

`Models & Types` · `AiDotNet.GaussianProcesses`

Beta Likelihood for Gaussian Processes with bounded outputs in [0, 1].

## For Beginners

Standard GP regression assumes Gaussian noise: y ~ N(f(x), σ²)
But what if your outputs are bounded between 0 and 1? (e.g., proportions, probabilities)

The Beta likelihood handles this:

1. GP models a latent function f(x) (unbounded)
2. Sigmoid transformation: μ = sigmoid(f) ∈ (0, 1)
3. Beta distribution: y ~ Beta(μ × ν, (1-μ) × ν)

Where:

- μ is the mean of the Beta (determined by sigmoid(f))
- ν is the "precision" (higher = less variance)

This is useful for:

- Modeling proportions (e.g., click-through rates)
- Modeling probabilities
- Any response bounded in [0, 1]

The Beta distribution naturally handles the bounded nature:

- Values near 0 or 1 have appropriately skewed distributions
- Variance is heteroscedastic (depends on mean)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BetaLikelihood(Double)` | Initializes a Beta Likelihood. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Precision` | Gets the precision parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Digamma(Double)` | Digamma function (derivative of log Gamma). |
| `FromData(Vector<>)` | Creates a default Beta Likelihood with automatic precision estimation. |
| `GetBetaParameters(Double)` | Computes Beta parameters (α, β) from mean μ. |
| `GetMeans(Vector<>)` | Transforms latent function values to Beta means via sigmoid. |
| `LogGamma(Double)` | Log Gamma function. |
| `LogLikelihood(Vector<>,Vector<>)` | Computes log-likelihood of observations given latent values. |
| `LogLikelihoodGradient(Vector<>,Vector<>)` | Computes gradient of log-likelihood with respect to f. |
| `LogLikelihoodHessianDiag(Vector<>,Vector<>)` | Computes Hessian (second derivative) of log-likelihood. |
| `PredictiveMoments(Double,Double)` | Computes predictive mean and variance for a new point. |
| `Sample(Double,Random)` | Samples from the Beta distribution given mean. |
| `SampleBeta(Double,Double,Random)` | Samples from Beta distribution using rejection sampling. |
| `SampleGamma(Double,Random)` | Samples from Gamma distribution using Marsaglia and Tsang's method. |
| `SampleNormal(Random)` | Samples from standard normal using Box-Muller. |
| `Sigmoid(Double)` | Sigmoid function. |
| `Trigamma(Double)` | Trigamma function (second derivative of log Gamma). |

## Fields

| Field | Summary |
|:-----|:--------|
| `Epsilon` | Small constant for numerical stability. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_precision` | Precision parameter ν. |

