---
title: "BernoulliLikelihood<T>"
description: "Implements the Bernoulli likelihood for binary classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.GaussianProcesses`

Implements the Bernoulli likelihood for binary classification.

## For Beginners

The Bernoulli likelihood is used for binary classification
(yes/no, 0/1, positive/negative outcomes).

The latent function f is passed through a sigmoid to get probability:
p(y=1|f) = σ(f) = 1 / (1 + exp(-f))

This means:

- f → +∞ gives p(y=1) → 1
- f → -∞ gives p(y=1) → 0
- f = 0 gives p(y=1) = 0.5

The GP models uncertainty in f, which translates to uncertainty in class probabilities.

Note: Bernoulli likelihood requires approximate inference (Laplace or EP).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BernoulliLikelihood` | Initializes a new Bernoulli likelihood. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this likelihood. |

## Methods

| Method | Summary |
|:-----|:--------|
| `LogLikelihood(Vector<>,Vector<>)` | Computes the log-likelihood of observations given latent function values. |
| `LogLikelihoodGradient(Vector<>,Vector<>)` | Computes the gradient of the log-likelihood with respect to f. |
| `LogLikelihoodHessianDiag(Vector<>,Vector<>)` | Computes the Hessian diagonal of the log-likelihood. |
| `LogOnePlusExp(Double)` | Computes log(1 + exp(x)) in a numerically stable way. |
| `PredictiveVariance(,)` | Computes predictive variance for classification probabilities. |
| `Sigmoid(Double)` | Numerically stable sigmoid function. |
| `TransformMean()` | Transforms latent function value to class probability using sigmoid. |

