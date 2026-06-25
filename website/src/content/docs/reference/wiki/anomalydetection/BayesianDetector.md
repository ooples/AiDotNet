---
title: "BayesianDetector<T>"
description: "Detects anomalies using Bayesian probability estimation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Probabilistic`

Detects anomalies using Bayesian probability estimation.

## For Beginners

This detector uses Bayesian probability to model normal data distribution.
It assumes data follows a multivariate Gaussian distribution with prior beliefs about
the parameters. Points with low likelihood under this model are considered anomalies.

## How It Works

The algorithm works by:

1. Estimate mean and covariance with Bayesian priors
2. Compute posterior predictive probability for each point
3. Low probability points are flagged as anomalies

**When to use:**

- When you have prior knowledge about the data distribution
- For probabilistic anomaly scoring
- When you want uncertainty estimates

**Industry Standard Defaults:**

- Prior strength (kappa0): 0.01 (weak prior)
- Prior degrees of freedom: n_features + 2
- Contamination: 0.1 (10%)

Reference: Murphy, K.P. (2012). "Machine Learning: A Probabilistic Perspective."
Chapter on Bayesian inference for MVN.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BayesianDetector(Double,Double,Int32)` | Creates a new Bayesian anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `PriorStrength` | Gets the prior strength parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

