---
title: "GMMDetector<T>"
description: "Detects anomalies using Gaussian Mixture Models (GMM)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Probabilistic`

Detects anomalies using Gaussian Mixture Models (GMM).

## For Beginners

GMM models data as a mixture of several Gaussian distributions.
Points with low probability under this model are anomalies - they don't fit well
into any of the learned Gaussian clusters.

## How It Works

The algorithm works by:

1. Fit a mixture of k Gaussians using Expectation-Maximization
2. For each point, compute its probability under the mixture model
3. Points with low probability are anomalies

**When to use:**

- Data is a mixture of Gaussian clusters
- Different clusters have different shapes/sizes
- You need probabilistic anomaly scores

**Industry Standard Defaults:**

- Components: 3 (typical range 2-10)
- Covariance type: diagonal (uses variance per feature for stability)
- Contamination: 0.1 (10%)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GMMDetector(Int32,Int32,Double,Int32)` | Creates a new GMM anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NComponents` | Gets the number of Gaussian components. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

