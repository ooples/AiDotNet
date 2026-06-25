---
title: "SOSDetector<T>"
description: "Detects anomalies using SOS (Stochastic Outlier Selection)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.DistanceBased`

Detects anomalies using SOS (Stochastic Outlier Selection).

## For Beginners

SOS is based on the concept of affinity - how likely a point is to be
selected as a neighbor by other points. If a point has low affinity (other points rarely
select it as a neighbor), it is likely an outlier.

## How It Works

The algorithm works by:

1. Compute pairwise affinities using Gaussian kernel
2. Normalize affinities (like t-SNE probabilities)
3. Compute binding probability for each point
4. Points with low binding probability are outliers

**When to use:**

- When you want a probabilistic interpretation of outlierness
- Medium-sized datasets
- When local density variations exist

**Industry Standard Defaults:**

- Perplexity: 30 (similar to t-SNE)
- Contamination: 0.1 (10%)

Reference: Janssens, J.H.M., et al. (2012). "Stochastic Outlier Selection." Technical Report.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SOSDetector(Double,Double,Int32)` | Creates a new SOS anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Perplexity` | Gets the perplexity parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

