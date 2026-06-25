---
title: "LoOPDetector<T>"
description: "Detects anomalies using Local Outlier Probability (LoOP)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.DistanceBased`

Detects anomalies using Local Outlier Probability (LoOP).

## For Beginners

LoOP improves on LOF by providing a probability score between 0 and 1,
making results easier to interpret. A score of 0 means the point is definitely not an outlier,
while 1 means it's definitely an outlier.

## How It Works

The algorithm works by:

1. Compute probabilistic set distance (PLOF) using Gaussian error function
2. Normalize using local standard deviation of LOF values
3. Convert to probability using error function

**When to use:**

- When you need interpretable probability scores
- When comparing outlier scores across different datasets
- Similar use cases as LOF but with better interpretability

**Industry Standard Defaults:**

- K (neighbors): 10
- Lambda: 3 (controls sensitivity)
- Contamination: 0.1 (10%)

Reference: Kriegel, H., et al. (2009). "LoOP: Local Outlier Probabilities." CIKM.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LoOPDetector(Int32,Double,Double,Int32)` | Creates a new LoOP anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `K` | Gets the number of neighbors used for detection. |
| `Lambda` | Gets the lambda parameter (standard deviations for probability). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ErrorFunction(Double)` | Approximation of the error function (erf). |
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

