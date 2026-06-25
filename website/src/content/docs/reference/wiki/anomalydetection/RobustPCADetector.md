---
title: "RobustPCADetector<T>"
description: "Detects anomalies using Robust PCA (Principal Component Pursuit)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Linear`

Detects anomalies using Robust PCA (Principal Component Pursuit).

## For Beginners

Robust PCA decomposes data into a low-rank component (normal patterns)
and a sparse component (anomalies). Unlike standard PCA, it's robust to outliers because
it explicitly models them as sparse corruptions.

## How It Works

The algorithm works by:

1. Decompose data matrix M = L + S
2. L is low-rank (captures normal patterns)
3. S is sparse (captures anomalies)
4. Solve via convex optimization (ADMM)

**When to use:**

- Data corrupted by sparse anomalies
- When standard PCA is affected by outliers
- Video surveillance (background subtraction)
- Network intrusion detection

**Industry Standard Defaults:**

- Lambda: 1/sqrt(max(n,m))
- Max iterations: 1000
- Tolerance: 1e-7
- Contamination: 0.1 (10%)

Reference: Candès, E.J., et al. (2011). "Robust Principal Component Analysis?" JACM.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RobustPCADetector(Double,Int32,Double,Double,Int32)` | Creates a new Robust PCA anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Lambda` | Gets the lambda parameter (sparsity penalty). |
| `MaxIterations` | Gets the maximum iterations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

