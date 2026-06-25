---
title: "LSCPDetector<T>"
description: "Detects anomalies using LSCP (Locally Selective Combination in Parallel Outlier Ensembles)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Ensemble`

Detects anomalies using LSCP (Locally Selective Combination in Parallel Outlier Ensembles).

## For Beginners

LSCP improves ensemble detection by selecting the most competent
detectors for each test point locally. Instead of combining all detectors equally,
it selects the best performing detectors based on local pseudo ground truth.

## How It Works

The algorithm works by:

1. Train multiple diverse base detectors
2. For each test point, find local region
3. Evaluate detector competence in that region
4. Combine scores from most competent detectors

**When to use:**

- When different detectors work better in different regions
- For heterogeneous anomaly types
- When ensemble averaging isn't optimal

**Industry Standard Defaults:**

- N estimators: 10
- Local region size: 30 neighbors
- Contamination: 0.1 (10%)

Reference: Zhao, Y., et al. (2019). "LSCP: Locally Selective Combination in Parallel
Outlier Ensembles." SDM.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LSCPDetector(Int32,Int32,Double,Int32)` | Creates a new LSCP anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LocalRegionSize` | Gets the local region size. |
| `NEstimators` | Gets the number of estimators. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

