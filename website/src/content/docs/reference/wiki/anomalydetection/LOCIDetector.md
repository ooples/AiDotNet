---
title: "LOCIDetector<T>"
description: "Detects anomalies using Local Correlation Integral (LOCI)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.DistanceBased`

Detects anomalies using Local Correlation Integral (LOCI).

## For Beginners

LOCI is a density-based outlier detection method that automatically
determines the appropriate neighborhood size. It computes a Multi-Granularity Deviation
Factor (MDEF) that indicates how much a point's local density deviates from its
neighborhood's density.

## How It Works

The algorithm works by:

1. For multiple radius values, compute local density
2. Compare point's density to average density of neighbors (MDEF)
3. Flag points with MDEF exceeding a threshold based on standard deviation

**When to use:**

- When you don't want to manually tune neighborhood size
- Data has varying local densities
- You want automatic threshold determination

**Industry Standard Defaults:**

- Alpha: 0.5 (neighborhood multiplier)
- Contamination: 0.1 (10%)

Reference: Papadimitriou, S., et al. (2003). "LOCI: Fast Outlier Detection Using
the Local Correlation Integral." ICDE.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LOCIDetector(Double,Int32,Double,Int32)` | Creates a new LOCI anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets the alpha parameter (sampling neighborhood ratio). |
| `KMax` | Gets the maximum number of neighbors to consider. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `IsolatedPointScore` | MDEF score assigned to points that have counting neighbors but zero sampling neighbors, indicating extreme isolation. |
| `NoNeighborScore` | Score for points with no neighbors at any radius. |
| `NumRadiiSteps` | Number of radius steps used to sweep from zero to `_maxRadius`. |
| `RadiusOvershootFactor` | Multiplier applied to `_maxRadius` so the sweep slightly exceeds it, avoiding floating-point edge cases. |

