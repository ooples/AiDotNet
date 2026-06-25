---
title: "COFDetector<T>"
description: "Detects anomalies using Connectivity-Based Outlier Factor (COF)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.DistanceBased`

Detects anomalies using Connectivity-Based Outlier Factor (COF).

## For Beginners

COF improves on LOF by considering the connectivity pattern
of a point's neighborhood. It detects outliers that are connected differently
from their neighbors (e.g., points in low-density corridors).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `COFDetector(Int32,Double,Int32)` | Creates a new COF anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `K` | Gets the number of neighbors used for detection. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

