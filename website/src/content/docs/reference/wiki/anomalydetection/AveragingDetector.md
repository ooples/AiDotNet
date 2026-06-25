---
title: "AveragingDetector<T>"
description: "Combines multiple anomaly detectors using score averaging."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Ensemble`

Combines multiple anomaly detectors using score averaging.

## For Beginners

This ensemble method combines the predictions of multiple anomaly
detectors by averaging their normalized scores. This is one of the simplest and most
effective ensemble techniques.

## How It Works

The algorithm works by:

1. Train each base detector on the data
2. Normalize scores from each detector to [0,1]
3. Average the normalized scores
4. Points with high average score are anomalies

**When to use:**

- When you want robust predictions from multiple detectors
- To reduce variance in anomaly scores
- As a simple baseline ensemble

**Industry Standard Defaults:**

- Default detectors: LOF, IsolationForest, KNN
- Contamination: 0.1 (10%)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AveragingDetector(Double,Int32)` | Creates a new Averaging ensemble anomaly detector with default base detectors. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

