---
title: "DDM<T>"
description: "Implements DDM (Drift Detection Method) for concept drift detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DriftDetection`

Implements DDM (Drift Detection Method) for concept drift detection.

## For Beginners

DDM is one of the simplest and most popular drift detectors.
It monitors the error rate of a classifier and triggers when the error rate increases
significantly compared to the minimum observed error rate.

## How It Works

**How DDM works:**

- Track the error rate (p) and its standard deviation (s)
- Remember the minimum p + s observed (the "best" state)
- Warning: Current p + s exceeds minimum by more than 2σ
- Drift: Current p + s exceeds minimum by more than 3σ

**Key Concepts:**

- **Error rate (p):** Running average of errors (0 = correct, 1 = error)
- **Standard deviation (s):** sqrt(p * (1-p) / n)
- **Minimum p + s:** The best observed performance
- **Warning zone:** Performance degraded but not enough for drift
- **Drift zone:** Significant performance degradation detected

**Advantages:** Simple, fast, low memory, works well with gradual drift.

**Disadvantages:** Only works with error rates (0/1), may miss small drifts.

**Reference:** Gama et al., "Learning with Drift Detection" (2004)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DDM(Double,Double,Int32,Int32)` | Creates a new DDM drift detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DriftThreshold` | Gets the drift threshold (default: 3.0 standard deviations). |
| `WarningDelay` | Gets the delay between warning and drift detection (for gradual drift). |
| `WarningThreshold` | Gets the warning threshold (default: 2.0 standard deviations). |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddObservation()` | Adds a new observation (error indicator) to the detector. |
| `GetErrorRate` | Gets the current error rate. |
| `GetMinimumPsi` | Gets the minimum error rate observed (the baseline). |
| `Reset` |  |

