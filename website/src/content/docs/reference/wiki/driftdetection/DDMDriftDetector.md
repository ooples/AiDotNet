---
title: "DDMDriftDetector<T>"
description: "Drift Detection Method (DDM) for concept drift detection based on error rate monitoring."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DriftDetection`

Drift Detection Method (DDM) for concept drift detection based on error rate monitoring.

## For Beginners

DDM monitors the error rate of a classifier over time. When
the error rate increases significantly beyond what was observed during a stable period,
drift is detected. DDM is simple, fast, and effective for detecting sudden drift.

## How It Works

**How it works:**

- Track the error rate p and its standard deviation s = sqrt(p(1-p)/n)
- Remember the minimum observed value of p + s (the baseline)
- Warning level: p + s > p_min + s_min + warning_threshold × s_min
- Drift level: p + s > p_min + s_min + drift_threshold × s_min

**Key insight:** For a stable distribution, p + s should remain relatively constant.
A significant increase indicates the underlying distribution has changed.

**Parameters:**

- **warningThreshold:** Number of standard deviations for warning (default: 2)
- **driftThreshold:** Number of standard deviations for drift (default: 3)

**Advantages:**

- Simple and computationally efficient
- Two-stage detection (warning before drift)
- Well-suited for sudden drift

**Limitations:**

- Assumes binary errors (0 or 1)
- May be slow to detect gradual drift
- Can miss drift if error rate decreases

**Reference:** Gama et al. (2004). "Learning with Drift Detection"

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DDMDriftDetector(Double,Double,Int32)` | Creates a new DDM drift detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ErrorRate` | Gets the current error rate. |
| `IsInWarning` | Gets whether the detector is in the warning zone. |
| `MinimumPPlusS` | Gets the minimum p + s observed. |
| `SamplesInWarning` | Gets the number of samples since entering warning zone. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddObservation()` | Adds a new observation (typically a prediction error: 1 for wrong, 0 for correct). |
| `Reset` | Resets the detector to its initial state. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_errorSum` | Running sum of errors. |
| `_minPPlusS` | Minimum value of p + s observed. |
| `_pAtMin` | Error rate p at minimum p + s. |
| `_sAtMin` | Standard deviation s at minimum p + s. |
| `_warningStartCount` | Sample count at the warning point. |

