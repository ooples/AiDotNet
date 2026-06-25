---
title: "EDDMDriftDetector<T>"
description: "Early Drift Detection Method (EDDM) for concept drift detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DriftDetection`

Early Drift Detection Method (EDDM) for concept drift detection.

## For Beginners

EDDM is an improvement over DDM that monitors the distance between
errors rather than just the error rate. This makes it better at detecting gradual drift because
even if the error rate stays similar, the spacing between errors may change.

## How It Works

**How it works:**

- Track the distance (number of samples) between consecutive errors
- Calculate the mean distance p' and standard deviation s'
- Monitor p' + 2s' and remember its maximum value
- Warning: when (p' + 2s') / (p'_max + 2s'_max) drops below warning threshold
- Drift: when (p' + 2s') / (p'_max + 2s'_max) drops below drift threshold

**Key insight:** When performance is good, errors are far apart (high mean distance).
When drift occurs, errors become more frequent and closer together (lower mean distance).

**Advantages over DDM:**

- Better at detecting gradual drift
- More sensitive to changes in error patterns
- Can detect drift even when error rate is low

**Parameters:**

- **warningThreshold:** Ratio for warning level (default: 0.95)
- **driftThreshold:** Ratio for drift level (default: 0.90)

**Reference:** Baena-García et al. (2006). "Early Drift Detection Method"

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EDDMDriftDetector(Double,Double,Int32,Int32)` | Creates a new EDDM drift detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AverageErrorDistance` | Gets the average distance between errors. |
| `CurrentRatio` | Gets the current ratio (lower means closer to drift). |
| `ErrorCount` | Gets the number of errors detected. |
| `MaximumPPrime2S` | Gets the maximum p' + 2s' observed. |
| `MinimumErrors` | Gets the minimum number of errors required before detection starts. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddObservation()` | Adds a new observation (typically a prediction error: 1 for wrong, 0 for correct). |
| `Reset` | Resets the detector to its initial state. |

