---
title: "EDDM<T>"
description: "Implements EDDM (Early Drift Detection Method) for concept drift detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DriftDetection`

Implements EDDM (Early Drift Detection Method) for concept drift detection.

## For Beginners

EDDM is an improvement over DDM that focuses on detecting
drift earlier, especially for gradual drift. Instead of monitoring just the error rate,
EDDM monitors the distance (number of samples) between consecutive errors.

## How It Works

**How EDDM works:**

- Track the distance between consecutive errors (not just error rate)
- Maintain running mean (p') and standard deviation (s') of these distances
- Monitor the ratio: current (p' + 2*s') / maximum (p' + 2*s')
- Warning: ratio drops below warning threshold (e.g., 0.95)
- Drift: ratio drops below drift threshold (e.g., 0.90)

**Why distance between errors?** When a model starts degrading:

- Errors become more frequent → distances decrease
- Error pattern becomes more consistent → standard deviation decreases
- Both effects contribute to earlier detection than pure error rate

**Advantages over DDM:**

- Earlier detection of gradual drift
- More stable in presence of noise
- Works well when errors are relatively rare

**Reference:** Baena-García et al., "Early Drift Detection Method" (2006)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EDDM(Double,Double,Int32,Int32)` | Creates a new EDDM drift detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DriftThreshold` | Gets the drift threshold (ratio of current to maximum p' + 2s'). |
| `ErrorCount` | Gets the total number of errors observed. |
| `MinimumErrors` | Gets the minimum number of errors required before detection starts. |
| `WarningThreshold` | Gets the warning threshold (ratio of current to maximum p' + 2s'). |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddObservation()` | Adds a new observation (error indicator) to the detector. |
| `GetCurrentRatio` | Gets the current ratio of performance to maximum observed. |
| `GetDistanceStd` | Gets the current standard deviation of distances between errors. |
| `GetMeanDistance` | Gets the current mean distance between errors. |
| `Reset` |  |

