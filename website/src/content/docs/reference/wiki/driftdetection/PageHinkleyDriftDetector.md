---
title: "PageHinkleyDriftDetector<T>"
description: "Page-Hinkley test for concept drift detection using cumulative sum analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DriftDetection`

Page-Hinkley test for concept drift detection using cumulative sum analysis.

## For Beginners

The Page-Hinkley test is a sequential analysis technique that
monitors a cumulative sum of deviations from the mean. When the cumulative sum exceeds a
threshold, drift is detected. It's particularly good at detecting changes in the mean of a stream.

## How It Works

**How it works:**

- Calculate the running mean x̄ of observations
- For each new value x, update cumulative sum: m = m + (x - x̄ - δ)
- Track the minimum value M of the cumulative sum
- When (m - M) > λ, drift is detected

**Parameters:**

- **delta (δ):** Magnitude of allowed changes - helps filter noise (default: 0.005)
- **lambda (λ):** Detection threshold - higher = less sensitive (default: 50)

**Key insight:** The cumulative sum tracks deviations from expected behavior.
Normal fluctuations cancel out over time, but a true change causes monotonic increase.

**Advantages:**

- Based on solid statistical foundation (sequential analysis)
- Low computational cost
- Works well for detecting changes in mean
- Can be adapted for two-sided detection (increases and decreases)

**Limitations:**

- Requires tuning delta and lambda parameters
- Primarily detects shifts in mean, not other distribution changes

**Reference:** Page, E. S. (1954). "Continuous Inspection Schemes"

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PageHinkleyDriftDetector(Double,Double,Double,Boolean,Int32)` | Creates a new Page-Hinkley drift detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Delta` | Gets the magnitude threshold (delta). |
| `IsInWarning` | Gets whether the detector is in warning zone. |
| `IsTwoSided` | Gets whether two-sided detection is enabled. |
| `PageHinkleyStatisticDown` | Gets the current Page-Hinkley statistic for mean decrease detection (if two-sided). |
| `PageHinkleyStatisticUp` | Gets the current Page-Hinkley statistic for mean increase detection. |
| `Threshold` | Gets the detection threshold (lambda). |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddObservation()` | Adds a new observation to the detector. |
| `Reset` | Resets the detector to its initial state. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_cumulativeSumDown` | Cumulative sum statistic (for decrease detection, in two-sided mode). |
| `_cumulativeSumUp` | Cumulative sum statistic (for increase detection). |
| `_maxCumulativeSumDown` | Maximum value of cumulative sum (for decrease detection). |
| `_minCumulativeSumUp` | Minimum value of cumulative sum (for increase detection). |
| `_sum` | Running sum of observations. |

