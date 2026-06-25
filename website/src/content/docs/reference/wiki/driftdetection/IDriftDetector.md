---
title: "IDriftDetector<T>"
description: "Interface for concept drift detectors in streaming/online learning scenarios."
section: "API Reference"
---

`Interfaces` · `AiDotNet.DriftDetection`

Interface for concept drift detectors in streaming/online learning scenarios.

## For Beginners

Concept drift occurs when the statistical properties of data
change over time. For example, customer behavior patterns may shift seasonally, or
spam email characteristics may evolve. Drift detectors monitor incoming data and alert
when significant changes are detected, signaling that a model may need retraining.

## How It Works

**Types of Drift:**

- **Sudden drift:** Abrupt change (e.g., system upgrade, policy change)
- **Gradual drift:** Slow transition between concepts over time
- **Incremental drift:** Small continuous changes that accumulate
- **Recurring drift:** Concepts that reappear periodically (e.g., seasonal)

**Common Approaches:**

- **Error-rate based:** Monitor classifier errors (DDM, EDDM)
- **Distribution-based:** Compare data distributions over windows (ADWIN)
- **Sequential analysis:** Cumulative sum tests (Page-Hinkley)

**When to use:** Any online learning scenario where data distribution may change:
fraud detection, recommendation systems, sensor monitoring, financial trading, etc.

## Properties

| Property | Summary |
|:-----|:--------|
| `DriftProbability` | Gets the estimated probability that drift has occurred. |
| `EstimatedMean` | Gets the current estimated mean of the stream. |
| `IsInDrift` | Gets whether drift has been detected. |
| `IsInWarning` | Gets whether a warning signal has been triggered (pre-drift indicator). |
| `ObservationCount` | Gets the total number of observations processed since the last reset. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddObservation()` | Adds a new observation to the drift detector. |
| `Reset` | Resets the detector to its initial state. |

