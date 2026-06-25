---
title: "PageHinkley<T>"
description: "Implements Page-Hinkley Test for concept drift detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DriftDetection`

Implements Page-Hinkley Test for concept drift detection.

## For Beginners

The Page-Hinkley test is a sequential analysis method that
detects changes in the mean of a process. Unlike DDM/EDDM which are designed for binary
errors, Page-Hinkley works with continuous values (like loss values or accuracy scores).

## How It Works

**How Page-Hinkley works:**

- Track the cumulative sum of deviations from the running mean
- Monitor the difference between cumulative sum and its minimum value
- Drift is detected when this difference exceeds a threshold (λ)

The test accumulates evidence of change over time, making it robust to noise.

**Key Parameters:**

- **λ (lambda):** Detection threshold - larger values = fewer false alarms but slower detection
- **α (alpha):** Magnitude of allowed change - helps ignore small fluctuations

**Variants:**

- **One-sided (decrease):** Detects when values decrease (e.g., accuracy dropping)
- **One-sided (increase):** Detects when values increase (e.g., loss increasing)
- **Two-sided:** Detects changes in either direction

**When to use:**

- Monitoring continuous metrics (loss, accuracy, scores)
- When you need to detect changes in mean value
- When you want control over detection sensitivity

**Reference:** Page, "Continuous Inspection Schemes" (1954)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PageHinkley(Double,Double,PageHinkley<>.DetectionMode,Int32)` | Creates a new Page-Hinkley drift detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets the tolerance parameter (alpha). |
| `Lambda` | Gets the detection threshold (lambda). |
| `Mode` | Gets the detection mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddObservation()` | Adds a new observation to the Page-Hinkley test. |
| `GetCumulativeSum` | Gets the current cumulative sum value. |
| `GetTestStatistic` | Gets the current test statistic value. |
| `Reset` |  |

