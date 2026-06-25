---
title: "IDriftDetector<T>"
description: "Defines the interface for concept drift detection."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the interface for concept drift detection.

## For Beginners

Concept drift happens when the patterns in your data change over time:

Examples of concept drift:

- Spam: New types of spam emails emerge
- Shopping: Customer preferences change seasonally
- Fraud: Fraudsters develop new techniques
- Weather: Climate patterns shift over years

Types of drift:

- Sudden drift: Abrupt change (e.g., policy change)
- Gradual drift: Slow transition between concepts
- Recurring drift: Patterns come back (e.g., seasonal)
- Incremental drift: Small, continuous changes

Without drift detection, your model's accuracy will silently degrade as
the data it was trained on becomes less relevant.

References:

- Gama et al. (2004). "Learning with Drift Detection"
- Bifet & Gavaldà (2007). "Learning from Time-Changing Data with Adaptive Windowing"

## How It Works

Concept drift detectors monitor the data stream for changes in the underlying
data distribution, signaling when the model may need to adapt or retrain.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsDriftDetected` | Checks if drift has been detected. |
| `IsWarning` | Checks if a warning (potential drift) has been detected. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetChangePoint` | Gets the estimated change point (when drift started). |
| `GetStatistics` | Gets the current detection statistics. |
| `Reset` | Resets the detector to its initial state. |
| `Update()` | Updates the detector with a new observation. |

