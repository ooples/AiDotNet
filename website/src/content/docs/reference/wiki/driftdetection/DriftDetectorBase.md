---
title: "DriftDetectorBase<T>"
description: "Base class for drift detectors providing common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.DriftDetection`

Base class for drift detectors providing common functionality.

## For Beginners

This base class provides the common infrastructure needed by
all drift detectors: tracking observations, managing state, and calculating statistics.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DriftDetectorBase` | Creates a new drift detector base. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DriftProbability` | Gets or sets the estimated drift probability. |
| `EstimatedMean` | Gets or sets the estimated mean of observations. |
| `IsInDrift` | Gets or sets whether drift has been detected. |
| `IsInWarning` | Gets or sets whether a warning has been triggered. |
| `MinimumObservations` | Minimum number of observations required before drift detection begins. |
| `ObservationCount` | Gets the total observation count. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddObservation()` | Adds a new observation to the detector. |
| `Reset` | Resets the detector to its initial state. |
| `ToDouble()` | Converts a value to double for calculations. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations for type T. |

