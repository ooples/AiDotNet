---
title: "OutlierRemovalAdapter<T, TInput, TOutput>"
description: "Adapts any `IAnomalyDetector` to the legacy `IOutlierRemoval` interface."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection`

Adapts any `IAnomalyDetector` to the legacy `IOutlierRemoval` interface.

## For Beginners

This adapter class allows you to use any of the new anomaly detection
algorithms (like Isolation Forest, Local Outlier Factor, etc.) with the existing data
preprocessing pipeline that expects the older `IOutlierRemoval` interface.

## How It Works

**Usage Example:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OutlierRemovalAdapter(IAnomalyDetector<>)` | Creates a new adapter that wraps an anomaly detector for use with the legacy outlier removal interface. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Detector` | Gets the underlying anomaly detector. |

## Methods

| Method | Summary |
|:-----|:--------|
| `RemoveOutliers(,)` | Removes outliers from the input data using the configured anomaly detector. |

