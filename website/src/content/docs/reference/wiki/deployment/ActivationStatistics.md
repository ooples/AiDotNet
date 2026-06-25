---
title: "ActivationStatistics<T>"
description: "Holds activation statistics collected during calibration forward passes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Optimization.Quantization.Calibration`

Holds activation statistics collected during calibration forward passes.

## For Beginners

When quantizing a model, we need to know the typical range of values
that flow through each layer during inference. This class stores those statistics, such as the
minimum, maximum, mean, and standard deviation of activations.

## How It Works

**Why This Matters:**

**Usage:** These statistics are collected by running calibration data through the model
and observing the intermediate activations at each layer.

## Properties

| Property | Summary |
|:-----|:--------|
| `CalibrationWarnings` | Warnings generated during calibration (e.g., high failure rate). |
| `GlobalActivationMagnitudes` | Global (flattened) activation magnitudes across all layers, normalized to [0,1]. |
| `GlobalMaxAbsActivations` | Global maximum absolute activation values per parameter position. |
| `IsFromRealForwardPasses` | Whether the statistics were collected from actual forward passes (true) or estimated from parameter magnitudes (false). |
| `LayerStats` | Per-layer activation statistics. |
| `SampleCount` | Number of calibration samples processed. |

