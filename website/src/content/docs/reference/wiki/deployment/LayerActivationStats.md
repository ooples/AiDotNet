---
title: "LayerActivationStats<T>"
description: "Activation statistics for a single layer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Optimization.Quantization.Calibration`

Activation statistics for a single layer.

## Properties

| Property | Summary |
|:-----|:--------|
| `LayerName` | Layer name or identifier. |
| `MaxAbsValue` | Maximum absolute activation value observed. |
| `MaxValue` | Maximum activation value observed. |
| `Mean` | Running mean of activation values. |
| `MinValue` | Minimum activation value observed. |
| `PerChannelMaxAbs` | Per-channel maximum absolute values (for per-channel quantization). |
| `SampleCount` | Number of samples accumulated. |
| `StandardDeviation` | Gets the standard deviation of activations. |
| `Variance` | Running variance of activation values (for standard deviation). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Update(Tensor<>)` | Updates statistics with a new batch of activations. |
| `UpdatePerChannelStats(Tensor<>)` | Updates per-channel maximum absolute values. |

