---
title: "QuantizationState"
description: "Stores quantization state for a layer during QAT."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Optimization.Quantization.Training`

Stores quantization state for a layer during QAT.

## Properties

| Property | Summary |
|:-----|:--------|
| `BitWidth` | Bit width for quantization. |
| `IsSymmetric` | Whether using symmetric quantization. |
| `LayerName` | Name of the layer. |
| `MaxValue` | Observed maximum value. |
| `MinValue` | Observed minimum value. |
| `QuantMax` | Maximum quantized value (e.g., 127 for INT8 symmetric). |
| `QuantMin` | Minimum quantized value (e.g., -128 for INT8 symmetric). |
| `SamplesObserved` | Number of samples observed for statistics. |
| `Scale` | Quantization scale factor. |
| `ZeroPoint` | Zero point for asymmetric quantization. |

