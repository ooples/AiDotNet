---
title: "LayerQuantizationInfo"
description: "Contains quantization information for a specific layer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Contains quantization information for a specific layer.

## Properties

| Property | Summary |
|:-----|:--------|
| `BitWidth` | Gets the bit width used for this layer's weights. |
| `IsQuantized` | Gets whether this layer was quantized. |
| `LayerName` | Gets the layer name. |
| `LayerType` | Gets the layer type (e.g., "Dense", "Conv2D"). |
| `ParameterCount` | Gets the number of parameters in this layer. |
| `QuantizationError` | Gets the quantization error (mean squared error) for this layer. |
| `Scale` | Gets the scale factor used for quantization. |
| `SkipReason` | Gets the reason if this layer was skipped during quantization. |
| `ZeroPoint` | Gets the zero point used for asymmetric quantization. |

